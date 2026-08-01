"""Streaming TorNet file- and frame-manifest construction."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


MANIFEST_SCHEMA_VERSION = "1.0.0"

EXPECTED_SPLITS = frozenset({"train", "test"})
EXPECTED_CATEGORIES = frozenset({"NUL", "WRN", "TOR"})

PRIMARY_RADAR_VARIABLES = (
    "DBZ",
    "KDP",
    "RHOHV",
    "VEL",
    "WIDTH",
    "ZDR",
)

REQUIRED_VARIABLES = frozenset(
    {
        *PRIMARY_RADAR_VARIABLES,
        "frame_labels",
        "range_folded_mask",
        "elevation",
        "nyquist_velocity",
        "time",
    }
)

FILENAME_PATTERN = re.compile(
    r"^(?P<category>[A-Z]+)_"
    r"(?P<date>\d{6})_"
    r"(?P<time>\d{6})_"
    r"(?P<radar_site>[A-Z0-9]{4})_"
    r"(?P<event_token>[^_]+)_"
    r"(?P<scit_id>[^.]+)\.nc$"
)


@dataclass(frozen=True)
class ParsedMember:
    """Structured metadata parsed from one TorNet archive member."""

    split: str
    year: int
    category: str
    filename: str
    filename_timestamp_utc: str
    radar_site: str
    filename_event_token: str
    filename_scit_id: str


@dataclass(frozen=True)
class ManifestBuildResult:
    """Results of streaming one TorNet annual archive."""

    archive_path: str
    archive_size_bytes: int
    year: int
    netcdf_member_count: int
    file_manifest: pd.DataFrame
    frame_manifest: pd.DataFrame
    schema_summary: pd.DataFrame
    errors: pd.DataFrame


@dataclass(frozen=True)
class ManifestValidation:
    """Validation tables for a manifest build."""

    checks: pd.DataFrame
    event_split_overlap: pd.DataFrame
    episode_split_overlap: pd.DataFrame
    category_frame_summary: pd.DataFrame

    @property
    def all_required_passed(self) -> bool:
        required = self.checks.loc[self.checks["required"]]
        return bool(required["passed"].all())

    def assert_valid(self) -> None:
        failed = self.checks.loc[
            self.checks["required"] & ~self.checks["passed"]
        ]

        if failed.empty:
            return

        details = "\n".join(
            f"- {row.check}: observed={row.observed}; "
            f"expected={row.expected}"
            for row in failed.itertuples(index=False)
        )

        raise AssertionError(
            "Required TorNet manifest validations failed:\n"
            f"{details}"
        )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, np.generic):
        return _json_safe(value.item())

    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, Path):
        return str(value)

    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
    )


def _display_value(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def _normalize_identifier(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, float) and not np.isfinite(value):
        return None

    text = str(value).strip()
    return text or None


def _optional_text(value: Any) -> str | None:
    return _normalize_identifier(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None

    return parsed if np.isfinite(parsed) else None


def _timestamp_to_utc_iso(value: Any) -> str | None:
    timestamp = pd.Timestamp(value)

    if pd.isna(timestamp):
        return None

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp.isoformat()


def parse_tornet_member(
    member_name: str,
    *,
    expected_year: int,
) -> ParsedMember:
    """Parse and validate one TorNet annual-archive member path."""

    parts = tuple(
        part
        for part in PurePosixPath(member_name).parts
        if part not in {"", "."}
    )

    if len(parts) != 3:
        raise ValueError(
            "Expected path '<split>/<year>/<filename>.nc'; "
            f"found {member_name!r}"
        )

    split, year_text, filename = parts

    if split not in EXPECTED_SPLITS:
        raise ValueError(
            f"Unexpected split {split!r} in {member_name!r}"
        )

    try:
        year = int(year_text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid year {year_text!r} in {member_name!r}"
        ) from exc

    if year != expected_year:
        raise ValueError(
            f"Expected year {expected_year}; found {year} "
            f"in {member_name!r}"
        )

    match = FILENAME_PATTERN.fullmatch(filename)

    if match is None:
        raise ValueError(
            f"Filename does not match the TorNet pattern: {filename!r}"
        )

    category = match.group("category")

    if category not in EXPECTED_CATEGORIES:
        raise ValueError(
            f"Unexpected category {category!r} in {member_name!r}"
        )

    timestamp = pd.to_datetime(
        match.group("date") + match.group("time"),
        format="%y%m%d%H%M%S",
        utc=True,
    )

    return ParsedMember(
        split=split,
        year=year,
        category=category,
        filename=filename,
        filename_timestamp_utc=timestamp.isoformat(),
        radar_site=match.group("radar_site"),
        filename_event_token=match.group("event_token"),
        filename_scit_id=match.group("scit_id"),
    )


def _copy_member_with_sha256(
    source,
    destination: Path,
) -> str:
    digest = hashlib.sha256()

    with destination.open("wb") as output:
        while True:
            chunk = source.read(1024 * 1024)

            if not chunk:
                break

            output.write(chunk)
            digest.update(chunk)

    return digest.hexdigest()


def _schema_payload(raw_dataset: xr.Dataset) -> dict[str, Any]:
    variables: dict[str, Any] = {}

    for variable_name in sorted(raw_dataset.variables):
        variable = raw_dataset[variable_name]

        variables[variable_name] = {
            "dimensions": list(variable.dims),
            "shape": [int(size) for size in variable.shape],
            "dtype": str(variable.dtype),
            "fill_value": _json_safe(
                variable.attrs.get("_FillValue")
            ),
        }

    return {
        "dimensions": {
            name: int(size)
            for name, size in sorted(raw_dataset.sizes.items())
        },
        "data_variables": sorted(raw_dataset.data_vars),
        "coordinates": sorted(raw_dataset.coords),
        "variables": variables,
    }


def _inspect_netcdf(
    sample_path: Path,
    *,
    archive_name: str,
    archive_index: int,
    archive_member: str,
    member_size_bytes: int,
    member_sha256: str,
    parsed: ParsedMember,
) -> tuple[dict[str, Any], list[dict[str, Any]], str, str]:
    with (
        xr.open_dataset(
            sample_path,
            engine="netcdf4",
            decode_cf=False,
            mask_and_scale=False,
            decode_times=False,
        ) as raw_dataset,
        xr.open_dataset(
            sample_path,
            engine="netcdf4",
            decode_cf=True,
            mask_and_scale=True,
            decode_times=True,
        ) as decoded_dataset,
    ):
        missing_variables = (
            REQUIRED_VARIABLES - set(decoded_dataset.variables)
        )

        if missing_variables:
            raise ValueError(
                "Missing required variables: "
                f"{sorted(missing_variables)}"
            )

        category_attribute = _normalize_identifier(
            decoded_dataset.attrs.get("category")
        )
        event_id = _normalize_identifier(
            decoded_dataset.attrs.get("event_id")
        )
        episode_id = _normalize_identifier(
            decoded_dataset.attrs.get("episode_id")
        )
        scit_id = _normalize_identifier(
            decoded_dataset.attrs.get("scit_id")
        )

        labels = np.asarray(
            decoded_dataset["frame_labels"].values
        )

        if labels.ndim != 1:
            raise ValueError(
                f"frame_labels must be one-dimensional; "
                f"found {labels.shape}"
            )

        labels = labels.astype(np.int64, copy=False)

        unique_labels = set(np.unique(labels).tolist())

        if not unique_labels.issubset({0, 1}):
            raise ValueError(
                "frame_labels contains values other than 0 and 1: "
                f"{sorted(unique_labels)}"
            )

        frame_times = np.asarray(decoded_dataset["time"].values)

        if frame_times.shape != labels.shape:
            raise ValueError(
                "time and frame_labels shapes differ: "
                f"time={frame_times.shape}, "
                f"labels={labels.shape}"
            )

        elevation = np.asarray(
            decoded_dataset["elevation"].values
        ).reshape(-1)

        nyquist_variable = decoded_dataset["nyquist_velocity"]

        if "time" not in nyquist_variable.dims:
            raise ValueError(
                "nyquist_velocity does not contain a time dimension"
            )

        range_folded_variable = decoded_dataset[
            "range_folded_mask"
        ]

        if "time" not in range_folded_variable.dims:
            raise ValueError(
                "range_folded_mask does not contain a time dimension"
            )

        for variable_name in PRIMARY_RADAR_VARIABLES:
            if "time" not in decoded_dataset[variable_name].dims:
                raise ValueError(
                    f"{variable_name} does not contain a time dimension"
                )

        payload = _schema_payload(raw_dataset)
        payload_json = _canonical_json(payload)
        schema_fingerprint = _sha256_text(payload_json)

        file_id = _sha256_text(
            f"{archive_name}\n{archive_member}"
        )

        frame_time_values = [
            _timestamp_to_utc_iso(value)
            for value in frame_times
        ]

        if any(value is None for value in frame_time_values):
            raise ValueError(
                "One or more frame timestamps could not be decoded"
            )

        frame_rows: list[dict[str, Any]] = []

        for frame_index, frame_label in enumerate(labels):
            finite_fractions: dict[str, float] = {}

            for variable_name in PRIMARY_RADAR_VARIABLES:
                values = np.asarray(
                    decoded_dataset[variable_name]
                    .isel(time=frame_index)
                    .values
                )

                finite_fractions[variable_name] = float(
                    np.isfinite(values).mean()
                )

            range_folded_values = np.asarray(
                range_folded_variable
                .isel(time=frame_index)
                .values
            )

            nyquist_values = np.asarray(
                nyquist_variable
                .isel(time=frame_index)
                .values
            ).reshape(-1)

            frame_id = f"{file_id}:{frame_index}"

            frame_rows.append(
                {
                    "manifest_schema_version": (
                        MANIFEST_SCHEMA_VERSION
                    ),
                    "frame_id": frame_id,
                    "file_id": file_id,
                    "archive_member": archive_member,
                    "split": parsed.split,
                    "year": parsed.year,
                    "category": parsed.category,
                    "radar_site": parsed.radar_site,
                    "event_group_id": event_id,
                    "event_id": event_id,
                    "episode_id": episode_id,
                    "scit_id": scit_id,
                    "ef_number": _optional_float(
                        decoded_dataset.attrs.get("ef_number")
                    ),
                    "frame_index": frame_index,
                    "frame_time_utc": frame_time_values[
                        frame_index
                    ],
                    "frame_label": int(frame_label),
                    "elevation_degrees_json": _canonical_json(
                        elevation.tolist()
                    ),
                    "nyquist_velocity_mps_json": (
                        _canonical_json(
                            nyquist_values.tolist()
                        )
                    ),
                    "dbz_finite_fraction": finite_fractions["DBZ"],
                    "kdp_finite_fraction": finite_fractions["KDP"],
                    "rhohv_finite_fraction": (
                        finite_fractions["RHOHV"]
                    ),
                    "vel_finite_fraction": finite_fractions["VEL"],
                    "width_finite_fraction": (
                        finite_fractions["WIDTH"]
                    ),
                    "zdr_finite_fraction": finite_fractions["ZDR"],
                    "mean_primary_finite_fraction": float(
                        np.mean(list(finite_fractions.values()))
                    ),
                    "range_folded_fraction": float(
                        np.mean(range_folded_values != 0)
                    ),
                }
            )

        file_row = {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "file_id": file_id,
            "archive_name": archive_name,
            "archive_index": archive_index,
            "archive_member": archive_member,
            "filename": parsed.filename,
            "split": parsed.split,
            "year": parsed.year,
            "category": parsed.category,
            "category_attribute": category_attribute,
            "category_matches_attribute": (
                category_attribute == parsed.category
            ),
            "filename_timestamp_utc": (
                parsed.filename_timestamp_utc
            ),
            "radar_site": parsed.radar_site,
            "filename_event_token": (
                parsed.filename_event_token
            ),
            "filename_scit_id": parsed.filename_scit_id,
            "scit_id": scit_id,
            "scit_id_matches_filename": (
                scit_id == parsed.filename_scit_id
            ),
            "event_group_id": event_id,
            "event_id": event_id,
            "episode_id": episode_id,
            "ef_number": _optional_float(
                decoded_dataset.attrs.get("ef_number")
            ),
            "tornado_start_time": _optional_text(
                decoded_dataset.attrs.get("tornado_start_time")
            ),
            "tornado_end_time": _optional_text(
                decoded_dataset.attrs.get("tornado_end_time")
            ),
            "site_lat": _optional_float(
                decoded_dataset.attrs.get("site_lat")
            ),
            "site_lon": _optional_float(
                decoded_dataset.attrs.get("site_lon")
            ),
            "storm_event_url": _optional_text(
                decoded_dataset.attrs.get("storm_event_url")
            ),
            "member_size_bytes": member_size_bytes,
            "member_sha256": member_sha256,
            "time_count": int(decoded_dataset.sizes.get("time", 0)),
            "sweep_count": int(
                decoded_dataset.sizes.get("sweep", 0)
            ),
            "azimuth_count": int(
                decoded_dataset.sizes.get("azimuth", 0)
            ),
            "range_count": int(
                decoded_dataset.sizes.get("range", 0)
            ),
            "lims_count": int(
                decoded_dataset.sizes.get("lims", 0)
            ),
            "frame_count": int(len(labels)),
            "positive_frame_count": int(labels.sum()),
            "has_positive_frame": bool(labels.sum() > 0),
            "has_mixed_frame_labels": bool(
                len(np.unique(labels)) > 1
            ),
            "frame_labels_json": _canonical_json(
                labels.tolist()
            ),
            "frame_time_start_utc": frame_time_values[0],
            "frame_time_end_utc": frame_time_values[-1],
            "elevation_degrees_json": _canonical_json(
                elevation.tolist()
            ),
            "data_variable_names_json": _canonical_json(
                sorted(decoded_dataset.data_vars)
            ),
            "coordinate_names_json": _canonical_json(
                sorted(decoded_dataset.coords)
            ),
            "schema_fingerprint": schema_fingerprint,
        }

        return (
            file_row,
            frame_rows,
            schema_fingerprint,
            payload_json,
        )


def _build_schema_summary(
    file_manifest: pd.DataFrame,
    schema_payloads: Mapping[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if file_manifest.empty:
        return pd.DataFrame(
            columns=[
                "schema_fingerprint",
                "file_count",
                "splits_json",
                "categories_json",
                "first_archive_member",
                "schema_payload_json",
            ]
        )

    for fingerprint, group in file_manifest.groupby(
        "schema_fingerprint",
        sort=True,
    ):
        rows.append(
            {
                "schema_fingerprint": fingerprint,
                "file_count": int(len(group)),
                "splits_json": _canonical_json(
                    sorted(group["split"].unique())
                ),
                "categories_json": _canonical_json(
                    sorted(group["category"].unique())
                ),
                "first_archive_member": (
                    group.sort_values("archive_index")
                    .iloc[0]["archive_member"]
                ),
                "schema_payload_json": schema_payloads[
                    fingerprint
                ],
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["file_count", "schema_fingerprint"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def build_archive_manifests(
    archive_path: str | Path,
    *,
    expected_year: int,
    working_directory: str | Path | None = None,
    progress_every: int = 250,
    progress: Callable[[str], None] | None = print,
) -> ManifestBuildResult:
    """Stream one annual TorNet archive into file and frame manifests."""

    archive_path = Path(archive_path)

    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    if progress_every <= 0:
        raise ValueError("progress_every must be positive")

    if working_directory is not None:
        working_directory = Path(working_directory)
        working_directory.mkdir(parents=True, exist_ok=True)

    file_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    schema_payloads: dict[str, str] = {}
    netcdf_member_count = 0

    temporary_directory_parent = (
        str(working_directory)
        if working_directory is not None
        else None
    )

    with tempfile.TemporaryDirectory(
        prefix="tornet-manifest-",
        dir=temporary_directory_parent,
    ) as temporary_directory:
        sample_path = Path(temporary_directory) / "current.nc"

        with tarfile.open(archive_path, mode="r|gz") as archive:
            for archive_index, member in enumerate(archive):
                if not member.isfile():
                    continue

                if not member.name.lower().endswith(".nc"):
                    continue

                netcdf_member_count += 1

                try:
                    parsed = parse_tornet_member(
                        member.name,
                        expected_year=expected_year,
                    )

                    source = archive.extractfile(member)

                    if source is None:
                        raise RuntimeError(
                            "tarfile.extractfile returned None"
                        )

                    with source:
                        member_sha256 = _copy_member_with_sha256(
                            source,
                            sample_path,
                        )

                    extracted_size = sample_path.stat().st_size

                    if extracted_size != member.size:
                        raise IOError(
                            "Extracted size does not match archive "
                            f"metadata: expected={member.size}, "
                            f"actual={extracted_size}"
                        )

                    (
                        file_row,
                        member_frame_rows,
                        schema_fingerprint,
                        schema_payload_json,
                    ) = _inspect_netcdf(
                        sample_path,
                        archive_name=archive_path.name,
                        archive_index=archive_index,
                        archive_member=member.name,
                        member_size_bytes=member.size,
                        member_sha256=member_sha256,
                        parsed=parsed,
                    )

                    file_rows.append(file_row)
                    frame_rows.extend(member_frame_rows)
                    schema_payloads.setdefault(
                        schema_fingerprint,
                        schema_payload_json,
                    )

                except Exception as exc:
                    error_rows.append(
                        {
                            "archive_index": archive_index,
                            "archive_member": member.name,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    )

                finally:
                    sample_path.unlink(missing_ok=True)

                if (
                    progress is not None
                    and netcdf_member_count % progress_every == 0
                ):
                    progress(
                        f"Scanned {netcdf_member_count:,} NetCDF "
                        f"members; built {len(file_rows):,} file rows "
                        f"and {len(frame_rows):,} frame rows; "
                        f"errors={len(error_rows):,}"
                    )

    file_manifest = pd.DataFrame(file_rows)
    frame_manifest = pd.DataFrame(frame_rows)
    errors = pd.DataFrame(
        error_rows,
        columns=[
            "archive_index",
            "archive_member",
            "error_type",
            "error_message",
        ],
    )

    if not file_manifest.empty:
        file_manifest = (
            file_manifest
            .sort_values("archive_index")
            .reset_index(drop=True)
        )

    if not frame_manifest.empty:
        frame_manifest = (
            frame_manifest
            .sort_values(["archive_member", "frame_index"])
            .reset_index(drop=True)
        )

    schema_summary = _build_schema_summary(
        file_manifest,
        schema_payloads,
    )

    if progress is not None:
        progress(
            f"Completed {archive_path.name}: "
            f"{netcdf_member_count:,} NetCDF members, "
            f"{len(file_manifest):,} file rows, "
            f"{len(frame_manifest):,} frame rows, "
            f"{len(schema_summary):,} schema variants, "
            f"{len(errors):,} errors"
        )

    return ManifestBuildResult(
        archive_path=str(archive_path),
        archive_size_bytes=archive_path.stat().st_size,
        year=expected_year,
        netcdf_member_count=netcdf_member_count,
        file_manifest=file_manifest,
        frame_manifest=frame_manifest,
        schema_summary=schema_summary,
        errors=errors,
    )


def _split_overlap(
    file_manifest: pd.DataFrame,
    key: str,
) -> pd.DataFrame:
    columns = [
        key,
        "splits_json",
        "file_count",
        "archive_members_json",
    ]

    if file_manifest.empty or key not in file_manifest:
        return pd.DataFrame(columns=columns)

    valid = file_manifest.loc[
        file_manifest[key].notna()
        & file_manifest[key].astype(str).str.len().gt(0)
    ]

    rows: list[dict[str, Any]] = []

    for value, group in valid.groupby(key, sort=True):
        splits = sorted(group["split"].unique())

        if len(splits) <= 1:
            continue

        rows.append(
            {
                key: value,
                "splits_json": _canonical_json(splits),
                "file_count": int(len(group)),
                "archive_members_json": _canonical_json(
                    sorted(group["archive_member"].tolist())
                ),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def _category_frame_summary(
    file_manifest: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "split",
        "category",
        "file_count",
        "frame_count",
        "positive_frame_count",
        "files_with_positive_frames",
        "files_with_mixed_frame_labels",
        "positive_frame_prevalence",
    ]

    if file_manifest.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []

    for (split, category), group in file_manifest.groupby(
        ["split", "category"],
        sort=True,
    ):
        frame_count = int(group["frame_count"].sum())
        positive_frame_count = int(
            group["positive_frame_count"].sum()
        )

        rows.append(
            {
                "split": split,
                "category": category,
                "file_count": int(len(group)),
                "frame_count": frame_count,
                "positive_frame_count": positive_frame_count,
                "files_with_positive_frames": int(
                    group["has_positive_frame"].sum()
                ),
                "files_with_mixed_frame_labels": int(
                    group["has_mixed_frame_labels"].sum()
                ),
                "positive_frame_prevalence": (
                    positive_frame_count / frame_count
                    if frame_count
                    else None
                ),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def validate_manifests(
    result: ManifestBuildResult,
    *,
    expected_file_count: int | None = None,
    expected_frame_count: int | None = None,
    expected_frames_per_file: int | None = None,
    expected_split_category_counts: (
        Mapping[tuple[str, str], int] | None
    ) = None,
    expected_dimensions: Mapping[str, int] | None = None,
) -> ManifestValidation:
    """Validate manifest integrity and official-split isolation."""

    file_manifest = result.file_manifest
    frame_manifest = result.frame_manifest

    checks: list[dict[str, Any]] = []

    def record(
        check: str,
        *,
        required: bool,
        passed: bool,
        observed: Any,
        expected: Any,
        detail: str = "",
    ) -> None:
        checks.append(
            {
                "check": check,
                "required": required,
                "passed": bool(passed),
                "observed": _display_value(observed),
                "expected": _display_value(expected),
                "detail": detail,
            }
        )

    record(
        "build_errors",
        required=True,
        passed=result.errors.empty,
        observed=len(result.errors),
        expected=0,
    )

    record(
        "file_row_count",
        required=expected_file_count is not None,
        passed=(
            expected_file_count is None
            or len(file_manifest) == expected_file_count
        ),
        observed=len(file_manifest),
        expected=expected_file_count,
    )

    record(
        "frame_row_count",
        required=expected_frame_count is not None,
        passed=(
            expected_frame_count is None
            or len(frame_manifest) == expected_frame_count
        ),
        observed=len(frame_manifest),
        expected=expected_frame_count,
    )

    record(
        "unique_archive_members",
        required=True,
        passed=(
            not file_manifest.empty
            and file_manifest["archive_member"].is_unique
        ),
        observed=(
            int(file_manifest["archive_member"].nunique())
            if not file_manifest.empty
            else 0
        ),
        expected=len(file_manifest),
    )

    record(
        "unique_file_ids",
        required=True,
        passed=(
            not file_manifest.empty
            and file_manifest["file_id"].is_unique
        ),
        observed=(
            int(file_manifest["file_id"].nunique())
            if not file_manifest.empty
            else 0
        ),
        expected=len(file_manifest),
    )

    record(
        "unique_frame_ids",
        required=True,
        passed=(
            not frame_manifest.empty
            and frame_manifest["frame_id"].is_unique
        ),
        observed=(
            int(frame_manifest["frame_id"].nunique())
            if not frame_manifest.empty
            else 0
        ),
        expected=len(frame_manifest),
    )

    if file_manifest.empty or frame_manifest.empty:
        frame_count_mismatches = len(file_manifest)
        label_sum_mismatches = len(file_manifest)
        frame_index_mismatches = len(file_manifest)
    else:
        expected_counts = file_manifest.set_index(
            "file_id"
        )["frame_count"]

        actual_counts = (
            frame_manifest.groupby("file_id")
            .size()
            .reindex(expected_counts.index, fill_value=-1)
        )

        frame_count_mismatches = int(
            (actual_counts != expected_counts).sum()
        )

        expected_positive = file_manifest.set_index(
            "file_id"
        )["positive_frame_count"]

        actual_positive = (
            frame_manifest.groupby("file_id")["frame_label"]
            .sum()
            .reindex(expected_positive.index, fill_value=-1)
        )

        label_sum_mismatches = int(
            (actual_positive != expected_positive).sum()
        )

        frame_index_mismatches = 0

        for file_id, group in frame_manifest.groupby(
            "file_id",
            sort=False,
        ):
            observed_indices = sorted(
                group["frame_index"].astype(int).tolist()
            )
            expected_indices = list(range(len(group)))

            if observed_indices != expected_indices:
                frame_index_mismatches += 1

    record(
        "frame_rows_match_file_frame_counts",
        required=True,
        passed=frame_count_mismatches == 0,
        observed=frame_count_mismatches,
        expected=0,
    )

    record(
        "frame_label_sums_match_file_manifest",
        required=True,
        passed=label_sum_mismatches == 0,
        observed=label_sum_mismatches,
        expected=0,
    )

    record(
        "frame_indices_are_contiguous",
        required=True,
        passed=frame_index_mismatches == 0,
        observed=frame_index_mismatches,
        expected=0,
    )

    if expected_frames_per_file is not None:
        mismatches = int(
            (
                file_manifest["frame_count"]
                != expected_frames_per_file
            ).sum()
        )

        record(
            "expected_frames_per_file",
            required=True,
            passed=mismatches == 0,
            observed=mismatches,
            expected=0,
            detail=(
                f"Expected {expected_frames_per_file} frames "
                "for every file"
            ),
        )

    record(
        "event_ids_present",
        required=True,
        passed=(
            not file_manifest.empty
            and file_manifest["event_group_id"].notna().all()
            and file_manifest[
                "event_group_id"
            ].astype(str).str.len().gt(0).all()
        ),
        observed=int(
            file_manifest["event_group_id"].isna().sum()
            if not file_manifest.empty
            else 0
        ),
        expected=0,
    )

    record(
        "path_and_attribute_categories_match",
        required=True,
        passed=(
            not file_manifest.empty
            and file_manifest[
                "category_matches_attribute"
            ].all()
        ),
        observed=int(
            (
                ~file_manifest["category_matches_attribute"]
            ).sum()
            if not file_manifest.empty
            else 0
        ),
        expected=0,
    )

    record(
        "filename_and_attribute_scit_ids_match",
        required=True,
        passed=(
            not file_manifest.empty
            and file_manifest[
                "scit_id_matches_filename"
            ].all()
        ),
        observed=int(
            (
                ~file_manifest["scit_id_matches_filename"]
            ).sum()
            if not file_manifest.empty
            else 0
        ),
        expected=0,
    )

    event_split_overlap = _split_overlap(
        file_manifest,
        "event_group_id",
    )

    episode_split_overlap = _split_overlap(
        file_manifest,
        "episode_id",
    )

    record(
        "event_groups_do_not_cross_official_splits",
        required=True,
        passed=event_split_overlap.empty,
        observed=len(event_split_overlap),
        expected=0,
    )

    record(
        "episode_groups_crossing_official_splits",
        required=False,
        passed=episode_split_overlap.empty,
        observed=len(episode_split_overlap),
        expected=0,
        detail=(
            "Informational until episode-level grouping semantics "
            "are reviewed"
        ),
    )

    if expected_split_category_counts is not None:
        actual_counts = {
            (row.split, row.category): int(row.file_count)
            for row in (
                file_manifest.groupby(
                    ["split", "category"],
                    sort=True,
                )
                .size()
                .rename("file_count")
                .reset_index()
                .itertuples(index=False)
            )
        }

        normalized_expected = {
            (split, category): int(count)
            for (split, category), count
            in expected_split_category_counts.items()
        }

        record(
            "split_category_file_counts",
            required=True,
            passed=actual_counts == normalized_expected,
            observed={
                f"{split}/{category}": count
                for (split, category), count
                in sorted(actual_counts.items())
            },
            expected={
                f"{split}/{category}": count
                for (split, category), count
                in sorted(normalized_expected.items())
            },
        )

    if expected_dimensions is not None:
        dimension_mismatches: dict[str, int] = {}

        for dimension_name, expected_size in (
            expected_dimensions.items()
        ):
            column = f"{dimension_name}_count"

            if column not in file_manifest:
                dimension_mismatches[
                    dimension_name
                ] = len(file_manifest)
                continue

            dimension_mismatches[dimension_name] = int(
                (
                    file_manifest[column] != expected_size
                ).sum()
            )

        record(
            "expected_dimensions",
            required=True,
            passed=all(
                count == 0
                for count in dimension_mismatches.values()
            ),
            observed=dimension_mismatches,
            expected={
                name: 0
                for name in expected_dimensions
            },
        )

    record(
        "schema_variant_count",
        required=False,
        passed=True,
        observed=len(result.schema_summary),
        expected="explicitly counted",
    )

    category_frame_summary = _category_frame_summary(
        file_manifest
    )

    return ManifestValidation(
        checks=pd.DataFrame(checks),
        event_split_overlap=event_split_overlap,
        episode_split_overlap=episode_split_overlap,
        category_frame_summary=category_frame_summary,
    )


def write_manifest_artifacts(
    result: ManifestBuildResult,
    validation: ManifestValidation,
    output_directory: str | Path,
    *,
    overwrite: bool = False,
    allow_invalid: bool = False,
) -> dict[str, str]:
    """Write manifest artifacts with a terminal status marker.

    Fully valid manifests receive ``_SUCCESS.json``.

    When ``allow_invalid`` is true, manifests that fail required
    validation may still be written for auditing. Those artifacts
    receive ``_INVALID.json`` and never ``_SUCCESS.json``.
    """

    all_required_passed = (
        validation.all_required_passed
    )

    if not all_required_passed and not allow_invalid:
        validation.assert_valid()

    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    status = (
        "valid"
        if all_required_passed
        else "invalid"
    )
    marker_name = (
        "_SUCCESS.json"
        if all_required_passed
        else "_INVALID.json"
    )
    marker_path = output_directory / marker_name

    terminal_marker_names = [
        "_SUCCESS.json",
        "_INVALID.json",
    ]

    artifact_names = [
        "file_manifest.parquet",
        "frame_manifest.parquet",
        "schema_summary.csv",
        "build_errors.csv",
        "validation_checks.csv",
        "event_split_overlap.csv",
        "episode_split_overlap.csv",
        "category_frame_summary.csv",
        "manifest_summary.json",
    ]

    existing = [
        output_directory / name
        for name in [
            *artifact_names,
            *terminal_marker_names,
        ]
        if (output_directory / name).exists()
    ]

    if existing and not overwrite:
        raise FileExistsError(
            "Manifest output already exists: "
            + ", ".join(
                str(path)
                for path in existing
            )
        )

    for terminal_marker_name in terminal_marker_names:
        (
            output_directory
            / terminal_marker_name
        ).unlink(missing_ok=True)

    failed_required = validation.checks.loc[
        validation.checks["required"]
        & ~validation.checks["passed"]
    ]

    failed_required_checks = [
        {
            "check": str(row.check),
            "observed": _json_safe(row.observed),
            "expected": _json_safe(row.expected),
            "detail": str(row.detail or ""),
        }
        for row in failed_required.itertuples(
            index=False
        )
    ]

    with tempfile.TemporaryDirectory(
        prefix="tornet-manifest-output-"
    ) as staging_directory:
        staging_directory = Path(
            staging_directory
        )

        result.file_manifest.to_parquet(
            staging_directory
            / "file_manifest.parquet",
            index=False,
        )
        result.frame_manifest.to_parquet(
            staging_directory
            / "frame_manifest.parquet",
            index=False,
        )
        result.schema_summary.to_csv(
            staging_directory
            / "schema_summary.csv",
            index=False,
        )
        result.errors.to_csv(
            staging_directory
            / "build_errors.csv",
            index=False,
        )
        validation.checks.to_csv(
            staging_directory
            / "validation_checks.csv",
            index=False,
        )
        validation.event_split_overlap.to_csv(
            staging_directory
            / "event_split_overlap.csv",
            index=False,
        )
        validation.episode_split_overlap.to_csv(
            staging_directory
            / "episode_split_overlap.csv",
            index=False,
        )
        validation.category_frame_summary.to_csv(
            staging_directory
            / "category_frame_summary.csv",
            index=False,
        )

        summary = {
            "manifest_schema_version": (
                MANIFEST_SCHEMA_VERSION
            ),
            "built_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "artifact_status": status,
            "archive_path": result.archive_path,
            "archive_size_bytes": (
                result.archive_size_bytes
            ),
            "year": result.year,
            "netcdf_member_count": (
                result.netcdf_member_count
            ),
            "file_row_count": len(
                result.file_manifest
            ),
            "frame_row_count": len(
                result.frame_manifest
            ),
            "schema_variant_count": len(
                result.schema_summary
            ),
            "build_error_count": len(
                result.errors
            ),
            "all_required_validations_passed": (
                all_required_passed
            ),
            "failed_required_validation_count": (
                len(failed_required_checks)
            ),
            "failed_required_checks": (
                failed_required_checks
            ),
        }

        (
            staging_directory
            / "manifest_summary.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        artifact_hashes: dict[str, str] = {}

        for artifact_name in artifact_names:
            source = (
                staging_directory
                / artifact_name
            )
            destination = (
                output_directory
                / artifact_name
            )

            shutil.copy2(
                source,
                destination,
            )

            source_hash = _sha256_path(source)
            destination_hash = _sha256_path(
                destination
            )

            if source_hash != destination_hash:
                raise IOError(
                    "Artifact hash mismatch after copy: "
                    f"{artifact_name}"
                )

            artifact_hashes[
                artifact_name
            ] = destination_hash

    marker_payload = {
        "manifest_schema_version": (
            MANIFEST_SCHEMA_VERSION
        ),
        "completed_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": status,
        "all_required_validations_passed": (
            all_required_passed
        ),
        "failed_required_checks": (
            failed_required_checks
        ),
        "artifact_sha256": artifact_hashes,
    }

    marker_path.write_text(
        json.dumps(
            marker_payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    return {
        name: str(output_directory / name)
        for name in [
            *artifact_names,
            marker_name,
        ]
    }
