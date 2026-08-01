"""Derived modeling manifests built from raw TorNet audits."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from tornado_detection.data.manifest import (
    MANIFEST_SCHEMA_VERSION,
    ManifestBuildResult,
    ManifestValidation,
    validate_manifests,
    write_manifest_artifacts,
)


@dataclass(frozen=True)
class ModelingExclusion:
    """One explicit raw-manifest exclusion."""

    archive_member: str
    reason: str
    resolution: str
    expected_event_id: str | None = None
    expected_episode_id: str | None = None


@dataclass(frozen=True)
class ModelingManifestBuild:
    """A validated derived modeling manifest."""

    manifest: ManifestBuildResult
    validation: ManifestValidation
    exclusion_ledger: pd.DataFrame
    source_directory: str
    source_summary: Mapping[str, Any]
    source_terminal_marker: str


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def _load_raw_manifest(
    raw_directory: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    str,
]:
    required_paths = {
        "file_manifest": (
            raw_directory / "file_manifest.parquet"
        ),
        "frame_manifest": (
            raw_directory / "frame_manifest.parquet"
        ),
        "schema_summary": (
            raw_directory / "schema_summary.csv"
        ),
        "build_errors": (
            raw_directory / "build_errors.csv"
        ),
        "manifest_summary": (
            raw_directory / "manifest_summary.json"
        ),
    }

    missing = [
        str(path)
        for path in required_paths.values()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Raw manifest artifacts are missing: "
            + ", ".join(missing)
        )

    terminal_markers = [
        marker_name
        for marker_name in (
            "_SUCCESS.json",
            "_INVALID.json",
        )
        if (
            raw_directory / marker_name
        ).is_file()
    ]

    if len(terminal_markers) != 1:
        raise RuntimeError(
            "Expected exactly one raw terminal marker; "
            f"found {terminal_markers}"
        )

    file_manifest = pd.read_parquet(
        required_paths["file_manifest"]
    )
    frame_manifest = pd.read_parquet(
        required_paths["frame_manifest"]
    )
    schema_summary = pd.read_csv(
        required_paths["schema_summary"]
    )
    build_errors = pd.read_csv(
        required_paths["build_errors"]
    )
    manifest_summary = json.loads(
        required_paths[
            "manifest_summary"
        ].read_text()
    )

    expected_file_rows = int(
        manifest_summary["file_row_count"]
    )
    expected_frame_rows = int(
        manifest_summary["frame_row_count"]
    )

    if len(file_manifest) != expected_file_rows:
        raise AssertionError(
            "Raw file-manifest row count differs from "
            "manifest_summary.json: "
            f"{len(file_manifest)} != {expected_file_rows}"
        )

    if len(frame_manifest) != expected_frame_rows:
        raise AssertionError(
            "Raw frame-manifest row count differs from "
            "manifest_summary.json: "
            f"{len(frame_manifest)} != {expected_frame_rows}"
        )

    return (
        file_manifest,
        frame_manifest,
        schema_summary,
        build_errors,
        manifest_summary,
        terminal_markers[0],
    )


def _rebuild_schema_summary(
    file_manifest: pd.DataFrame,
    raw_schema_summary: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "schema_fingerprint",
        "file_count",
        "splits_json",
        "categories_json",
        "first_archive_member",
        "schema_payload_json",
    ]

    if file_manifest.empty:
        return pd.DataFrame(columns=columns)

    payload_by_fingerprint = dict(
        zip(
            raw_schema_summary[
                "schema_fingerprint"
            ],
            raw_schema_summary[
                "schema_payload_json"
            ],
            strict=True,
        )
    )

    rows = []

    for fingerprint, group in (
        file_manifest.groupby(
            "schema_fingerprint",
            sort=True,
        )
    ):
        if fingerprint not in payload_by_fingerprint:
            raise KeyError(
                "Missing schema payload for fingerprint "
                f"{fingerprint}"
            )

        first_row = (
            group.sort_values("archive_index")
            .iloc[0]
        )

        rows.append(
            {
                "schema_fingerprint": fingerprint,
                "file_count": int(len(group)),
                "splits_json": _canonical_json(
                    sorted(
                        group["split"].unique()
                    )
                ),
                "categories_json": _canonical_json(
                    sorted(
                        group["category"].unique()
                    )
                ),
                "first_archive_member": (
                    first_row["archive_member"]
                ),
                "schema_payload_json": (
                    payload_by_fingerprint[
                        fingerprint
                    ]
                ),
            }
        )

    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            [
                "file_count",
                "schema_fingerprint",
            ],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def build_modeling_manifest(
    raw_directory: str | Path,
    exclusions: Sequence[ModelingExclusion],
    *,
    expected_dimensions: (
        Mapping[str, int] | None
    ) = None,
) -> ModelingManifestBuild:
    """Derive and validate a modeling manifest from a raw audit."""

    raw_directory = Path(raw_directory)

    (
        raw_files,
        raw_frames,
        raw_schema_summary,
        raw_errors,
        source_summary,
        source_terminal_marker,
    ) = _load_raw_manifest(raw_directory)

    exclusion_members = [
        exclusion.archive_member
        for exclusion in exclusions
    ]

    if len(exclusion_members) != len(
        set(exclusion_members)
    ):
        raise ValueError(
            "Duplicate archive members appear in exclusions"
        )

    ledger_rows: list[dict[str, Any]] = []
    removed_file_ids: set[str] = set()

    for exclusion in exclusions:
        matches = raw_files.loc[
            raw_files["archive_member"].eq(
                exclusion.archive_member
            )
        ]

        if len(matches) != 1:
            raise AssertionError(
                "Expected exactly one raw file for exclusion "
                f"{exclusion.archive_member!r}; "
                f"found {len(matches)}"
            )

        row = matches.iloc[0]

        if row["split"] != "train":
            raise AssertionError(
                "Modeling exclusions may not remove official "
                "test files: "
                f"{exclusion.archive_member}"
            )

        actual_event_id = str(row["event_id"])
        actual_episode_id = str(row["episode_id"])

        if (
            exclusion.expected_event_id is not None
            and actual_event_id
            != exclusion.expected_event_id
        ):
            raise AssertionError(
                "Unexpected event ID for exclusion "
                f"{exclusion.archive_member}: "
                f"{actual_event_id} != "
                f"{exclusion.expected_event_id}"
            )

        if (
            exclusion.expected_episode_id
            is not None
            and actual_episode_id
            != exclusion.expected_episode_id
        ):
            raise AssertionError(
                "Unexpected episode ID for exclusion "
                f"{exclusion.archive_member}: "
                f"{actual_episode_id} != "
                f"{exclusion.expected_episode_id}"
            )

        file_id = str(row["file_id"])
        removed_file_ids.add(file_id)

        ledger_rows.append(
            {
                "manifest_schema_version": (
                    MANIFEST_SCHEMA_VERSION
                ),
                "year": int(row["year"]),
                "archive_member": (
                    row["archive_member"]
                ),
                "file_id": file_id,
                "split": row["split"],
                "category": row["category"],
                "event_id": actual_event_id,
                "episode_id": actual_episode_id,
                "radar_site": row["radar_site"],
                "removed_frame_count": int(
                    row["frame_count"]
                ),
                "removed_positive_frame_count": int(
                    row["positive_frame_count"]
                ),
                "reason": exclusion.reason,
                "resolution": exclusion.resolution,
            }
        )

    source_test_file_ids = set(
        raw_files.loc[
            raw_files["split"].eq("test"),
            "file_id",
        ].astype(str)
    )

    modeling_files = (
        raw_files.loc[
            ~raw_files["file_id"]
            .astype(str)
            .isin(removed_file_ids)
        ]
        .copy()
        .reset_index(drop=True)
    )

    modeling_frames = (
        raw_frames.loc[
            ~raw_frames["file_id"]
            .astype(str)
            .isin(removed_file_ids)
        ]
        .copy()
        .reset_index(drop=True)
    )

    modeling_test_file_ids = set(
        modeling_files.loc[
            modeling_files["split"].eq("test"),
            "file_id",
        ].astype(str)
    )

    if modeling_test_file_ids != source_test_file_ids:
        raise AssertionError(
            "Derived modeling manifest altered the "
            "official test file set"
        )

    expected_removed_frames = sum(
        row["removed_frame_count"]
        for row in ledger_rows
    )

    actual_removed_frames = (
        len(raw_frames) - len(modeling_frames)
    )

    if actual_removed_frames != expected_removed_frames:
        raise AssertionError(
            "Removed frame count does not match the "
            "exclusion ledger: "
            f"{actual_removed_frames} != "
            f"{expected_removed_frames}"
        )

    schema_summary = _rebuild_schema_summary(
        modeling_files,
        raw_schema_summary,
    )

    year = int(source_summary["year"])

    manifest = ManifestBuildResult(
        archive_path=str(
            source_summary["archive_path"]
        ),
        archive_size_bytes=int(
            source_summary["archive_size_bytes"]
        ),
        year=year,
        netcdf_member_count=len(
            modeling_files
        ),
        file_manifest=modeling_files,
        frame_manifest=modeling_frames,
        schema_summary=schema_summary,
        errors=raw_errors.copy(),
    )

    validation = validate_manifests(
        manifest,
        expected_file_count=len(
            modeling_files
        ),
        expected_frame_count=len(
            modeling_frames
        ),
        expected_frames_per_file=4,
        expected_dimensions=expected_dimensions,
    )

    validation.assert_valid()

    exclusion_ledger = pd.DataFrame(
        ledger_rows,
        columns=[
            "manifest_schema_version",
            "year",
            "archive_member",
            "file_id",
            "split",
            "category",
            "event_id",
            "episode_id",
            "radar_site",
            "removed_frame_count",
            "removed_positive_frame_count",
            "reason",
            "resolution",
        ],
    )

    return ModelingManifestBuild(
        manifest=manifest,
        validation=validation,
        exclusion_ledger=exclusion_ledger,
        source_directory=str(raw_directory),
        source_summary=source_summary,
        source_terminal_marker=(
            source_terminal_marker
        ),
    )


def write_modeling_manifest_artifacts(
    build: ModelingManifestBuild,
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, str]:
    """Write a validated modeling manifest and exclusion ledger."""

    build.validation.assert_valid()

    output_directory = Path(output_directory)

    if (
        output_directory.exists()
        and any(output_directory.iterdir())
    ):
        if not overwrite:
            raise FileExistsError(
                "Modeling manifest output already exists: "
                f"{output_directory}"
            )

        shutil.rmtree(output_directory)

    base_artifacts = write_manifest_artifacts(
        build.manifest,
        build.validation,
        output_directory,
        overwrite=False,
    )

    success_path = (
        output_directory / "_SUCCESS.json"
    )

    if not success_path.is_file():
        raise RuntimeError(
            "Base manifest writer did not create "
            "_SUCCESS.json"
        )

    success_path.unlink()

    exclusion_path = (
        output_directory / "exclusion_ledger.csv"
    )

    build.exclusion_ledger.to_csv(
        exclusion_path,
        index=False,
    )

    source_file_count = int(
        build.source_summary["file_row_count"]
    )
    source_frame_count = int(
        build.source_summary["frame_row_count"]
    )
    modeling_file_count = len(
        build.manifest.file_manifest
    )
    modeling_frame_count = len(
        build.manifest.frame_manifest
    )

    modeling_summary = {
        "manifest_schema_version": (
            MANIFEST_SCHEMA_VERSION
        ),
        "built_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "artifact_kind": "modeling_manifest",
        "status": "valid",
        "year": build.manifest.year,
        "source_raw_manifest_directory": (
            build.source_directory
        ),
        "source_raw_terminal_marker": (
            build.source_terminal_marker
        ),
        "source_file_row_count": (
            source_file_count
        ),
        "source_frame_row_count": (
            source_frame_count
        ),
        "modeling_file_row_count": (
            modeling_file_count
        ),
        "modeling_frame_row_count": (
            modeling_frame_count
        ),
        "excluded_file_count": len(
            build.exclusion_ledger
        ),
        "excluded_frame_count": int(
            build.exclusion_ledger[
                "removed_frame_count"
            ].sum()
        ),
        "excluded_positive_frame_count": int(
            build.exclusion_ledger[
                "removed_positive_frame_count"
            ].sum()
        ),
        "event_split_overlap_count": len(
            build.validation.event_split_overlap
        ),
        "episode_split_overlap_count": len(
            build.validation.episode_split_overlap
        ),
        "all_required_validations_passed": (
            build.validation.all_required_passed
        ),
    }

    modeling_summary_path = (
        output_directory
        / "modeling_summary.json"
    )

    modeling_summary_path.write_text(
        json.dumps(
            modeling_summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    artifact_paths = sorted(
        path
        for path in output_directory.iterdir()
        if (
            path.is_file()
            and path.name
            not in {
                "_SUCCESS.json",
                "_INVALID.json",
            }
        )
    )

    artifact_hashes = {
        path.name: _sha256_path(path)
        for path in artifact_paths
    }

    success_payload = {
        "manifest_schema_version": (
            MANIFEST_SCHEMA_VERSION
        ),
        "completed_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "artifact_kind": "modeling_manifest",
        "status": "valid",
        "all_required_validations_passed": True,
        "artifact_sha256": artifact_hashes,
    }

    success_path.write_text(
        json.dumps(
            success_payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    return {
        path.name: str(path)
        for path in [
            *artifact_paths,
            success_path,
        ]
    }
