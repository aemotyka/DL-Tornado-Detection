"""Canonical frame index for TorNet model development."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import pandas as pd

from tornado_detection.data.manifest import (
    NON_GROUPING_IDENTIFIERS,
)


EXPECTED_CANONICAL_FRAME_COUNT = 812_432
EXPECTED_YEARS = tuple(range(2013, 2023))
EXPECTED_OFFICIAL_SPLITS = frozenset({"train", "test"})
EXPECTED_FRAME_LABELS = frozenset({0, 1})

CANONICAL_MANIFEST_SOURCES: Mapping[int, str] = (
    MappingProxyType(
        {
            2013: "raw",
            2014: "modeling",
            2015: "raw",
            2016: "raw",
            2017: "raw",
            2018: "modeling",
            2019: "raw",
            2020: "raw",
            2021: "modeling",
            2022: "raw",
        }
    )
)

REQUIRED_FRAME_COLUMNS = frozenset(
    {
        "manifest_schema_version",
        "frame_id",
        "file_id",
        "archive_member",
        "split",
        "year",
        "category",
        "radar_site",
        "event_group_id",
        "event_id",
        "episode_id",
        "scit_id",
        "ef_number",
        "frame_index",
        "frame_time_utc",
        "frame_label",
        "dbz_finite_fraction",
        "kdp_finite_fraction",
        "rhohv_finite_fraction",
        "vel_finite_fraction",
        "width_finite_fraction",
        "zdr_finite_fraction",
        "mean_primary_finite_fraction",
        "range_folded_fraction",
    }
)


def canonical_manifest_directory(
    manifests_root: str | Path,
    year: int,
) -> Path:
    """Return the canonical manifest directory for one year."""

    try:
        source = CANONICAL_MANIFEST_SOURCES[year]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported canonical TorNet year: {year}"
        ) from exc

    manifests_root = Path(manifests_root)

    if source == "raw":
        return manifests_root / "v1" / str(year)

    return (
        manifests_root
        / "modeling"
        / "v1"
        / str(year)
    )


def _normalize_identifier_series(
    values: pd.Series,
) -> pd.Series:
    normalized = values.astype("string").str.strip()

    return normalized.replace(
        {
            "-1.0": "-1",
        }
    )


def _validate_columns(
    frame_index: pd.DataFrame,
) -> None:
    missing_columns = sorted(
        REQUIRED_FRAME_COLUMNS
        - set(frame_index.columns)
    )

    if missing_columns:
        raise AssertionError(
            "Canonical frame index is missing required "
            f"columns: {missing_columns}"
        )


def _validate_source_selection(
    frame_index: pd.DataFrame,
) -> None:
    expected = pd.DataFrame(
        {
            "year": list(CANONICAL_MANIFEST_SOURCES),
            "manifest_source": list(
                CANONICAL_MANIFEST_SOURCES.values()
            ),
        }
    )

    actual = (
        frame_index[
            [
                "year",
                "manifest_source",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "year",
                "manifest_source",
            ]
        )
        .reset_index(drop=True)
    )

    expected = expected.sort_values(
        [
            "year",
            "manifest_source",
        ]
    ).reset_index(drop=True)

    if not actual.equals(expected):
        raise AssertionError(
            (
                "Canonical manifest-source selection differs "
                "from the required year mapping. Expected: "
                f"{expected.to_dict(orient='records')}; "
                "actual: "
                f"{actual.to_dict(orient='records')}"
            )
        )


def _validate_official_event_disjointness(
    frame_index: pd.DataFrame,
) -> None:
    event_groups = _normalize_identifier_series(
        frame_index["event_group_id"]
    )

    real_event_mask = (
        event_groups.notna()
        & ~event_groups.isin(
            NON_GROUPING_IDENTIFIERS
        )
    )

    real_events = frame_index.loc[
        real_event_mask,
        [
            "split",
        ],
    ].copy()

    real_events["event_group_id"] = (
        event_groups.loc[real_event_mask].values
    )

    overlaps = (
        real_events.groupby(
            "event_group_id",
            sort=True,
        )["split"]
        .nunique()
    )

    overlaps = overlaps.loc[overlaps > 1]

    if not overlaps.empty:
        examples = overlaps.index.astype(str).tolist()[:10]

        raise AssertionError(
            "Real event groups cross the official "
            "train/test boundary. "
            f"Count={len(overlaps)}; examples={examples}"
        )


def validate_canonical_frame_index(
    frame_index: pd.DataFrame,
    *,
    expected_frame_count: int = (
        EXPECTED_CANONICAL_FRAME_COUNT
    ),
) -> None:
    """Validate the combined canonical frame index."""

    _validate_columns(frame_index)

    if len(frame_index) != expected_frame_count:
        raise AssertionError(
            "Unexpected canonical frame count: "
            f"{len(frame_index):,} != "
            f"{expected_frame_count:,}"
        )

    actual_years = tuple(
        sorted(
            int(year)
            for year in frame_index["year"].unique()
        )
    )

    if actual_years != EXPECTED_YEARS:
        raise AssertionError(
            "Unexpected canonical years: "
            f"{actual_years} != {EXPECTED_YEARS}"
        )

    if frame_index["frame_id"].isna().any():
        raise AssertionError(
            "Canonical frame index contains null frame IDs"
        )

    duplicate_frame_ids = frame_index[
        "frame_id"
    ].duplicated(keep=False)

    if duplicate_frame_ids.any():
        examples = (
            frame_index.loc[
                duplicate_frame_ids,
                "frame_id",
            ]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        raise AssertionError(
            "Canonical frame IDs are not globally unique. "
            f"Examples={examples}"
        )

    if frame_index["split"].isna().any():
        raise AssertionError(
            "Canonical frame index contains null splits"
        )

    actual_splits = frozenset(
        frame_index["split"].astype(str).unique()
    )

    if actual_splits != EXPECTED_OFFICIAL_SPLITS:
        raise AssertionError(
            "Unexpected official splits: "
            f"{sorted(actual_splits)} != "
            f"{sorted(EXPECTED_OFFICIAL_SPLITS)}"
        )

    if frame_index["frame_label"].isna().any():
        raise AssertionError(
            "Canonical frame index contains null labels"
        )

    numeric_labels = pd.to_numeric(
        frame_index["frame_label"],
        errors="raise",
    )

    non_integral_labels = (
        numeric_labels
        != numeric_labels.astype("int64")
    )

    if non_integral_labels.any():
        raise AssertionError(
            "Canonical frame labels contain non-integer "
            "values"
        )

    actual_labels = frozenset(
        numeric_labels.astype("int64").unique()
    )

    if actual_labels != EXPECTED_FRAME_LABELS:
        raise AssertionError(
            "Unexpected frame labels: "
            f"{sorted(actual_labels)} != "
            f"{sorted(EXPECTED_FRAME_LABELS)}"
        )

    _validate_source_selection(frame_index)
    _validate_official_event_disjointness(frame_index)


def load_canonical_frame_index(
    manifests_root: str | Path,
    *,
    expected_frame_count: int = (
        EXPECTED_CANONICAL_FRAME_COUNT
    ),
) -> pd.DataFrame:
    """Load and validate all canonical annual frame manifests."""

    annual_frames: list[pd.DataFrame] = []

    for year in EXPECTED_YEARS:
        source = CANONICAL_MANIFEST_SOURCES[year]
        directory = canonical_manifest_directory(
            manifests_root,
            year,
        )
        frame_manifest_path = (
            directory / "frame_manifest.parquet"
        )
        success_marker_path = (
            directory / "_SUCCESS.json"
        )

        if not directory.is_dir():
            raise FileNotFoundError(
                "Canonical manifest directory does not "
                f"exist for {year}: {directory}"
            )

        if not success_marker_path.is_file():
            raise FileNotFoundError(
                "Canonical manifest is missing its success "
                f"marker for {year}: {success_marker_path}"
            )

        if not frame_manifest_path.is_file():
            raise FileNotFoundError(
                "Canonical frame manifest does not exist "
                f"for {year}: {frame_manifest_path}"
            )

        annual = pd.read_parquet(
            frame_manifest_path
        ).copy()

        _validate_columns(annual)

        observed_years = set(
            pd.to_numeric(
                annual["year"],
                errors="raise",
            )
            .astype("int64")
            .unique()
        )

        if observed_years != {year}:
            raise AssertionError(
                f"Manifest for {year} contains years "
                f"{sorted(observed_years)}"
            )

        annual["year"] = year
        annual["manifest_source"] = source
        annual["manifest_directory"] = str(
            directory
        )

        annual_frames.append(annual)

    frame_index = pd.concat(
        annual_frames,
        ignore_index=True,
        copy=False,
    )

    frame_index = frame_index.sort_values(
        [
            "year",
            "split",
            "archive_member",
            "frame_index",
        ],
        kind="stable",
    ).reset_index(drop=True)

    validate_canonical_frame_index(
        frame_index,
        expected_frame_count=expected_frame_count,
    )

    return frame_index


def summarize_canonical_frame_index(
    frame_index: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Return the first canonical ML audit summaries."""

    _validate_columns(frame_index)

    by_split = (
        frame_index.groupby(
            "split",
            sort=True,
            observed=True,
        )
        .agg(
            frame_count=("frame_id", "size"),
            positive_frames=("frame_label", "sum"),
        )
        .reset_index()
    )

    by_year_split = (
        frame_index.groupby(
            [
                "year",
                "split",
            ],
            sort=True,
            observed=True,
        )
        .agg(
            frame_count=("frame_id", "size"),
            positive_frames=("frame_label", "sum"),
        )
        .reset_index()
    )

    by_category_split = (
        frame_index.groupby(
            [
                "category",
                "split",
            ],
            sort=True,
            observed=True,
        )
        .agg(
            frame_count=("frame_id", "size"),
            positive_frames=("frame_label", "sum"),
        )
        .reset_index()
    )

    for summary in (
        by_split,
        by_year_split,
        by_category_split,
    ):
        summary["negative_frames"] = (
            summary["frame_count"]
            - summary["positive_frames"]
        )
        summary["positive_prevalence"] = (
            summary["positive_frames"]
            / summary["frame_count"]
        )

    train = frame_index.loc[
        frame_index["split"].eq("train")
    ].copy()

    train_event_groups = _normalize_identifier_series(
        train["event_group_id"]
    )

    real_event_mask = (
        train_event_groups.notna()
        & ~train_event_groups.isin(
            NON_GROUPING_IDENTIFIERS
        )
    )

    grouping_summary = pd.DataFrame(
        [
            {
                "official_train_frames": int(len(train)),
                "distinct_real_event_groups": int(
                    train_event_groups.loc[
                        real_event_mask
                    ].nunique()
                ),
                "sentinel_or_missing_event_frames": int(
                    (~real_event_mask).sum()
                ),
                "sentinel_or_missing_files": int(
                    train.loc[
                        ~real_event_mask,
                        "file_id",
                    ].nunique()
                ),
            }
        ]
    )

    return {
        "by_split": by_split,
        "by_year_split": by_year_split,
        "by_category_split": by_category_split,
        "training_groups": grouping_summary,
    }


def build_validation_group_keys(
    frame_index: pd.DataFrame,
) -> pd.Series:
    """Build stable event groups with file fallback for sentinels."""

    _validate_columns(frame_index)

    if frame_index["file_id"].isna().any():
        raise AssertionError(
            "Cannot construct validation groups with null file IDs"
        )

    event_groups = _normalize_identifier_series(
        frame_index["event_group_id"]
    )
    file_ids = frame_index["file_id"].astype(
        "string"
    ).str.strip()

    real_event_mask = (
        event_groups.notna()
        & ~event_groups.isin(
            NON_GROUPING_IDENTIFIERS
        )
    )

    group_keys = pd.Series(
        index=frame_index.index,
        dtype="string",
        name="validation_group_key",
    )

    group_keys.loc[real_event_mask] = (
        "event:"
        + event_groups.loc[real_event_mask]
    )

    group_keys.loc[~real_event_mask] = (
        "file:"
        + file_ids.loc[~real_event_mask]
    )

    if group_keys.isna().any():
        raise AssertionError(
            "Validation group construction produced null keys"
        )

    return group_keys


def _validation_hash_score(
    group_key: str,
    *,
    seed: int,
) -> float:
    import hashlib

    payload = (
        f"{seed}\n{group_key}"
    ).encode("utf-8")

    digest = hashlib.sha256(payload).digest()
    integer = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    return integer / float(2 ** 64)


def assign_model_splits(
    frame_index: pd.DataFrame,
    *,
    validation_fraction: float = 0.20,
    seed: int = 20260913,
    expected_frame_count: int = (
        EXPECTED_CANONICAL_FRAME_COUNT
    ),
) -> pd.DataFrame:
    """Assign deterministic group-disjoint train/validation/test."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be strictly "
            "between 0 and 1"
        )

    validate_canonical_frame_index(
        frame_index,
        expected_frame_count=expected_frame_count,
    )

    assigned = frame_index.copy()
    assigned["validation_group_key"] = (
        build_validation_group_keys(assigned)
    )

    official_train_mask = assigned[
        "split"
    ].eq("train")
    official_test_mask = assigned[
        "split"
    ].eq("test")

    train_group_keys = (
        assigned.loc[
            official_train_mask,
            "validation_group_key",
        ]
        .drop_duplicates()
        .sort_values()
    )

    group_scores = {
        group_key: _validation_hash_score(
            str(group_key),
            seed=seed,
        )
        for group_key in train_group_keys
    }

    validation_groups = {
        group_key
        for group_key, score in group_scores.items()
        if score < validation_fraction
    }

    if not validation_groups:
        raise AssertionError(
            "Validation assignment selected no groups"
        )

    validation_mask = (
        official_train_mask
        & assigned[
            "validation_group_key"
        ].isin(validation_groups)
    )

    assigned["model_split"] = "train"
    assigned.loc[
        validation_mask,
        "model_split",
    ] = "validation"
    assigned.loc[
        official_test_mask,
        "model_split",
    ] = "test"

    assigned["validation_fraction"] = float(
        validation_fraction
    )
    assigned["validation_seed"] = int(seed)

    if not (
        assigned.loc[
            official_test_mask,
            "model_split",
        ]
        .eq("test")
        .all()
    ):
        raise AssertionError(
            "Official test rows were reassigned"
        )

    if assigned.loc[
        official_train_mask,
        "model_split",
    ].eq("test").any():
        raise AssertionError(
            "Official training rows were assigned to test"
        )

    internal = assigned.loc[
        assigned["model_split"].isin(
            [
                "train",
                "validation",
            ]
        ),
        [
            "validation_group_key",
            "model_split",
        ],
    ].drop_duplicates()

    crossing_groups = (
        internal.groupby(
            "validation_group_key",
            sort=True,
        )["model_split"]
        .nunique()
    )

    crossing_groups = crossing_groups.loc[
        crossing_groups > 1
    ]

    if not crossing_groups.empty:
        raise AssertionError(
            "Validation groups cross internal train/"
            "validation splits"
        )

    actual_model_splits = set(
        assigned["model_split"].unique()
    )

    if actual_model_splits != {
        "train",
        "validation",
        "test",
    }:
        raise AssertionError(
            "Unexpected model splits: "
            f"{sorted(actual_model_splits)}"
        )

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        labels = set(
            assigned.loc[
                assigned["model_split"].eq(
                    split_name
                ),
                "frame_label",
            ]
            .astype("int64")
            .unique()
        )

        if labels != {0, 1}:
            raise AssertionError(
                f"Model split {split_name!r} does not "
                f"contain both labels: {sorted(labels)}"
            )

    return assigned


def summarize_model_splits(
    assigned_frame_index: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize assigned model splits and class prevalence."""

    required = {
        "model_split",
        "validation_group_key",
        "frame_id",
        "file_id",
        "frame_label",
    }
    missing = sorted(
        required
        - set(assigned_frame_index.columns)
    )

    if missing:
        raise AssertionError(
            "Assigned frame index is missing columns: "
            f"{missing}"
        )

    summary = (
        assigned_frame_index.groupby(
            "model_split",
            sort=False,
            observed=True,
        )
        .agg(
            frame_count=("frame_id", "size"),
            file_count=("file_id", "nunique"),
            group_count=(
                "validation_group_key",
                "nunique",
            ),
            positive_frames=("frame_label", "sum"),
        )
        .reset_index()
    )

    split_order = {
        "train": 0,
        "validation": 1,
        "test": 2,
    }

    summary["_order"] = summary[
        "model_split"
    ].map(split_order)

    summary = (
        summary.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    summary["negative_frames"] = (
        summary["frame_count"]
        - summary["positive_frames"]
    )
    summary["positive_prevalence"] = (
        summary["positive_frames"]
        / summary["frame_count"]
    )
    summary["frame_fraction"] = (
        summary["frame_count"]
        / summary["frame_count"].sum()
    )

    return summary
