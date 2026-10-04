"""Tests for the canonical TorNet frame index."""

from pathlib import Path

import pandas as pd
import pytest

from tornado_detection.data.training_index import (
    CANONICAL_MANIFEST_SOURCES,
    EXPECTED_YEARS,
    assign_model_splits,
    build_validation_group_keys,
    canonical_manifest_directory,
    load_canonical_frame_index,
    summarize_canonical_frame_index,
    summarize_model_splits,
)


def _frame_row(
    *,
    year: int,
    split: str,
    label: int,
    event_group_id: str,
    suffix: str,
) -> dict[str, object]:
    file_id = f"file-{year}-{suffix}"

    return {
        "manifest_schema_version": "1",
        "frame_id": f"{file_id}:0",
        "file_id": file_id,
        "archive_member": (
            f"{split}/{year}/"
            f"NUL_{year}_{suffix}.nc"
        ),
        "split": split,
        "year": year,
        "category": (
            "TOR" if label == 1 else "NUL"
        ),
        "radar_site": "KAAA",
        "event_group_id": event_group_id,
        "event_id": event_group_id,
        "episode_id": f"episode-{year}-{suffix}",
        "scit_id": f"S{suffix}",
        "ef_number": -1.0,
        "frame_index": 0,
        "frame_time_utc": (
            f"{year}-01-01T00:00:00+00:00"
        ),
        "frame_label": label,
        "dbz_finite_fraction": 1.0,
        "kdp_finite_fraction": 1.0,
        "rhohv_finite_fraction": 1.0,
        "vel_finite_fraction": 1.0,
        "width_finite_fraction": 1.0,
        "zdr_finite_fraction": 1.0,
        "mean_primary_finite_fraction": 1.0,
        "range_folded_fraction": 0.0,
    }


def _write_fixture_manifests(
    root: Path,
    *,
    crossing_event: bool = False,
) -> int:
    total_rows = 0

    for year in EXPECTED_YEARS:
        directory = canonical_manifest_directory(
            root,
            year,
        )
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        (directory / "_SUCCESS.json").write_text(
            '{"status": "valid"}\n'
        )

        train_event = (
            "crossing-event"
            if crossing_event and year == 2013
            else f"train-event-{year}"
        )
        test_event = (
            "crossing-event"
            if crossing_event and year == 2013
            else f"test-event-{year}"
        )

        rows = [
            _frame_row(
                year=year,
                split="train",
                label=year % 2,
                event_group_id=train_event,
                suffix="train",
            ),
            _frame_row(
                year=year,
                split="test",
                label=(year + 1) % 2,
                event_group_id=test_event,
                suffix="test",
            ),
            _frame_row(
                year=year,
                split="train",
                label=0,
                event_group_id="-1",
                suffix="sentinel",
            ),
        ]

        pd.DataFrame(rows).to_parquet(
            directory / "frame_manifest.parquet",
            index=False,
        )

        total_rows += len(rows)

    return total_rows


def test_canonical_manifest_mapping() -> None:
    assert tuple(
        CANONICAL_MANIFEST_SOURCES
    ) == EXPECTED_YEARS

    assert {
        year
        for year, source
        in CANONICAL_MANIFEST_SOURCES.items()
        if source == "modeling"
    } == {
        2014,
        2018,
        2021,
    }


def test_canonical_manifest_paths(
    tmp_path: Path,
) -> None:
    assert canonical_manifest_directory(
        tmp_path,
        2013,
    ) == (
        tmp_path / "v1" / "2013"
    )

    assert canonical_manifest_directory(
        tmp_path,
        2014,
    ) == (
        tmp_path
        / "modeling"
        / "v1"
        / "2014"
    )

    with pytest.raises(
        ValueError,
        match="Unsupported canonical TorNet year",
    ):
        canonical_manifest_directory(
            tmp_path,
            2012,
        )


def test_load_and_summarize_canonical_index(
    tmp_path: Path,
) -> None:
    expected_rows = _write_fixture_manifests(
        tmp_path
    )

    frame_index = load_canonical_frame_index(
        tmp_path,
        expected_frame_count=expected_rows,
    )

    assert len(frame_index) == expected_rows
    assert frame_index["frame_id"].is_unique
    assert set(frame_index["year"]) == set(
        EXPECTED_YEARS
    )

    actual_sources = dict(
        frame_index[
            [
                "year",
                "manifest_source",
            ]
        ]
        .drop_duplicates()
        .itertuples(
            index=False,
            name=None,
        )
    )

    assert actual_sources == dict(
        CANONICAL_MANIFEST_SOURCES
    )

    summaries = summarize_canonical_frame_index(
        frame_index
    )

    assert set(summaries) == {
        "by_split",
        "by_year_split",
        "by_category_split",
        "training_groups",
    }

    grouping = summaries[
        "training_groups"
    ].iloc[0]

    assert (
        grouping["sentinel_or_missing_event_frames"]
        == len(EXPECTED_YEARS)
    )
    assert (
        grouping["sentinel_or_missing_files"]
        == len(EXPECTED_YEARS)
    )


def test_rejects_official_event_overlap(
    tmp_path: Path,
) -> None:
    expected_rows = _write_fixture_manifests(
        tmp_path,
        crossing_event=True,
    )

    with pytest.raises(
        AssertionError,
        match=(
            "Real event groups cross the official "
            "train/test boundary"
        ),
    ):
        load_canonical_frame_index(
            tmp_path,
            expected_frame_count=expected_rows,
        )


def test_rejects_missing_success_marker(
    tmp_path: Path,
) -> None:
    expected_rows = _write_fixture_manifests(
        tmp_path
    )

    marker = (
        canonical_manifest_directory(
            tmp_path,
            2018,
        )
        / "_SUCCESS.json"
    )
    marker.unlink()

    with pytest.raises(
        FileNotFoundError,
        match=(
            "Canonical manifest is missing its "
            "success marker for 2018"
        ),
    ):
        load_canonical_frame_index(
            tmp_path,
            expected_frame_count=expected_rows,
        )



def test_validation_groups_use_file_fallback(
    tmp_path: Path,
) -> None:
    expected_rows = _write_fixture_manifests(
        tmp_path
    )
    frame_index = load_canonical_frame_index(
        tmp_path,
        expected_frame_count=expected_rows,
    )

    group_keys = build_validation_group_keys(
        frame_index
    )

    sentinel_rows = frame_index.loc[
        frame_index["event_group_id"].eq("-1")
    ]

    sentinel_keys = group_keys.loc[
        sentinel_rows.index
    ]

    assert sentinel_keys.str.startswith(
        "file:"
    ).all()
    assert (
        sentinel_keys.nunique()
        == sentinel_rows["file_id"].nunique()
    )

    real_rows = frame_index.loc[
        ~frame_index["event_group_id"].eq("-1")
    ]
    real_keys = group_keys.loc[
        real_rows.index
    ]

    assert real_keys.str.startswith(
        "event:"
    ).all()


def test_model_split_is_deterministic_and_disjoint(
    tmp_path: Path,
) -> None:
    expected_rows = _write_fixture_manifests(
        tmp_path
    )
    frame_index = load_canonical_frame_index(
        tmp_path,
        expected_frame_count=expected_rows,
    )

    first = assign_model_splits(
        frame_index,
        validation_fraction=0.50,
        seed=7,
        expected_frame_count=expected_rows,
    )
    second = assign_model_splits(
        frame_index,
        validation_fraction=0.50,
        seed=7,
        expected_frame_count=expected_rows,
    )

    pd.testing.assert_series_equal(
        first["model_split"],
        second["model_split"],
    )
    pd.testing.assert_series_equal(
        first["validation_group_key"],
        second["validation_group_key"],
    )

    official_test = first.loc[
        first["split"].eq("test")
    ]

    assert official_test["model_split"].eq(
        "test"
    ).all()

    internal = first.loc[
        first["model_split"].isin(
            [
                "train",
                "validation",
            ]
        )
    ]

    assert (
        internal.groupby(
            "validation_group_key"
        )["model_split"]
        .nunique()
        .max()
        == 1
    )

    summary = summarize_model_splits(first)

    assert set(summary["model_split"]) == {
        "train",
        "validation",
        "test",
    }


def test_rejects_invalid_validation_fraction(
    tmp_path: Path,
) -> None:
    expected_rows = _write_fixture_manifests(
        tmp_path
    )
    frame_index = load_canonical_frame_index(
        tmp_path,
        expected_frame_count=expected_rows,
    )

    for invalid_fraction in (
        0.0,
        1.0,
        -0.1,
        1.1,
    ):
        with pytest.raises(
            ValueError,
            match="validation_fraction",
        ):
            assign_model_splits(
                frame_index,
                validation_fraction=invalid_fraction,
            )
