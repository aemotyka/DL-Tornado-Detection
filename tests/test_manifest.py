from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
import xarray as xr

from tornado_detection.data.manifest import (
    build_archive_manifests,
    parse_tornet_member,
    validate_manifests,
)


def test_parse_tornet_member() -> None:
    parsed = parse_tornet_member(
        "train/2013/NUL_131101_063025_KRLX_476088s_F5.nc",
        expected_year=2013,
    )

    assert parsed.split == "train"
    assert parsed.year == 2013
    assert parsed.category == "NUL"
    assert parsed.radar_site == "KRLX"
    assert parsed.filename_event_token == "476088s"
    assert parsed.filename_scit_id == "F5"
    assert parsed.filename_timestamp_utc.startswith(
        "2013-11-01T06:30:25"
    )


def _synthetic_dataset(
    *,
    category: str,
    event_id: str,
    episode_id: str,
    scit_id: str,
    frame_labels: list[int],
) -> xr.Dataset:
    shape = (4, 2, 3, 2)
    base = np.arange(
        np.prod(shape),
        dtype=np.float32,
    ).reshape(shape)

    base[0, 0, 0, 0] = np.nan

    data_vars = {
        name: (
            ("time", "azimuth", "range", "sweep"),
            base.copy(),
        )
        for name in (
            "DBZ",
            "KDP",
            "RHOHV",
            "VEL",
            "WIDTH",
            "ZDR",
        )
    }

    data_vars.update(
        {
            "frame_labels": (
                ("time",),
                np.asarray(frame_labels, dtype=np.uint8),
            ),
            "range_folded_mask": (
                ("time", "azimuth", "range", "sweep"),
                np.zeros(shape, dtype=np.uint8),
            ),
            "elevation": (
                ("sweep",),
                np.asarray([0.5, 0.9], dtype=np.float32),
            ),
            "nyquist_velocity": (
                ("time", "sweep"),
                np.full((4, 2), 30.0, dtype=np.float32),
            ),
        }
    )

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": (
                "time",
                np.asarray(
                    [1356998400, 1356998700, 1356999000, 1356999300],
                    dtype=np.float64,
                ),
                {
                    "units": "seconds since 1970-01-01",
                    "calendar": "proleptic_gregorian",
                },
            ),
            "azimuth": (
                "azimuth",
                np.asarray([0.0, 0.5], dtype=np.float32),
            ),
            "range": (
                "range",
                np.asarray(
                    [1000.0, 1250.0, 1500.0],
                    dtype=np.float32,
                ),
            ),
            "sweep": (
                "sweep",
                np.asarray([0, 1], dtype=np.int32),
            ),
        },
        attrs={
            "category": category,
            "event_id": event_id,
            "episode_id": episode_id,
            "scit_id": scit_id,
            "ef_number": 1.0 if category == "TOR" else -1.0,
            "site_lat": 35.0,
            "site_lon": -97.0,
            "tornado_start_time": (
                "2013-01-01 00:05:00"
                if category == "TOR"
                else ""
            ),
            "tornado_end_time": (
                "2013-01-01 00:15:00"
                if category == "TOR"
                else ""
            ),
        },
    )


def _write_member(
    archive: tarfile.TarFile,
    temporary_path: Path,
    *,
    archive_member: str,
    dataset: xr.Dataset,
) -> None:
    local_path = temporary_path / Path(archive_member).name

    encoding = {
        variable_name: {
            "_FillValue": -999.0,
            "zlib": True,
            "complevel": 1,
        }
        for variable_name in (
            "DBZ",
            "KDP",
            "RHOHV",
            "VEL",
            "WIDTH",
            "ZDR",
        )
    }

    dataset.to_netcdf(
        local_path,
        engine="netcdf4",
        encoding=encoding,
    )

    archive.add(
        local_path,
        arcname=archive_member,
    )

    local_path.unlink()


def test_build_and_validate_synthetic_archive(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "tornet_2013.tar.gz"

    with tarfile.open(archive_path, mode="w:gz") as archive:
        _write_member(
            archive,
            tmp_path,
            archive_member=(
                "train/2013/"
                "TOR_130101_000000_KAAA_100_A1.nc"
            ),
            dataset=_synthetic_dataset(
                category="TOR",
                event_id="100",
                episode_id="10",
                scit_id="A1",
                frame_labels=[0, 0, 1, 1],
            ),
        )

        _write_member(
            archive,
            tmp_path,
            archive_member=(
                "test/2013/"
                "NUL_130101_010000_KBBB_200_B2.nc"
            ),
            dataset=_synthetic_dataset(
                category="NUL",
                event_id="200",
                episode_id="20",
                scit_id="B2",
                frame_labels=[0, 0, 0, 0],
            ),
        )

    result = build_archive_manifests(
        archive_path,
        expected_year=2013,
        progress=None,
    )

    assert result.errors.empty
    assert len(result.file_manifest) == 2
    assert len(result.frame_manifest) == 8

    validation = validate_manifests(
        result,
        expected_file_count=2,
        expected_frame_count=8,
        expected_frames_per_file=4,
        expected_split_category_counts={
            ("train", "TOR"): 1,
            ("test", "NUL"): 1,
        },
        expected_dimensions={
            "time": 4,
            "sweep": 2,
            "azimuth": 2,
            "range": 3,
        },
    )

    validation.assert_valid()

    train_tor = result.file_manifest.loc[
        (result.file_manifest["split"] == "train")
        & (result.file_manifest["category"] == "TOR")
    ].iloc[0]

    assert train_tor["positive_frame_count"] == 2
    assert train_tor["has_mixed_frame_labels"]
    assert validation.event_split_overlap.empty
