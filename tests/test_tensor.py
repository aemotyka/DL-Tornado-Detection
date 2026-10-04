"""Tests for frame-level radar tensor construction."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tornado_detection.data.tensor import (
    build_file_tensor,
    build_frame_tensor,
    read_netcdf_file,
    read_netcdf_frame,
)


def _dataset() -> xr.Dataset:
    shape = (
        4,
        120,
        240,
        2,
    )

    dbz = np.arange(
        np.prod(shape),
        dtype=np.float32,
    ).reshape(shape)

    vel = (
        dbz * np.float32(0.5)
    ).astype(np.float32)

    dbz[2, 0, 0, 0] = np.nan
    vel[2, 0, 0, 1] = np.nan

    return xr.Dataset(
        data_vars={
            "DBZ": (
                (
                    "time",
                    "azimuth",
                    "range",
                    "sweep",
                ),
                dbz,
            ),
            "VEL": (
                (
                    "time",
                    "azimuth",
                    "range",
                    "sweep",
                ),
                vel,
            ),
            "frame_labels": (
                ("time",),
                np.asarray(
                    [
                        0,
                        0,
                        1,
                        1,
                    ],
                    dtype=np.uint8,
                ),
            ),
        },
        coords={
            "time": np.arange(4),
            "azimuth": np.arange(120),
            "range": np.arange(240),
            "sweep": np.arange(2),
        },
    )


def test_build_dbz_vel_tensor() -> None:
    dataset = _dataset()

    result = build_frame_tensor(
        dataset,
        2,
    )

    assert result.values.shape == (
        120,
        240,
        4,
    )
    assert result.values.dtype == np.float32
    assert result.label == 1
    assert result.frame_index == 2
    assert result.variables == (
        "DBZ",
        "VEL",
    )

    np.testing.assert_array_equal(
        result.values[:, :, 0:2],
        dataset["DBZ"]
        .isel(time=2)
        .values,
    )
    np.testing.assert_array_equal(
        result.values[:, :, 2:4],
        dataset["VEL"]
        .isel(time=2)
        .values,
    )

    assert np.isnan(
        result.values[0, 0, 0]
    )
    assert np.isnan(
        result.values[0, 0, 3]
    )


def test_read_netcdf_frame(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.nc"

    _dataset().to_netcdf(
        path,
        engine="netcdf4",
    )

    result = read_netcdf_frame(
        path,
        3,
    )

    assert result.values.shape == (
        120,
        240,
        4,
    )
    assert result.values.dtype == np.float32
    assert result.label == 1


def test_build_file_tensor() -> None:
    dataset = _dataset()

    result = build_file_tensor(dataset)

    assert result.values.shape == (
        4,
        120,
        240,
        4,
    )
    assert result.values.dtype == np.float32
    assert result.labels.dtype == np.uint8
    np.testing.assert_array_equal(
        result.labels,
        np.asarray([0, 0, 1, 1], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        result.values[3],
        build_frame_tensor(dataset, 3).values,
    )


def test_read_netcdf_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.nc"

    _dataset().to_netcdf(
        path,
        engine="netcdf4",
    )

    result = read_netcdf_file(path)

    assert result.values.shape == (
        4,
        120,
        240,
        4,
    )
    np.testing.assert_array_equal(
        result.labels,
        np.asarray([0, 0, 1, 1], dtype=np.uint8),
    )


def test_rejects_wrong_dimension_order() -> None:
    dataset = _dataset().transpose(
        "time",
        "sweep",
        "azimuth",
        "range",
    )

    with pytest.raises(
        AssertionError,
        match="DBZ dimensions",
    ):
        build_frame_tensor(
            dataset,
            0,
        )


def test_rejects_invalid_frame_index() -> None:
    dataset = _dataset()

    with pytest.raises(
        IndexError,
        match="outside the available range",
    ):
        build_frame_tensor(
            dataset,
            4,
        )

    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        build_frame_tensor(
            dataset,
            1.0,
        )


def test_rejects_missing_variable() -> None:
    dataset = _dataset().drop_vars(
        "VEL"
    )

    with pytest.raises(
        KeyError,
        match="VEL",
    ):
        build_frame_tensor(
            dataset,
            0,
        )


def test_rejects_duplicate_variables() -> None:
    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        build_frame_tensor(
            _dataset(),
            0,
            variables=(
                "DBZ",
                "DBZ",
            ),
        )


def test_rejects_nonbinary_label() -> None:
    dataset = _dataset()
    dataset["frame_labels"][0] = 2

    with pytest.raises(
        AssertionError,
        match="Unexpected frame label",
    ):
        build_frame_tensor(
            dataset,
            0,
        )
