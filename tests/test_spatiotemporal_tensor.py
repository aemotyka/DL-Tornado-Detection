"""Tests for explicit TorNet spatiotemporal tensors."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tornado_detection.data import (
    SPATIOTEMPORAL_COORDINATE_NAMES,
    build_spatiotemporal_file_tensor,
    read_spatiotemporal_netcdf_file,
)


VARIABLES = (
    "DBZ",
    "KDP",
    "RHOHV",
    "VEL",
    "WIDTH",
    "ZDR",
)


def _dataset() -> xr.Dataset:
    shape = (4, 120, 240, 2)
    data_vars = {}

    for index, name in enumerate(VARIABLES):
        values = np.full(
            shape,
            float(index),
            dtype=np.float32,
        )
        data_vars[name] = (
            (
                "time",
                "azimuth",
                "range",
                "sweep",
            ),
            values,
        )

    data_vars["DBZ"][1][0, 0, 0, 0] = np.nan
    data_vars["range_folded_mask"] = (
        (
            "time",
            "azimuth",
            "range",
            "sweep",
        ),
        np.zeros(shape, dtype=np.float32),
    )
    data_vars["frame_labels"] = (
        ("time",),
        np.asarray([0, 0, 1, 1], dtype=np.uint8),
    )
    data_vars["azimuth_limits"] = (
        ("lims",),
        np.asarray([210.0, 270.0], dtype=np.float32),
    )
    data_vars["range_limits"] = (
        ("lims",),
        np.asarray([20_000.0, 80_000.0], dtype=np.float32),
    )

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": np.asarray(
                [
                    "2020-01-01T00:00:00",
                    "2020-01-01T00:05:00",
                    "2020-01-01T00:10:00",
                    "2020-01-01T00:15:00",
                ],
                dtype="datetime64[s]",
            ),
            "azimuth": np.arange(120),
            "range": np.arange(240),
            "sweep": np.arange(2),
            "lims": np.arange(2),
        },
    )


def test_builds_explicit_sequence_axes() -> None:
    result = build_spatiotemporal_file_tensor(
        _dataset(),
        variables=VARIABLES,
    )

    assert result.values.shape == (
        4,
        2,
        6,
        120,
        240,
    )
    assert result.values.dtype == np.float32
    assert result.finite_mask.shape == result.values.shape
    assert result.finite_mask.dtype == np.bool_
    assert not result.finite_mask[0, 0, 0, 0, 0]
    assert result.range_folded_mask.shape == (
        4,
        2,
        120,
        240,
    )
    assert result.coordinates.shape == (
        2,
        5,
        120,
        240,
    )
    assert result.coordinate_names == (
        SPATIOTEMPORAL_COORDINATE_NAMES
    )
    np.testing.assert_array_equal(
        result.labels,
        np.asarray([0, 0, 1, 1], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        np.diff(result.times_unix_seconds),
        np.asarray([300, 300, 300]),
    )


def test_preserves_variable_and_sweep_identity() -> None:
    result = build_spatiotemporal_file_tensor(
        _dataset(),
        variables=VARIABLES,
    )

    for variable_index in range(len(VARIABLES)):
        if variable_index == 0:
            finite = result.finite_mask[:, :, variable_index]
            np.testing.assert_array_equal(
                result.values[:, :, variable_index][finite],
                np.zeros(int(finite.sum()), dtype=np.float32),
            )
        else:
            np.testing.assert_array_equal(
                result.values[:, :, variable_index],
                np.full(
                    (4, 2, 120, 240),
                    float(variable_index),
                    dtype=np.float32,
                ),
            )

    np.testing.assert_allclose(
        result.coordinates[:, 4, 0, 0],
        np.asarray([0.5, 0.9], dtype=np.float32),
    )


def test_reads_sequence_from_netcdf(tmp_path: Path) -> None:
    path = tmp_path / "sequence.nc"
    _dataset().to_netcdf(path, engine="netcdf4")

    result = read_spatiotemporal_netcdf_file(
        path,
        variables=VARIABLES,
    )

    assert result.values.shape == (4, 2, 6, 120, 240)
    assert result.variables == VARIABLES


def test_rejects_wrong_sequence_length() -> None:
    dataset = _dataset().isel(time=slice(0, 3))

    with pytest.raises(
        AssertionError,
        match="time size 3 != 4",
    ):
        build_spatiotemporal_file_tensor(
            dataset,
            variables=VARIABLES,
        )


def test_rejects_nonmonotonic_time() -> None:
    dataset = _dataset().assign_coords(
        time=np.asarray(
            [
                "2020-01-01T00:00:00",
                "2020-01-01T00:10:00",
                "2020-01-01T00:05:00",
                "2020-01-01T00:15:00",
            ],
            dtype="datetime64[s]",
        )
    )

    with pytest.raises(
        AssertionError,
        match="strictly increasing",
    ):
        build_spatiotemporal_file_tensor(
            dataset,
            variables=VARIABLES,
        )
