"""Frame-level radar tensor construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr


EXPECTED_RADAR_DIMENSIONS = (
    "time",
    "azimuth",
    "range",
    "sweep",
)
EXPECTED_FRAME_SHAPE = (
    120,
    240,
)
EXPECTED_SWEEP_COUNT = 2
EXPECTED_SEQUENCE_LENGTH = 4
TORNET_SWEEP_ELEVATIONS_DEGREES = (
    0.5,
    0.9,
)
DEFAULT_RADAR_VARIABLES = (
    "DBZ",
    "VEL",
)


@dataclass(frozen=True)
class FrameTensor:
    """One manifest-selected radar tensor and label."""

    values: np.ndarray
    label: int
    frame_index: int
    variables: tuple[str, ...]


@dataclass(frozen=True)
class FileTensor:
    """Every frame from one TorNet file, decoded in one open."""

    values: np.ndarray
    labels: np.ndarray
    variables: tuple[str, ...]


@dataclass(frozen=True)
class SpatiotemporalFileTensor:
    """One complete TorNet sequence with explicit physical axes."""

    values: np.ndarray
    finite_mask: np.ndarray
    range_folded_mask: np.ndarray
    coordinates: np.ndarray
    labels: np.ndarray
    times_unix_seconds: np.ndarray
    variables: tuple[str, ...]
    coordinate_names: tuple[str, ...]


SPATIOTEMPORAL_COORDINATE_NAMES = (
    "range_100km",
    "inverse_range_100km",
    "sin_azimuth",
    "cos_azimuth",
    "elevation_degrees",
)


def _required_variable(
    dataset: xr.Dataset,
    name: str,
) -> xr.DataArray:
    if name not in dataset:
        raise KeyError(
            f"Dataset is missing required variable {name!r}"
        )
    return dataset[name]


def _limits(
    dataset: xr.Dataset,
    name: str,
) -> tuple[float, float]:
    variable = _required_variable(dataset, name)

    if tuple(variable.dims) != ("lims",):
        raise AssertionError(
            f"{name} dimensions {tuple(variable.dims)} != "
            "('lims',)"
        )

    values = np.asarray(variable.values, dtype=np.float64)

    if values.shape != (2,):
        raise AssertionError(
            f"{name} shape {values.shape} != (2,)"
        )
    if not np.isfinite(values).all():
        raise AssertionError(
            f"{name} contains non-finite values"
        )
    if not values[0] < values[1]:
        raise AssertionError(
            f"{name} limits are not increasing"
        )

    return float(values[0]), float(values[1])


def _build_spatiotemporal_coordinates(
    dataset: xr.Dataset,
) -> np.ndarray:
    azimuth_lower, azimuth_upper = _limits(
        dataset,
        "azimuth_limits",
    )
    range_lower, range_upper = _limits(
        dataset,
        "range_limits",
    )

    azimuth_radians = np.deg2rad(
        np.linspace(
            azimuth_lower,
            azimuth_upper,
            EXPECTED_FRAME_SHAPE[0],
            dtype=np.float64,
        )
    )
    range_100km = (
        np.linspace(
            range_lower,
            range_upper,
            EXPECTED_FRAME_SHAPE[1],
            dtype=np.float64,
        )
        * 1.0e-5
    )
    safe_range = np.maximum(
        range_100km,
        0.02125,
    )

    range_grid = np.broadcast_to(
        range_100km[None, :],
        EXPECTED_FRAME_SHAPE,
    )
    inverse_range_grid = np.broadcast_to(
        (1.0 / safe_range)[None, :],
        EXPECTED_FRAME_SHAPE,
    )
    sin_azimuth_grid = np.broadcast_to(
        np.sin(azimuth_radians)[:, None],
        EXPECTED_FRAME_SHAPE,
    )
    cos_azimuth_grid = np.broadcast_to(
        np.cos(azimuth_radians)[:, None],
        EXPECTED_FRAME_SHAPE,
    )

    sweep_coordinates = []

    for elevation in TORNET_SWEEP_ELEVATIONS_DEGREES:
        sweep_coordinates.append(
            np.stack(
                (
                    range_grid,
                    inverse_range_grid,
                    sin_azimuth_grid,
                    cos_azimuth_grid,
                    np.full(
                        EXPECTED_FRAME_SHAPE,
                        elevation,
                        dtype=np.float64,
                    ),
                ),
                axis=0,
            )
        )

    return np.stack(
        sweep_coordinates,
        axis=0,
    ).astype(np.float32)


def build_spatiotemporal_file_tensor(
    dataset: xr.Dataset,
    *,
    variables: Sequence[str],
) -> SpatiotemporalFileTensor:
    """Build the complete four-frame model input without flattening axes."""

    variable_names = tuple(variables)

    if not variable_names:
        raise ValueError(
            "At least one radar variable is required"
        )
    if len(variable_names) != len(set(variable_names)):
        raise ValueError(
            "Radar variables must be unique"
        )

    expected_sizes = {
        "time": EXPECTED_SEQUENCE_LENGTH,
        "sweep": EXPECTED_SWEEP_COUNT,
        "azimuth": EXPECTED_FRAME_SHAPE[0],
        "range": EXPECTED_FRAME_SHAPE[1],
    }

    for dimension, expected_size in expected_sizes.items():
        actual_size = dataset.sizes.get(dimension)
        if actual_size != expected_size:
            raise AssertionError(
                f"{dimension} size {actual_size} != {expected_size}"
            )

    arrays = []

    for name in variable_names:
        variable = _required_variable(dataset, name)

        if tuple(variable.dims) != EXPECTED_RADAR_DIMENSIONS:
            raise AssertionError(
                f"{name} dimensions {tuple(variable.dims)} != "
                f"{EXPECTED_RADAR_DIMENSIONS}"
            )

        arrays.append(
            np.asarray(
                variable.transpose(
                    "time",
                    "sweep",
                    "azimuth",
                    "range",
                ).values,
                dtype=np.float32,
            )
        )

    values = np.stack(arrays, axis=2)
    expected_values_shape = (
        EXPECTED_SEQUENCE_LENGTH,
        EXPECTED_SWEEP_COUNT,
        len(variable_names),
        *EXPECTED_FRAME_SHAPE,
    )

    if values.shape != expected_values_shape:
        raise AssertionError(
            f"Sequence values shape {values.shape} != "
            f"{expected_values_shape}"
        )

    finite_mask = np.isfinite(values)
    range_folded = _required_variable(
        dataset,
        "range_folded_mask",
    )

    if tuple(range_folded.dims) != EXPECTED_RADAR_DIMENSIONS:
        raise AssertionError(
            "range_folded_mask dimensions "
            f"{tuple(range_folded.dims)} != "
            f"{EXPECTED_RADAR_DIMENSIONS}"
        )

    range_folded_mask = np.asarray(
        range_folded.transpose(
            "time",
            "sweep",
            "azimuth",
            "range",
        ).values,
        dtype=np.float32,
    )

    labels_variable = _required_variable(
        dataset,
        "frame_labels",
    )

    if tuple(labels_variable.dims) != ("time",):
        raise AssertionError(
            "frame_labels dimensions "
            f"{tuple(labels_variable.dims)} != ('time',)"
        )

    labels = np.asarray(
        labels_variable.values,
        dtype=np.uint8,
    )

    if labels.shape != (EXPECTED_SEQUENCE_LENGTH,):
        raise AssertionError(
            f"labels shape {labels.shape} is invalid"
        )
    if not np.isin(labels, (0, 1)).all():
        raise AssertionError(
            "Sequence contains nonbinary frame labels"
        )

    time_variable = _required_variable(dataset, "time")
    times = np.asarray(time_variable.values)

    if times.shape != (EXPECTED_SEQUENCE_LENGTH,):
        raise AssertionError(
            f"time shape {times.shape} is invalid"
        )

    if np.issubdtype(times.dtype, np.datetime64):
        times_unix_seconds = times.astype(
            "datetime64[s]"
        ).astype(np.int64)
    elif np.issubdtype(times.dtype, np.number):
        times_unix_seconds = times.astype(np.int64)
    else:
        raise AssertionError(
            f"Unsupported time dtype: {times.dtype}"
        )

    if not np.all(np.diff(times_unix_seconds) > 0):
        raise AssertionError(
            "Sequence times must be strictly increasing"
        )

    coordinates = _build_spatiotemporal_coordinates(dataset)

    return SpatiotemporalFileTensor(
        values=values,
        finite_mask=finite_mask,
        range_folded_mask=range_folded_mask,
        coordinates=coordinates,
        labels=labels,
        times_unix_seconds=times_unix_seconds,
        variables=variable_names,
        coordinate_names=SPATIOTEMPORAL_COORDINATE_NAMES,
    )


def read_spatiotemporal_netcdf_file(
    path: str | Path,
    *,
    variables: Sequence[str],
) -> SpatiotemporalFileTensor:
    """Read one complete TorNet sequence for the temporal model."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"NetCDF file does not exist: {path}"
        )

    with xr.open_dataset(path, engine="netcdf4") as dataset:
        result = build_spatiotemporal_file_tensor(
            dataset,
            variables=variables,
        )

        return SpatiotemporalFileTensor(
            values=result.values.copy(),
            finite_mask=result.finite_mask.copy(),
            range_folded_mask=(
                result.range_folded_mask.copy()
            ),
            coordinates=result.coordinates.copy(),
            labels=result.labels.copy(),
            times_unix_seconds=(
                result.times_unix_seconds.copy()
            ),
            variables=result.variables,
            coordinate_names=result.coordinate_names,
        )


def _validate_frame_index(
    dataset: xr.Dataset,
    frame_index: int,
) -> None:
    if not isinstance(frame_index, int):
        raise TypeError(
            "frame_index must be an integer"
        )

    if "time" not in dataset.sizes:
        raise AssertionError(
            "Dataset does not contain a time dimension"
        )

    time_size = int(dataset.sizes["time"])

    if not 0 <= frame_index < time_size:
        raise IndexError(
            f"frame_index {frame_index} is outside "
            f"the available range 0..{time_size - 1}"
        )


def build_frame_tensor(
    dataset: xr.Dataset,
    frame_index: int,
    *,
    variables: Sequence[str] = (
        DEFAULT_RADAR_VARIABLES
    ),
) -> FrameTensor:
    """Construct an azimuth/range/channel tensor by dimension name."""

    _validate_frame_index(
        dataset,
        frame_index,
    )

    variable_names = tuple(variables)

    if not variable_names:
        raise ValueError(
            "At least one radar variable is required"
        )

    if len(variable_names) != len(
        set(variable_names)
    ):
        raise ValueError(
            "Radar variables must be unique"
        )

    arrays: list[np.ndarray] = []

    for variable_name in variable_names:
        if variable_name not in dataset:
            raise KeyError(
                f"Dataset is missing radar variable "
                f"{variable_name!r}"
            )

        variable = dataset[variable_name]
        actual_dimensions = tuple(variable.dims)

        if (
            actual_dimensions
            != EXPECTED_RADAR_DIMENSIONS
        ):
            raise AssertionError(
                f"{variable_name} dimensions "
                f"{actual_dimensions} != "
                f"{EXPECTED_RADAR_DIMENSIONS}"
            )

        frame = (
            variable.isel(time=frame_index)
            .transpose(
                "azimuth",
                "range",
                "sweep",
            )
            .values
        )

        expected_variable_shape = (
            *EXPECTED_FRAME_SHAPE,
            EXPECTED_SWEEP_COUNT,
        )

        if (
            frame.shape
            != expected_variable_shape
        ):
            raise AssertionError(
                f"{variable_name} frame shape "
                f"{frame.shape} != "
                f"{expected_variable_shape}"
            )

        arrays.append(
            np.asarray(
                frame,
                dtype=np.float32,
            )
        )

    if "frame_labels" not in dataset:
        raise KeyError(
            "Dataset is missing frame_labels"
        )

    label_variable = dataset["frame_labels"]

    if tuple(label_variable.dims) != (
        "time",
    ):
        raise AssertionError(
            "frame_labels dimensions "
            f"{tuple(label_variable.dims)} != "
            "('time',)"
        )

    label = int(
        label_variable.isel(
            time=frame_index
        ).item()
    )

    if label not in {
        0,
        1,
    }:
        raise AssertionError(
            f"Unexpected frame label: {label}"
        )

    values = np.concatenate(
        arrays,
        axis=-1,
    )

    expected_tensor_shape = (
        *EXPECTED_FRAME_SHAPE,
        (
            len(variable_names)
            * EXPECTED_SWEEP_COUNT
        ),
    )

    if values.shape != expected_tensor_shape:
        raise AssertionError(
            f"Tensor shape {values.shape} != "
            f"{expected_tensor_shape}"
        )

    if values.dtype != np.float32:
        raise AssertionError(
            f"Tensor dtype {values.dtype} != float32"
        )

    return FrameTensor(
        values=values,
        label=label,
        frame_index=frame_index,
        variables=variable_names,
    )


def build_file_tensor(
    dataset: xr.Dataset,
    *,
    variables: Sequence[str] = (
        DEFAULT_RADAR_VARIABLES
    ),
) -> FileTensor:
    """Construct tensors for every time frame in an open dataset."""

    if "time" not in dataset.sizes:
        raise AssertionError(
            "Dataset does not contain a time dimension"
        )

    frame_count = int(dataset.sizes["time"])

    if frame_count < 1:
        raise AssertionError(
            "Dataset contains no time frames"
        )

    frames = [
        build_frame_tensor(
            dataset,
            frame_index,
            variables=variables,
        )
        for frame_index in range(frame_count)
    ]

    values = np.stack(
        [frame.values for frame in frames],
        axis=0,
    ).astype(np.float32, copy=False)
    labels = np.asarray(
        [frame.label for frame in frames],
        dtype=np.uint8,
    )

    return FileTensor(
        values=values,
        labels=labels,
        variables=frames[0].variables,
    )


def read_netcdf_frame(
    path: str | Path,
    frame_index: int,
    *,
    variables: Sequence[str] = (
        DEFAULT_RADAR_VARIABLES
    ),
) -> FrameTensor:
    """Read one frame from one extracted TorNet NetCDF file."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"NetCDF file does not exist: {path}"
        )

    with xr.open_dataset(
        path,
        engine="netcdf4",
    ) as dataset:
        result = build_frame_tensor(
            dataset,
            frame_index,
            variables=variables,
        )

        values = result.values.copy()

    return FrameTensor(
        values=values,
        label=result.label,
        frame_index=result.frame_index,
        variables=result.variables,
    )


def read_netcdf_file(
    path: str | Path,
    *,
    variables: Sequence[str] = (
        DEFAULT_RADAR_VARIABLES
    ),
) -> FileTensor:
    """Read every frame from one extracted TorNet NetCDF file."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"NetCDF file does not exist: {path}"
        )

    with xr.open_dataset(
        path,
        engine="netcdf4",
    ) as dataset:
        result = build_file_tensor(
            dataset,
            variables=variables,
        )

        values = result.values.copy()
        labels = result.labels.copy()

    return FileTensor(
        values=values,
        labels=labels,
        variables=result.variables,
    )
