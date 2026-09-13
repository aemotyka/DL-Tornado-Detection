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
