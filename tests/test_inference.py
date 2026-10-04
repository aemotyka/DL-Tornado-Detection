"""Tests for supported bundle inference and the command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

torch = pytest.importorskip("torch")

from tornado_detection.cli import main  # noqa: E402
from tornado_detection.inference import (  # noqa: E402
    EXPECTED_CHECKPOINT_ARTIFACT_KIND,
    TornadoDetector,
)
from tornado_detection.models import (  # noqa: E402
    SpatiotemporalTornadoDetector,
)


VARIABLES = (
    "DBZ",
    "KDP",
    "RHOHV",
    "VEL",
    "WIDTH",
    "ZDR",
)
CHANNEL_ORDER = tuple(
    f"{variable}_sweep_{sweep}"
    for variable in VARIABLES
    for sweep in range(2)
)
COORDINATE_NAMES = (
    "range_100km",
    "inverse_range_100km",
    "sin_azimuth",
    "cos_azimuth",
    "elevation_degrees",
)


def _write_input(path: Path) -> None:
    shape = (4, 120, 240, 2)
    data_vars = {
        name: (
            ("time", "azimuth", "range", "sweep"),
            np.full(shape, index, dtype=np.float32),
        )
        for index, name in enumerate(VARIABLES)
    }
    data_vars["DBZ"][1][0, 0, 0, 0] = np.nan
    data_vars["range_folded_mask"] = (
        ("time", "azimuth", "range", "sweep"),
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
    dataset = xr.Dataset(
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
    dataset.to_netcdf(path, engine="netcdf4")


def _config() -> dict:
    return {
        "schema_version": 1,
        "package_version": "1.0.0",
        "model_class": "SpatiotemporalTornadoDetector",
        "model": {
            "variable_count": 6,
            "sweep_count": 2,
            "coordinate_count": 5,
            "encoder_widths": [2],
            "temporal_channels": 2,
            "pooling_temperature": 1.0,
            "pooling_topk_fraction": 0.05,
            "bidirectional_temporal": True,
        },
        "variables": list(VARIABLES),
        "channel_order": list(CHANNEL_ORDER),
        "coordinate_names": list(COORDINATE_NAMES),
        "sequence_shape": [4, 2, 6, 120, 240],
        "temporal_context": "full four-frame offline sequence",
        "offline_only": True,
        "normalization": {
            "bundle_filename": "normalization.json",
            "source_experiment": "test",
        },
        "data_split": {
            "validation_fraction": 0.20,
            "validation_seed": 20260913,
        },
        "training_seed": 20260928,
        "checkpoint": {
            "bundle_filename": "best_model.pt",
            "epoch": 23,
        },
        "decision_threshold": 0.6,
        "score_semantics": "uncalibrated sigmoid score",
    }


def _write_bundle(path: Path) -> None:
    path.mkdir()
    config = _config()
    (path / "model_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    (path / "normalization.json").write_text(
        json.dumps(
            {
                "variables": list(VARIABLES),
                "channel_order": list(CHANNEL_ORDER),
                "means": [0.0] * 12,
                "standard_deviations": [1.0] * 12,
            },
            indent=2,
        )
        + "\n"
    )

    model = SpatiotemporalTornadoDetector(**config["model"])
    for parameter in model.parameters():
        parameter.data.zero_()
    torch.save(
        {
            "artifact_kind": EXPECTED_CHECKPOINT_ARTIFACT_KIND,
            "epoch": 23,
            "model_state_dict": model.state_dict(),
        },
        path / "best_model.pt",
    )


def test_predict_file_is_deterministic(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    input_path = tmp_path / "sequence.nc"
    _write_bundle(bundle)
    _write_input(input_path)

    detector = TornadoDetector.from_bundle(bundle, device="cpu")
    first = detector.predict_file(input_path)
    second = detector.predict_file(input_path)

    assert first.to_dict() == second.to_dict()
    assert first.frame_scores == pytest.approx([0.5] * 4)
    assert first.frame_predictions == [0, 0, 0, 0]
    assert first.threshold == 0.6
    assert first.variables == list(VARIABLES)
    assert first.sequence_shape == [4, 2, 6, 120, 240]
    assert first.bidirectional_temporal is True
    assert first.offline_only is True
    assert first.checkpoint_epoch == 23
    assert first.score_semantics == "uncalibrated sigmoid score"


def test_cli_writes_prediction_json(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    input_path = tmp_path / "sequence.nc"
    output_path = tmp_path / "prediction.json"
    _write_bundle(bundle)
    _write_input(input_path)

    exit_code = main(
        [
            "--bundle",
            str(bundle),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    output = json.loads(output_path.read_text())
    assert output["frame_scores"] == pytest.approx([0.5] * 4)
    assert output["frame_predictions"] == [0, 0, 0, 0]
    assert "probabilities" not in output


def test_rejects_missing_bundle_files(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    with pytest.raises(
        FileNotFoundError,
        match=(
            "best_model.pt, model_config.json, normalization.json"
        ),
    ):
        TornadoDetector.from_bundle(bundle, device="cpu")


def test_rejects_normalization_channel_mismatch(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    normalization_path = bundle / "normalization.json"
    normalization = json.loads(normalization_path.read_text())
    normalization["channel_order"] = list(reversed(CHANNEL_ORDER))
    normalization_path.write_text(json.dumps(normalization))

    with pytest.raises(
        ValueError,
        match="Normalization channel_order",
    ):
        TornadoDetector.from_bundle(bundle, device="cpu")


def test_checkpoint_loading_is_strict(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    checkpoint_path = bundle / "best_model.pt"
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    checkpoint["model_state_dict"].pop(
        next(iter(checkpoint["model_state_dict"]))
    )
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(RuntimeError, match="Missing key"):
        TornadoDetector.from_bundle(bundle, device="cpu")
