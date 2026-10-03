"""Supported inference interface for the frozen V4 model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tornado_detection import __version__
from tornado_detection.data import (
    SPATIOTEMPORAL_COORDINATE_NAMES,
    read_spatiotemporal_netcdf_file,
)
from tornado_detection.models import SpatiotemporalTornadoDetector


EXPECTED_CHECKPOINT_ARTIFACT_KIND = (
    "spatiotemporal_bidirectional_all_year_v4_best_model"
)
REQUIRED_BUNDLE_FILES = (
    "best_model.pt",
    "model_config.json",
    "normalization.json",
)


@dataclass(frozen=True)
class PredictionResult:
    """Serializable scores and decisions for one four-frame sequence."""

    input_path: str
    frame_scores: list[float]
    frame_predictions: list[int]
    frame_times_unix_seconds: list[int]
    threshold: float
    variables: list[str]
    sequence_shape: list[int]
    bidirectional_temporal: bool
    offline_only: bool
    checkpoint_epoch: int
    score_semantics: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error

    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    return value


def _require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{context} keys do not match; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    return device


class TornadoDetector:
    """Load and run the frozen offline V4 tornado detector."""

    def __init__(
        self,
        *,
        model: SpatiotemporalTornadoDetector,
        config: dict[str, Any],
        means: np.ndarray,
        standard_deviations: np.ndarray,
        checkpoint_epoch: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.config = config
        self.means = means
        self.standard_deviations = standard_deviations
        self.checkpoint_epoch = checkpoint_epoch
        self.device = device

    @classmethod
    def from_bundle(
        cls,
        bundle_directory: str | Path,
        *,
        device: str = "auto",
    ) -> TornadoDetector:
        bundle = Path(bundle_directory)
        if not bundle.is_dir():
            raise FileNotFoundError(
                f"Bundle directory does not exist: {bundle}"
            )

        missing = [
            name
            for name in REQUIRED_BUNDLE_FILES
            if not (bundle / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Bundle is missing required files: "
                + ", ".join(missing)
            )

        config = _load_json(bundle / "model_config.json")
        cls._validate_config(config)
        normalization = _load_json(bundle / "normalization.json")
        means, standard_deviations = cls._validate_normalization(
            normalization,
            config,
        )

        resolved_device = _resolve_device(device)
        model = SpatiotemporalTornadoDetector(**config["model"])

        checkpoint = torch.load(
            bundle / config["checkpoint"]["bundle_filename"],
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must contain a dictionary")
        if checkpoint.get("artifact_kind") != (
            EXPECTED_CHECKPOINT_ARTIFACT_KIND
        ):
            raise ValueError("Unexpected checkpoint artifact_kind")

        checkpoint_epoch = int(checkpoint.get("epoch", -1))
        expected_epoch = int(config["checkpoint"]["epoch"])
        if checkpoint_epoch != expected_epoch:
            raise ValueError(
                f"Checkpoint epoch {checkpoint_epoch} != "
                f"configured epoch {expected_epoch}"
            )
        if "model_state_dict" not in checkpoint:
            raise ValueError("Checkpoint is missing model_state_dict")

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
        model.to(resolved_device)
        model.eval()

        return cls(
            model=model,
            config=config,
            means=means,
            standard_deviations=standard_deviations,
            checkpoint_epoch=checkpoint_epoch,
            device=resolved_device,
        )

    @staticmethod
    def _validate_config(config: dict[str, Any]) -> None:
        expected_keys = {
            "schema_version",
            "package_version",
            "model_class",
            "model",
            "variables",
            "channel_order",
            "coordinate_names",
            "sequence_shape",
            "temporal_context",
            "offline_only",
            "normalization",
            "data_split",
            "training_seed",
            "checkpoint",
            "decision_threshold",
            "score_semantics",
        }
        _require_exact_keys(
            config,
            expected_keys,
            context="model_config.json",
        )

        if config["schema_version"] != 1:
            raise ValueError("Unsupported model configuration schema")
        if config["package_version"] != __version__:
            raise ValueError(
                f"Bundle package version {config['package_version']!r} "
                f"does not match installed version {__version__!r}"
            )
        if config["model_class"] != "SpatiotemporalTornadoDetector":
            raise ValueError("Unsupported model_class")
        if config["offline_only"] is not True:
            raise ValueError("The frozen V4 model must be offline-only")
        if config["model"].get("bidirectional_temporal") is not True:
            raise ValueError("The frozen V4 model must be bidirectional")

        variables = config["variables"]
        if not isinstance(variables, list) or not variables:
            raise ValueError("variables must be a nonempty list")
        if len(variables) != len(set(variables)):
            raise ValueError("variables must be unique")

        model_config = config["model"]
        if model_config.get("variable_count") != len(variables):
            raise ValueError("variable_count does not match variables")
        if model_config.get("sweep_count") != 2:
            raise ValueError("sweep_count must be two")

        expected_shape = [4, 2, len(variables), 120, 240]
        if config["sequence_shape"] != expected_shape:
            raise ValueError(
                f"sequence_shape must be {expected_shape}"
            )

        expected_channels = [
            f"{variable}_sweep_{sweep}"
            for variable in variables
            for sweep in range(2)
        ]
        if config["channel_order"] != expected_channels:
            raise ValueError("channel_order does not match variables")

        coordinates = list(SPATIOTEMPORAL_COORDINATE_NAMES)
        if config["coordinate_names"] != coordinates:
            raise ValueError("coordinate_names do not match the reader")
        if model_config.get("coordinate_count") != len(coordinates):
            raise ValueError(
                "coordinate_count does not match coordinate_names"
            )

        threshold = config["decision_threshold"]
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError("decision_threshold must be in [0, 1]")
        if config["score_semantics"] != "uncalibrated sigmoid score":
            raise ValueError("Unsupported score_semantics")

        checkpoint = config["checkpoint"]
        normalization = config["normalization"]
        if checkpoint.get("bundle_filename") != "best_model.pt":
            raise ValueError("checkpoint bundle filename must be best_model.pt")
        if normalization.get("bundle_filename") != "normalization.json":
            raise ValueError(
                "normalization bundle filename must be normalization.json"
            )

    @staticmethod
    def _validate_normalization(
        normalization: dict[str, Any],
        config: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        for key in (
            "variables",
            "channel_order",
            "means",
            "standard_deviations",
        ):
            if key not in normalization:
                raise ValueError(
                    f"normalization.json is missing {key!r}"
                )

        if normalization["variables"] != config["variables"]:
            raise ValueError("Normalization variables do not match config")
        if normalization["channel_order"] != config["channel_order"]:
            raise ValueError(
                "Normalization channel_order does not match config"
            )

        variable_count = len(config["variables"])
        expected_count = 2 * variable_count
        means_flat = np.asarray(
            normalization["means"],
            dtype=np.float32,
        )
        stds_flat = np.asarray(
            normalization["standard_deviations"],
            dtype=np.float32,
        )
        if means_flat.shape != (expected_count,):
            raise ValueError(
                f"Expected {expected_count} normalization means"
            )
        if stds_flat.shape != (expected_count,):
            raise ValueError(
                f"Expected {expected_count} standard deviations"
            )
        if not np.isfinite(means_flat).all():
            raise ValueError("Normalization means must be finite")
        if not np.isfinite(stds_flat).all() or not (stds_flat > 0).all():
            raise ValueError(
                "Normalization standard deviations must be finite and positive"
            )

        means = means_flat.reshape(variable_count, 2).T
        standard_deviations = stds_flat.reshape(variable_count, 2).T
        return (
            means.reshape(1, 2, variable_count, 1, 1),
            standard_deviations.reshape(1, 2, variable_count, 1, 1),
        )

    def predict_file(self, netcdf_path: str | Path) -> PredictionResult:
        input_path = Path(netcdf_path)
        variables = tuple(self.config["variables"])
        sequence = read_spatiotemporal_netcdf_file(
            input_path,
            variables=variables,
        )

        if list(sequence.values.shape) != self.config["sequence_shape"]:
            raise ValueError("Decoded sequence shape does not match config")
        if list(sequence.coordinate_names) != self.config["coordinate_names"]:
            raise ValueError("Decoded coordinates do not match config")

        normalized = (
            (sequence.values - self.means)
            / self.standard_deviations
        ).astype(np.float32, copy=False)

        values = torch.from_numpy(normalized[None]).to(self.device)
        finite_mask = torch.from_numpy(
            sequence.finite_mask[None]
        ).to(self.device)
        range_folded_mask = torch.from_numpy(
            sequence.range_folded_mask[None]
        ).to(self.device)
        coordinates = torch.from_numpy(
            sequence.coordinates[None]
        ).to(self.device)

        with torch.inference_mode():
            logits = self.model(
                values,
                finite_mask,
                range_folded_mask,
                coordinates,
            )
            scores = torch.sigmoid(logits).cpu().numpy()[0]

        threshold = float(self.config["decision_threshold"])
        predictions = (scores >= threshold).astype(np.uint8)

        return PredictionResult(
            input_path=str(input_path),
            frame_scores=[float(value) for value in scores],
            frame_predictions=[int(value) for value in predictions],
            frame_times_unix_seconds=[
                int(value) for value in sequence.times_unix_seconds
            ],
            threshold=threshold,
            variables=list(variables),
            sequence_shape=list(sequence.values.shape),
            bidirectional_temporal=bool(
                self.config["model"]["bidirectional_temporal"]
            ),
            offline_only=bool(self.config["offline_only"]),
            checkpoint_epoch=self.checkpoint_epoch,
            score_semantics=self.config["score_semantics"],
        )
