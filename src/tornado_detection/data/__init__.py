"""Data inspection and manifest-building utilities."""

from tornado_detection.data.manifest import (
    EXPECTED_CATEGORIES,
    EXPECTED_SPLITS,
    MANIFEST_SCHEMA_VERSION,
    PRIMARY_RADAR_VARIABLES,
    ManifestBuildResult,
    ManifestValidation,
    ParsedMember,
    build_archive_manifests,
    parse_tornet_member,
    validate_manifests,
    write_manifest_artifacts,
)

from tornado_detection.data.modeling import (
    ModelingExclusion,
    ModelingManifestBuild,
    build_modeling_manifest,
    write_modeling_manifest_artifacts,
)

from tornado_detection.data.training_index import (
    CANONICAL_MANIFEST_SOURCES,
    EXPECTED_CANONICAL_FRAME_COUNT,
    EXPECTED_YEARS,
    assign_model_splits,
    build_validation_group_keys,
    canonical_manifest_directory,
    load_canonical_frame_index,
    summarize_canonical_frame_index,
    summarize_model_splits,
    validate_canonical_frame_index,
)

from tornado_detection.data.tensor import (
    DEFAULT_RADAR_VARIABLES,
    EXPECTED_FRAME_SHAPE,
    EXPECTED_RADAR_DIMENSIONS,
    EXPECTED_SWEEP_COUNT,
    FrameTensor,
    build_frame_tensor,
    read_netcdf_frame,
)

__all__ = [
    "DEFAULT_RADAR_VARIABLES",
    "EXPECTED_FRAME_SHAPE",
    "EXPECTED_RADAR_DIMENSIONS",
    "EXPECTED_SWEEP_COUNT",
    "FrameTensor",
    "build_frame_tensor",
    "read_netcdf_frame",
    "summarize_model_splits",
    "build_validation_group_keys",
    "assign_model_splits",
    "validate_canonical_frame_index",
    "summarize_canonical_frame_index",
    "load_canonical_frame_index",
    "canonical_manifest_directory",
    "EXPECTED_YEARS",
    "EXPECTED_CANONICAL_FRAME_COUNT",
    "CANONICAL_MANIFEST_SOURCES",
    "EXPECTED_CATEGORIES",
    "EXPECTED_SPLITS",
    "MANIFEST_SCHEMA_VERSION",
    "PRIMARY_RADAR_VARIABLES",
    "ManifestBuildResult",
    "ManifestValidation",
    "ModelingExclusion",
    "ModelingManifestBuild",
    "ParsedMember",
    "build_archive_manifests",
    "build_modeling_manifest",
    "parse_tornet_member",
    "validate_manifests",
    "write_manifest_artifacts",
    "write_modeling_manifest_artifacts",
]
