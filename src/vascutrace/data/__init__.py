"""VascuTrace data pipeline package (Phase 2).

Research prototype. Trained and evaluated using simulated vascular-like
abnormalities, not confirmed human post-angioplasty lesions.

Re-exports the public surface of ``ingest`` (dataset discovery + label
-semantics pin + geometry validation), ``split`` (deterministic
sex-stratified subject split + leakage proof), and ``crops``/``contract``
(bilateral iliac-corridor crop pipeline + versioned crop/tensor schema).
"""

from src.vascutrace.data.contract import (
    ADJACENT_SLICE_COUNT_K,
    AXIAL_ADJACENT_SLICE_AXIS,
    CROP_SCHEMA_VERSION,
    FIXED_CROP_SHAPE,
    ILIAC_LABEL_LEFT,
    ILIAC_LABEL_RIGHT,
    CropBundle,
    CropIntegrityError,
    CropIntegrityErrorCode,
    NETWORK_TENSOR_FIELDS,
    load_crop_bundle,
    make_crop_bundle,
    reflect_volume,
    save_crop_bundle,
)
from src.vascutrace.data.crops import (
    DEFAULT_CROP_MARGIN_MM,
    REFLECTION_RESIDUAL_QC_THRESHOLD_MM,
    CropPipelineReport,
    compute_session_crop,
    run_crop_pipeline,
)
from src.vascutrace.data.ingest import (
    RESEARCH_PROTOTYPE_WARNING,
    DatasetManifest,
    IngestError,
    IngestResult,
    LabelPinError,
    SessionPaths,
    discover_dataset,
    ingest_dataset,
    pin_label_semantics,
    validate_dataset_grids,
)
from src.vascutrace.data.split import (
    N_TEST_SUBJECTS,
    N_VAL_SUBJECTS,
    SPLIT_SEED,
    LeakageProofResult,
    SubjectSplit,
    leakage_proof,
    load_subject_sex_table,
    load_subject_split,
    save_subject_split,
    stratified_subject_split,
)

__all__ = [
    "RESEARCH_PROTOTYPE_WARNING",
    # ingest
    "DatasetManifest",
    "SessionPaths",
    "IngestError",
    "IngestResult",
    "LabelPinError",
    "discover_dataset",
    "ingest_dataset",
    "pin_label_semantics",
    "validate_dataset_grids",
    # split
    "SPLIT_SEED",
    "N_VAL_SUBJECTS",
    "N_TEST_SUBJECTS",
    "SubjectSplit",
    "LeakageProofResult",
    "stratified_subject_split",
    "leakage_proof",
    "load_subject_sex_table",
    "save_subject_split",
    "load_subject_split",
    # crops
    "DEFAULT_CROP_MARGIN_MM",
    "REFLECTION_RESIDUAL_QC_THRESHOLD_MM",
    "CropPipelineReport",
    "compute_session_crop",
    "run_crop_pipeline",
    # contract
    "CROP_SCHEMA_VERSION",
    "FIXED_CROP_SHAPE",
    "AXIAL_ADJACENT_SLICE_AXIS",
    "ADJACENT_SLICE_COUNT_K",
    "ILIAC_LABEL_LEFT",
    "ILIAC_LABEL_RIGHT",
    "NETWORK_TENSOR_FIELDS",
    "CropBundle",
    "CropIntegrityError",
    "CropIntegrityErrorCode",
    "make_crop_bundle",
    "save_crop_bundle",
    "load_crop_bundle",
    "reflect_volume",
]
