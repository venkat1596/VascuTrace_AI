"""Versioned contracts at the imaging-to-agent boundary."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QuantitativeMeasurements(StrictModel):
    target_suvmax: float | None = Field(default=None, ge=0)
    target_suvmean: float | None = Field(default=None, ge=0)
    contralateral_suvmax: float | None = Field(default=None, ge=0)
    contralateral_suvmean: float | None = Field(default=None, ge=0)
    asymmetry_index: float | None = None
    metabolic_volume_ml: float | None = Field(default=None, ge=0)
    longitudinal_extent_mm: float | None = Field(default=None, ge=0)
    quality_flags: list[str] = Field(default_factory=list)
    null_reasons: dict[str, str] = Field(default_factory=dict)


class ModelInput(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    case_id: str
    pet_left_path: Path
    pet_right_path: Path
    ct_left_path: Path
    ct_right_path: Path
    voxel_spacing_mm: tuple[float, float, float]
    case_type: Literal["synthetic", "research_scan"]
    simulation_parameters_path: Path | None = None


class ModelOutput(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    case_id: str
    model_name: str
    model_version: str
    laterality: Literal["left", "right", "bilateral", "none"]
    abnormality_score: float = Field(
        ge=0,
        le=1,
        description="Uncalibrated model score; not a clinical probability.",
    )
    mask_path: Path
    metrics_path: Path
    overlay_path: Path
    runtime_seconds: float = Field(ge=0)


class Finding(StrictModel):
    laterality: Literal["left", "right", "bilateral", "none"]
    target_region: str
    abnormality_score: float = Field(
        ge=0,
        le=1,
        description="Uncalibrated model score copied from ModelOutput.",
    )


class QualityControl(StrictModel):
    partial_volume_risk: bool
    misregistration_risk: bool
    flags: list[str] = Field(default_factory=list)


class EvidenceReference(StrictModel):
    citation_id: str
    title: str
    source_url: str | None = None
    supporting_text: str


class ResearchReport(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    case_id: str
    case_type: Literal["synthetic_research_case", "research_scan"]
    finding: Finding
    quantitative_measurements: QuantitativeMeasurements
    quality_control: QualityControl
    interpretation: str
    evidence: list[EvidenceReference] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    research_only_warning: Literal[True] = True
