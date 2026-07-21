"""Deterministic verification for generated VascuTrace reports."""

import re
from dataclasses import dataclass

from vascutrace.contracts import QuantitativeMeasurements, ResearchReport


@dataclass(frozen=True)
class VerificationIssue:
    code: str
    message: str


@dataclass(frozen=True)
class VerificationResult:
    accepted: bool
    issues: tuple[VerificationIssue, ...]


_PROHIBITED_CLAIMS = {
    "diagnosis": re.compile(r"\b(diagnos(?:e|ed|is|tic)|confirms? disease)\b", re.I),
    "outcome_prediction": re.compile(
        r"\b(will (?:develop|progress|recur)|patient outcome|prognosis is)\b", re.I
    ),
    "real_pathology_for_synthetic": re.compile(
        r"\b(patient has|demonstrates? (?:restenosis|vasculitis|atherosclerosis))\b",
        re.I,
    ),
}


def verify_report(
    report: ResearchReport,
    source_metrics: QuantitativeMeasurements,
    *,
    expected_laterality: str,
    numerical_tolerance: float = 1e-6,
) -> VerificationResult:
    """Compare a generated report with deterministic measurements and safety rules."""

    issues: list[VerificationIssue] = []
    if report.finding.laterality != expected_laterality:
        issues.append(
            VerificationIssue(
                "laterality_mismatch",
                f"Expected {expected_laterality!r}, got {report.finding.laterality!r}.",
            )
        )

    source = source_metrics.model_dump()
    reported = report.quantitative_measurements.model_dump()
    for name, expected in source.items():
        actual = reported[name]
        matches = (
            abs(expected - actual) <= numerical_tolerance
            if isinstance(expected, float)
            else expected == actual
        )
        if not matches:
            issues.append(
                VerificationIssue(
                    "measurement_mismatch",
                    f"{name}: expected {expected!r}, got {actual!r}.",
                )
            )

    prose = " ".join([report.interpretation, *report.limitations])
    for code, pattern in _PROHIBITED_CLAIMS.items():
        if pattern.search(prose):
            issues.append(
                VerificationIssue(code, "Report contains a prohibited claim.")
            )

    if (
        report.case_type == "synthetic_research_case"
        and "synthetic" not in prose.lower()
    ):
        issues.append(
            VerificationIssue(
                "synthetic_status_omitted",
                "Synthetic reports must explicitly describe the case as synthetic.",
            )
        )

    return VerificationResult(accepted=not issues, issues=tuple(issues))
