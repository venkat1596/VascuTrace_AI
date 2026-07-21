from vascutrace.contracts import (
    Finding,
    QualityControl,
    QuantitativeMeasurements,
    ResearchReport,
)
from vascutrace.report_verifier import verify_report


def metrics() -> QuantitativeMeasurements:
    return QuantitativeMeasurements(
        target_suvmax=1.48,
        target_suvmean=1.17,
        contralateral_suvmax=0.91,
        contralateral_suvmean=0.78,
        asymmetry_index=0.63,
        metabolic_volume_ml=1.74,
        longitudinal_extent_mm=42.0,
        quality_flags=["partial_volume_risk"],
    )


def report(**changes) -> ResearchReport:
    values = {
        "case_id": "quadra_subject_006_scan_1_seed_182",
        "case_type": "synthetic_research_case",
        "finding": Finding(
            laterality="right",
            target_region="iliac_proximal_femoral_corridor",
            abnormality_score=0.91,
        ),
        "quantitative_measurements": metrics(),
        "quality_control": QualityControl(
            partial_volume_risk=True,
            misregistration_risk=False,
            flags=["partial_volume_risk"],
        ),
        "interpretation": "Synthetic research case with simulated asymmetric uptake.",
        "limitations": ["Not for clinical use."],
    }
    values.update(changes)
    return ResearchReport(**values)


def test_matching_report_is_accepted() -> None:
    result = verify_report(report(), metrics(), expected_laterality="right")
    assert result.accepted
    assert result.issues == ()


def test_changed_measurement_is_rejected() -> None:
    changed = metrics().model_copy(update={"target_suvmax": 9.9})
    result = verify_report(
        report(quantitative_measurements=changed),
        metrics(),
        expected_laterality="right",
    )
    assert not result.accepted
    assert "measurement_mismatch" in {issue.code for issue in result.issues}


def test_laterality_and_diagnostic_claim_are_rejected() -> None:
    changed_finding = report().finding.model_copy(update={"laterality": "left"})
    result = verify_report(
        report(
            finding=changed_finding,
            interpretation="This synthetic scan confirms diagnosis of disease.",
        ),
        metrics(),
        expected_laterality="right",
    )
    codes = {issue.code for issue in result.issues}
    assert {"laterality_mismatch", "diagnosis"} <= codes
