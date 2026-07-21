import json
from pathlib import Path

from vascutrace.evaluation import run_evaluation_suite, write_evaluation_summary


def test_product_evaluation_suite_passes(tmp_path: Path) -> None:
    summary = run_evaluation_suite(tmp_path)

    assert summary.all_passed
    assert summary.passed == 6
    assert summary.failed == 0
    assert {case.category for case in summary.cases} == {
        "schema",
        "numeric_fidelity",
        "claim_safety",
        "cache_isolation",
        "service_failure",
        "trace",
    }


def test_evaluation_summary_is_serializable(tmp_path: Path) -> None:
    summary = run_evaluation_suite(tmp_path / "run")
    path = write_evaluation_summary(summary, tmp_path / "summary.json")

    saved = json.loads(path.read_text())
    assert saved["all_passed"] is True
    assert saved["passed"] == 6
