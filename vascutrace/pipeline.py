"""One-command synthetic case pipeline and artifact manifest generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from vascutrace.orchestrator import run_first_checkpoint


SCIENTIFIC_WARNING = (
    "Research prototype. Trained and evaluated using simulated vascular-like "
    "abnormalities, not confirmed human post-angioplasty lesions."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _format_measurement(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.4f}"


def run_complete_case(output_root: Path | str = "outputs/complete_case") -> Path:
    root = Path(output_root)
    result = run_first_checkpoint(str(root))
    payload = result.payload
    case_dir = Path(payload["case"]["case_dir"])
    report_json = case_dir / "report.json"
    report_md = case_dir / "report.md"
    trace_json = case_dir / "trace.json"
    report = dict(payload["report"])
    report["limitations"] = [*report["limitations"], SCIENTIFIC_WARNING]
    report_json.write_text(json.dumps(report, indent=2) + "\n")
    metrics = report["quantitative_measurements"]
    markdown_limitations = [
        item for item in report["limitations"] if item != SCIENTIFIC_WARNING
    ]
    report_md.write_text(
        "\n".join(
            [
                "# VascuTrace synthetic research report",
                "",
                f"**{SCIENTIFIC_WARNING}**",
                "",
                f"Laterality: {report['finding']['laterality']}",
                f"Target SUVmax: {_format_measurement(metrics['target_suvmax'])}",
                "Contralateral SUVmax: "
                f"{_format_measurement(metrics['contralateral_suvmax'])}",
                f"Asymmetry index: {_format_measurement(metrics['asymmetry_index'])}",
                "",
                report["interpretation"],
                "",
                "## Limitations",
                *[f"- {item}" for item in markdown_limitations],
                "",
            ]
        ),
        encoding="utf-8",
    )
    trace_json.write_text(json.dumps(result.trace, indent=2) + "\n")
    artifacts = sorted(path for path in case_dir.iterdir() if path.is_file())
    manifest = {
        "case_id": payload["case"]["case_id"],
        "research_only": True,
        "verification": payload["verification"],
        "artifacts": [
            {
                "name": path.name,
                "relative_path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in artifacts
        ],
    }
    manifest_path = case_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path
