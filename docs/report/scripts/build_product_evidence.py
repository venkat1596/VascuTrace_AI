"""Build generated-only product and collaboration evidence for the report.

The builder executes the deterministic reference product path in a temporary
directory. It publishes only synthetic views, code-owned values, ordered tool
names, evaluation outcomes, and curated public collaboration outcomes. Runtime
durations, temporary paths, private development material, raw data, and model
weights are never copied into the receipt or figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import textwrap
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPORT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = REPORT_DIR.parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vascutrace.evaluation import run_evaluation_suite  # noqa: E402
from vascutrace.pipeline import run_complete_case  # noqa: E402
from vascutrace.services import create_demo_views  # noqa: E402


WARNING = (
    "Research prototype. Trained and evaluated using simulated vascular-like "
    "abnormalities, not confirmed human post-angioplasty lesions."
)

NAVY = "#17324D"
BLUE = "#2F6690"
CYAN = "#3A8D9B"
GREEN = "#4C956C"
GOLD = "#D9A441"
RED = "#B54A4A"
PURPLE = "#6C5B7B"
LIGHT = "#EEF3F6"
MID = "#CAD7DF"
DARK = "#263238"
GRAY = "#66727A"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def _deterministic_backends() -> Iterator[None]:
    settings = {
        "VASCUTRACE_DETECTION_BACKEND": "reference",
        "VASCUTRACE_REPORT_BACKEND": "template",
        "VASCUTRACE_EVIDENCE_BACKEND": "keyword",
    }
    previous = {name: os.environ.get(name) for name in settings}
    os.environ.update(settings)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": MID,
            "axes.linewidth": 0.8,
            "text.color": DARK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, name: str, figure_dir: Path) -> None:
    fig.savefig(
        figure_dir / name,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Title": name, "Author": "VascuTrace"},
    )
    plt.close(fig)


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    color: str = BLUE,
    fill: str = LIGHT,
    fontsize: float = 9.5,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.5,
        edgecolor=color,
        facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=DARK,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.6,
            color=GRAY,
        )
    )


def _build_receipt(case_dir: Path, evaluation: Any) -> dict[str, Any]:
    parameters = _load_json(case_dir / "simulation_parameters.json")
    metrics = _load_json(case_dir / "metrics.json")
    report = _load_json(case_dir / "report.json")
    report_markdown = (case_dir / "report.md").read_text(encoding="utf-8")
    trace = _load_json(case_dir / "trace.json")
    manifest = _load_json(case_dir / "artifact_manifest.json")

    if report["quantitative_measurements"] != metrics:
        raise RuntimeError("Report measurements differ from deterministic metrics")
    if WARNING not in report["limitations"] or WARNING not in report_markdown:
        raise RuntimeError("Generated reports do not contain the permanent warning")
    if "\N{EM DASH}" in report_markdown:
        raise RuntimeError("Generated Markdown report contains an em dash")
    if not manifest["verification"]["accepted"]:
        raise RuntimeError("Generated report did not pass deterministic verification")
    if not evaluation.all_passed:
        raise RuntimeError("Product evaluation contains a failed check")

    view_names = ("pet", "ct", "fused", "overlay", "bilateral")
    views = [
        {
            "name": name,
            "source_file": f"{name}.png",
            "sha256": _sha256(case_dir / f"{name}.png"),
        }
        for name in view_names
    ]

    collaboration_outcomes = [
        {
            "role": "Codex with GPT-5.6",
            "outcome": (
                "Performed every project workflow role other than code "
                "implementation: planning, primary technical decisions, scientific "
                "review, report writing, delivery review, Git and release handling, "
                "and humanizer and editorial review. It also authored the plans "
                "and instructions used for coding."
            ),
            "public_evidence": (
                "plans/VascuTrace_Publication_and_Reproducibility_Plan_2026-07-20.md; "
                "docs/report/VascuTrace_Technical_Report_2026-07-20.tex; "
                "docs/CODEX_COLLABORATION.md; docs/HACKATHON_SUBMISSION.md"
            ),
        },
        {
            "role": "Claude",
            "outcome": (
                "Implemented code only, following Codex-authored plans and "
                "instructions."
            ),
            "public_evidence": (
                "app.py; scripts/run_complete_case.py; "
                "scripts/run_product_evaluation.py; vascutrace/; tests/"
            ),
        },
        {
            "role": "Project owner",
            "outcome": (
                "Retained final authority over product, engineering, scientific, "
                "publication, licensing, repository, category, video, and "
                "submission decisions."
            ),
            "public_evidence": (
                "docs/CODEX_COLLABORATION.md; docs/HACKATHON_SUBMISSION.md"
            ),
        },
    ]

    return {
        "schema_version": "1.0",
        "evidence_id": "DEMO-001",
        "evidence_date": "2026-07-21",
        "scientific_warning": WARNING,
        "evidence_class": "generated deterministic product execution",
        "evidence_binding": {
            "receipt": "docs/report/evidence/product_demo_receipt.json",
            "figures": [
                "docs/report/figures/09_generated_product_case.png",
                "docs/report/figures/10_verified_product_output.png",
                "docs/report/figures/11_collaboration_evidence.png",
            ],
        },
        "execution": {
            "case_id": parameters["case_id"],
            "case_type": parameters["case_type"],
            "synthetic": True,
            "research_only": parameters["research_only"],
            "backends": {
                "detection": "reference",
                "report": "template",
                "evidence": "keyword",
            },
            "stored_nominal_fixture_metadata": {
                "side": parameters["side"],
                "radius_mm": {
                    "value": parameters["radius_mm"],
                    "used_by_array_generator": False,
                },
                "length_mm": {
                    "value": parameters["length_mm"],
                    "used_by_array_generator": False,
                },
                "uptake_multiplier": {
                    "value": parameters["uptake_multiplier"],
                    "applied_by_array_generator": False,
                },
                "seed": {
                    "value": parameters["seed"],
                    "used_by_array_generator": False,
                },
            },
            "array_generator": {
                "pet_increment_suv": 0.76,
                "rule": (
                    "A fixed 0.76 SUV increment is added inside the fixed binary "
                    "source mask."
                ),
            },
        },
        "finding": report["finding"],
        "deterministic_measurements": metrics,
        "measurement_scope": (
            "Default reference-backend values are deterministic integration-fixture "
            "conventions, not physical-space 3D scientific measurements."
        ),
        "report_verification": manifest["verification"],
        "ordered_runtime_trace": trace,
        "product_evaluation": {
            "suite_version": evaluation.suite_version,
            "passed": evaluation.passed,
            "failed": evaluation.failed,
            "all_passed": evaluation.all_passed,
            "cases": [case.model_dump(mode="json") for case in evaluation.cases],
        },
        "generated_views": views,
        "development_collaboration": {
            "evidence_class": "curated public outcomes",
            "separate_from_product_runtime": True,
            "outcomes": collaboration_outcomes,
            "excluded": [
                "private development instructions and configuration",
                "prompts and transcripts",
                "hidden reasoning and creation mechanics",
                "credentials, workstation state, and private logs",
            ],
        },
        "sanitization": {
            "temporary_paths_removed": True,
            "runtime_durations_removed": True,
            "patient_or_raw_medical_input_used": False,
            "model_weights_used": False,
            "network_or_api_used": False,
        },
        "interpretation_limits": [
            "This execution validates default product integration and safeguards.",
            "It is not a detector benchmark or a native-space 3D evaluation.",
            "It does not support diagnosis or any clinical performance claim.",
        ],
    }


def _validate_receipt(receipt: dict[str, Any]) -> None:
    serialized = json.dumps(receipt, sort_keys=True)
    prohibited = (
        "/home/",
        "/tmp/",
        "runtime_seconds",
        "model_output.json",
        "artifact_manifest.json",
    )
    hits = [term for term in prohibited if term in serialized]
    if hits:
        raise RuntimeError(f"Receipt contains prohibited values: {hits}")


def _figure_generated_case(
    case_dir: Path, receipt: dict[str, Any], figure_dir: Path
) -> None:
    labels = [
        ("pet", "PET", "Generated synthetic PET view"),
        ("ct", "CT", "Generated synthetic CT view"),
        ("fused", "Fused", "Generated PET and CT fusion"),
        ("overlay", "Simulated mask overlay", "Red is the simulated source mask"),
        (
            "bilateral",
            "Bilateral comparison",
            "Contralateral reference and target views",
        ),
    ]
    fig = plt.figure(figsize=(14, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    for index, (name, title, detail) in enumerate(labels):
        ax = fig.add_subplot(grid[index // 3, index % 3])
        ax.imshow(plt.imread(case_dir / f"{name}.png"), interpolation="nearest")
        ax.set_title(f"{chr(65 + index)}. {title}", loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(MID)
        ax.text(
            0.5,
            -0.08,
            detail,
            transform=ax.transAxes,
            ha="center",
            va="top",
            color=GRAY,
            fontsize=9,
        )

    ax = fig.add_subplot(grid[1, 2])
    ax.axis("off")
    metadata = receipt["execution"]["stored_nominal_fixture_metadata"]
    generator = receipt["execution"]["array_generator"]
    ax.text(
        0.02,
        0.92,
        "Stored nominal metadata",
        weight="bold",
        color=NAVY,
        fontsize=12,
    )
    rows = [
        ("Case", receipt["execution"]["case_id"]),
        ("Side", metadata["side"].title()),
        ("Radius", f"{metadata['radius_mm']['value']:.1f} mm, nominal"),
        ("Length", f"{metadata['length_mm']['value']:.1f} mm, nominal"),
        (
            "Multiplier",
            f"{metadata['uptake_multiplier']['value']:.1f}, not applied",
        ),
        ("Seed", f"{metadata['seed']['value']}, unused"),
        ("PET change", f"+{generator['pet_increment_suv']:.2f} SUV, fixed"),
    ]
    for row_index, (label, value) in enumerate(rows):
        y = 0.78 - row_index * 0.09
        ax.text(0.04, y, label, weight="bold", color=NAVY)
        ax.text(0.42, y, value, color=DARK)
    ax.text(
        0.04,
        0.07,
        textwrap.fill(
            "Generated fixture only. No patient image, raw medical volume, model "
            "weight, network service, or API was used.",
            width=43,
        ),
        color=RED,
        weight="bold",
        fontsize=9.3,
        va="bottom",
    )

    fig.suptitle(
        "Actual generated VascuTrace product views | DEMO-001",
        fontsize=17,
        weight="bold",
        color=NAVY,
    )
    _save(fig, "09_generated_product_case.png", figure_dir)


def _figure_verified_output(receipt: dict[str, Any], figure_dir: Path) -> None:
    metrics = receipt["deterministic_measurements"]
    evaluation = receipt["product_evaluation"]
    trace = receipt["ordered_runtime_trace"]

    fig = plt.figure(figsize=(10.5, 9.0), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=[1.45, 0.9, 2.05])

    ax = fig.add_subplot(grid[0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    accepted = receipt["report_verification"]["accepted"]
    values = [
        (
            "Report verification",
            "ACCEPTED\n0 issues" if accepted else "FAILED",
            "",
        ),
        ("Target SUVmax", metrics["target_suvmax"], "SUV"),
        ("Target SUVmean", metrics["target_suvmean"], "SUV"),
        ("Contralateral SUVmax", metrics["contralateral_suvmax"], "SUV"),
        ("Contralateral SUVmean", metrics["contralateral_suvmean"], "SUV"),
        ("Asymmetry index", metrics["asymmetry_index"], ""),
        ("Nominal volume", metrics["metabolic_volume_ml"], "mL"),
        ("Nominal extent", metrics["longitudinal_extent_mm"], "mm"),
    ]
    width = 0.225
    height = 0.31
    x_positions = [0.015, 0.265, 0.515, 0.765]
    y_positions = [0.57, 0.15]
    for index, (label, value, unit) in enumerate(values):
        row, column = divmod(index, 4)
        x = x_positions[column]
        y = y_positions[row]
        stored_value = value if isinstance(value, str) else json.dumps(value)
        is_verification = index == 0
        _box(
            ax,
            (x, y),
            width,
            height,
            f"{label}\n{stored_value} {unit}".strip(),
            color=GREEN if is_verification else (BLUE if index < 6 else GOLD),
            fill=(
                "#EFF8F1"
                if is_verification
                else ("#EDF5FB" if index < 6 else "#FFF8E7")
            ),
            fontsize=9.0,
        )

    ax = fig.add_subplot(grid[1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.01,
        0.91,
        "Ordered five-tool runtime trace",
        weight="bold",
        color=NAVY,
        fontsize=11.5,
    )
    display_names = {
        "load_case": "1. Load case",
        "run_vascular_detection": "2. Run detection",
        "calculate_pet_metrics": "3. Calculate metrics",
        "generate_research_report": "4. Generate report",
        "verify_report": "5. Verify report",
    }
    box_width = 0.175
    starts = np.linspace(0.01, 0.81, len(trace))
    for index, (x, tool_name) in enumerate(zip(starts, trace, strict=True)):
        _box(
            ax,
            (float(x), 0.24),
            box_width,
            0.42,
            display_names[tool_name],
            color=CYAN if index < len(trace) - 1 else GREEN,
            fill="#EEF8F8" if index < len(trace) - 1 else "#EFF8F1",
            fontsize=9.2,
        )
        if index < len(trace) - 1:
            _arrow(ax, (float(x) + box_width, 0.45), (float(starts[index + 1]), 0.45))

    ax = fig.add_subplot(grid[2])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.01,
        0.95,
        f"Product evaluation: {evaluation['passed']} passed, {evaluation['failed']} failed",
        weight="bold",
        color=GREEN,
        fontsize=12.5,
    )
    for index, case in enumerate(evaluation["cases"]):
        row = index // 2
        col = index % 2
        x = 0.015 + col * 0.495
        y = 0.68 - row * 0.27
        title = case["name"].replace("_", " ").title()
        detail = textwrap.fill(case["detail"], width=48)
        _box(
            ax,
            (x, y),
            0.475,
            0.20,
            f"PASS  |  {title}\n{detail}",
            color=GREEN,
            fill="#EFF8F1",
            fontsize=9.0,
        )
    ax.text(
        0.01,
        0.02,
        "Integration-fixture evidence only. Values are deterministic conventions, not physical-space 3D measurements.",
        color=RED,
        weight="bold",
        fontsize=9.2,
    )

    fig.suptitle(
        "Verified default-path product execution | DEMO-001",
        fontsize=16.5,
        weight="bold",
        color=NAVY,
    )
    _save(fig, "10_verified_product_output.png", figure_dir)


def _figure_collaboration(receipt: dict[str, Any], figure_dir: Path) -> None:
    outcomes = receipt["development_collaboration"]["outcomes"]
    fig, ax = plt.subplots(figsize=(10.5, 9.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.955,
        "Development collaboration evidence | DEMO-001",
        ha="center",
        weight="bold",
        color=NAVY,
        fontsize=16.5,
    )
    ax.text(
        0.5,
        0.915,
        "Codex with GPT-5.6 performed every non-coding workflow role",
        ha="center",
        color=GRAY,
        fontsize=11.0,
    )

    _box(
        ax,
        (0.025, 0.47),
        0.25,
        0.35,
        "Codex with GPT-5.6\nALL NON-CODING WORK",
        color=BLUE,
        fill="#EDF5FB",
        fontsize=10.2,
    )
    _arrow(ax, (0.275, 0.645), (0.31, 0.645))
    ax.text(
        0.315,
        0.795,
        "Owned and performed",
        weight="bold",
        color=NAVY,
        fontsize=11.0,
        va="center",
    )
    workstreams = [
        "Planning and architecture",
        "Primary technical decisions",
        "Scientific review",
        "Report writing",
        "Delivery review and\nGit/release handling",
        "Humanizer and editorial review",
    ]
    for index, label in enumerate(workstreams):
        row, column = divmod(index, 2)
        x = 0.315 + column * 0.335
        y = 0.70 - row * 0.095
        _box(
            ax,
            (x, y),
            0.31,
            0.068,
            label,
            color=GREEN if index in {2, 5} else BLUE,
            fill="#EFF8F1" if index in {2, 5} else "#F5F9FC",
            fontsize=8.8,
        )
    ax.text(
        0.315,
        0.474,
        textwrap.fill(
            "Also authored the plans and instructions used for code implementation.",
            width=75,
        ),
        color=DARK,
        fontsize=9.2,
        va="center",
        weight="bold",
    )
    ax.text(
        0.315,
        0.434,
        "Public evidence: plans, technical report, collaboration record, and submission guide",
        color=GRAY,
        fontsize=8.3,
        va="center",
    )

    claude = outcomes[1]
    _box(
        ax,
        (0.025, 0.265),
        0.25,
        0.125,
        "Claude\nCODING ONLY",
        color=PURPLE,
        fill="#F5F0F7",
        fontsize=10.0,
    )
    _arrow(ax, (0.275, 0.3275), (0.31, 0.3275))
    ax.text(
        0.315,
        0.345,
        textwrap.fill(claude["outcome"], width=72),
        color=DARK,
        fontsize=10.0,
        va="center",
        weight="bold",
    )
    ax.text(
        0.315,
        0.287,
        "Public evidence: dashboard, commands, runtime package, and tests",
        color=GRAY,
        fontsize=8.5,
        va="center",
    )

    owner = outcomes[2]
    _box(
        ax,
        (0.025, 0.115),
        0.25,
        0.09,
        "Project owner\nFINAL AUTHORITY",
        color=GOLD,
        fill="#FFF8E7",
        fontsize=9.5,
    )
    _arrow(ax, (0.275, 0.16), (0.31, 0.16))
    ax.text(
        0.315,
        0.16,
        textwrap.fill(owner["outcome"], width=72),
        color=DARK,
        fontsize=9.5,
        va="center",
    )

    ax.add_patch(
        FancyBboxPatch(
            (0.025, 0.012),
            0.95,
            0.055,
            boxstyle="round,pad=0.008,rounding_size=0.02",
            linewidth=1.4,
            edgecolor=RED,
            facecolor="#FBEFEF",
        )
    )
    ax.text(
        0.5,
        0.0395,
        textwrap.fill(
            "Not published: private instructions, prompts, configuration, "
            "transcripts, hidden reasoning, creation mechanics, or private logs.",
            width=96,
        ),
        ha="center",
        va="center",
        color=RED,
        weight="bold",
        fontsize=8.4,
    )
    _save(fig, "11_collaboration_evidence.png", figure_dir)


def build(output_root: Path) -> None:
    _style()
    figure_dir = output_root / "figures"
    evidence_dir = output_root / "evidence"
    receipt_path = evidence_dir / "product_demo_receipt.json"
    figure_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vascutrace-product-evidence-") as temp:
        temporary_root = Path(temp)
        with _deterministic_backends():
            manifest_path = run_complete_case(temporary_root / "complete_case")
            case_dir = manifest_path.parent
            create_demo_views(case_dir)
            evaluation = run_evaluation_suite(temporary_root / "evaluation")
        receipt = _build_receipt(case_dir, evaluation)
        _validate_receipt(receipt)
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _figure_generated_case(case_dir, receipt, figure_dir)
        _figure_verified_output(receipt, figure_dir)
        _figure_collaboration(receipt, figure_dir)
    print("Built generated product receipt and figures 09 through 11")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPORT_DIR,
        help="Report-style output root containing evidence/ and figures/.",
    )
    args = parser.parse_args()
    build(args.output_root)


if __name__ == "__main__":
    main()
