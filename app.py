"""VascuTrace AI synthetic research workspace."""

import os

import streamlit as st

from dashboard.components import (
    evidence_card,
    hero,
    metric_card,
    report_text,
    safety_banner,
    section,
    status_row,
    trace_steps,
)
from dashboard.theme import apply_theme
from vascutrace.evaluation import run_evaluation_suite
from vascutrace.orchestrator import (
    run_evidence_request,
    run_experiment_request,
    run_first_checkpoint,
)
from vascutrace.tools import create_overlay


def _format_measurement(value: float | None, digits: int = 3) -> str:
    return "Unavailable" if value is None else f"{value:.{digits}f}"


apply_theme()

with st.spinner("Preparing deterministic research workspace…"):
    result = run_first_checkpoint()
    payload = result.payload
    output = payload["model_output"]
    metrics = payload["metrics"]
    report = payload["report"]
    quality = report["quality_control"]
    views = create_overlay(payload["case"]["case_dir"])


with st.sidebar:
    st.markdown(
        '<div class="vt-brand">VascuTrace<span class="vt-brand-dot"> AI</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("RESEARCH WORKSPACE · v0.1")
    st.divider()
    st.markdown("**Workspace**")
    st.markdown("Overview  ")
    st.markdown("Imaging workspace  ")
    st.markdown("Research report  ")
    st.markdown("Evidence library  ")
    st.markdown("Experiments & audit")
    st.divider()
    st.markdown("**Active case**")
    st.caption(payload["case"]["case_id"])
    st.markdown("**Laterality**")
    st.caption(output["laterality"].title())
    st.markdown("**Abnormality score**")
    st.progress(output["abnormality_score"])
    st.caption(f"{output['abnormality_score']:.3f} uncalibrated research-model output")
    st.divider()
    st.markdown("**Active backends**")
    _active_backends = (
        (
            "Detection",
            "VASCUTRACE_DETECTION_BACKEND",
            "reference",
            {
                "reference": "Deterministic reference",
                "siamese": "Siamese B2 (trained)",
            },
        ),
        (
            "Reasoning report",
            "VASCUTRACE_REPORT_BACKEND",
            "template",
            {
                "template": "Deterministic template",
                "llm": "gpt-5-mini (grounded)",
            },
        ),
        (
            "Evidence",
            "VASCUTRACE_EVIDENCE_BACKEND",
            "keyword",
            {
                "keyword": "Keyword store",
                "rag": "Qwen RAG (embed + rerank)",
            },
        ),
    )
    for _label, _env, _default, _labels in _active_backends:
        _key = os.environ.get(_env, _default)
        st.caption(f"{_label}: {_labels.get(_key, _key)}")
    st.divider()
    st.caption("Synthetic data · Auditable trace · No patient information")

safety_banner()
hero(
    payload["case"]["case_id"],
    payload["verification"]["accepted"],
    output["model_name"],
)

section(
    "Case overview",
    "Quantitative signal at a glance",
    "Deterministic measurements calculated from the active synthetic PET/CT case.",
)
metric_columns = st.columns(4)
metric_specs = (
    (
        "Target SUVmax",
        _format_measurement(metrics["target_suvmax"]),
        "SUV",
        "Target corridor peak",
    ),
    (
        "Contralateral SUVmax",
        _format_measurement(metrics["contralateral_suvmax"]),
        "SUV",
        "Mirrored control peak",
    ),
    (
        "Asymmetry index",
        _format_measurement(metrics["asymmetry_index"]),
        "",
        "Target-to-control difference",
    ),
    (
        "Metabolic volume",
        _format_measurement(metrics["metabolic_volume_ml"]),
        "mL",
        "Simulated target volume",
    ),
)
for column, spec in zip(metric_columns, metric_specs, strict=True):
    with column:
        metric_card(*spec)

section(
    "Multimodal review",
    "Imaging workspace",
    "Move between aligned modalities, segmentation context, and bilateral comparison.",
)
image_column, qc_column = st.columns([2.45, 1], gap="large")
with image_column:
    pet_tab, ct_tab, fusion_tab, overlay_tab, bilateral_tab = st.tabs(
        ["PET", "CT", "Fused", "Mask overlay", "Bilateral"]
    )
    image_tabs = (
        (pet_tab, "pet_path", "Synthetic PET reference · axial slice"),
        (ct_tab, "ct_path", "Synthetic CT anatomy reference · axial slice"),
        (fusion_tab, "fused_path", "PET/CT research fusion · aligned reference"),
        (overlay_tab, "overlay_path", "Target mask in red · simulated ground truth"),
        (
            bilateral_tab,
            "bilateral_path",
            "Mirrored left control ↔ right target comparison",
        ),
    )
    for tab, view_key, caption in image_tabs:
        with tab:
            st.image(views[view_key], caption=caption, width="stretch")

with qc_column:
    st.markdown("### Quality control")
    st.caption("Automated checks attached to this deterministic reference.")
    status_row("Partial-volume effects", quality["partial_volume_risk"])
    status_row("PET/CT misregistration", quality["misregistration_risk"])
    st.markdown("#### Review flags")
    if quality["flags"]:
        for flag in quality["flags"]:
            st.warning(flag.replace("_", " ").title(), icon="⚠️")
    else:
        st.success("No quality-control flags", icon="✅")
    st.markdown("#### Model context")
    st.caption(f"{output['model_name']} · version {output['model_version']}")
    st.caption(f"Runtime {output['runtime_seconds']:.3f} seconds")

section(
    "Verified output",
    "Structured research report",
    "Human-readable interpretation with exact source measurements and safety checks.",
)
report_column, finding_column = st.columns([1.65, 1], gap="large")
with report_column:
    if payload["verification"]["accepted"]:
        st.success("Deterministic verification passed", icon="✅")
    else:
        st.error("Report verification failed", icon="⚠️")
    grounding = payload.get("grounding_score")
    if grounding is not None:
        st.progress(grounding)
        st.caption(
            f"Evidence grounding {grounding:.0%} — share of interpretation "
            "sentences supported by cited passages (spec §12)."
        )
    report_text(report["interpretation"])
    st.markdown("#### Limitations")
    for limitation in report["limitations"]:
        st.markdown(f"- {limitation}")
    report_evidence = report.get("evidence") or []
    if report_evidence:
        st.markdown("#### Cited evidence")
        for reference in report_evidence:
            evidence_card(
                reference["title"],
                reference["supporting_text"],
                reference.get("source_url"),
            )

with finding_column:
    st.markdown("#### Finding summary")
    finding = report["finding"]
    st.markdown(f"**Laterality**  \n{finding['laterality'].title()}")
    st.markdown(
        "**Target region**  \n" + finding["target_region"].replace("_", " ").title()
    )
    st.markdown(
        f"**Abnormality score**  \n{finding['abnormality_score']:.3f} (uncalibrated)"
    )
    with st.expander("View machine-readable report"):
        st.json(report)

section(
    "Research context",
    "Evidence library",
    "Retrieve provenance-aware background evidence without mixing scan-specific data into the cache.",
)
evidence_query = st.text_input(
    "Evidence question",
    value="Why can partial-volume effects bias PET uptake?",
    placeholder="Ask a research evidence question…",
)
evidence_result = run_evidence_request(evidence_query)
if not evidence_result.payload["cache_eligible"]:
    st.warning("Case-specific queries bypass the semantic cache.", icon="🔒")
for reference in evidence_result.payload["evidence"]:
    evidence_card(
        reference["title"],
        reference["supporting_text"],
        reference["source_url"],
    )

section(
    "Reference study",
    "Synthetic detectability experiment",
    "Explore deterministic PET-only and PET+CT reference behavior across controlled conditions.",
)
with st.spinner("Loading deterministic experiment…"):
    experiment = run_experiment_request().payload
chart_rows = [
    {
        "Radius (mm)": row["radius_mm"],
        "Uptake multiplier": row["uptake_multiplier"],
        "PET-only Dice": row["pet_only"]["dice"],
        "PET+CT Dice": row["pet_ct_non_siamese"]["dice"],
    }
    for row in experiment["rows"]
    if row["blur_fwhm_mm"] == 5.0 and row["source_offset_mm"] == 2.0
]
chart_tab, table_tab = st.tabs(["Performance chart", "Detailed data"])
with chart_tab:
    st.line_chart(
        chart_rows,
        x="Radius (mm)",
        y=["PET-only Dice", "PET+CT Dice"],
        color=["#36d9d2", "#9b7bff"],
        height=360,
    )
    st.caption(
        "Fixed blur FWHM 5.0 mm and misalignment 2.0 mm. Multiple uptake conditions "
        "are shown for each radius."
    )
with table_tab:
    st.dataframe(chart_rows, width="stretch", hide_index=True)
st.info(
    "These are deterministic classical references—not trained U-Net results or "
    "estimates of clinical performance.",
    icon="ℹ️",
)

section(
    "System integrity",
    "Evaluation and audit trail",
    "Executable safeguards covering schemas, measurement fidelity, claim safety, caching, and trace completeness.",
)
evaluation = run_evaluation_suite()
evaluation_column, trace_column = st.columns([1.4, 1], gap="large")
with evaluation_column:
    passed_column, failed_column = st.columns(2)
    with passed_column:
        metric_card("Checks passed", str(evaluation.passed), "", "Product safeguards")
    with failed_column:
        metric_card("Checks failed", str(evaluation.failed), "", "Requires attention")
    for case in evaluation.cases:
        label = "✓" if case.passed else "✕"
        with st.expander(f"{label}  {case.name.replace('_', ' ').title()}"):
            st.caption(case.category.replace("_", " ").upper())
            st.write(case.detail)

with trace_column:
    st.markdown("#### Orchestration trace")
    st.caption("Ordered tools used to produce and verify this case.")
    trace_steps(result.trace)
    st.markdown("#### Experiment exports")
    st.caption(experiment["results_csv_path"])
    st.caption(experiment["results_json_path"])

st.divider()
st.caption(
    "VascuTrace AI · Synthetic research workspace · Deterministic outputs · "
    "Not for clinical use"
)
