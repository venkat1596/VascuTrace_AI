"""Build a deterministic, privacy-limited Codex session evidence projection.

The source JSONL files remain local. This module emits only allowlisted session
identifiers, timestamps, surface categories, model identifiers, event counts,
and explicitly curated themes and public outcomes. Message text, reasoning,
prompts, instructions, tool payloads, working directories, and source paths are
never copied into the receipt or figures.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import tempfile
import textwrap
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


EVIDENCE_FILENAME = "codex_session_evidence.json"
TIMELINE_FILENAME = "12_codex_session_timeline.png"
ACTIVITY_FILENAME = "13_codex_session_activity.png"

COUNT_KEYS = (
    "user_turns",
    "assistant_updates",
    "tasks",
    "tool_calls",
    "patch_events",
    "web_searches",
    "bounded_review_activities",
    "compactions",
)

EVENT_COUNT_MAP = {
    "user_message": "user_turns",
    "agent_message": "assistant_updates",
    "task_started": "tasks",
    "patch_apply_end": "patch_events",
    "web_search_end": "web_searches",
    "sub_agent_activity": "bounded_review_activities",
}
TOOL_CALL_TYPES = frozenset(
    {
        "function_call",
        "custom_tool_call",
        "mcp_tool_call",
    }
)
SESSION_ID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$")
SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,127}$")

NAVY = "#17324D"
BLUE = "#2F6690"
CYAN = "#3A8D9B"
GREEN = "#4C956C"
GOLD = "#D9A441"
PURPLE = "#6C5B7B"
GRAY = "#66727A"
LIGHT = "#EEF3F6"


@dataclass(frozen=True)
class SessionSpec:
    """Curated public annotations for one allowlisted root session."""

    session_id: str
    primary: bool
    task_themes: tuple[str, ...]
    public_outcomes: tuple[str, ...]


DEFAULT_SESSION_SPECS = (
    SessionSpec(
        session_id="019f5c4e-1f8d-7190-8a04-2c6f7e7a2e18",
        primary=False,
        task_themes=(
            "Translate the initial specification into a two-track PET/CT architecture with fixed interfaces.",
            "Define Codex-led planning, reporting, Git stewardship, and competition alignment.",
            "Keep private development-support material outside the public release.",
        ),
        public_outcomes=(
            "Architecture and implementation inventory: README.md; plans/VascuTrace_Publication_and_Reproducibility_Plan_2026-07-20.md",
            "Versioned imaging and product contracts: src/vascutrace/data/contract.py; vascutrace/contracts.py",
            "Research demonstrator: app.py; vascutrace/orchestrator.py; scripts/run_complete_case.py",
        ),
    ),
    SessionSpec(
        session_id="019f5c9a-0be0-7351-b9c1-9d7b342d8108",
        primary=False,
        task_themes=(
            "Develop a Codex-led seven-day plan with scientific, product, evaluation, and submission gates.",
            "Establish bounded planning and review workflows across imaging, ML, product, testing, and documentation.",
            "Keep development-support material separate from the VascuTrace release.",
        ),
        public_outcomes=(
            "Publication and verification gates: plans/VascuTrace_Publication_and_Reproducibility_Plan_2026-07-20.md",
            "Privacy-safe collaboration record: docs/CODEX_COLLABORATION.md",
            "Release safeguards: .gitignore; .github/workflows/ci.yml",
        ),
    ),
    SessionSpec(
        session_id="019f6015-b646-70b3-ab11-6d5b20d9eeee",
        primary=False,
        task_themes=(
            "Perform local EDA from imaging, ML, mathematical, and statistical perspectives.",
            "Explain plots, inputs, outputs, and project value in plain language.",
            "Prepare Phase 1 and workstation resource planning before experiments.",
        ),
        public_outcomes=(
            "Reproducible EDA entry point: scripts/eda_quadra.py",
            "Aggregate cohort evidence: docs/report/figures/01_cohort_eda.png; docs/report/figures/02_test_retest_summary.png",
            "Plain-language project summary: README.md",
        ),
    ),
    SessionSpec(
        session_id="019f60c0-ba06-77f0-be21-37da29651799",
        primary=False,
        task_themes=(
            "Define an acceptance-first Phase 1 foundation for provenance, geometry, QA, splitting, and resources.",
            "Complete EDA and investigate label provenance and licensing.",
            "Synchronize Git safely while preserving remote work.",
        ),
        public_outcomes=(
            "Affine-aware PET-grid geometry: src/vascutrace/geometry.py; tests/test_geometry.py",
            "Strict ingestion and bilateral crops: src/vascutrace/data/ingest.py; src/vascutrace/data/split.py; src/vascutrace/data/crops.py",
            "Geometry report evidence: docs/report/figures/03_geometry_and_resampling.png; docs/report/VascuTrace_Technical_Report_2026-07-20.tex",
        ),
    ),
    SessionSpec(
        session_id="019f6187-81f0-7db1-b5c1-d9cdbba313dc",
        primary=True,
        task_themes=(
            "Reconcile the specification, EDA, and product workplan after resuming from Git.",
            "Expose execution plans and assess 2D and 2.5D U-Net readiness.",
            "Approve and progress Phase 1 geometry delivery and reconstruction.",
            "Diagnose geometry and verification failures caused by candidate or worktree state.",
        ),
        public_outcomes=(
            "Physical-grid geometry: src/vascutrace/geometry.py; tests/test_geometry.py",
            "Bilateral crop and data contracts: src/vascutrace/data/crops.py; src/vascutrace/data/contract.py",
            "Geometry-to-model synthesis: docs/report/VascuTrace_Technical_Report_2026-07-20.tex",
        ),
    ),
    SessionSpec(
        session_id="019f661f-b909-7a22-b165-7d5c8e5da35f",
        primary=False,
        task_themes=(
            "Report the concrete status and contents of Phase 1.",
            "Continue geometry recovery under bounded approvals.",
            "Move from the imaging foundation toward model implementation and training.",
        ),
        public_outcomes=(
            "Recovered geometry foundation: src/vascutrace/geometry.py; tests/test_geometry.py",
            "Executable Phase 2 data pipeline: scripts/run_p2_pipeline.py; src/vascutrace/data/ingest.py; src/vascutrace/data/crops.py",
            "Siamese model and training path: src/vascutrace/ml/model.py; src/vascutrace/ml/train.py; configs/train_siamese_v1.yaml",
        ),
    ),
    SessionSpec(
        session_id="019f6773-1bc8-73d2-9a39-b7ade344260f",
        primary=False,
        task_themes=(
            "Resolve remaining Phase 1 binding and completion blockers.",
            "Produce an all-phases implementation handoff with explicit quality requirements.",
            "Define end-to-end requirements and tests needed for reliable training.",
        ),
        public_outcomes=(
            "Controlled source simulation: src/vascutrace/simulation/anomaly.py; tests/test_simulation.py",
            "Deterministic 3D quantification: src/vascutrace/quantification/measure.py; tests/test_quantification.py",
            "Training and evaluation: src/vascutrace/ml/train.py; src/vascutrace/ml/evaluate.py; tests/test_ml_evaluate.py",
        ),
    ),
    SessionSpec(
        session_id="019f6bd0-46c4-7121-bf6d-88481daf4566",
        primary=False,
        task_themes=(
            "Assess whether the reported IoU was near a defensible maximum.",
            "Evaluate whether higher IoU could coexist with strong precision and F1.",
            "Ground improvement work in literature and failure-specific evaluation.",
        ),
        public_outcomes=(
            "Failure-specific evaluation: src/vascutrace/ml/evaluate.py; src/vascutrace/ml/metrics.py; tests/test_ml_evaluate.py",
            "Reproducible model configuration: configs/train_siamese_v4_big.yaml",
            "Mechanism-specific activation probe: scripts/fp_cnr_gate_probe.py",
        ),
    ),
    SessionSpec(
        session_id="019f6d22-f305-7ca0-86ae-20e1ac55dba6",
        primary=False,
        task_themes=(
            "Falsify unsupported IoU-ceiling claims and correct statistical and reproducibility defects.",
            "Resolve source-package and path-validation blockers while retaining Codex Git stewardship.",
            "Design a bounded soft-target experiment without universal-ceiling claims.",
            "Separate exploratory science from certification and compute logistics.",
        ),
        public_outcomes=(
            "Restored public data-pipeline source: src/vascutrace/data/contract.py; src/vascutrace/data/crops.py; src/vascutrace/data/ingest.py",
            "Boundary and soft-target losses: src/vascutrace/ml/losses.py; tests/test_ml_boundary_aux.py; tests/test_ml_b3_soft_term.py",
            "Outcome-blind mechanism probe: scripts/p3_lambda_probe.py",
        ),
    ),
    SessionSpec(
        session_id="019f7398-2350-79d0-a657-fca01373af04",
        primary=False,
        task_themes=(
            "Integrate reviewed planning, source-restoration, and validation repairs.",
            "Analyze overlap, detection, and negative-clean tradeoffs in later models.",
            "Develop the next training and component-filter strategy with tests.",
            "Prepare public code while excluding weights and data.",
        ),
        public_outcomes=(
            "Next-model configurations: configs/train_siamese_v7b.yaml; configs/train_siamese_v7soft.yaml",
            "Config-gated training and evaluation: src/vascutrace/ml/losses.py; src/vascutrace/ml/train.py; src/vascutrace/ml/evaluate.py",
            "Operating-point coverage: src/vascutrace/ml/metrics.py; tests/test_operating_point_knobs.py",
        ),
    ),
    SessionSpec(
        session_id="019f81d0-b3be-7e21-a19a-491a13e360e8",
        primary=False,
        task_themes=(
            "Produce a professional EDA-to-pipeline report with figures and independent checks.",
            "Include product GenAI code while excluding private development material, weights, and large data.",
            "Document Codex collaboration, competition, video, repository, and judge-test requirements.",
            "Add sanitized details for the primary and supporting root sessions.",
        ),
        public_outcomes=(
            "Technical report: docs/report/VascuTrace_Technical_Report_2026-07-20.tex; docs/report/VascuTrace_Technical_Report_2026-07-20.pdf",
            "Codex collaboration chronology: docs/CODEX_COLLABORATION.md",
            "Competition and judge-test guide: docs/HACKATHON_SUBMISSION.md",
        ),
    ),
)


def _parse_timestamp(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing or invalid {label} timestamp")
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"Invalid {label} timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{label.capitalize()} timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    rendered = value.astimezone(timezone.utc).isoformat()
    return rendered.replace("+00:00", "Z")


def _validate_public_text(value: str, *, label: str, allow_relative_path: bool) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    normalized = " ".join(value.split())
    if not normalized or len(normalized) > 240:
        raise ValueError(f"{label} must contain 1 to 240 characters")
    if "\N{EM DASH}" in normalized or "-" * 3 in normalized:
        raise ValueError(f"{label} contains a prohibited typographic artifact")
    if "\x00" in normalized:
        raise ValueError(f"{label} contains a prohibited control character")
    path_candidate = PurePath(normalized)
    if path_candidate.is_absolute():
        raise ValueError(f"{label} must not contain an absolute path")
    if not allow_relative_path:
        absolute_fragment = re.search(
            r"(?:^|\s)(?:/[A-Za-z0-9_.-]+/|[A-Za-z]:\\)",
            normalized,
        )
        if absolute_fragment or "\\" in normalized:
            raise ValueError(f"{label} must not contain an absolute path")
    return normalized


def _validate_specs(session_specs: Sequence[SessionSpec]) -> tuple[SessionSpec, ...]:
    specs = tuple(session_specs)
    if len(specs) != 11:
        raise ValueError("Session allowlist must contain exactly eleven entries")
    identifiers = [spec.session_id for spec in specs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Session allowlist contains a duplicate identifier")
    if sum(bool(spec.primary) for spec in specs) != 1:
        raise ValueError("Session allowlist must identify exactly one primary session")

    validated: list[SessionSpec] = []
    for spec in specs:
        if not SESSION_ID_PATTERN.fullmatch(spec.session_id):
            raise ValueError("Session allowlist contains an invalid identifier")
        themes = tuple(
            _validate_public_text(
                value,
                label="Task theme",
                allow_relative_path=False,
            )
            for value in spec.task_themes
        )
        outcomes = tuple(
            _validate_public_text(
                value,
                label="Public outcome",
                allow_relative_path=True,
            )
            for value in spec.public_outcomes
        )
        if not themes or not outcomes:
            raise ValueError("Every session requires a theme and public outcome")
        validated.append(
            SessionSpec(
                session_id=spec.session_id,
                primary=bool(spec.primary),
                task_themes=themes,
                public_outcomes=outcomes,
            )
        )
    return tuple(validated)


def _safe_surface(value: Any) -> str:
    if not isinstance(value, str):
        return "other"
    normalized = value.strip().lower()
    surface_markers = (
        ("vscode", "vscode"),
        ("visual studio code", "vscode"),
        ("app-server", "app-server"),
        ("app_server", "app-server"),
        ("cli", "cli"),
        ("exec", "exec"),
        ("mcp", "mcp"),
    )
    for marker, public_name in surface_markers:
        if marker in normalized:
            return public_name
    return "other"


def _safe_model_identifier(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not MODEL_ID_PATTERN.fullmatch(normalized):
        raise ValueError("Recorded model identifier has an unsafe format")
    return normalized


def _safe_metadata_token(value: Any) -> str:
    if not isinstance(value, str):
        return "other"
    normalized = value.strip()
    if not MODEL_ID_PATTERN.fullmatch(normalized):
        return "other"
    return normalized


def _validate_workspace_name(workspace_name: str) -> str:
    if not isinstance(workspace_name, str) or not SAFE_NAME_PATTERN.fullmatch(
        workspace_name
    ):
        raise ValueError("Workspace name must be a safe basename")
    return workspace_name


def _resolve_allowlisted_files(
    session_root: Path,
    session_specs: Sequence[SessionSpec],
) -> dict[str, Path]:
    if not session_root.is_dir():
        raise ValueError("Session root is missing or is not a directory")
    available = tuple(path for path in session_root.rglob("*.jsonl") if path.is_file())
    resolved: dict[str, Path] = {}
    for spec in session_specs:
        suffix = f"-{spec.session_id}.jsonl"
        candidates = [path for path in available if path.name.endswith(suffix)]
        if not candidates:
            raise ValueError(f"Missing allowlisted session {spec.session_id}")
        if len(candidates) > 1:
            raise ValueError(f"Duplicate allowlisted session {spec.session_id}")
        resolved[spec.session_id] = candidates[0]
    return resolved


def _empty_counts() -> dict[str, int]:
    return dict.fromkeys(COUNT_KEYS, 0)


def _read_json_object(line: str, *, line_number: int) -> Mapping[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSONL record at line {line_number}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSONL record at line {line_number} is not an object")
    return value


def _parse_session(
    path: Path,
    spec: SessionSpec,
    cutoff: datetime,
    workspace_name: str,
) -> dict[str, Any]:
    counts = _empty_counts()
    instrumentation_counts = {
        "mcp_tool_calls": 0,
        "tasks_completed": 0,
    }
    timestamps: list[datetime] = []
    models: Counter[str] = Counter()
    surfaces: set[str] = set()
    root_metadata_seen = False
    originator = "other"
    cli_version = "other"
    model_provider = "other"

    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = _read_json_object(line, line_number=line_number)
            timestamp = _parse_timestamp(
                record.get("timestamp"),
                label="record",
            )
            if timestamp > cutoff:
                continue
            timestamps.append(timestamp)

            record_type = record.get("type")
            payload = record.get("payload")
            if not isinstance(record_type, str) or not isinstance(payload, dict):
                continue

            if record_type == "session_meta":
                metadata_id = payload.get("id")
                if not root_metadata_seen:
                    if metadata_id != spec.session_id:
                        raise ValueError(
                            f"Session metadata mismatch for {spec.session_id}"
                        )
                    cwd = payload.get("cwd")
                    if not isinstance(cwd, str) or PurePath(cwd).name != workspace_name:
                        raise ValueError(
                            f"Session workspace mismatch for {spec.session_id}"
                        )
                    root_metadata_seen = True
                    surfaces.add(_safe_surface(payload.get("source")))
                    originator = _safe_metadata_token(payload.get("originator"))
                    cli_version = _safe_metadata_token(payload.get("cli_version"))
                    model_provider = _safe_metadata_token(payload.get("model_provider"))
                elif metadata_id == spec.session_id:
                    surfaces.add(_safe_surface(payload.get("source")))
                continue

            if record_type == "turn_context":
                model = _safe_model_identifier(payload.get("model"))
                if model is not None:
                    models[model] += 1
                continue

            if record_type == "event_msg":
                event_type = payload.get("type")
                count_key = EVENT_COUNT_MAP.get(event_type)
                if count_key is not None:
                    counts[count_key] += 1
                elif event_type == "task_complete":
                    instrumentation_counts["tasks_completed"] += 1
                elif event_type == "mcp_tool_call_end":
                    instrumentation_counts["mcp_tool_calls"] += 1
                continue

            if record_type == "response_item":
                if payload.get("type") in TOOL_CALL_TYPES:
                    counts["tool_calls"] += 1
                continue

            if record_type == "compacted":
                counts["compactions"] += 1

    if not root_metadata_seen:
        raise ValueError(f"Missing root metadata for session {spec.session_id}")
    if not timestamps:
        raise ValueError(f"No records before cutoff for session {spec.session_id}")
    if not models:
        raise ValueError(f"Missing model metadata for session {spec.session_id}")

    start = min(timestamps)
    end = max(timestamps)
    observed_span_hours = round((end - start).total_seconds() / 3600, 2)
    return {
        "session_id": spec.session_id,
        "primary": spec.primary,
        "start_utc": _format_timestamp(start),
        "end_utc": _format_timestamp(end),
        "observed_span_hours": observed_span_hours,
        "source_records_at_cutoff": len(timestamps),
        "recorded_surfaces": sorted(surfaces),
        "recorded_originator": originator,
        "recorded_cli_version": cli_version,
        "recorded_model_provider": model_provider,
        "recorded_model_identifiers": sorted(models),
        "recorded_model_context_counts": dict(sorted(models.items())),
        "counts": counts,
        "instrumentation_counts": instrumentation_counts,
        "task_themes": list(spec.task_themes),
        "public_outcomes": list(spec.public_outcomes),
    }


def _build_receipt(
    rows: Sequence[dict[str, Any]],
    specs: Sequence[SessionSpec],
    cutoff_utc: str,
    workspace_name: str,
) -> dict[str, Any]:
    aggregates = {
        count_key: sum(int(row["counts"][count_key]) for row in rows)
        for count_key in COUNT_KEYS
    }
    aggregates["session_count"] = len(rows)
    aggregates["source_records_at_cutoff"] = sum(
        int(row["source_records_at_cutoff"]) for row in rows
    )
    aggregates["recorded_model_contexts"] = sum(
        sum(int(count) for count in row["recorded_model_context_counts"].values())
        for row in rows
    )
    aggregates["tasks_completed"] = sum(
        int(row["instrumentation_counts"]["tasks_completed"]) for row in rows
    )
    aggregates["mcp_tool_calls"] = sum(
        int(row["instrumentation_counts"]["mcp_tool_calls"]) for row in rows
    )
    aggregates["observed_span_hours_sum"] = round(
        sum(float(row["observed_span_hours"]) for row in rows),
        2,
    )
    primary_session_id = next(spec.session_id for spec in specs if spec.primary)
    return {
        "schema_version": 1,
        "workspace_name": workspace_name,
        "cutoff_utc": cutoff_utc,
        "session_count": len(rows),
        "primary_session_id": primary_session_id,
        "sessions": list(rows),
        "aggregates": aggregates,
        "evidence_scope": (
            "Sanitized development-process metadata and event counts. Event volume "
            "is not a quality or scientific-performance metric."
        ),
        "privacy_method": (
            "The builder reads only allowlisted metadata fields and structural "
            "event types through the fixed cutoff. Raw conversation text and "
            "private development material remain local."
        ),
        "excluded_source_lanes": [
            "message bodies and transcripts",
            "hidden reasoning and summaries",
            "system and developer text",
            "prompts and private instructions",
            "tool arguments and tool results",
            "credentials and workstation state",
            "absolute source paths",
        ],
    }


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.edgecolor": "#CAD7DF",
            "axes.linewidth": 0.8,
            "text.color": "#263238",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _short_session_label(index: int, session_id: str, primary: bool) -> str:
    marker = "*" if primary else " "
    return f"{marker} S{index + 1:02d}  {session_id[:8]}..{session_id[-6:]}"


def _save_png_bytes(fig: plt.Figure, *, title: str) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=120,
        facecolor="white",
        metadata={
            "Title": title,
            "Author": "VascuTrace",
            "Software": "VascuTrace deterministic evidence builder",
        },
    )
    plt.close(fig)
    return buffer.getvalue()


def _render_timeline(rows: Sequence[dict[str, Any]]) -> bytes:
    _style()
    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.25, 1.75), wspace=0.04)
    text_axis = fig.add_subplot(grid[0, 0])
    timeline_axis = fig.add_subplot(grid[0, 1])
    y_positions = np.arange(len(rows))
    starts = [
        mdates.date2num(_parse_timestamp(row["start_utc"], label="start"))
        for row in rows
    ]
    ends = [
        mdates.date2num(_parse_timestamp(row["end_utc"], label="end")) for row in rows
    ]
    for index, row in enumerate(rows):
        color = GOLD if row["primary"] else BLUE
        timeline_axis.hlines(
            y=index,
            xmin=starts[index],
            xmax=ends[index],
            color=color,
            linewidth=4,
            zorder=2,
        )
        timeline_axis.scatter(
            [starts[index], ends[index]],
            [index, index],
            color=color,
            s=28,
            zorder=3,
        )
        session_label = _short_session_label(
            index,
            row["session_id"],
            bool(row["primary"]),
        )
        text_axis.text(
            0.0,
            index,
            session_label,
            fontsize=8.4,
            ha="left",
            va="center",
            color=NAVY if row["primary"] else "#263238",
            weight="bold" if row["primary"] else "normal",
        )
        text_axis.text(
            0.34,
            index,
            textwrap.fill(str(row["task_themes"][0]), width=42),
            fontsize=7.8,
            ha="left",
            va="center",
            color=GRAY,
        )

    text_axis.set_xlim(0, 1)
    text_axis.set_ylim(len(rows) - 0.5, -0.5)
    text_axis.axis("off")
    text_axis.text(
        0.0,
        -0.55,
        "Root session",
        fontsize=9.3,
        weight="bold",
        color=NAVY,
    )
    text_axis.text(
        0.34,
        -0.55,
        "Primary task theme",
        fontsize=9.3,
        weight="bold",
        color=NAVY,
    )

    timeline_axis.set_ylim(len(rows) - 0.5, -0.5)
    timeline_axis.set_yticks(y_positions, labels=[])
    timeline_axis.tick_params(axis="y", left=False)
    timeline_axis.set_xlim(min(starts) - 0.1, max(ends) + 0.1)
    timeline_axis.xaxis.set_major_locator(
        mdates.AutoDateLocator(minticks=4, maxticks=9)
    )
    timeline_axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M UTC"))
    timeline_axis.grid(axis="x", color=LIGHT, linewidth=0.9)
    timeline_axis.set_xlabel("Allowlisted evidence window")
    for spine in ("top", "right", "left"):
        timeline_axis.spines[spine].set_visible(False)

    fig.suptitle(
        "Codex root session timeline and curated task themes",
        fontsize=15,
        weight="bold",
        color=NAVY,
        x=0.05,
        ha="left",
    )
    fig.text(
        0.05,
        0.03,
        "* Primary session. Timings and themes describe development-process evidence.",
        fontsize=8.5,
        color=GRAY,
    )
    fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.15)
    return _save_png_bytes(fig, title=TIMELINE_FILENAME)


def _render_activity(rows: Sequence[dict[str, Any]]) -> bytes:
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    x_positions = np.arange(len(rows))
    panels: tuple[tuple[plt.Axes, tuple[tuple[str, str, str], ...]], ...] = (
        (
            axes[0, 0],
            (
                ("user_turns", "User turns", BLUE),
                ("assistant_updates", "Assistant updates", CYAN),
                ("tasks", "Tasks started", GREEN),
            ),
        ),
        (
            axes[0, 1],
            (("tool_calls", "Tool calls", GOLD),),
        ),
        (
            axes[1, 0],
            (
                ("patch_events", "Patch events", PURPLE),
                (
                    "bounded_review_activities",
                    "Bounded review activity",
                    CYAN,
                ),
            ),
        ),
        (
            axes[1, 1],
            (
                ("web_searches", "Web searches", NAVY),
                ("compactions", "Compactions", GRAY),
            ),
        ),
    )
    for panel_index, (axis, series) in enumerate(panels):
        bar_width = min(0.72 / len(series), 0.52)
        offsets = np.arange(len(series)) - (len(series) - 1) / 2
        for offset, (count_key, label, color) in zip(offsets, series, strict=True):
            values = [int(row["counts"][count_key]) for row in rows]
            bars = axis.bar(
                x_positions + offset * bar_width,
                values,
                width=bar_width,
                label=label,
                color=color,
            )
            if len(series) == 1:
                axis.bar_label(bars, fontsize=7, padding=2)
        axis.grid(axis="y", color=LIGHT, linewidth=0.9)
        axis.set_ylabel("Event count")
        axis.legend(
            loc="upper right" if panel_index >= 2 else "upper left",
            ncol=min(3, len(series)),
            frameon=False,
            fontsize=8,
        )
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    axes[0, 0].set_title("User turns, Codex updates, and tasks", loc="left")
    axes[0, 1].set_title("Tool calls", loc="left")
    axes[1, 0].set_title("Patch and bounded review activity", loc="left")
    axes[1, 1].set_title("Search and context compaction", loc="left")
    labels = [
        f"S{index + 1:02d}{'*' if row['primary'] else ''}"
        for index, row in enumerate(rows)
    ]
    for axis in axes[1, :]:
        axis.set_xticks(x_positions, labels=labels)
        axis.set_xlabel("Root session, primary marked by *")
    fig.suptitle(
        "Allowlisted Codex root session activity counts",
        fontsize=15,
        weight="bold",
        color=NAVY,
        x=0.08,
        ha="left",
    )
    fig.text(
        0.01,
        0.01,
        "Event volume documents workflow activity. It is not a quality or scientific-performance metric.",
        fontsize=8.5,
        color=GRAY,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.89,
        bottom=0.13,
        hspace=0.33,
        wspace=0.20,
    )
    return _save_png_bytes(fig, title=ACTIVITY_FILENAME)


def _serialize_receipt(receipt: Mapping[str, Any]) -> bytes:
    text = json.dumps(
        receipt,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    return f"{text}\n".encode("utf-8")


def _write_atomic(path: Path, content: bytes) -> None:
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            temporary_name = stream.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            temporary_path = Path(temporary_name)
            if temporary_path.exists():
                temporary_path.unlink()


def build_evidence(
    session_root: Path | str,
    output_root: Path | str,
    cutoff_utc: str,
    session_specs: Sequence[SessionSpec],
    workspace_name: str,
    report_layout: bool = False,
) -> dict[str, Any]:
    """Build the sanitized receipt and two deterministic figures."""

    source_root = Path(session_root)
    destination_root = Path(output_root)
    validated_workspace = _validate_workspace_name(workspace_name)
    validated_specs = _validate_specs(session_specs)
    cutoff = _parse_timestamp(cutoff_utc, label="cutoff")
    resolved = _resolve_allowlisted_files(source_root, validated_specs)

    rows = [
        _parse_session(
            resolved[spec.session_id],
            spec,
            cutoff,
            validated_workspace,
        )
        for spec in validated_specs
    ]
    receipt = _build_receipt(
        rows,
        validated_specs,
        cutoff_utc,
        validated_workspace,
    )
    artifacts = {
        EVIDENCE_FILENAME: _serialize_receipt(receipt),
        TIMELINE_FILENAME: _render_timeline(rows),
        ACTIVITY_FILENAME: _render_activity(rows),
    }

    destinations = {
        EVIDENCE_FILENAME: (
            destination_root / "evidence" / EVIDENCE_FILENAME
            if report_layout
            else destination_root / EVIDENCE_FILENAME
        ),
        TIMELINE_FILENAME: (
            destination_root / "figures" / TIMELINE_FILENAME
            if report_layout
            else destination_root / TIMELINE_FILENAME
        ),
        ACTIVITY_FILENAME: (
            destination_root / "figures" / ACTIVITY_FILENAME
            if report_layout
            else destination_root / ACTIVITY_FILENAME
        ),
    }
    for filename, content in artifacts.items():
        destination = destinations[filename]
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(destination, content)
    return receipt


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sanitized Codex session evidence for VascuTrace.",
    )
    parser.add_argument(
        "--session-root",
        type=Path,
        required=True,
        help="Local root containing Codex JSONL session files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory for the public receipt and figures.",
    )
    parser.add_argument(
        "--cutoff",
        required=True,
        help="Inclusive UTC evidence cutoff in ISO 8601 format.",
    )
    parser.add_argument(
        "--workspace-name",
        default="VascuTrace_AI",
        help="Expected workspace basename. The source path is not published.",
    )
    parser.add_argument(
        "--report-layout",
        action="store_true",
        help="Write the receipt under evidence/ and figures under figures/.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    build_evidence(
        session_root=args.session_root,
        output_root=args.output_root,
        cutoff_utc=args.cutoff,
        session_specs=DEFAULT_SESSION_SPECS,
        workspace_name=args.workspace_name,
        report_layout=args.report_layout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
