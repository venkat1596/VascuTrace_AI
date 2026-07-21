"""Reusable visual components for the VascuTrace dashboard."""

from __future__ import annotations

import html
from collections.abc import Iterable

import streamlit as st


def _escape(value: object) -> str:
    return html.escape(str(value))


def safety_banner() -> None:
    st.markdown(
        """
        <div class="vt-safety" role="alert">
          <span>⚠</span><div><strong>Synthetic research demonstration</strong> —
          generated data and deterministic reference outputs only. Not for clinical use,
          diagnosis, or treatment decisions.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(case_id: str, verified: bool, model_name: str) -> None:
    state = "Verification passed" if verified else "Verification requires review"
    st.markdown(
        f"""
        <div class="vt-hero">
          <div class="vt-kicker">Agentic vascular imaging research</div>
          <h1>See the signal.<br>Trace the evidence.</h1>
          <p>An auditable PET/CT workspace connecting bilateral imaging, deterministic
          measurements, quality control, and research evidence in one focused view.</p>
          <div class="vt-hero-meta">
            <span class="vt-pill"><span class="vt-dot"></span>{_escape(state)}</span>
            <span class="vt-pill">Case&nbsp; {_escape(case_id)}</span>
            <span class="vt-pill">Model&nbsp; {_escape(model_name)}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="vt-section">
          <div class="vt-eyebrow">{_escape(eyebrow)}</div>
          <h2>{_escape(title)}</h2>
          <p>{_escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, unit: str = "", note: str = "") -> None:
    st.markdown(
        f"""
        <div class="vt-card vt-metric">
          <div class="vt-metric-label">{_escape(label)}</div>
          <div class="vt-metric-value">{_escape(value)}<span class="vt-metric-unit">{_escape(unit)}</span></div>
          <div class="vt-metric-note">{_escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_row(label: str, active: bool) -> None:
    state, css = ("Review", "warn") if active else ("Clear", "good")
    st.markdown(
        f"""
        <div class="vt-status">
          <span class="vt-status-label">{_escape(label)}</span>
          <span class="vt-badge vt-badge-{css}">{state}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def report_text(text: str) -> None:
    st.markdown(f'<div class="vt-report">{_escape(text)}</div>', unsafe_allow_html=True)


def evidence_card(title: str, text: str, source: str | None) -> None:
    source_markup = (
        f'<div class="vt-source">↗ {_escape(source)}</div>' if source else ""
    )
    st.markdown(
        f"""
        <div class="vt-evidence">
          <div class="vt-evidence-title">{_escape(title)}</div>
          <div class="vt-evidence-text">{_escape(text)}</div>
          {source_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


def trace_steps(steps: Iterable[str]) -> None:
    markup = "".join(
        f'<div class="vt-step"><span class="vt-step-num">{index}</span><span>{_escape(step.replace("_", " ").title())}</span></div>'
        for index, step in enumerate(steps, start=1)
    )
    st.markdown(markup, unsafe_allow_html=True)
