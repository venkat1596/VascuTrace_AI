"""Build publication-safe VascuTrace figures from aggregate evidence.

Every panel is either a schematic or a rendering of values in the aggregate
evidence ledger. The script never reads medical volumes, case tables, model
weights, identifiers, or cached predictions.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

try:
    import matplotlib
except ModuleNotFoundError:
    repository_root = Path(__file__).resolve().parents[3]
    project_python = repository_root / ".venv" / "bin" / "python"
    if (
        project_python.is_file()
        and os.environ.get("VASCUTRACE_REPORT_FIGURE_BOOTSTRAP") != "1"
    ):
        os.environ["VASCUTRACE_REPORT_FIGURE_BOOTSTRAP"] = "1"
        matplotlib_cache = Path(tempfile.gettempdir()) / "vascutrace-report-matplotlib"
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
        os.execv(
            str(project_python), [str(project_python), str(Path(__file__).resolve())]
        )
    raise

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


REPORT_DIR = Path(__file__).resolve().parents[1]
LEDGER_PATH = REPORT_DIR / "evidence" / "aggregate_evidence.json"
FIGURE_DIR = REPORT_DIR / "figures"

NAVY = "#17324D"
BLUE = "#2F6690"
CYAN = "#3A8D9B"
GREEN = "#4C956C"
GOLD = "#D9A441"
ORANGE = "#D97841"
RED = "#B54A4A"
PURPLE = "#6C5B7B"
LIGHT = "#EEF3F6"
MID = "#CAD7DF"
DARK = "#263238"
GRAY = "#66727A"


def _load() -> dict:
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def _record(ledger: dict, record_id: str) -> dict:
    return next(item for item in ledger["records"] if item["id"] == record_id)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": MID,
            "axes.linewidth": 0.8,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "text.color": DARK,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(
        FIGURE_DIR / name,
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
    fontsize: float = 10,
    linewidth: float = 1.6,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=linewidth,
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
    return patch


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = GRAY,
    style: str = "-|>",
    width: float = 1.8,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=13,
            linewidth=width,
            color=color,
            connectionstyle="arc3",
        )
    )


def cohort_eda(ledger: dict) -> None:
    eda1 = _record(ledger, "EDA-001")["values"]
    eda2 = _record(ledger, "EDA-002")["values"]
    pet = _record(ledger, "EDA-005")["values"]
    ct = _record(ledger, "EDA-006")["values"]

    fig = plt.figure(figsize=(14, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1, 1])

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("A. Release inventory and paired structure", loc="left")
    _box(
        ax,
        (0.03, 0.55),
        0.24,
        0.27,
        f"{eda1['subjects']} subjects\n25 female, 23 male",
        color=NAVY,
    )
    _box(
        ax,
        (0.38, 0.55),
        0.24,
        0.27,
        f"{eda1['sessions']} sessions\nTest and Retest",
        color=CYAN,
    )
    _box(
        ax,
        (0.72, 0.55),
        0.25,
        0.27,
        f"{eda1['nifti_files']} NIfTI volumes\n96 PET, 96 CT\n768 labels",
        color=GREEN,
    )
    _arrow(ax, (0.27, 0.685), (0.38, 0.685))
    _arrow(ax, (0.62, 0.685), (0.72, 0.685))
    ax.text(
        0.5,
        0.31,
        "Independent unit for model assessment: subject",
        ha="center",
        weight="bold",
        color=NAVY,
    )
    ax.text(
        0.5,
        0.20,
        "Sessions and synthetic conditions are repeated observations",
        ha="center",
        color=GRAY,
    )

    ax = fig.add_subplot(gs[0, 1])
    labels = ["Female", "Male"]
    values = [eda1["female"], eda1["male"]]
    bars = ax.bar(labels, values, color=[PURPLE, BLUE], width=0.58)
    ax.set_ylim(0, 30)
    ax.set_ylabel("Subjects")
    ax.set_title("B. Cohort sex counts", loc="left")
    ax.grid(axis="y", color=LIGHT, linewidth=1)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.6,
            str(value),
            ha="center",
            weight="bold",
        )
    ax.text(
        0.98, 0.94, "n = 48", ha="right", va="top", transform=ax.transAxes, color=GRAY
    )

    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    ax.set_title("C. Aggregate demographic descriptors", loc="left")
    rows = [
        (
            "Age",
            f"{eda2['age_mean_years']:.2f} years",
            f"SD {eda2['age_sd_years']:.2f}; range {eda2['age_range_years'][0]} to {eda2['age_range_years'][1]}",
        ),
        (
            "Test BMI",
            f"{eda2['test_bmi_mean_kg_m2']:.2f} kg/m2",
            f"SD {eda2['test_bmi_sd_kg_m2']:.2f}; range {eda2['test_bmi_range_kg_m2'][0]} to {eda2['test_bmi_range_kg_m2'][1]}",
        ),
        (
            "Missing values",
            str(eda2["missing_demographic_values"]),
            "reported in the demographic workbook",
        ),
    ]
    for idx, (label, headline, detail) in enumerate(rows):
        y = 0.72 - idx * 0.26
        ax.add_patch(
            Rectangle(
                (0.02, y - 0.1),
                0.96,
                0.19,
                facecolor=LIGHT if idx % 2 == 0 else "white",
                edgecolor=MID,
            )
        )
        ax.text(0.06, y, label, va="center", weight="bold", color=NAVY)
        ax.text(0.34, y, headline, va="center", weight="bold")
        ax.text(0.58, y, detail, va="center", color=GRAY)

    ax = fig.add_subplot(gs[1, 1])
    labels = ["PET median", "PET p95", "PET p99", "CT median", "CT p95", "CT p99"]
    values = [
        pet["median_suv"],
        pet["p95_suv"],
        pet["p99_suv"],
        ct["median_hu"],
        ct["p95_hu"],
        ct["p99_hu"],
    ]
    colors = [BLUE, BLUE, BLUE, ORANGE, ORANGE, ORANGE]
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, alpha=0.9)
    ax.axvline(0, color=DARK, linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_title("D. Deterministic intensity samples", loc="left")
    ax.set_xlabel("Native units: SUV for PET, HU for CT")
    ax.grid(axis="x", color=LIGHT)
    for yi, value in zip(y, values, strict=True):
        if value < 0:
            ax.text(
                value + 22,
                yi,
                f"{value:g}",
                va="center",
                ha="left",
                fontsize=9,
                color="white",
                weight="bold",
            )
        else:
            ax.annotate(
                f"{value:g}",
                (value, yi),
                xytext=(18, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=9,
            )
    ax.text(
        0.99,
        0.02,
        "Different samples and units; shown only as corruption screens",
        transform=ax.transAxes,
        ha="right",
        color=GRAY,
        fontsize=8.5,
    )

    fig.suptitle(
        "QUADRA_HC aggregate exploratory data analysis",
        fontsize=17,
        weight="bold",
        color=NAVY,
    )
    _save(fig, "01_cohort_eda.png")


def test_retest_summary(ledger: dict) -> None:
    values = _record(ledger, "EDA-003")["values"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), constrained_layout=True)

    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _box(ax, (0.05, 0.52), 0.32, 0.25, "Test\n48 scans", color=BLUE)
    _box(ax, (0.63, 0.52), 0.32, 0.25, "Retest\n48 scans", color=GREEN)
    _arrow(ax, (0.37, 0.645), (0.63, 0.645), color=GOLD)
    ax.text(0.5, 0.86, "A. Paired design", ha="center", weight="bold", fontsize=12)
    ax.text(
        0.5,
        0.35,
        "Both visits remain in the same split",
        ha="center",
        weight="bold",
        color=NAVY,
    )
    ax.text(0.5, 0.22, "48 independent subject pairs", ha="center", color=GRAY)

    ax = axes[1]
    low, high = values["interval_range_days"]
    median = values["interval_median_days"]
    mean = values["interval_mean_days"]
    ax.hlines(0, low, high, color=BLUE, linewidth=8, alpha=0.3)
    ax.scatter([low, high], [0, 0], s=85, color=BLUE, zorder=3, label="Range")
    ax.scatter([median], [0], marker="D", s=110, color=NAVY, zorder=4, label="Median")
    ax.scatter(
        [mean],
        [0],
        marker="o",
        s=100,
        color=GOLD,
        edgecolor=DARK,
        zorder=4,
        label="Mean",
    )
    ax.set_xlim(15, 105)
    ax.set_ylim(-0.65, 0.65)
    ax.set_yticks([])
    ax.set_xlabel("Days")
    ax.set_title("B. Reported test/retest interval")
    ax.grid(axis="x", color=LIGHT)
    ax.legend(loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    ax.text(mean, 0.23, f"mean {mean:.2f}", ha="center", color=DARK)
    ax.text(median, -0.24, f"median {median:g}", ha="center", color=NAVY)

    ax = axes[2]
    activities = [
        values["test_injected_activity_mean_mbq"],
        values["retest_injected_activity_mean_mbq"],
    ]
    bars = ax.bar(["Test", "Retest"], activities, color=[BLUE, GREEN], width=0.58)
    ax.set_ylim(0, 130)
    ax.set_ylabel("Mean injected activity (MBq)")
    ax.set_title("C. Aggregate acquisition descriptor")
    ax.grid(axis="y", color=LIGHT)
    for bar, value in zip(bars, activities, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.2f}",
            ha="center",
            weight="bold",
        )
    ax.text(
        0.5,
        -0.19,
        "Means do not measure PET repeatability",
        transform=ax.transAxes,
        ha="center",
        color=GRAY,
        fontsize=9,
    )

    fig.suptitle(
        "Test/retest structure and aggregate acquisition context",
        fontsize=16,
        weight="bold",
        color=NAVY,
    )
    _save(fig, "02_test_retest_summary.png")


def geometry_and_resampling(ledger: dict) -> None:
    geom = _record(ledger, "EDA-004")["values"]
    fit = _record(ledger, "FIT-002")["values"]
    fig, ax = plt.subplots(figsize=(14, 7.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(0.2, 7.55, "Source grids", fontsize=13, weight="bold", color=NAVY)
    _box(
        ax,
        (0.35, 5.75),
        2.75,
        1.05,
        f"CT\n{geom['ct_shape'][0]} x {geom['ct_shape'][1]} x {geom['ct_shape'][2]}\n{geom['ct_spacing_mm'][0]:.6f} x {geom['ct_spacing_mm'][1]:.6f} x {geom['ct_spacing_mm'][2]:.1f} mm",
        color=ORANGE,
        fill="#FFF4EC",
        fontsize=8.7,
    )
    _box(
        ax,
        (0.35, 4.12),
        2.75,
        1.05,
        "CT-derived labels\ninteger anatomical anchors\nsame native grid as CT",
        color=GREEN,
        fill="#EFF8F1",
        fontsize=9,
    )
    _box(
        ax,
        (0.35, 2.49),
        2.75,
        1.05,
        f"PET SUVbw reference\n{geom['pet_shape'][0]} x {geom['pet_shape'][1]} x {geom['pet_shape'][2]}\n{geom['pet_spacing_mm'][0]:.2f} x {geom['pet_spacing_mm'][1]:.2f} x {geom['pet_spacing_mm'][2]:.1f} mm",
        color=BLUE,
        fill="#EDF5FB",
        fontsize=8.9,
    )
    ax.text(1.72, 2.08, "Both modalities stored as LAS", ha="center", color=GRAY)

    _arrow(ax, (3.1, 6.28), (4.35, 6.28), color=ORANGE)
    _arrow(ax, (3.1, 4.65), (4.35, 4.65), color=GREEN)
    _box(
        ax,
        (4.4, 5.75),
        2.55,
        1.05,
        "CT to PET\nphysical coordinates\nlinear interpolation",
        color=ORANGE,
        fill="#FFF4EC",
        fontsize=9,
    )
    _box(
        ax,
        (4.4, 4.12),
        2.55,
        1.05,
        "Labels to PET\nphysical coordinates\nnearest neighbor",
        color=GREEN,
        fill="#EFF8F1",
        fontsize=9,
    )

    _arrow(ax, (6.95, 6.28), (8.15, 5.48), color=ORANGE)
    _arrow(ax, (6.95, 4.65), (8.15, 5.08), color=GREEN)
    _arrow(ax, (3.1, 3.02), (8.15, 4.72), color=BLUE)
    _box(
        ax,
        (8.2, 4.25),
        2.45,
        1.55,
        "Common PET grid\nraw SUV preserved\nCT and anchors aligned",
        color=CYAN,
        fill="#EEF8F8",
    )

    _arrow(ax, (10.65, 5.03), (11.45, 5.03), color=CYAN)
    _box(
        ax,
        (11.5, 4.25),
        2.1,
        1.55,
        f"Fixed crop\n{fit['fixed_crop_voxels'][0]} x {fit['fixed_crop_voxels'][1]} x {fit['fixed_crop_voxels'][2]}",
        color=NAVY,
        fill="#EEF3F6",
    )
    ax.text(
        7.0,
        7.05,
        "Every resampling uses finite affines and named transforms",
        ha="center",
        color=GRAY,
    )

    ax.add_patch(
        Rectangle(
            (4.55, 0.55), 5.0, 1.35, facecolor="white", edgecolor=MID, linewidth=1.4
        )
    )
    ax.plot([7.05, 7.05], [0.72, 1.72], color=GOLD, linewidth=2.2, linestyle=":")
    ax.add_patch(Circle((6.05, 1.22), 0.35, facecolor=BLUE, alpha=0.35, edgecolor=BLUE))
    ax.add_patch(
        Circle((8.05, 1.22), 0.35, facecolor=GREEN, alpha=0.35, edgecolor=GREEN)
    )
    _arrow(ax, (6.45, 1.22), (7.65, 1.22), color=GOLD, style="<->")
    ax.text(
        7.05,
        0.28,
        "Reflect through a fitted patient-space plane",
        ha="center",
        color=NAVY,
        weight="bold",
    )
    ax.text(0.35, 1.35, "Evidence", weight="bold", color=NAVY)
    ax.text(
        0.35,
        0.95,
        f"PET coverage inside CT: {geom['pet_within_ct_sessions']} sessions",
        color=DARK,
    )
    ax.text(
        0.35,
        0.55,
        f"PET to CT physical volume ratio: {geom['pet_to_ct_physical_volume_percent']:.4f}%",
        color=DARK,
    )
    ax.text(10.0, 1.35, "Known limitation", weight="bold", color=RED)
    ax.text(10.0, 0.95, "Legacy crop reflection differs", color=DARK)
    ax.text(10.0, 0.55, "from the intended method", color=DARK)

    ax.set_title(
        "Physical-coordinate geometry and resampling contract",
        fontsize=17,
        weight="bold",
        color=NAVY,
        pad=15,
    )
    _save(fig, "03_geometry_and_resampling.png")


def simulation_contract(ledger: dict) -> None:
    sim = _record(ledger, "SIM-001")["values"]
    fig = plt.figure(figsize=(14, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(-7, 7)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.axis("off")
    x = np.linspace(-5, 5, 400)
    center = 0.55 * np.sin(x / 2.2)
    ax.plot(x, center, color=NAVY, linewidth=2)
    ax.fill_between(
        x,
        center - 0.8,
        center + 0.8,
        color=GOLD,
        alpha=0.38,
        label="Fractional source occupancy",
    )
    ax.scatter(
        np.linspace(-5, 5, 14),
        0.55 * np.sin(np.linspace(-5, 5, 14) / 2.2),
        s=12,
        color=NAVY,
    )
    ax.text(
        0,
        3.15,
        "A. Supersampled vascular-like capsule",
        ha="center",
        weight="bold",
        fontsize=12,
    )
    ax.text(
        0,
        -2.85,
        f"radius {sim['radius_mm'][0]} to {sim['radius_mm'][1]} mm; length {sim['length_mm']} mm; supersampling {sim['supersampling_factor']}",
        ha="center",
        color=GRAY,
    )

    ax = fig.add_subplot(gs[0, 1])
    xx = np.linspace(-16, 16, 500)
    for fwhm, color in [(4, BLUE), (6, GREEN), (8, ORANGE)]:
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        yy = np.exp(-(xx**2) / (2 * sigma**2))
        ax.plot(xx, yy, color=color, linewidth=2.4, label=f"FWHM {fwhm} mm")
    ax.set_title("B. Controlled Gaussian image blur")
    ax.set_xlabel("Distance from source center (mm)")
    ax.set_ylabel("Normalized excess")
    ax.set_ylim(0, 1.08)
    ax.grid(color=LIGHT)
    ax.legend(frameon=False)
    ax.text(
        0.5,
        -0.24,
        "Image-domain factor, not a measured scanner point-spread function",
        transform=ax.transAxes,
        ha="center",
        color=GRAY,
    )

    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    ax.set_title("C. Additive source equation", loc="left")
    _box(ax, (0.03, 0.44), 0.19, 0.27, "Original\nPET SUVbw", color=BLUE)
    ax.text(0.255, 0.575, "+", fontsize=24, weight="bold", ha="center", va="center")
    _box(ax, (0.30, 0.44), 0.16, 0.27, "Blur\nG sigma", color=ORANGE, fill="#FFF4EC")
    ax.text(0.495, 0.575, "x", fontsize=20, weight="bold", ha="center", va="center")
    _box(
        ax,
        (0.54, 0.44),
        0.42,
        0.27,
        "(m - 1) x B x F x H\ncontrast, baseline, occupancy, heterogeneity",
        color=GREEN,
        fill="#EFF8F1",
        fontsize=9,
    )
    ax.text(
        0.5,
        0.25,
        f"uptake multiplier {sim['uptake_multiplier'][0]} to {sim['uptake_multiplier'][1]}; heterogeneity parameter {sim['heterogeneity_parameter']}",
        ha="center",
        color=GRAY,
    )

    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.set_title("D. Invariants and explicit exclusions", loc="left")
    entries = [
        (GREEN, "Sham identity", "m = 1 reproduces the original PET"),
        (
            GREEN,
            "Quantitative grid",
            "raw SUV remains separate from network normalization",
        ),
        (
            GREEN,
            "Supervision",
            f"source fraction threshold {sim['binary_fraction_threshold']:.1f}",
        ),
        (RED, "Not implemented", "no image-wide acquisition-noise model"),
        (RED, "Not evaluated", "stored CT shift was not applied in the standard cache"),
    ]
    for idx, (color, title, detail) in enumerate(entries):
        y = 0.82 - idx * 0.16
        ax.add_patch(Circle((0.06, y), 0.024, facecolor=color, edgecolor=color))
        ax.text(0.11, y + 0.025, title, weight="bold", va="center")
        ax.text(0.11, y - 0.035, detail, va="center", color=GRAY, fontsize=9)

    fig.suptitle(
        "Synthetic-source design and numerical meaning",
        fontsize=17,
        weight="bold",
        color=NAVY,
    )
    _save(fig, "04_simulation_contract.png")


def model_architecture(ledger: dict) -> None:
    model = _record(ledger, "MODEL-001")["values"]
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")

    _box(
        ax,
        (0.3, 5.55),
        2.25,
        1.2,
        f"Left view\n{model['pet_slices_per_branch']} PET + {model['ct_slices_per_branch']} CT slices",
        color=BLUE,
        fill="#EDF5FB",
    )
    _box(
        ax,
        (0.3, 3.55),
        2.25,
        1.2,
        f"Reflected view\n{model['pet_slices_per_branch']} PET + {model['ct_slices_per_branch']} CT slices",
        color=GREEN,
        fill="#EFF8F1",
    )
    _box(
        ax,
        (0.3, 1.55),
        2.25,
        1.2,
        f"PET difference\n{model['difference_channels']} channels",
        color=GOLD,
        fill="#FFF8E7",
    )

    _box(ax, (3.3, 5.55), 2.2, 1.2, "Shared encoder\nweights reused", color=NAVY)
    _box(ax, (3.3, 3.55), 2.2, 1.2, "Shared encoder\nsame parameters", color=NAVY)
    _box(
        ax,
        (3.3, 1.55),
        2.2,
        1.2,
        "Difference stem\nfull resolution",
        color=GOLD,
        fill="#FFF8E7",
    )
    _arrow(ax, (2.55, 6.15), (3.3, 6.15), color=BLUE)
    _arrow(ax, (2.55, 4.15), (3.3, 4.15), color=GREEN)
    _arrow(ax, (2.55, 2.15), (3.3, 2.15), color=GOLD)

    _box(
        ax,
        (6.15, 4.15),
        2.45,
        1.55,
        "Multi-scale fusion\nleft, right, abs difference\n1 x 1 projection",
        color=PURPLE,
        fill="#F5F0F7",
    )
    _arrow(ax, (5.5, 6.15), (6.15, 5.35), color=BLUE)
    _arrow(ax, (5.5, 4.15), (6.15, 4.75), color=GREEN)

    _box(
        ax,
        (9.25, 4.15),
        2.25,
        1.55,
        "U-Net decoder\nGroupNorm + GELU\nskip connections",
        color=CYAN,
        fill="#EEF8F8",
    )
    _arrow(ax, (8.6, 4.93), (9.25, 4.93), color=PURPLE)
    _arrow(ax, (5.5, 2.15), (10.1, 4.15), color=GOLD)

    _box(
        ax,
        (12.2, 4.35),
        2.35,
        1.15,
        f"Center-slice logits\n{model['output_shape'][0]} x {model['output_shape'][1]}",
        color=RED,
        fill="#FBEFEF",
    )
    _arrow(ax, (11.5, 4.93), (12.2, 4.93), color=CYAN)

    ax.add_patch(
        Rectangle(
            (6.2, 1.05), 5.3, 1.35, facecolor="white", edgecolor=MID, linewidth=1.4
        )
    )
    ax.text(6.45, 2.05, "Training only", weight="bold", color=NAVY)
    ax.text(
        6.45,
        1.58,
        f"Auxiliary heads at x{model['deep_supervision_scales'][0]} and x{model['deep_supervision_scales'][1]}",
        color=DARK,
    )
    ax.text(
        6.45,
        1.25,
        f"loss weights {model['deep_supervision_weights'][0]} and {model['deep_supervision_weights'][1]}",
        color=GRAY,
    )

    ax.text(13.38, 3.75, "sigmoid", ha="center", color=GRAY)
    ax.text(
        13.38,
        3.34,
        "uncalibrated abnormality_score",
        ha="center",
        color=NAVY,
        weight="bold",
    )
    ax.text(
        7.5,
        7.35,
        "2.5D shared-weight Siamese U-Net",
        ha="center",
        fontsize=17,
        weight="bold",
        color=NAVY,
    )
    ax.text(
        7.5,
        0.38,
        "One output covers one 2D center slice; native-space 3D stitching remains an evaluation requirement",
        ha="center",
        color=RED,
        weight="bold",
    )
    _save(fig, "05_model_architecture.png")


def detectability_atlas(ledger: dict) -> None:
    model2 = _record(ledger, "MODEL-002")["values"]
    model3 = _record(ledger, "MODEL-003")["values"]
    atlas = _record(ledger, "ATLAS-001")["values"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    ax = axes[0, 0]
    labels = ["Mean IoU", "Clean rate"]
    b0 = [model3["b0_mean_iou"], model3["b0_clean_rate"]]
    b2 = [model3["b2_mean_iou"], model3["b2_clean_rate"]]
    x = np.arange(2)
    width = 0.34
    bars0 = ax.bar(x - width / 2, b0, width, label="B0", color=GRAY)
    bars2 = ax.bar(x + width / 2, b2, width, label="B2", color=BLUE)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("Proportion")
    ax.set_title("A. Aggregate exploratory validation")
    ax.grid(axis="y", color=LIGHT)
    ax.legend(frameon=False)
    for bars in (bars0, bars2):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{bar.get_height():.3f}",
                ha="center",
                fontsize=9,
            )
    ax.text(
        0.98,
        0.96,
        "Seven subject clusters",
        transform=ax.transAxes,
        color=GRAY,
        fontsize=9,
        ha="right",
        va="top",
    )

    ax = axes[0, 1]
    levels = np.arange(3)
    factor_colors = {"radius": BLUE, "uptake": GREEN, "blur": ORANGE}
    factor_labels = {"radius": "Radius", "uptake": "Uptake", "blur": "Blur"}
    for key in ("radius", "uptake", "blur"):
        ax.plot(
            levels,
            atlas[key]["mean_iou"],
            marker="o",
            linewidth=2.2,
            markersize=7,
            color=factor_colors[key],
            label=factor_labels[key],
        )
    ax.set_xticks(levels, ["Low", "Mid", "High"])
    ax.set_ylim(0.38, 0.78)
    ax.set_ylabel("Mean IoU")
    ax.set_title("B. Descriptive factor strata")
    ax.grid(color=LIGHT)
    ax.legend(frameon=False, ncol=3, loc="lower center")

    ax = axes[1, 0]
    matrix = np.asarray(atlas["radius_by_uptake_mean_iou"], dtype=float)
    counts = np.asarray(atlas["radius_by_uptake_n"], dtype=int)
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0.4, vmax=0.85, aspect="auto")
    for row in range(3):
        for col in range(3):
            color = "white" if matrix[row, col] > 0.68 else DARK
            ax.text(
                col,
                row,
                f"{matrix[row, col]:.3f}\nn={counts[row, col]}",
                ha="center",
                va="center",
                color=color,
                weight="bold",
            )
    ax.set_xticks(range(3), ["Low", "Mid", "High"])
    ax.set_yticks(range(3), ["Low", "Mid", "High"])
    ax.set_xlabel("Uptake tertile")
    ax.set_ylabel("Radius tertile")
    ax.set_title("C. Radius by uptake mean IoU")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Mean IoU")

    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("D. Interpretation boundary", loc="left")
    statements = [
        (
            NAVY,
            f"75/78 target-overlapping predictions ({model2['target_overlap_percent']:.1f}%)",
        ),
        (NAVY, f"Mean positive IoU {model2['positive_mean_iou']:.3f}"),
        (
            NAVY,
            f"Activation on 37/130 negative center slices ({model2['negative_activation_percent']:.1f}%)",
        ),
        (RED, "Validation, not a sealed test set"),
        (RED, "2D center slices, not native-space 3D scans"),
        (RED, "Repeated observations within seven subjects"),
        (RED, "No clinical performance interpretation"),
    ]
    for idx, (color, text) in enumerate(statements):
        y = 0.86 - idx * 0.115
        ax.add_patch(Circle((0.04, y), 0.017, facecolor=color, edgecolor=color))
        ax.text(
            0.08,
            y,
            text,
            va="center",
            color=DARK,
            fontsize=10.5,
            weight="bold" if idx < 3 else "normal",
        )

    fig.suptitle(
        "Exploratory B2 center-slice detectability atlas",
        fontsize=17,
        weight="bold",
        color=NAVY,
    )
    _save(fig, "06_detectability_atlas.png")


def activation_by_noise_proxy(ledger: dict) -> None:
    values = _record(ledger, "ATLAS-002")["values"]
    levels = values["derived_texture_proxy_levels"]
    rates = values["activation_percent"]
    counts = values["n"]
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    bars = ax.bar(levels, rates, color=[GREEN, GOLD, RED], width=0.6)
    ax.set_ylim(0, 72)
    ax.set_ylabel("Center slices with activation (%)")
    ax.set_xlabel("Derived raw-PET texture proxy stratum")
    ax.set_title(
        "Activation on negative validation center slices",
        fontsize=16,
        weight="bold",
        color=NAVY,
    )
    ax.grid(axis="y", color=LIGHT)
    for bar, rate, count in zip(bars, rates, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 1.8,
            f"{rate:.1f}%\nn={count}",
            ha="center",
            weight="bold",
        )
    ax.text(
        0.5,
        -0.20,
        "130 repeated center-slice observations from seven subject clusters",
        transform=ax.transAxes,
        ha="center",
        color=GRAY,
    )
    ax.text(
        0.5,
        -0.27,
        "The proxy is not simulated acquisition noise; confirmatory clustered inference was not performed",
        transform=ax.transAxes,
        ha="center",
        color=RED,
        weight="bold",
    )
    _save(fig, "07_activation_by_noise_proxy.png")


def end_to_end_pipeline(ledger: dict) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 9))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 9)
    ax.axis("off")

    stages = [
        (0.3, "1. Ingest and QA", "manifest, affine, labels", GREEN),
        (2.75, "2. PET-grid crop", "resample, reflect, hash", GOLD),
        (5.2, "3. Synthetic source", "fraction, contrast, blur", GREEN),
        (7.65, "4. Detection", "reference or Siamese", GOLD),
        (10.1, "5. Quantification", "SUV and geometry", GOLD),
        (12.55, "6. Report and UI", "verify, render, export", GREEN),
    ]
    for x, title, detail, status in stages:
        _box(
            ax,
            (x, 5.95),
            2.05,
            1.15,
            f"{title}\n{detail}",
            color=status,
            fill="#EFF8F1" if status == GREEN else "#FFF8E7",
            fontsize=9.5,
        )
    for idx in range(len(stages) - 1):
        _arrow(
            ax, (stages[idx][0] + 2.05, 6.52), (stages[idx + 1][0], 6.52), color=GRAY
        )

    _box(
        ax,
        (0.8, 3.65),
        2.6,
        1.05,
        "Deterministic\ntool registry\nstructured I/O",
        color=BLUE,
        fill="#EDF5FB",
        fontsize=8.8,
    )
    _box(
        ax,
        (4.0, 3.65),
        2.6,
        1.05,
        "Local MCP boundary\noutput-root checks",
        color=BLUE,
        fill="#EDF5FB",
        fontsize=8.8,
    )
    _box(
        ax,
        (7.2, 3.65),
        2.6,
        1.05,
        "Evidence retrieval\npublic rebuild pending",
        color=GOLD,
        fill="#FFF8E7",
        fontsize=8.8,
    )
    _box(
        ax,
        (10.4, 3.65),
        2.6,
        1.05,
        "Report generation\ntemplate default\noptional prose",
        color=BLUE,
        fill="#EDF5FB",
        fontsize=8.8,
    )
    _box(
        ax,
        (13.3, 3.65),
        1.7,
        1.05,
        "Numeric verifier\nclaim checks",
        color=GREEN,
        fill="#EFF8F1",
        fontsize=8.5,
    )
    for start, end in [
        ((3.4, 4.17), (4.0, 4.17)),
        ((6.6, 4.17), (7.2, 4.17)),
        ((9.8, 4.17), (10.4, 4.17)),
        ((13.0, 4.17), (13.3, 4.17)),
    ]:
        _arrow(ax, start, end, color=GRAY)

    ax.text(0.35, 2.75, "Current boundaries", fontsize=13, weight="bold", color=RED)
    boundaries = [
        "Reference backend is the default and returns a known synthetic mask",
        "Learned backend processes one selected validation center slice",
        "Product metric assembly does not yet call the complete 3D quantifier",
        "Native-space 3D stitching and sealed test evaluation are not complete",
        "The publication-safe retrieval index must be rebuilt and reevaluated",
    ]
    for idx, text in enumerate(boundaries):
        col = 0 if idx < 3 else 1
        row = idx if idx < 3 else idx - 3
        x = 0.45 if col == 0 else 8.0
        y = 2.2 - row * 0.72
        ax.add_patch(Circle((x, y), 0.04, facecolor=RED, edgecolor=RED))
        ax.text(x + 0.14, y, text, va="center", fontsize=10, color=DARK)

    ax.text(
        7.75,
        8.25,
        "VascuTrace end-to-end research pipeline",
        ha="center",
        fontsize=18,
        weight="bold",
        color=NAVY,
    )
    ax.text(
        7.75,
        7.72,
        "Green indicates implemented with an aligned interface; gold indicates implemented with a known contract or integration gap",
        ha="center",
        color=GRAY,
    )
    _save(fig, "08_end_to_end_pipeline.png")


def main() -> None:
    _style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ledger = _load()
    cohort_eda(ledger)
    test_retest_summary(ledger)
    geometry_and_resampling(ledger)
    simulation_contract(ledger)
    model_architecture(ledger)
    detectability_atlas(ledger)
    activation_by_noise_proxy(ledger)
    end_to_end_pipeline(ledger)
    print(f"Built 8 publication figures in {FIGURE_DIR.relative_to(REPORT_DIR)}")


if __name__ == "__main__":
    main()
