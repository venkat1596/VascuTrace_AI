"""Bilateral-CNR gate probe (FP_REDUCTION Phase-2 evidence).

Research prototype. Image-domain false-activation only
NOT clinical specificity.

The autopsy showed thr is inert (FP scores saturate at 1.0) and size is a Pareto
trade. This probe tests the mechanism-matched lever from the PET note (Rank 1):
a noise-adaptive bilateral consistency gate. For each predicted component of
B2 @ thr=0.5, define

    signal   = |mean(pet_diff) over the component|      # coherent one-sided L-R excess
    sigma_loc= 1.4826 * MAD(pet_diff) in a local annulus around the component
    CNR      = signal / sigma_loc

A true unilateral lesion is a coherent excess in relatively uniform tissue (high
CNR)
a high-uptake-structure edge artifact sits in locally textured pet_diff
(high sigma_loc) => low CNR. We compare how well CNR vs size vs max-score separate
FP from TP, and sweep kappa to report (TP kept, FP removed, high-noise FP removed).

pet_diff is the network-normalized L-R difference (5 channels)
we reduce to a
single signed map by the mean over channels. CNR is a ratio so the normalization
largely cancels. Runs on CPU by default to avoid contending with a GPU grid.

Outputs -> runs/diagnostics/fp_autopsy_2026-07-20/
  cnr_components.csv, cnr_summary.md, cnr_gate.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from src.vascutrace.ml.infer import load_inference_model, predict_abnormality_score
from src.vascutrace.ml.metrics import connected_components

CKPT = Path("runs/siamese_p4b2_deepsup/best_constrained_iou.pt")
CACHE = Path("data/processed/p6_cache_big/val")
OUT = Path("runs/diagnostics/fp_autopsy_2026-07-20")
THR = 0.5
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"  # grid done
NOISE_EDGES = (0.4514, 0.5939)
ANNULUS_DILATE = 6  # pixels of local context around a component


def robust_std(vals: np.ndarray) -> float:
    if vals.size < 4:
        return float("nan")
    med = np.median(vals)
    return float(1.4826 * np.median(np.abs(vals - med)))


def noise_tertile(v: float) -> str:
    return "low" if v <= NOISE_EDGES[0] else ("mid" if v <= NOISE_EDGES[1] else "high")


def fp_removed_at_tp_kept(fp_vals, tp_vals, keep_frac=0.95, higher_is_lesion=True):
    """Choose the gate cutoff that keeps `keep_frac` of TP components, then report
    the fraction of FP components removed. `higher_is_lesion`: larger value => more
    lesion-like (CNR, size, score). Returns (cutoff, tp_kept, fp_removed)."""
    fp_vals = np.asarray(fp_vals, float)
    tp_vals = np.asarray(tp_vals, float)
    if higher_is_lesion:
        cutoff = float(np.quantile(tp_vals, 1 - keep_frac))  # keep TP >= cutoff
        tp_kept = float(np.mean(tp_vals >= cutoff))
        fp_removed = float(np.mean(fp_vals < cutoff))
    else:
        cutoff = float(np.quantile(tp_vals, keep_frac))
        tp_kept = float(np.mean(tp_vals <= cutoff))
        fp_removed = float(np.mean(fp_vals > cutoff))
    return cutoff, tp_kept, fp_removed


def auc(fp_vals, tp_vals):
    """P(TP value > FP value) -- rank separation; 0.5 = no separation."""
    fp_vals = np.asarray(fp_vals, float)
    tp_vals = np.asarray(tp_vals, float)
    if len(fp_vals) == 0 or len(tp_vals) == 0:
        return float("nan")
    gt = sum((tp_vals[:, None] > fp_vals[None, :]).sum(axis=1))
    eq = sum((tp_vals[:, None] == fp_vals[None, :]).sum(axis=1))
    return float((gt + 0.5 * eq) / (len(fp_vals) * len(tp_vals)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model, meta = load_inference_model(CKPT, device=DEVICE)
    print(f"loaded {meta.model_name} dev={DEVICE}")
    files = sorted(CACHE.glob("sample_*.npz"))
    rows: list[dict] = []
    for f in files:
        d = np.load(f, allow_pickle=False)
        pdiff = d["pet_diff"].astype(np.float32)
        signed = pdiff.mean(axis=0)  # (H,W) signed L-R difference
        tgt = d["target_mask"][0] >= 0.5
        valid = d["valid_mask"][0] >= 0.5
        raw = d["raw_pet"][0].astype(np.float32)
        subject = json.loads(str(d["meta_json"])).get("subject", "NA")
        positive = bool(tgt.sum() > 0)
        noise = robust_std(raw[valid & (~tgt)])
        tert = noise_tertile(noise)
        score = predict_abnormality_score(
            model,
            torch.from_numpy(d["left_view"]),
            torch.from_numpy(d["right_view"]),
            torch.from_numpy(pdiff),
            device=DEVICE,
        )
        labeled, n = connected_components(score, threshold=THR, valid_mask=valid)
        idx = int(f.stem.split("_")[1])
        for lab in range(1, n + 1):
            comp = labeled == lab
            # local annulus: dilate the component, remove it, stay inside valid
            dil = ndimage.binary_dilation(comp, iterations=ANNULUS_DILATE)
            annulus = dil & (~comp) & valid
            signal = abs(float(signed[comp].mean()))
            sigma_loc = robust_std(signed[annulus])
            cnr = (
                signal / sigma_loc
                if (sigma_loc and np.isfinite(sigma_loc) and sigma_loc > 1e-6)
                else np.nan
            )
            rows.append(
                dict(
                    idx=idx,
                    subject=subject,
                    positive=positive,
                    noise_tertile=tert,
                    size=int(comp.sum()),
                    max_score=float(score[comp].max()),
                    signal=signal,
                    sigma_loc=sigma_loc,
                    cnr=float(cnr),
                    is_tp=bool((comp & tgt).sum() > 0),
                )
            )

    # write csv
    cols = list(rows[0].keys())
    (OUT / "cnr_components.csv").write_text(
        ",".join(cols)
        + "\n"
        + "\n".join(",".join(str(r[c]) for c in cols) for r in rows)
        + "\n"
    )

    valid_rows = [r for r in rows if np.isfinite(r["cnr"])]
    tp = [r for r in valid_rows if r["is_tp"]]
    fp = [r for r in valid_rows if not r["is_tp"]]
    fp_hi = [r for r in fp if r["noise_tertile"] == "high"]

    def col(rs, k):
        return np.array([r[k] for r in rs], float)

    lines = [
        "# Bilateral-CNR gate probe (banked B2, thr=0.5)",
        "",
        "> Research prototype. Image-domain false-activation only; NOT clinical specificity.",
        f"Components with finite CNR: TP={len(tp)}, FP={len(fp)} (high-noise FP={len(fp_hi)}).",
        f"CNR = |mean(pet_diff) in comp| / robust_sigma(pet_diff in {ANNULUS_DILATE}px annulus).",
        "",
        "## Which lever separates FP from TP?  (rank AUC: P(TP lever-value > FP), 0.5=none)",
        "| lever | AUC(TP>FP) | note |",
        "|---|--:|---|",
        f"| CNR (bilateral) | {auc(col(fp, 'cnr'), col(tp, 'cnr')):.3f} | higher = more lesion-like |",
        f"| size (px) | {auc(col(fp, 'size'), col(tp, 'size')):.3f} | |",
        f"| max_score | {auc(col(fp, 'max_score'), col(tp, 'max_score')):.3f} | ~0.5 => saturated, no signal |",
        f"| signal only (|mean pet_diff|) | {auc(col(fp, 'signal'), col(tp, 'signal')):.3f} | before local-noise norm |",
        "",
        "## Gate operating point: keep 95% of TP components, how many FP removed?",
        "| lever | cutoff | TP kept | FP removed (all) | FP removed (high-noise) |",
        "|---|--:|--:|--:|--:|",
    ]
    for name, key, hi in (
        ("CNR", "cnr", True),
        ("size", "size", True),
        ("max_score", "max_score", True),
    ):
        cut, tpk, fpr = fp_removed_at_tp_kept(col(fp, key), col(tp, key), 0.95, hi)
        _, _, fpr_hi = (
            fp_removed_at_tp_kept(col(fp_hi, key), col(tp, key), 0.95, hi)
            if fp_hi
            else (0, 0, float("nan"))
        )
        lines.append(f"| {name} | {cut:.3f} | {tpk:.3f} | {fpr:.3f} | {fpr_hi:.3f} |")

    # explicit CNR kappa sweep (Rose-like 1..4)
    lines += [
        "",
        "## CNR kappa sweep (component kept iff CNR >= kappa)",
        "| kappa | TP kept | FP removed (all) | FP removed (high-noise) |",
        "|--:|--:|--:|--:|",
    ]
    tp_cnr, fp_cnr, fphi_cnr = col(tp, "cnr"), col(fp, "cnr"), col(fp_hi, "cnr")
    for k in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        lines.append(
            f"| {k:.1f} | {np.mean(tp_cnr >= k):.3f} | {np.mean(fp_cnr < k):.3f} | "
            f"{(np.mean(fphi_cnr < k) if len(fphi_cnr) else float('nan')):.3f} |"
        )
    lines += [
        "",
        "## Distributions (median [p25, p75])",
        f"- CNR: TP {np.median(tp_cnr):.2f} [{np.percentile(tp_cnr, 25):.2f}, {np.percentile(tp_cnr, 75):.2f}] "
        f"| FP {np.median(fp_cnr):.2f} [{np.percentile(fp_cnr, 25):.2f}, {np.percentile(fp_cnr, 75):.2f}]",
    ]
    (OUT / "cnr_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[6:]))

    _plot(tp_cnr, fp_cnr, fphi_cnr, col(tp, "max_score"), col(fp, "max_score"))
    print(f"\nwrote {OUT}/cnr_summary.md, cnr_components.csv, cnr_gate.png")


def _plot(tp_cnr, fp_cnr, fphi_cnr, tp_sc, fp_sc) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    bins = np.linspace(
        0, max(6, np.percentile(np.concatenate([tp_cnr, fp_cnr]), 98)), 30
    )
    ax[0].hist(
        tp_cnr,
        bins=bins,
        alpha=0.6,
        label=f"TP (n={len(tp_cnr)})",
        color="#2c7fb8",
        density=True,
    )
    ax[0].hist(
        fp_cnr,
        bins=bins,
        alpha=0.6,
        label=f"FP (n={len(fp_cnr)})",
        color="#d95f0e",
        density=True,
    )
    ax[0].hist(
        fphi_cnr,
        bins=bins,
        alpha=0.5,
        label=f"FP high-noise (n={len(fphi_cnr)})",
        color="#e34a33",
        histtype="step",
        lw=2,
        density=True,
    )
    ax[0].set_title("Bilateral CNR: TP vs FP")
    ax[0].set_xlabel("CNR = signal/sigma_local")
    ax[0].legend(fontsize=8)

    sb = np.linspace(0.5, 1.0, 26)
    ax[1].hist(tp_sc, bins=sb, alpha=0.6, label="TP", color="#2c7fb8", density=True)
    ax[1].hist(fp_sc, bins=sb, alpha=0.6, label="FP", color="#d95f0e", density=True)
    ax[1].set_title("Max score: SATURATED, no signal")
    ax[1].set_xlabel("component max sigmoid score")
    ax[1].legend(fontsize=8)

    ks = np.linspace(0, 4, 60)
    ax[2].plot(
        ks, [np.mean(tp_cnr >= k) for k in ks], label="TP kept", color="#2c7fb8", lw=2
    )
    ax[2].plot(
        ks, [np.mean(fp_cnr < k) for k in ks], label="FP removed", color="#d95f0e", lw=2
    )
    ax[2].plot(
        ks,
        [np.mean(fphi_cnr < k) for k in ks],
        label="FP high-noise removed",
        color="#e34a33",
        lw=2,
        ls="--",
    )
    ax[2].set_title("CNR gate operating curve")
    ax[2].set_xlabel("kappa")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)
    fig.suptitle(
        "Why the bilateral-CNR gate beats threshold: score is saturated, CNR is not",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "cnr_gate.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
