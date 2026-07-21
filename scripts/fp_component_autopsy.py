"""Phase 0 FP autopsy for the banked B2 checkpoint (FP_REDUCTION plan H0).

Research prototype. Simulated vascular-like abnormalities only. Image-domain
clean/false-activation rates, NOT clinical specificity.

Runs banked B2 (main head, thr=0.5, m=0) over the frozen val cache and extracts,
per PREDICTED connected component (8-connectivity, valid-masked):
  size(px), max_score, mean_score, |pet_diff| in-component, raw_pet in-component,
  and a TP/FP label (component overlaps the GT target => TP
  else FP).

Answers H0: are FP components smaller / lower-score than TP components, and do
FPs stay small/low-score in the HIGH-NOISE tertile (=> thr/m can fix them), or do
they become large + high-score there (=> thr/m cannot
need a CNR gate)?

Correctness self-check: reproduces the banked clean rate (~0.715) and the atlas
noise-tertile FP rates (16.1 / 23.1 / 62.1 %) or it aborts -- so the autopsy is
only trusted if it lands on the already-audited numbers.

Outputs -> runs/diagnostics/fp_autopsy_2026-07-20/
  fp_components.csv, tp_components.csv, per_sample.csv, summary.md, gallery.png
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Atlas-audited noise tertile edges (noise_suv_robust_std over all 208 samples).
NOISE_EDGES = (0.4514, 0.5939)
# Atlas-audited reference values -- the self-check targets.
ATLAS_CLEAN = 0.7154  # 93/130
ATLAS_FP_TERTILE = (0.1613, 0.2308, 0.6207)  # low/mid/high on negatives


def robust_std(vals: np.ndarray) -> float:
    """1.4826 * MAD -- identical to the atlas noise proxy."""
    if vals.size == 0:
        return float("nan")
    med = np.median(vals)
    return float(1.4826 * np.median(np.abs(vals - med)))


def noise_tertile(v: float) -> str:
    if v <= NOISE_EDGES[0]:
        return "low"
    if v <= NOISE_EDGES[1]:
        return "mid"
    return "high"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model, meta = load_inference_model(CKPT, device=DEVICE)
    print(
        f"loaded {meta.model_name} cfg={meta.config_hash} ep={meta.epoch} dev={DEVICE}"
    )

    files = sorted(CACHE.glob("sample_*.npz"))
    struct = ndimage.generate_binary_structure(2, 2)  # 8-connectivity, matches metrics
    fp_rows: list[dict] = []
    tp_rows: list[dict] = []
    samples: list[dict] = []

    for f in files:
        d = np.load(f, allow_pickle=False)
        left = d["left_view"]
        right = d["right_view"]
        pdiff = d["pet_diff"]
        raw = d["raw_pet"][0].astype(np.float32)  # (H,W) SUVbw
        tgt = d["target_mask"][0] >= 0.5
        valid = d["valid_mask"][0] >= 0.5
        meta_j = json.loads(str(d["meta_json"]))
        subject = meta_j.get("subject", "NA")
        positive = bool(tgt.sum() > 0)

        score = predict_abnormality_score(
            model,
            torch.from_numpy(left),
            torch.from_numpy(right),
            torch.from_numpy(pdiff),
            device=DEVICE,
        )  # (H,W)
        # |pet_diff| summed over its channels -> a single per-pixel residual magnitude
        pdiff_mag = np.abs(pdiff).sum(axis=0).astype(np.float32)  # (H,W)

        # noise proxy over the valid, NON-target region (=> whole crop for negatives)
        bg = valid & (~tgt)
        noise = robust_std(raw[bg])
        tert = noise_tertile(noise)

        labeled, n = connected_components(score, threshold=THR, valid_mask=valid)
        idx = int(f.stem.split("_")[1])
        samples.append(
            dict(
                idx=idx,
                subject=subject,
                positive=positive,
                noise=noise,
                noise_tertile=tert,
                n_pred_components=n,
                clean=(n == 0),
                gt_px=int(tgt.sum()),
            )
        )
        for lab in range(1, n + 1):
            comp = labeled == lab
            size = int(comp.sum())
            overlaps_gt = bool((comp & tgt).sum() > 0)
            row = dict(
                idx=idx,
                subject=subject,
                positive=positive,
                noise=noise,
                noise_tertile=tert,
                size=size,
                max_score=float(score[comp].max()),
                mean_score=float(score[comp].mean()),
                pdiff_max=float(pdiff_mag[comp].max()),
                pdiff_mean=float(pdiff_mag[comp].mean()),
                raw_mean=float(raw[comp].mean()),
                is_tp=overlaps_gt,
            )
            (tp_rows if overlaps_gt else fp_rows).append(row)

    # ---- write CSVs ----
    def write_csv(rows: list[dict], path: Path) -> None:
        if not rows:
            path.write_text("")
            return
        cols = list(rows[0].keys())
        lines = [",".join(cols)]
        for r in rows:
            lines.append(",".join(str(r[c]) for c in cols))
        path.write_text("\n".join(lines) + "\n")

    write_csv(fp_rows, OUT / "fp_components.csv")
    write_csv(tp_rows, OUT / "tp_components.csv")
    write_csv(samples, OUT / "per_sample.csv")

    negs = [s for s in samples if not s["positive"]]
    poss = [s for s in samples if s["positive"]]
    clean = np.mean([s["clean"] for s in negs])
    # FP rate = P(activation) per noise tertile on NEGATIVES
    fp_tert = {}
    for t in ("low", "mid", "high"):
        sub = [s for s in negs if s["noise_tertile"] == t]
        fp_tert[t] = (
            np.mean([not s["clean"] for s in sub]) if sub else float("nan"),
            len(sub),
        )

    # ---- self-check against audited numbers ----
    ok_clean = abs(clean - ATLAS_CLEAN) < 0.01
    ok_tert = all(
        abs(fp_tert[t][0] - a) < 0.02
        for t, a in zip(("low", "mid", "high"), ATLAS_FP_TERTILE)
    )
    print(f"SELF-CHECK clean={clean:.4f} (atlas {ATLAS_CLEAN}) ok={ok_clean}")
    print(
        f"SELF-CHECK fp_tertile low/mid/high={[round(fp_tert[t][0], 3) for t in ('low', 'mid', 'high')]} "
        f"(atlas {ATLAS_FP_TERTILE}) ok={ok_tert}"
    )

    fp_sz = np.array([r["size"] for r in fp_rows])
    fp_sc = np.array([r["max_score"] for r in fp_rows])
    tp_sz = np.array([r["size"] for r in tp_rows])
    tp_sc = np.array([r["max_score"] for r in tp_rows])
    fp_hi = [r for r in fp_rows if r["noise_tertile"] == "high"]
    fp_hi_sz = np.array([r["size"] for r in fp_hi])
    fp_hi_sc = np.array([r["max_score"] for r in fp_hi])

    def pct(a, q):
        return float(np.percentile(a, q)) if len(a) else float("nan")

    lines = [
        "# FP autopsy (banked B2, thr=0.5, m=0) -- Phase 0 / H0",
        "",
        "> Research prototype. Image-domain false-activation only; NOT clinical specificity.",
        "",
        f"Checkpoint: `{CKPT}` (cfg {meta.config_hash}, ep {meta.epoch}); device {DEVICE}.",
        f"Samples: {len(samples)} ({len(poss)} pos / {len(negs)} neg).",
        "",
        "## Self-check vs audited atlas",
        f"- clean rate = **{clean:.4f}** (atlas {ATLAS_CLEAN}) -> {'OK' if ok_clean else 'MISMATCH'}",
        f"- FP by noise tertile low/mid/high = **{fp_tert['low'][0]:.3f} / {fp_tert['mid'][0]:.3f} / {fp_tert['high'][0]:.3f}** "
        f"(n={fp_tert['low'][1]}/{fp_tert['mid'][1]}/{fp_tert['high'][1]}; atlas {ATLAS_FP_TERTILE}) -> {'OK' if ok_tert else 'MISMATCH'}",
        "",
        "## Component populations",
        f"- FP components: **{len(fp_rows)}** across {len({r['idx'] for r in fp_rows})} negatives+positives",
        f"- TP components: **{len(tp_rows)}**",
        "",
        "### Size (pixels)  [H_size: are FPs smaller than TPs?]",
        "| pop | n | p10 | p25 | median | p75 | p90 |",
        "|---|--:|--:|--:|--:|--:|--:|",
        f"| FP (all) | {len(fp_sz)} | {pct(fp_sz, 10):.1f} | {pct(fp_sz, 25):.1f} | {pct(fp_sz, 50):.1f} | {pct(fp_sz, 75):.1f} | {pct(fp_sz, 90):.1f} |",
        f"| FP (high-noise) | {len(fp_hi_sz)} | {pct(fp_hi_sz, 10):.1f} | {pct(fp_hi_sz, 25):.1f} | {pct(fp_hi_sz, 50):.1f} | {pct(fp_hi_sz, 75):.1f} | {pct(fp_hi_sz, 90):.1f} |",
        f"| TP | {len(tp_sz)} | {pct(tp_sz, 10):.1f} | {pct(tp_sz, 25):.1f} | {pct(tp_sz, 50):.1f} | {pct(tp_sz, 75):.1f} | {pct(tp_sz, 90):.1f} |",
        "",
        "### Max score  [H_score: are FPs lower-score than TPs?]",
        "| pop | n | p10 | p25 | median | p75 | p90 |",
        "|---|--:|--:|--:|--:|--:|--:|",
        f"| FP (all) | {len(fp_sc)} | {pct(fp_sc, 10):.3f} | {pct(fp_sc, 25):.3f} | {pct(fp_sc, 50):.3f} | {pct(fp_sc, 75):.3f} | {pct(fp_sc, 90):.3f} |",
        f"| FP (high-noise) | {len(fp_hi_sc)} | {pct(fp_hi_sc, 10):.3f} | {pct(fp_hi_sc, 25):.3f} | {pct(fp_hi_sc, 50):.3f} | {pct(fp_hi_sc, 75):.3f} | {pct(fp_hi_sc, 90):.3f} |",
        f"| TP | {len(tp_sc)} | {pct(tp_sc, 10):.3f} | {pct(tp_sc, 25):.3f} | {pct(tp_sc, 50):.3f} | {pct(tp_sc, 75):.3f} | {pct(tp_sc, 90):.3f} |",
        "",
        "### How many FPs would a thr/m filter survive?",
    ]
    for m in (5, 10, 15):
        surv = int((fp_sz >= m).sum())
        surv_hi = int((fp_hi_sz >= m).sum())
        tp_lost = int((tp_sz < m).sum())
        lines.append(
            f"- m={m}: FP survivors {surv}/{len(fp_sz)} (high-noise {surv_hi}/{len(fp_hi_sz)}); TP lost {tp_lost}/{len(tp_sz)}"
        )
    for t in (0.6, 0.7):
        surv = int((fp_sc >= t).sum())
        surv_hi = int((fp_hi_sc >= t).sum())
        tp_lost = int((tp_sc < t).sum())
        lines.append(
            f"- thr={t}: FP survivors {surv}/{len(fp_sc)} (high-noise {surv_hi}/{len(fp_hi_sc)}); TP-component max<thr {tp_lost}/{len(tp_sc)}"
        )

    # symmetry-break signal: do FP components sit in high-|pet_diff|?
    fp_pd = np.array([r["pdiff_mean"] for r in fp_rows])
    tp_pd = np.array([r["pdiff_mean"] for r in tp_rows])
    lines += [
        "",
        "### Symmetry-break check (|pet_diff| in-component, mean)",
        f"- FP median |pet_diff| = {pct(fp_pd, 50):.3f}; TP median |pet_diff| = {pct(tp_pd, 50):.3f}",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[13:]))

    # ---- gallery: worst high-noise FP negatives ----
    _gallery(model, files, samples, struct)
    print(f"\nwrote {OUT}/summary.md, *.csv, gallery.png")
    if not (ok_clean and ok_tert):
        raise SystemExit(
            "SELF-CHECK FAILED -- autopsy numbers do not match audited atlas; do not trust."
        )


def _gallery(model, files, samples, struct) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # pick the high-noise NEGATIVES that fired (worst FP cases), by n_components then noise
    dirty_hi = sorted(
        [
            s
            for s in samples
            if (not s["positive"]) and s["noise_tertile"] == "high" and not s["clean"]
        ],
        key=lambda s: (-s["n_pred_components"], -s["noise"]),
    )[:6]
    if not dirty_hi:
        return
    fig, axes = plt.subplots(len(dirty_hi), 4, figsize=(13, 3.1 * len(dirty_hi)))
    if len(dirty_hi) == 1:
        axes = axes[None, :]
    for r, s in enumerate(dirty_hi):
        f = files[
            [i for i, ff in enumerate(files) if int(ff.stem.split("_")[1]) == s["idx"]][
                0
            ]
        ]
        d = np.load(f, allow_pickle=False)
        raw = d["raw_pet"][0]
        pdiff = d["pet_diff"]
        valid = d["valid_mask"][0] >= 0.5
        pdiff_mag = np.abs(pdiff).sum(axis=0)
        score = predict_abnormality_score(
            model,
            torch.from_numpy(d["left_view"]),
            torch.from_numpy(d["right_view"]),
            torch.from_numpy(pdiff),
            device=DEVICE,
        )
        labeled, n = connected_components(score, threshold=THR, valid_mask=valid)
        panels = [
            (raw, "raw_pet (SUVbw)", "magma"),
            (pdiff_mag, "|pet_diff| (L-R residual)", "viridis"),
            (score, "B2 score (sigmoid)", "inferno"),
            (score, f"pred @0.5 -> {n} FP comp", "inferno"),
        ]
        for c, (img, title, cmap) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=9)
            if c == 3:
                for lab in range(1, n + 1):
                    ys, xs = np.where(labeled == lab)
                    ax.add_patch(
                        Rectangle(
                            (xs.min() - 1, ys.min() - 1),
                            xs.max() - xs.min() + 2,
                            ys.max() - ys.min() + 2,
                            fill=False,
                            edgecolor="lime",
                            lw=1.3,
                        )
                    )
                    ax.text(
                        xs.min(),
                        ys.min() - 2,
                        f"{len(xs)}px\n{score[ys, xs].max():.2f}",
                        color="lime",
                        fontsize=6,
                        va="bottom",
                    )
            if c == 0:
                ax.set_ylabel(f"neg #{s['idx']}\nnoise {s['noise']:.2f}", fontsize=8)
    fig.suptitle(
        "Worst high-noise FALSE-POSITIVE negatives (banked B2, thr=0.5) -- healthy crops the model fires on",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT / "gallery.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
