"""Focal Tversky + Combo losses (VascuTrace Phase 6) -- an under
-segmentation-resistant alternative/complement to ``model.py``'s plain
``dice_bce_loss``.

RESEARCH_PROTOTYPE_WARNING
---------------------------------------------------------------------------
Research prototype. Trained and evaluated using simulated vascular-like
abnormalities, not confirmed human post-angioplasty lesions.
---------------------------------------------------------------------------

Implementation notes
============================================================================
This module exists because of a concrete, measured failure this project's
own evaluation found: the trained checkpoint's honest positive-only Dice
(0.487) sits far below what its training-time SELECTION metric (a blended
val Dice of 0.740) suggested, because a blended Dice trivially rewards
predicting empty on the majority-healthy validation slices -- the
"empty-reference" pitfall (Reinke, Tizabi, Sudre et al., 2024, *Understanding
Metric-Related Pitfalls in Image Analysis Validation*, Nature Methods,
"Pitfall 8: Ignoring the effect of the reference annotation" /
empty-reference handling). Two independent, literature-grounded causes
contribute, and this module fixes the loss-shape half of the problem
(``train.py`` fixes the checkpoint-selection half -- see its own module
docstring, item 9): plain Dice/BCE loss is unstable and collapse-prone
under this class of foreground/background imbalance (Sudre, Li, Vercauteren,
Ourselin & Cardoso, 2017, *Generalised Dice Overlap as a Deep Learning Loss
Function for Highly Unbalanced Segmentations*, arXiv:1707.03237; Kervadec,
Bouchtiba, Desrosiers, Granger, Dolz & Ben Ayed, 2019, *Boundary Loss for
Highly Unbalanced Segmentation*, arXiv:1812.07032, both documenting Dice
-family loss instability/local-minima collapse under severe class
imbalance), and a plain Dice loss's gradient vanishes near convergence
precisely where a small lesion needs the most encouragement to keep
improving (see item 2 below).

1. Tversky index -- Dice with independently-tunable FN/FP weights
   ------------------------------------------------------------------------
   The Tversky index (Salehi, Erdogmus & Gholipour, 2017, *Tversky Loss
   Function for Image Segmentation Using 3D Fully Convolutional Deep
   Networks*, arXiv:1706.05721, Eq. 2-3) generalizes Dice/F1 with two
   independent penalty weights on the two error types. This module follows
   the parameterization and hyperparameter convention of the cited Focal
   Tversky variant by Abraham &
   Khan, 2019, *A Novel Focal Tversky Loss Function With Improved Attention
   U-Net For Lesion Segmentation*, arXiv:1810.07842, Eq. 3, and (more
   authoritatively, since the paper's own typeset equation is ambiguous
   about which weight multiplies which error term -- see the note below)
   their own released reference implementation
   (https://github.com/nabsabraham/focal-tversky-unet, function
   ``tversky``, verified directly, not recalled):

       T(alpha, beta) = TP / (TP + alpha*FN + beta*FP)

   i.e. **alpha weights false negatives (FN, missed lesion voxels /
   under-segmentation), beta weights false positives (FP)**. Plain Dice is
   the special case alpha=beta=0.5. This implementation's default alpha=0.7,
   beta=0.3 matches Abraham & Khan's own reported operating point exactly
   (their paper, Sec. 2.1: "We hypothesize using a higher alpha in our
   generalized loss function will improve model convergence by shifting
   the focus to minimize FN predictions. Therefore, we train all models
   with alpha=0.7 and beta=0.3") -- **CONFIRMED: FN-weight = alpha = 0.7**,
   directly combating this project's own measured under-segmentation
   (honest positive-only Dice 0.487 vs. the collapse-prone blended-Dice
   -selected checkpoint) by making a missed-lesion voxel cost more than an
   over-eager one of the same size.

   Verification note (a genuine literature-cross-check finding, not a
   silent assumption): Abraham & Khan's own PAPER TEXT, read directly from
   the source PDF, typesets Eq. 3's two penalty terms as
   ``alpha * sum(p_ic * g_i,c-bar)`` (a false-POSITIVE-shaped term: predicted
   -lesion score times ground-truth-NON-lesion) and
   ``beta * sum(p_i,c-bar * g_ic)`` (a false-NEGATIVE-shaped term) -- i.e.
   the OPPOSITE assignment from the alpha=FN/beta=FP convention stated in
   their own prose and used in their own released code. This is an
   internal inconsistency in the paper's typeset equation (also present,
   identically, in the original Salehi et al. 2017 Tversky paper this one
   builds on, whose Eq. 2 the same way assigns alpha->FP, beta->FN). Given
   three independent, mutually-consistent sources -- Abraham & Khan's own
   prose ("higher alpha ... to minimize FN predictions"), their reported
   hyperparameter choice's stated purpose (alpha=0.7 for recall emphasis on
   small lesions), and their own released, directly-verified code -- all
   agree on alpha->FN/beta->FP, this module follows that convention (the
   authors' own implementation, not their own paper's possibly-mistyped
   equation) and flags the discrepancy here rather than silently picking a
   side.

2. Focal Tversky Loss (FTL) -- keeping gradient signal alive near
   convergence
   ------------------------------------------------------------------------
   :func:`focal_tversky_loss` = ``(1 - T)^gamma`` (Abraham & Khan, 2019,
   Sec. 2.1, Eq. 4 and their own released code's ``focal_tversky``
   function: ``K.pow((1-pt_1), gamma)``, ``gamma = 0.75`` -- the exact
   value this module defaults to). Verified numerically (not merely
   quoted): for ``0 < gamma < 1``, ``x^gamma >= x`` on ``x in [0, 1]``, and
   the RELATIVE amplification ``x^gamma / x = x^(gamma-1)`` is LARGEST
   precisely when ``x = (1 - T)`` is SMALL -- i.e. when the prediction is
   already close to correct (``T`` near 1). Concretely: at ``gamma=0.75``,
   ``(1-T)=0.01`` (a near-perfect T=0.99) is amplified ``0.01^0.75/0.01 ~=
   3.16x`` relative to the plain (gamma=1) Tversky loss, versus only
   ``~1.03x`` at ``(1-T)=0.9`` (a poor, T=0.1, prediction). This is the
   opposite of naively "focusing on hard examples" in the classical
   Lin-et-al focal-loss sense -- what it verifiably does is keep the loss
   (and therefore the gradient) from decaying to numerically negligible
   values as the model APPROACHES a good segmentation, which is exactly
   the vanishing-gradient-near-convergence failure mode a plain Dice
   /Tversky loss exhibits on a small-ROI target (the paper's own stated
   motivation, Sec. 1: Dice loss "struggle[s] to segment small ROIs as
   they do not contribute to the loss significantly"). This module states
   this mechanism precisely, verified by direct calculation, rather than
   repeating the paper's own "focuses on hard examples" framing verbatim
   -- that framing describes the paper's OWN alternate ``(1-T)^(1/gamma)``
   parameterization (their plotted ``gamma in [1,3]``, best at ``gamma=
   4/3``; note ``1/(4/3) = 0.75``, which is exactly this module's default
   and reconciles the two parameterizations to the same numeric operating
   point) and does not straightforwardly transfer to the direct-exponent
   form this module (and the reference code) actually uses.

3. Empty-mask / degenerate-input safety
   ------------------------------------------------------------------------
   ``TP``, ``FN``, ``FP`` are all sums of non-negative products of ``[0,1]``
   sigmoid scores and ``{0,1}``-ish targets, so the Tversky index
   ``T = (TP+eps)/(TP+alpha*FN+beta*FP+eps)`` is always in ``(0, 1]`` by
   construction (never negative, never NaN, never a 0/0) -- an all-empty
   prediction-and-target pair (TP=FN=FP=0) gives ``T = eps/eps = 1``, a
   perfect (if vacuous) score, matching ``model.py``'s own established
   dice-score empty-mask convention. ``(1 - T).clamp_min(0.0)`` before the
   ``**gamma`` fractional power is a deliberate numerical-safety belt (not
   a no-op): floating-point rounding can occasionally push ``T``
   infinitesimally above ``1.0``, and raising a negative float to a
   fractional power produces ``NaN`` in real-valued floating point --
   without the clamp, one such rounding event would silently poison the
   entire batch's loss (and every subsequent gradient) with ``NaN``.

4. ``combo_loss`` -- Focal Tversky (recall-driving) + BCE-with-logits
   (early-training-stabilizing)
   ------------------------------------------------------------------------
   :func:`combo_loss` sums a weighted Focal Tversky term with a
   ``binary_cross_entropy_with_logits`` term, matching this project's own
   established compound-loss precedent
   (``src.vascutrace.ml.model.dice_bce_loss``'s module docstring, item 7:
   combining a region-overlap term with a per-pixel term is standard
   practice in the biomedical-segmentation literature) and the general
   "Combo Loss" naming precedent (Taghanaki, Zheng, Zhou, Georgescu, Sharma
   Xu, Comaniciu & Hamarneh, 2019, *Combo Loss: Handling Input and Output
   Imbalance in Multi-Organ Segmentation*, arXiv:1805.02798) -- though this
   module's combination is the direct weighted sum the coordinator's own
   spec names (``tversky_weight * FTL + bce_weight * BCE``), not
   Taghanaki et al.'s own more elaborate weighted-cross-entropy variant; it
   is not claimed to be numerically identical to that paper's formula. BCE
   -with-logits has a well-behaved, never-vanishing gradient everywhere
   (unlike a region-overlap loss on a near-empty target) and is what makes
   early training (before the model has found the lesion at all, when the
   Tversky index is near its minimum and provides comparatively little
   directional signal) stable; the Focal Tversky term then drives the
   recall/under-segmentation correction once the model has something to
   refine. The BCE term's ``valid_mask`` handling is copied verbatim from
   ``dice_bce_loss`` (masked mean, not masked sum-then-mean-over-all
   -pixels) for exact consistency with this project's existing loss.

5. Soft-target losses -- :func:`soft_bce_loss`, :func:`soft_dice_semimetric
   _loss`, :func:`soft_combo_loss` (config-gated ``train.py`` soft-target
   experiment; do NOT modify ``combo_loss`` above -- these are new,
   additive functions)
   ------------------------------------------------------------------------
   This project's simulator computes a sharp, pre-blur fractional
   occupancy ``source_fraction in [0, 1]`` for every synthetic source
   (``simulation/anomaly.py``'s ``_supersampled_occupancy``;
   ``dataset.py``'s module docstring, item 9) and, until this item,
   discarded it after thresholding it into the binary ``target_mask``.
   ``combo_loss`` above is PROVABLY IMPROPER as a loss on that continuous
   field: its Tversky/Focal-Tversky machinery (item 1's ``_tversky_index``)
   sums PRODUCTS ``p*g``, ``g*(1-p)``, ``(1-g)*p`` -- exactly the "standard
   Soft Dice Loss" (SDL) construction Wang, Popordanoska, Bertels,
   Lemmens & Blaschko, 2023, *Dice Semimetric Losses: Optimizing the Dice
   Score with Soft Labels*, arXiv:2303.16296, prove (Sec. 2, the paper's
   own stated counterexample) does **not** reach its minimum at ``p = g``
   for a fractional ``g`` -- their own example: at ``g = 0.5`` uniformly,
   SDL is minimized at ``p = 1``, "which is clearly erroneous." This is
   a hard-vertex preference that defeats the purpose of retaining a
   continuous occupancy target. The soft-target route therefore does not
   use that Tversky form.

   :func:`soft_bce_loss` is the proper core (BCE-with-logits is a
   classical *strictly proper scoring rule* -- Gneiting & Raftery, 2007,
   *Strictly Proper Scoring Rules, Prediction, and Estimation*, JASA,
   for a soft/probabilistic target: for FIXED soft target ``g in [0, 1]``,
   ``L(p, g) = -(g*log(p) + (1-g)*log(1-p))`` is strictly convex in
   ``p in (0, 1)`` (second derivative ``g/p^2 + (1-g)/(1-p)^2 >= 0``,
   ``> 0`` whenever ``g`` is not simultaneously forcing both terms to
   vanish) with ``dL/dp = -(g/p - (1-g)/(1-p)) = 0`` at exactly ``p = g``
   -- a closed-form, directly-verified fact (see this implementation's numeric
   generated-fixture verification recorded with this implementation),
   not a citation taken on faith. Note the minimum VALUE at ``p=g`` is the
   binary entropy ``H(g) = -(g*log(g)+(1-g)*log(1-g))`` -- zero only when
   ``g`` is exactly 0 or 1 -- NOT zero for a genuinely fractional ``g``;
   properness here means "uniquely minimized in ``p`` at ``p=g``", the
   standard scoring-rule sense, not "reaches loss value zero at the
   target" (that second, Dice-specific sense is what item 5's Dice term,
   below, additionally provides).

   :func:`soft_dice_semimetric_loss` implements Wang et al.'s Eq. 4,
   ``DML_2`` (their second Dice Semimetric Loss; here generalized to
   independent FN/FP-style weights -- see the function's own docstring for
   the exact algebraic identity this uses and why it stays provably proper
   for any nonnegative weights, plus a precise disclosure of where this
   module's ``alpha``/``beta`` convention differs numerically from item
   1's ``_tversky_index`` ``alpha``/``beta``). Their Theorem 2.2 proves two
   properties this module relies on directly, at this function's own
   ``alpha=beta=1`` default: (i) reflexivity/positivity -- the loss is
   ZERO if and only if prediction equals target EXACTLY (unlike SDL, whose
   minimum is a hard vertex for fractional targets); (ii) it is IDENTICAL
   to standard hard Dice loss when both prediction and target are binary
   (``{0, 1}``-valued) -- so this function changes nothing about how this
   project's loss behaves on the existing, already-shipped hard-target
   path, only on a genuinely fractional one.

   :func:`soft_combo_loss` = ``bce_weight * soft_bce_loss + dice_weight *
   soft_dice_semimetric_loss`` (own module docstring, matching the
   Strategy-style additive-not-modifying pattern ``combo_loss`` itself
   established for combining a per-pixel term with a region-overlap term).
   Because BOTH component losses are individually minimized (not merely at
   a stationary point, but at their own true minimum value) at exactly
   ``p = g`` everywhere on the valid region -- ``soft_bce_loss`` per-pixel
   (strict convexity, shown above) and ``soft_dice_semimetric_loss``
   globally (Theorem 2.2) -- ANY positively-weighted sum of the two is
   ALSO minimized at exactly ``p = g``: for any other prediction ``p'``,
   ``bce_weight*soft_bce_loss(p',g) >= bce_weight*soft_bce_loss(g,g)``
   (pointwise, hence summed) and ``dice_weight*soft_dice_semimetric_loss
   (p',g) >= 0 = dice_weight*soft_dice_semimetric_loss(g,g)``, so their sum
   at ``p'`` is bounded below by their sum at ``p=g``, with equality iff
   ``p'=g`` (for positive weights). This is an elementary, directly
   -verifiable argument (also checked numerically -- see
   the generated-fixture verification), not a citation-only
   claim; it is what makes ``soft_combo_loss`` "proper on soft targets"
   in the same operational sense ``soft_bce_loss`` alone is, while still
   getting a region-overlap term's usual benefit of keeping gradient
   signal alive on a small lesion (item 2's motivation for Focal Tversky
   in the first place) without reintroducing item 1's vertex-collapse
   failure.

6. Boundary-local auxiliary loss -- :func:`boundary_auxiliary_loss`
   follows the frozen A/B/C fractional-boundary-supervision contract. It
   is a new, additive function, so
   ``combo_loss``/``focal_tversky_loss`` above remain byte-for-byte
   unchanged. The auxiliary is added as a new loss function that leaves the
   tested combo_loss closed for modification; A/B/C differ only by a
   swapped auxiliary target, i.e. a Strategy-style parameter, not by
   editing the shipped loss.")

   Unlike item 5's soft-target family (which REPLACES the training target
   everywhere with ``source_fraction``), this item keeps the shipped hard
   -mask ``combo_loss`` objective as the PRIMARY term and adds a
   separately-normalized auxiliary BCE term restricted to a narrow,
   privileged support set -- the pixels where the simulator's pre-blur
   fractional occupancy ``source_fraction`` is genuinely fractional
   (``0 < F < 1``), i.e. the simulator's own boundary. the boundary experiment Sec 3
   motivates this precisely as pre-blur simulator geometry, never as
   measured PET activity, a calibrated probability, or clinical truth --
   see that section for the full permitted-interpretation boundary.

   Exact contract (frozen protocol notation):

   ::

       F_i = source_fraction
       H_i = target_mask = 1[F_i >= 0.5]
       M_i = valid_mask
       W_i = M_i * 1[0 < F_i < 1]                    (support)
       G_i = H_i (arm B, "hard") or F_i (arm C, "fraction")

       numerator_i   = sum_j W_ij * BCEWithLogits(z_ij, G_ij)
       denominator_i = sum_j W_ij
       L_aux_i       = numerator_i / denominator_i,  if denominator_i > 0
                       exact differentiable zero,     otherwise
       L_aux         = sum_i eligible_i * L_aux_i / batch_size

   ``eligible_i = 1[denominator_i > 0]``. Reduction uses the SAME final
   ``batch_size`` denominator regardless of how many samples in the batch
   are eligible -- an ineligible sample (no fractional-boundary pixels)
   contributes an exact, structural zero, so a batch that happens to
   contain fewer boundary-bearing samples cannot silently inflate the
   effective per-eligible-sample auxiliary weight (the frozen protocol states, "An
   ineligible sample contributes exact zero auxiliary...").

   Zero-support numerical safety (no NaN, no non-differentiable branch):
   whenever ``denominator_i == 0``, every ``W_ij`` in that sample's sum is
   also ``0``, so ``numerator_i`` is ALREADY exactly ``0`` before any
   division happens (a product-with-zero, not a cancellation) -- dividing
   that already-zero numerator by ANY strictly-positive safe denominator
   (this implementation uses ``denominator_i.clamp_min(1.0)``) yields
   exactly ``0`` with exactly zero gradient, never a bare ``0/0``. The
   explicit ``eligible_i`` multiply in the batch reduction is therefore a
   belt-and-braces re-assertion of that same invariant (matching the
   plan's own literal ``sum_i eligible_i * L_aux_i`` wording), not the sole
   mechanism preventing NaN.

   ``W`` is a data-derived (comparison-based) mask -- ``0 < F < 1`` and
   ``valid_mask`` never carry a gradient of their own, so no gradient
   flows through the SUPPORT decision, only through the BCE term at
   supported pixels. This also gives the protocol's required "invalid pixels
   have exactly zero loss and gradient" and "background-expansion
   invariance" properties for free: any pixel with ``valid_mask=0`` or
   with ``F`` exactly ``0``/``1`` (i.e. not a boundary pixel) has
   ``W_ij=0`` and therefore contributes nothing to ``L_aux`` or its
   gradient, however many such pixels a fixture adds elsewhere in the
   crop.

   Arms B and C call this SAME function, differing only in
   ``target_mode`` ("hard" selects ``G=H``, "fraction" selects ``G=F``) --
   support, weighting (uniform ``W_ij in {0,1}``, per the protocol's
   "Uniform weights ask the narrowest causal question"), normalization,
   and topology are otherwise byte-identical between the two calls, which
   is what makes the C-minus-B contrast an attribution of
   "fractional values" and nothing else.

   BCE-with-logits is reused here for the same strictly-proper-scoring
   -rule reason item 5 uses it (Gneiting & Raftery, 2007): it is
   individually proper for whichever ``G`` (hard or fractional) is passed,
   so this one function is correct for both arms without special-casing.
============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.vascutrace.geometry import RESEARCH_PROTOTYPE_WARNING

__all__ = [
    "RESEARCH_PROTOTYPE_WARNING",
    "LOSSES_MODULE_VERSION",
    "focal_tversky_loss",
    "combo_loss",
    "soft_bce_loss",
    "soft_dice_semimetric_loss",
    "soft_combo_loss",
    "BoundaryAuxiliaryLoss",
    "boundary_auxiliary_loss",
]

LOSSES_MODULE_VERSION = "p6-losses-v1"


def _validate_shapes(
    logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor | None
) -> None:
    if logits.shape != target.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} != target shape {tuple(target.shape)}"
        )
    if valid_mask is not None and valid_mask.shape != target.shape:
        raise ValueError(
            f"valid_mask shape {tuple(valid_mask.shape)} != target shape {tuple(target.shape)}"
        )


def _tversky_index(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None,
    *,
    alpha: float,
    beta: float,
    eps: float,
) -> torch.Tensor:
    """Per-sample Tversky index ``T = (TP+eps)/(TP+alpha*FN+beta*FP+eps)``
    -- see module docstring, item 1, for the alpha->FN/beta->FP convention
    and item 3 for why this is always finite and in ``(0, 1]``.
    """
    score = torch.sigmoid(logits)
    b = logits.shape[0]
    flat_p = score.reshape(b, -1)
    flat_g = target.reshape(b, -1).to(dtype=flat_p.dtype)

    if valid_mask is not None:
        flat_m = valid_mask.reshape(b, -1).to(dtype=flat_p.dtype)
        flat_p = flat_p * flat_m
        flat_g = flat_g * flat_m

    tp = (flat_p * flat_g).sum(dim=1)
    fn = (flat_g * (1.0 - flat_p)).sum(dim=1)
    fp = ((1.0 - flat_g) * flat_p).sum(dim=1)

    return (tp + eps) / (tp + alpha * fn + beta * fp + eps)


def focal_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    alpha: float = 0.7,
    beta: float = 0.3,
    gamma: float = 0.75,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Focal Tversky Loss: ``mean_batch((1 - T(alpha, beta))^gamma)``. See
    module docstring, items 1-3. ``alpha`` weights false negatives (missed
    lesion voxels), ``beta`` weights false positives -- default
    ``alpha=0.7 > beta=0.3`` penalizes under-segmentation more than
    over-segmentation, matching Abraham & Khan 2019's own reported
    operating point exactly. Pure, deterministic (no RNG), and finite for
    any finite ``logits``/``{0,1}``-ish ``target`` (see item 3).
    """
    _validate_shapes(logits, target, valid_mask)
    tversky = _tversky_index(
        logits, target, valid_mask, alpha=alpha, beta=beta, eps=eps
    )
    ftl_per_sample = (1.0 - tversky).clamp_min(0.0).pow(gamma)
    return ftl_per_sample.mean()


def combo_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    tversky_weight: float = 1.0,
    bce_weight: float = 0.5,
    alpha: float = 0.7,
    beta: float = 0.3,
    gamma: float = 0.75,
    eps: float = 1e-6,
) -> torch.Tensor:
    """``tversky_weight * focal_tversky_loss(...) + bce_weight *
    BCE-with-logits(...)``. See module docstring, item 4. The BCE term's
    ``valid_mask`` handling (masked mean) is copied verbatim from
    ``src.vascutrace.ml.model.dice_bce_loss`` for exact consistency with
    this project's existing loss.
    """
    _validate_shapes(logits, target, valid_mask)

    ftl = focal_tversky_loss(
        logits, target, valid_mask, alpha=alpha, beta=beta, gamma=gamma, eps=eps
    )

    bce_elementwise = F.binary_cross_entropy_with_logits(
        logits, target.to(logits.dtype), reduction="none"
    )
    if valid_mask is not None:
        mask = valid_mask.to(logits.dtype)
        bce = (bce_elementwise * mask).sum() / mask.sum().clamp_min(eps)
    else:
        bce = bce_elementwise.mean()

    return tversky_weight * ftl + bce_weight * bce


# ---------------------------------------------------------------------------
# Soft-target losses -- see module docstring, item 5. New, additive
# functions; combo_loss/focal_tversky_loss above are UNCHANGED.
# ---------------------------------------------------------------------------


def soft_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """The proper core of the soft-target experiment (module docstring,
    item 5): plain ``binary_cross_entropy_with_logits``, masked-mean over
    ``valid_mask`` exactly like ``combo_loss``'s own BCE term (same
    formula, extracted as an independent, reusable primitive rather than
    calling into/duplicating ``combo_loss`` -- ``combo_loss`` itself is
    left completely unmodified).

    For a FIXED soft target ``g in [0, 1]`` at one pixel, this is a
    strictly proper scoring rule in ``p = sigmoid(logit) in (0, 1)``: it
    is strictly convex in ``p`` and its unique minimizer is exactly
    ``p = g`` (module docstring, item 5, for the closed-form derivative
    argument; verified numerically, not merely asserted -- see
    the generated-fixture verification). Works identically,
    with no special-casing, when ``target`` happens to be exactly ``{0,
    1}``-valued (the existing hard-target path): BCE-with-logits is
    already this project's own early-training-stabilizing term inside
    ``combo_loss`` (item 4).
    """
    _validate_shapes(logits, target, valid_mask)
    bce_elementwise = F.binary_cross_entropy_with_logits(
        logits, target.to(logits.dtype), reduction="none"
    )
    if valid_mask is not None:
        mask = valid_mask.to(logits.dtype)
        return (bce_elementwise * mask).sum() / mask.sum().clamp_min(eps)
    return bce_elementwise.mean()


def soft_dice_semimetric_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """A soft-label-compatible Dice/Tversky-style term (module docstring,
    item 5): generalizes Wang, Popordanoska, Bertels, Lemmens & Blaschko,
    2023, *Dice Semimetric Losses: Optimizing the Dice Score with Soft
    Labels*, arXiv:2303.16296, Eq. 4's second Dice Semimetric Loss,

        DML_2(x, y) = 1 - 2*||x (.) y||_1 / (2*||x (.) y||_1 + ||x - y||_1)

    (``x`` = prediction, ``y`` = soft target, both in ``[0, 1]^p``, ``(.)``
    = elementwise product, ``||.||_1`` = L1 norm), to independent FN/FP
    -style weights via the EXACT algebraic identity
    ``|x_i - y_i| = relu(x_i - y_i) + relu(y_i - x_i)`` (both terms are
    ``>= 0`` and exactly one is nonzero per element, so this is a
    decomposition, not an approximation):

        per-sample: intersection = sum(p*g)                    (~TP)
                    over  = sum(relu(p - g))     (over-prediction,  ~FP)
                    under = sum(relu(g - p))     (under-prediction, ~FN)
        score = (2*intersection + eps) / (2*intersection + beta*over
                                           + alpha*under + eps)
        loss  = mean_batch(1 - score)

    matching this project's own ``alpha``-weights-FN / ``beta``-weights-FP
    direction convention (item 1's ``_tversky_index``) -- but NOT that
    function's numeric scale: ``_tversky_index``'s denominator has no
    separate factor of 2 on its TP-like term (``tp + alpha*fn +
    beta*fp``), so ITS ``alpha=beta=0.5`` recovers plain Dice, while THIS
    function's ``alpha=beta=1`` recovers plain Dice (and is the literal
    ``DML_2`` object above) -- the two ``alpha``/``beta`` pairs are not
    directly comparable across the two functions; this is flagged here
    explicitly rather than left for a reviewer to discover by surprise.

    Provably proper for ANY ``alpha, beta >= 0`` (a strictly stronger
    property than Wang et al.'s own Theorem 2.2, which is stated for
    ``alpha=beta=1``): ``score <= 1`` always, since
    ``denominator - numerator = beta*over + alpha*under >= 0``; equality
    (``score = 1``, ``loss = 0``, the GLOBAL minimum -- not merely a
    stationary point, since loss is bounded below by 0 everywhere) holds
    if and only if ``over = under = 0`` on the valid region, i.e. ``p = g``
    everywhere valid, whenever ``alpha, beta > 0``. At ``alpha = beta = 1``
    this is EXACTLY Wang et al.'s Theorem 2.2 (reflexivity/positivity, and
    identical to standard Dice loss when ``p``/``g`` are both binary);
    ``alpha != beta`` is this module's own principled but NOT literally
    paper-proven Tversky-style extension (the properness argument above
    still applies verbatim -- it never used ``alpha = beta`` -- only the
    "identical to SDL/Dice under hard labels" claim is specific to
    ``alpha = beta = 1``).

    Degenerate-input safety matches item 3's established convention
    exactly (``(numerator + eps) / (denominator + eps)``, never negative,
    never NaN, never a bare ``0/0``): an all-empty prediction-and-target
    pair gives ``score = eps/eps = 1`` (a perfect, if vacuous, score).
    """
    _validate_shapes(logits, target, valid_mask)
    score = torch.sigmoid(logits)
    b = logits.shape[0]
    flat_p = score.reshape(b, -1)
    flat_g = target.reshape(b, -1).to(dtype=flat_p.dtype)

    if valid_mask is not None:
        flat_m = valid_mask.reshape(b, -1).to(dtype=flat_p.dtype)
        flat_p = flat_p * flat_m
        flat_g = flat_g * flat_m

    intersection = (flat_p * flat_g).sum(dim=1)
    over = F.relu(flat_p - flat_g).sum(dim=1)
    under = F.relu(flat_g - flat_p).sum(dim=1)

    numerator = 2.0 * intersection
    denominator = numerator + beta * over + alpha * under
    score = (numerator + eps) / (denominator + eps)
    return (1.0 - score).mean()


def soft_combo_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    bce_weight: float = 1.0,
    dice_weight: float = 0.5,
    alpha: float = 1.0,
    beta: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """``bce_weight * soft_bce_loss(...) + dice_weight *
    soft_dice_semimetric_loss(...)`` (module docstring, item 5). Default
    ``alpha=beta=1`` keeps the Dice term the literal, Theorem-2.2-proven
    ``DML_2`` object (see :func:`soft_dice_semimetric_loss`'s own
    docstring), not this module's unproven Tversky-style extension.

    Provably proper (module docstring, item 5, for the full argument):
    since ``soft_bce_loss`` is minimized (per-pixel, hence in aggregate)
    at exactly ``p = g`` and ``soft_dice_semimetric_loss`` is minimized
    (globally, value ``0``, the lowest a manifestly-nonnegative loss can
    reach) at exactly ``p = g``, ANY positively-weighted sum of the two is
    also minimized at exactly ``p = g`` -- a direct consequence of both
    summands individually attaining their OWN minimum simultaneously at
    that point, not merely a shared stationary point. Setting
    ``dice_weight=0.0`` reduces this exactly to ``bce_weight *
    soft_bce_loss(...)``.
    """
    _validate_shapes(logits, target, valid_mask)
    bce = soft_bce_loss(logits, target, valid_mask, eps=eps)
    if dice_weight == 0.0:
        return bce_weight * bce
    dice = soft_dice_semimetric_loss(
        logits, target, valid_mask, alpha=alpha, beta=beta, eps=eps
    )
    return bce_weight * bce + dice_weight * dice


# ---------------------------------------------------------------------------
# Boundary-local auxiliary loss -- see module docstring, item 6. New,
# additive function; combo_loss/focal_tversky_loss above are UNCHANGED.
# The frozen auxiliary-loss contract is the sole authority for
# this construction.
# ---------------------------------------------------------------------------

_BOUNDARY_TARGET_MODES: frozenset[str] = frozenset({"hard", "fraction"})


@dataclass(frozen=True)
class BoundaryAuxiliaryLoss:
    """Return value of :func:`boundary_auxiliary_loss` -- see module
    docstring, item 6.

    ``loss`` is the scalar, batch-reduced, differentiable auxiliary term
    (``L_aux`` in the protocol's notation) -- the only field a caller adds into
    a total loss.

    ``boundary_count``/``boundary_fraction`` are the protocol's required
    logging quantities ("Record fractional-boundary support: boundary_count
    = sum W, boundary_fraction = sum W / sum M, so sparse support is
    visible") -- plain 0-dim tensors with no gradient (``W``/``M`` are
    comparison-derived masks, never differentiable), summed over the WHOLE
    batch (not per-sample). Callers read them via ``.item()`` for
    ``metrics.jsonl`` logging.
    """

    loss: torch.Tensor
    boundary_count: torch.Tensor
    boundary_fraction: torch.Tensor


def boundary_auxiliary_loss(
    logits: torch.Tensor,
    source_fraction: torch.Tensor,
    target_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    target_mode: str,
    eps: float = 1e-12,
) -> BoundaryAuxiliaryLoss:
    """Boundary-local auxiliary BCE term -- see module docstring, item 6,
    for the full mechanism and the boundary experiment Sec 5 for the authoritative contract
    this reproduces verbatim.

    Support ``W_i = valid_mask_i * 1[0 < source_fraction_i < 1]``. Auxiliary
    target ``G_i`` is ``target_mask`` when ``target_mode="hard"`` (arm B) or
    ``source_fraction`` when ``target_mode="fraction"`` (arm C) -- the ONLY
    difference between the two arms; support, weighting, normalization, and
    batch reduction are byte-identical between the two calls (plan Sec 5:
    "B and C differ only in auxiliary target values").

    Per-sample ``L_aux_i = sum_j(W_ij * BCEWithLogits(z_ij, G_ij)) /
    sum_j(W_ij)`` when ``sum_j(W_ij) > 0``, else an exact differentiable
    zero (see module docstring, item 6, for why this never divides ``0/0``
    -- the numerator is already exactly zero in that case). Batch reduction
    is ``sum_i(eligible_i * L_aux_i) / batch_size`` -- the SAME fixed
    ``batch_size`` denominator regardless of how many samples are eligible,
    so a batch with fewer boundary-bearing samples cannot silently inflate
    the effective per-sample auxiliary weight.

    Raises ``ValueError`` for a shape mismatch among ``logits``/
    ``source_fraction``/``target_mask``/``valid_mask``, or for a
    ``target_mode`` outside ``{"hard", "fraction"}`` -- callers must state
    which arm they mean; there is no silent default.
    """
    if target_mode not in _BOUNDARY_TARGET_MODES:
        raise ValueError(
            f"target_mode must be one of {sorted(_BOUNDARY_TARGET_MODES)}, "
            f"got {target_mode!r}"
        )
    _validate_shapes(logits, source_fraction, valid_mask)
    _validate_shapes(logits, target_mask, valid_mask)

    b = logits.shape[0]
    flat_z = logits.reshape(b, -1)
    flat_f = source_fraction.reshape(b, -1).to(dtype=flat_z.dtype)
    flat_h = target_mask.reshape(b, -1).to(dtype=flat_z.dtype)
    flat_m = valid_mask.reshape(b, -1).to(dtype=flat_z.dtype)

    # W_ij = M_ij * 1[0 < F_ij < 1] -- plan Sec 5's exact support
    # definition. The comparisons below are boolean/data-derived and carry
    # no gradient of their own -- only the BCE term at supported pixels
    # does (module docstring, item 6, "no gradient flows through the
    # SUPPORT decision").
    support = (flat_f > 0.0) & (flat_f < 1.0)
    w = flat_m * support.to(dtype=flat_z.dtype)

    g = flat_h if target_mode == "hard" else flat_f

    bce_elementwise = F.binary_cross_entropy_with_logits(flat_z, g, reduction="none")

    numerator = (w * bce_elementwise).sum(dim=1)  # [B]; already 0 where w==0
    denominator = w.sum(dim=1)  # [B]
    eligible = (denominator > 0.0).to(dtype=flat_z.dtype)  # [B]

    # denominator==0 rows have numerator==0 already (every w_ij==0 there),
    # so dividing by ANY strictly-positive safe denominator yields an exact,
    # differentiable 0 -- never a bare 0/0 (module docstring, item 6). The
    # explicit `eligible *` below is the plan's own literal
    # "sum_i eligible_i * L_aux_i" wording, applied as a belt-and-braces
    # re-assertion, not the sole safety mechanism.
    safe_denominator = denominator.clamp_min(1.0)
    per_sample_loss = numerator / safe_denominator  # [B]

    batch_size = flat_z.shape[0]
    loss = (eligible * per_sample_loss).sum() / batch_size

    boundary_count = w.sum()
    valid_count = flat_m.sum()
    boundary_fraction = boundary_count / valid_count.clamp_min(eps)

    return BoundaryAuxiliaryLoss(
        loss=loss,
        boundary_count=boundary_count.detach(),
        boundary_fraction=boundary_fraction.detach(),
    )
