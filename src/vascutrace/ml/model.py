"""Compact shared-weight Siamese PET/CT 2.5D U-Net (VascuTrace Phase 6).

RESEARCH_PROTOTYPE_WARNING
---------------------------------------------------------------------------
Research prototype. Trained and evaluated using simulated vascular-like
abnormalities, not confirmed human post-angioplasty lesions.
---------------------------------------------------------------------------

Implementation notes
============================================================================
This module implements the single, fixed model design frozen for P6 -- not
an architecture search. It consumes the frozen tensor contract in
``src.vascutrace.ml.tensor_schema`` (``left_view``/``right_view``:
``[B, 2K, H, W]`` = PET K-slices then CT K-slices at ``(H, W) = (144, 80)``;
optional ``pet_diff``: ``[B, K, H, W]``) and returns raw segmentation
**logits** ``[B, 1, H, W]`` for the frozen crop's center-slice target mask.
No sigmoid is applied inside ``forward`` -- see :func:`abnormality_score`.

1. Why a *shared-weight Siamese* encoder, not two independent encoders
   ----------------------------------------------------------------------
   ``tensor_schema.py`` frames the detection problem precisely: "A
   unilateral synthetic lesion breaks left/right symmetry, so it is visible
   in the paired views and their difference." A single encoder instance
   (:class:`SharedEncoder`) is called *twice* -- once on ``left_view``, once
   on the physically-reflected ``right_view`` -- with the *same* parameter
   tensors both times (:meth:`SiameseBilateralUNet.encode` is the one
   code path both calls go through; there is no second, independently
   -initialized encoder anywhere in this module). This is the classical
   twin-network design: "two branches in the network share exactly the
   same architecture and the same set of weights" (Zagoruyko & Komodakis,
   2015, *Learning to Compare Image Patches via Convolutional Neural
   Networks*, arXiv:1504.03641, Sec. 3.1 "Basic models"). Weight sharing is
   the right prior here for two reasons: (a) "left" and "right" are not
   semantically distinct classes of anatomy -- they are the *same* anatomy
   viewed through a physical mirror, so whatever filters are useful for
   detecting vascular structure on one side are, by construction, equally
   useful on the other; forcing shared weights removes an entire
   redundant copy of the parameter space and halves the risk of
   overfitting to spurious left/right differences. (b) it guarantees, by
   construction rather than by training discipline, that any difference
   the *decoder* sees between the two branches' features is attributable
   to the input asymmetry alone, not to different encoder inductive biases
   accidentally offering an alternative, un-anatomical bilateral cue.

2. Feature-combination scheme -- concat-and-abs-diff, then fuse
   ----------------------------------------------------------------------
   At every encoder scale (the four skip levels plus the bottleneck),
   :class:`BilateralFuse` combines the left and right feature maps as
   ``concat([left, right, |left - right|])`` followed by a
   1x1 convolution + GroupNorm + GELU back down to the single-branch
   channel width. The explicit ``|left - right|`` channel hands the
   decoder the asymmetry signal directly -- exactly the quantity the
   target is defined by -- instead of forcing the decoder to learn
   subtraction from raw concatenated features at every scale. Keeping the
   raw ``left``/``right`` channels alongside the difference (rather than
   feeding the decoder the difference alone) preserves absolute-intensity
   context: a purely differential representation can degrade information
   about *which side* the raw signal sits on, or about background
   intensity useful for e.g. valid-FOV reasoning. This concat-then-project
   pattern mirrors the paired-branch feature combination in Zagoruyko &
   Komodakis 2015 Sec. 3.1. The 1x1 fusion projects the tripled channel
   count (``3 * C``) back to the single-branch width ``C`` at every level,
   so the decoder's channel budget is identical to a plain (non-Siamese)
   U-Net of the same base width -- the Siamese comparison is "free" in
   decoder parameter count.

3. Encoder/decoder backbone -- U-Net with resize-convolution upsampling
   ----------------------------------------------------------------------
   :class:`SharedEncoder` is a 4-level (``144x80 -> 72x40 -> 36x20 -> 18x10
   -> 9x5``) contracting path of stride-1 3x3 convolutions + 2x2 max
   pooling; the decoder is the symmetric expansive path that upsamples and
   concatenates the corresponding fused skip level at every step -- the
   canonical U-Net design (Ronneberger, Fischer & Brox, 2015, *U-Net:
   Convolutional Networks for Biomedical Image Segmentation*,
   arXiv:1505.04597, Sec. 2 "Network Architecture": "Every step in the
   expansive path consists of an upsampling of the feature map ... and a
   concatenation with the correspondingly cropped feature map from the
   contracting path"). Unlike the original U-Net, :class:`UpBlock` upsamples
   via nearest-neighbor interpolation followed by a 1x1 convolution rather
   than a learned transposed convolution: transposed convolutions have
   uneven kernel overlap that produces checkerboard artifacts, and
   resize-then-convolve is the documented fix ("We've had our best results
   with nearest-neighbor interpolation" -- Odena, Dumoulin & Olah, 2016,
   *Deconvolution and Checkerboard Artifacts*, Distill, Sec. "Better
   Upsampling"). Both dims of ``IN_PLANE_HW`` are divisible by 16
   (``tensor_schema.py``), so all four poolings and their matching
   upsamplings are exact integer halvings/doublings -- no padding/cropping
   mismatch of the kind the original U-Net's "cropped feature map" language
   has to handle.

4. Normalization -- GroupNorm, not BatchNorm
   ----------------------------------------------------------------------
   Every conv block normalizes with GroupNorm (Wu & He, 2018, *Group
   Normalization*, arXiv:1803.08494, Sec. 3.1 "Formulation", "Group Norm",
   Eq. (7): channels are partitioned into ``G`` groups and normalized by
   the mean/variance computed within each group, independent of the batch
   axis), not BatchNorm. GroupNorm's whole motivation is that BatchNorm's
   estimate of the normalization statistics degrades sharply at small
   batch sizes (Sec. 4.1 "Small batch sizes": at batch size 2, a ResNet-50
   trained with BN reaches 34.7% error vs. GroupNorm's ~24% -- a 10.6-point
   gap) while GroupNorm's error is stable across the whole batch-size
   range tested. A 2.5D per-slice sample here is large relative to typical
   GPU memory (``2 x 10 x 144 x 80`` floats per pair plus activations
   through a U-Net), so small training batch sizes are the expected
   regime -- GroupNorm is therefore the safer default independent of
   whatever batch size the (separately owned) training loop ultimately
   uses.

5. Activation -- GELU
   ----------------------------------------------------------------------
   GELU is used throughout (Hendrycks & Gimpel, 2016, *Gaussian Error
   Linear Units (GELUs)*, arXiv:1606.08415) as the "modern activation"
   called for by this design: smooth, unlike ReLU's hard kink, and suitable
   for gradient-based optimization.

6. Optional ``pet_diff`` fusion
   ----------------------------------------------------------------------
   ``pet_diff`` (``[B, K, H, W]``, "network-normalized left PET minus
   right PET" per ``tensor_schema.py``) is *already* a difference signal at
   full input resolution, so it is not pushed through the shared encoder's
   downsampling path at all. :class:`DiffStem` encodes it once at full
   resolution and :class:`SiameseBilateralUNet.forward` concatenates that
   feature map into the decoder's *final* (full-resolution) stage only,
   just before the 1x1 logit head -- giving the decoder a second,
   independently-derived asymmetry cue at exactly the resolution the
   target mask is defined at. When ``pet_diff`` is not supplied,
   ``forward`` substitutes an all-zero tensor of the expected shape so the
   diff-stem parameters remain part of the computation graph (and
   therefore always receive a defined, finite -- if zero -- gradient)
   regardless of whether the caller passes ``pet_diff``.

7. Loss -- Dice + BCE-with-logits
   ----------------------------------------------------------------------
   :func:`dice_bce_loss` combines a soft Dice term (Milletari, Navab &
   Ahmadi, 2016, *V-Net: Fully Convolutional Neural Networks for
   Volumetric Medical Image Segmentation*, arXiv:1606.04797, Sec. 3 "Dice
   loss layer": ``D = 2 * sum_i(p_i * g_i) / (sum_i(p_i^2) +
   sum_i(g_i^2))``, minimized here as ``1 - D`` with a small ``eps`` added
   to numerator/denominator for numerical stability on empty masks -- an
   addition not present in the original equation) with
   ``binary_cross_entropy_with_logits`` -- the canonical Bernoulli-output
   cross-entropy for a sigmoid target (Goodfellow, Bengio & Courville,
   2016, *Deep Learning*, Ch. 6 "Output Units": pair a sigmoid output with
   cross-entropy, not MSE, so gradients do not saturate). Combining a
   region-overlap term (Dice) with a per-pixel term (BCE) as a compound
   segmentation loss is standard practice in the biomedical-segmentation
   literature (e.g. nnU-Net, Isensee et al., 2021, *Nature Methods*).

8a. Deep supervision -- train-only auxiliary heads at decoder scales x2/x4
   ----------------------------------------------------------------------
   The frozen B2 design includes two corrections from an empirical
   downsample-sparsity check on 2026-07-19. This implementation uses scales
   ``{2, 4}`` only (the spec's third, x8, head is DROPPED -- even under
   max-pool it leaves only ~2px targets for the median 18px lesion,
   marginal), and downsampling of the hard ``target_mask`` uses MAX-POOL
   (``F.max_pool2d`` -- "any positive pixel in the block -> 1"), not
   strided/nearest sampling, because nearest-style downsampling vanishes
   24% of lesions at x4 and 79% at x8 for this dataset's median 18px
   lesion, making the aux targets mostly empty; max-pool preserves all 78
   val positives at every scale.

   When ``ModelConfig.deep_supervision=True``, two additional bare 1x1
   conv heads are constructed -- ``aux_head2`` on the ``up2`` decoder
   output (``[B, C1, 72, 40]``, x2 downsample vs the ``[B, 1, 144, 80]``
   full-res target) and ``aux_head3`` on the ``up3`` decoder output
   (``[B, C2, 36, 20]``, x4 downsample) -- mirroring the shipped ``head``'s
   own bare-``Conv2d(C, out_channels, 1)`` design (spec Sec 2: "raw logits
   at its native scale ... no upsampling of aux logits to full res", so
   ``combo_loss`` at each scale supervises against a same-scale
   nearest-downsampled -- here, max/min-pooled -- target). This is the
   ONLY architectural change deep supervision makes: no new loss family
   (the shipped, unmodified ``losses.combo_loss`` is reused by the
   training step at every scale -- see ``train.py`` item 21), no edit to
   ``up1``/ ``up2``/``up3``/``up4``/``head`` themselves.

   When ``deep_supervision=False`` (the default), ``aux_head2``/
   ``aux_head3`` are never constructed -- the module has exactly the same
   parameter set, ``state_dict`` keys, and ``model_signature()`` as
   pre-B2 code, so every existing v6/B0 checkpoint stays loadable
   unmodified (spec Sec 3, "Model contract"). ``forward``'s new
   ``return_aux: bool = False`` parameter defaults to returning the bare
   full-res ``logits`` tensor exactly as before for every existing caller
   (``evaluate.py``, ``train.py``'s own validation loop); only the
   training step, and only when ``TrainConfig.deep_supervision`` is
   True, passes ``return_aux=True`` to additionally receive
   ``aux_logits_by_scale: dict[int, Tensor]`` keyed by the integer
   downsample factor (``{2: ..., 4: ...}``) -- aux heads are structurally
   TRAIN-ONLY: no code path reachable from evaluation ever requests them.

9. Foreground-imbalance bias initialization
   ----------------------------------------------------------------------
   The final 1x1 logit-head bias is initialized to ``-log((1 - pi) / pi)``
   for a small assumed foreground prior ``pi`` (Lin, Goyal, Girshick, He &
   Dollar, 2017, *Focal Loss for Dense Object Detection*, arXiv:1708.02002,
   Sec. 4.1 "Inference and Training", "Initialization": "the loss due to
   the frequent class can dominate total loss and cause instability in
   early training" under a naive zero/uniform bias init). A vascular
   lesion target is expected to occupy a small fraction of the 144x80
   slice, so this initialization starts the network's prior close to the
   expected sparsity rather than at 50/50, which speeds early-training
   convergence -- directly relevant to this module's easy-overfit
   acceptance gate (Dice >= 0.95 within 1000 steps).
============================================================================
"""

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.vascutrace.geometry import RESEARCH_PROTOTYPE_WARNING
from src.vascutrace.ml.tensor_schema import (
    CHANNELS_PER_VIEW,
    IN_PLANE_HW,
    K,
    TENSOR_SCHEMA_VERSION,
)

__all__ = [
    "RESEARCH_PROTOTYPE_WARNING",
    "MODEL_ARCH_VERSION",
    "ModelConfig",
    "SharedEncoder",
    "BilateralFuse",
    "UpBlock",
    "DiffStem",
    "SiameseBilateralUNet",
    "build_model",
    "model_signature",
    "abnormality_score",
    "dice_bce_loss",
    "dice_score",
]

MODEL_ARCH_VERSION = "p6-siamese-unet-v1"

# Foreground-imbalance bias init prior (see module docstring, item 9).
_HEAD_BIAS_FOREGROUND_PRIOR = 0.02


@dataclass(frozen=True)
class ModelConfig:
    """Frozen architecture configuration for :class:`SiameseBilateralUNet`.

    ``channel_mult`` has one entry per U-Net level: index 0 is the
    full-resolution stem, indices 1-3 are the three intermediate
    downsampled skip levels, and index 4 is the bottleneck -- five levels
    total, matching the four 2x2 max-poolings the frozen ``IN_PLANE_HW =
    (144, 80)`` supports cleanly (both dims divisible by 16; see
    ``tensor_schema.py``). ``seed``, if set, is consumed by
    :func:`build_model` (via ``torch.manual_seed``) immediately before
    constructing the module, so weight initialization is reproducible;
    it is deliberately excluded from :func:`model_signature` (a training
    seed does not change the architecture or the checkpoint's parameter
    shapes).
    """

    base_channels: int = 16
    channel_mult: tuple[int, int, int, int, int] = (1, 2, 4, 8, 16)
    diff_stem_channels: int = 16
    groupnorm_groups: int = 8
    in_channels_per_view: int = CHANNELS_PER_VIEW
    pet_diff_channels: int = K
    out_channels: int = 1
    dropout_p: float = 0.0
    seed: int | None = None

    # Deep supervision (Phase 4 lever B2/L4) -- see module docstring, item
    # 8a. False (default) constructs the exact pre-B2 module (no aux-head
    # params, no state_dict change, model_signature() unchanged) -- this
    # is the byte-identical-to-v6 invariant the specification requires.
    deep_supervision: bool = False

    def __post_init__(self) -> None:
        if len(self.channel_mult) != 5:
            raise ValueError(
                "channel_mult must have exactly 5 entries (stem + 4 "
                f"downsampled levels); got {len(self.channel_mult)}"
            )
        if self.base_channels <= 0:
            raise ValueError(
                f"base_channels must be positive, got {self.base_channels}"
            )
        if not (0.0 <= self.dropout_p < 1.0):
            raise ValueError(f"dropout_p must be in [0, 1), got {self.dropout_p}")

    def level_channels(self) -> tuple[int, int, int, int, int]:
        """Per-level channel widths ``(C0, C1, C2, C3, C4)``."""
        c0, c1, c2, c3, c4 = (self.base_channels * m for m in self.channel_mult)
        return (c0, c1, c2, c3, c4)


def model_signature(config: ModelConfig | None = None) -> str:
    """Checkpoint-provenance string: architecture version + tensor-schema
    version + a short hash of the architecture-affecting config fields
    (``seed`` excluded -- see :class:`ModelConfig` docstring). Two
    checkpoints with the same signature are guaranteed state-dict
    -compatible with this code; a mismatched signature means either the
    tensor contract or the architecture changed since the checkpoint was
    written.
    """
    cfg = config or ModelConfig()
    arch_fields: tuple[Any, ...] = (
        cfg.base_channels,
        cfg.channel_mult,
        cfg.diff_stem_channels,
        cfg.groupnorm_groups,
        cfg.in_channels_per_view,
        cfg.pet_diff_channels,
        cfg.out_channels,
        cfg.dropout_p,
    )
    # Deep supervision (module docstring, item 8a) folds into the arch hash
    # ONLY when enabled -- appending nothing when False keeps arch_fields
    # byte-identical to the pre-B2 8-tuple above, so model_signature()
    # returns the EXACT pre-B2 v6/B0 string for every default/off config
    # (checkpoint compatibility preserved); appending the flag when True
    # gives the genuinely-different aux-head parameter set its own,
    # distinct signature.
    if cfg.deep_supervision:
        arch_fields = arch_fields + (cfg.deep_supervision,)
    digest = hashlib.sha256(repr(arch_fields).encode("utf-8")).hexdigest()[:12]
    return f"{MODEL_ARCH_VERSION}+{TENSOR_SCHEMA_VERSION}+cfg-{digest}"


def _group_norm(num_channels: int, groups: int) -> nn.GroupNorm:
    """GroupNorm with ``groups`` reduced to the largest divisor of
    ``num_channels`` that is ``<= groups`` -- a defensive fallback for
    ``ModelConfig`` values where ``base_channels`` is not a multiple of
    ``groupnorm_groups`` (the default config's channel widths are all
    multiples of 8, so this fallback is inert for defaults).
    """
    g = max(1, min(groups, num_channels))
    while num_channels % g != 0:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=num_channels)


class ConvGNAct(nn.Module):
    """3x3 conv (no bias -- redundant ahead of GroupNorm's own learned
    affine shift) + GroupNorm + GELU (+ optional spatial dropout). See
    module docstring items 4-5.
    """

    def __init__(
        self, in_ch: int, out_ch: int, groups: int, dropout_p: float = 0.0
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm = _group_norm(out_ch, groups)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


class DoubleConv(nn.Module):
    """Two stacked :class:`ConvGNAct` blocks -- the standard U-Net
    per-level unit (Ronneberger et al. 2015, Sec. 2: "two 3x3
    convolutions").
    """

    def __init__(
        self, in_ch: int, out_ch: int, groups: int, dropout_p: float = 0.0
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(in_ch, out_ch, groups, dropout_p),
            ConvGNAct(out_ch, out_ch, groups, dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SharedEncoder(nn.Module):
    """The single U-Net contracting-path instance. ``forward`` is called
    twice per model forward pass -- once for ``left_view``, once for
    ``right_view`` -- through the *same* parameter tensors (see module
    docstring item 1). Returns the 5-level feature pyramid
    ``[f0 (full res), f1, f2, f3, f4 (bottleneck)]``.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        c0, c1, c2, c3, c4 = config.level_channels()
        g, d = config.groupnorm_groups, config.dropout_p
        self.stem = DoubleConv(config.in_channels_per_view, c0, g, d)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c0, c1, g, d))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c1, c2, g, d))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c2, c3, g, d))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c3, c4, g, d))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f0 = self.stem(x)
        f1 = self.down1(f0)
        f2 = self.down2(f1)
        f3 = self.down3(f2)
        f4 = self.down4(f3)
        return [f0, f1, f2, f3, f4]


class BilateralFuse(nn.Module):
    """Combine one level's ``(left, right)`` feature maps into a single
    ``C``-channel tensor: ``concat([left, right, |left - right|])`` -> 1x1
    conv -> GroupNorm -> GELU. See module docstring item 2.
    """

    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.fuse = ConvGNAct(3 * channels, channels, groups)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        diff = (left - right).abs()
        return self.fuse(torch.cat([left, right, diff], dim=1))


class UpBlock(nn.Module):
    """One decoder stage: nearest-neighbor upsample to the skip's exact
    spatial size + 1x1 channel-reduction conv, concatenate with the fused
    skip connection, then a :class:`DoubleConv`. See module docstring
    item 3.
    """

    def __init__(
        self, in_ch: int, skip_ch: int, out_ch: int, groups: int, dropout_p: float = 0.0
    ) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, skip_ch, kernel_size=1)
        self.block = DoubleConv(2 * skip_ch, out_ch, groups, dropout_p)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class DiffStem(nn.Module):
    """Full-resolution encoder for the optional ``pet_diff`` auxiliary
    input. See module docstring item 6.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.stem = DoubleConv(
            config.pet_diff_channels,
            config.diff_stem_channels,
            config.groupnorm_groups,
            config.dropout_p,
        )

    def forward(self, pet_diff: torch.Tensor) -> torch.Tensor:
        return self.stem(pet_diff)


class SiameseBilateralUNet(nn.Module):
    """Compact shared-weight Siamese PET/CT 2.5D U-Net. See the module
    docstring for the full design rationale and paper citations.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        c0, c1, c2, c3, c4 = cfg.level_channels()
        g = cfg.groupnorm_groups

        self.encoder = SharedEncoder(cfg)  # single instance -- see class docstring

        self.fuse0 = BilateralFuse(c0, g)
        self.fuse1 = BilateralFuse(c1, g)
        self.fuse2 = BilateralFuse(c2, g)
        self.fuse3 = BilateralFuse(c3, g)
        self.fuse4 = BilateralFuse(c4, g)  # bottleneck

        self.up4 = UpBlock(c4, c3, c3, g, cfg.dropout_p)
        self.up3 = UpBlock(c3, c2, c2, g, cfg.dropout_p)
        self.up2 = UpBlock(c2, c1, c1, g, cfg.dropout_p)
        self.up1 = UpBlock(c1, c0, c0, g, cfg.dropout_p)

        self.diff_stem = DiffStem(cfg)
        self.diff_fuse = nn.Conv2d(c0 + cfg.diff_stem_channels, c0, kernel_size=1)
        self.head = nn.Conv2d(c0, cfg.out_channels, kernel_size=1)

        # Deep supervision (module docstring, item 8a) -- aux heads
        # constructed ONLY when cfg.deep_supervision=True, so the default
        # (False) model has EXACTLY the pre-B2 parameter set / state_dict
        # keys. aux_head2 reads the up2 output (C1 channels, x2 downsample
        # vs full res); aux_head3 reads the up3 output (C2 channels, x4
        # downsample). Both are bare Conv2d(C, out_channels, 1) -- raw
        # logits at native scale, mirroring self.head's own design; no
        # GroupNorm/GELU/upsample inside either head (spec Sec 2).
        self.aux_head2: nn.Module | None = None
        self.aux_head3: nn.Module | None = None
        if cfg.deep_supervision:
            self.aux_head2 = nn.Conv2d(c1, cfg.out_channels, kernel_size=1)
            self.aux_head3 = nn.Conv2d(c2, cfg.out_channels, kernel_size=1)

        self._init_head_bias(_HEAD_BIAS_FOREGROUND_PRIOR)

    def _init_head_bias(self, prior: float) -> None:
        """See module docstring item 9 (Lin et al. 2017, Sec. 4.1)."""
        bias_value = -math.log((1.0 - prior) / prior)
        with torch.no_grad():
            self.head.bias.fill_(bias_value)

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run the single shared :class:`SharedEncoder` instance on one
        Siamese branch's input. Both branches call this exact method with
        the exact same ``self.encoder`` parameters (see class docstring).
        """
        return self.encoder(x)

    def _validate_inputs(
        self,
        left_view: torch.Tensor,
        right_view: torch.Tensor,
        pet_diff: torch.Tensor | None,
    ) -> None:
        expected_view = (self.config.in_channels_per_view, *IN_PLANE_HW)
        for name, view in (("left_view", left_view), ("right_view", right_view)):
            if view.dim() != 4 or tuple(view.shape[1:]) != expected_view:
                raise ValueError(
                    f"{name} must have shape [B, {expected_view[0]}, "
                    f"{expected_view[1]}, {expected_view[2]}] "
                    f"(tensor_schema CHANNELS_PER_VIEW/IN_PLANE_HW); got {tuple(view.shape)}"
                )
        if left_view.shape[0] != right_view.shape[0]:
            raise ValueError(
                f"left_view batch ({left_view.shape[0]}) != right_view batch ({right_view.shape[0]})"
            )
        if pet_diff is not None:
            expected_diff = (self.config.pet_diff_channels, *IN_PLANE_HW)
            if pet_diff.dim() != 4 or tuple(pet_diff.shape[1:]) != expected_diff:
                raise ValueError(
                    f"pet_diff must have shape [B, {expected_diff[0]}, "
                    f"{expected_diff[1]}, {expected_diff[2]}] "
                    f"(tensor_schema K/IN_PLANE_HW); got {tuple(pet_diff.shape)}"
                )
            if pet_diff.shape[0] != left_view.shape[0]:
                raise ValueError(
                    f"pet_diff batch ({pet_diff.shape[0]}) != left_view batch ({left_view.shape[0]})"
                )

    def forward(
        self,
        left_view: torch.Tensor,
        right_view: torch.Tensor,
        pet_diff: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Return raw segmentation **logits** ``[B, 1, H, W]`` for the
        center-slice target mask, in the left (crop) frame. No sigmoid is
        applied -- see :func:`abnormality_score`.

        ``return_aux`` (default ``False``) is the deep-supervision escape
        hatch (module docstring, item 8a): every existing/default caller
        (this signature's own default, ``evaluate.py``, ``train.py``'s
        validation loop) gets EXACTLY the pre-B2 return value -- a bare
        ``Tensor`` -- unchanged. Only the training step, and only when
        ``self.config.deep_supervision`` is True, should pass
        ``return_aux=True``; doing so then additionally returns
        ``aux_logits_by_scale: dict[int, Tensor]`` keyed by integer
        downsample factor (``{2: [B,1,72,40], 4: [B,1,36,20]}``) -- raw
        logits at each aux head's own native scale, no upsampling.
        Requesting ``return_aux=True`` on a model built with
        ``deep_supervision=False`` is a caller error (no aux heads exist
        to call), raised explicitly rather than silently ignored.
        """
        self._validate_inputs(left_view, right_view, pet_diff)
        if return_aux and not self.config.deep_supervision:
            raise ValueError(
                "return_aux=True requires ModelConfig.deep_supervision=True "
                "-- this model was built without aux heads"
            )

        feats_left = self.encode(left_view)
        feats_right = self.encode(right_view)

        fused = [
            self.fuse0(feats_left[0], feats_right[0]),
            self.fuse1(feats_left[1], feats_right[1]),
            self.fuse2(feats_left[2], feats_right[2]),
            self.fuse3(feats_left[3], feats_right[3]),
            self.fuse4(feats_left[4], feats_right[4]),
        ]

        x4 = self.up4(fused[4], fused[3])
        x3 = self.up3(x4, fused[2])  # x4 downsample vs full res, C2 channels
        x2 = self.up2(x3, fused[1])  # x2 downsample vs full res, C1 channels
        x = self.up1(x2, fused[0])

        if pet_diff is None:
            b = left_view.shape[0]
            pet_diff = left_view.new_zeros(
                b, self.config.pet_diff_channels, *IN_PLANE_HW
            )
        diff_feat = self.diff_stem(pet_diff)
        x = self.diff_fuse(torch.cat([x, diff_feat], dim=1))
        logits = self.head(x)

        if not return_aux:
            return logits

        assert self.aux_head2 is not None and self.aux_head3 is not None
        aux_logits_by_scale = {
            2: self.aux_head2(x2),
            4: self.aux_head3(x3),
        }
        return logits, aux_logits_by_scale


def build_model(config: ModelConfig | None = None) -> SiameseBilateralUNet:
    """Factory: construct a :class:`SiameseBilateralUNet`. If
    ``config.seed`` is set, seeds ``torch.manual_seed`` immediately before
    constructing the module so weight initialization is reproducible
    (see the determinism test in ``tests/test_ml_model.py``).
    """
    cfg = config or ModelConfig()
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
    return SiameseBilateralUNet(cfg)


def abnormality_score(logits: torch.Tensor) -> torch.Tensor:
    """Monotonic ``[0, 1]`` transform of the raw segmentation logit
    (``torch.sigmoid``) -- **not** a calibrated probability. Per the
    project's scientific-boundary contract (``src.vascutrace.geometry.
    RESEARCH_PROTOTYPE_WARNING``): use ``abnormality_score``,
    never probability/confidence, unless a separate calibration experiment
    passes), this is deliberately not named ``predict_proba``.
    """
    return torch.sigmoid(logits)


def dice_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    dice_weight: float = 0.5,
    bce_weight: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compound soft-Dice + BCE-with-logits loss. See module docstring
    item 7 for the cited formulation. ``valid_mask`` (if given, any shape
    broadcastable to/matching ``target``) excludes invalid voxels from
    *both* terms via multiplicative masking before reduction -- masked-out
    positions contribute exactly zero to the Dice intersection/denominator
    and are excluded from the BCE mean's numerator and denominator.
    """
    if logits.shape != target.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} != target shape {tuple(target.shape)}"
        )
    if valid_mask is not None and valid_mask.shape != target.shape:
        raise ValueError(
            f"valid_mask shape {tuple(valid_mask.shape)} != target shape {tuple(target.shape)}"
        )

    score = torch.sigmoid(logits)
    b = logits.shape[0]
    flat_p = score.reshape(b, -1)
    flat_g = target.reshape(b, -1).to(dtype=flat_p.dtype)

    if valid_mask is not None:
        flat_m = valid_mask.reshape(b, -1).to(dtype=flat_p.dtype)
        flat_p = flat_p * flat_m
        flat_g = flat_g * flat_m

    intersection = (flat_p * flat_g).sum(dim=1)
    denom = flat_p.pow(2).sum(dim=1) + flat_g.pow(2).sum(dim=1)
    dice = (2.0 * intersection + eps) / (denom + eps)
    dice_loss = (1.0 - dice).mean()

    bce_elementwise = F.binary_cross_entropy_with_logits(
        logits, target.to(logits.dtype), reduction="none"
    )
    if valid_mask is not None:
        mask = valid_mask.to(logits.dtype)
        bce_loss = (bce_elementwise * mask).sum() / mask.sum().clamp_min(eps)
    else:
        bce_loss = bce_elementwise.mean()

    return dice_weight * dice_loss + bce_weight * bce_loss


def dice_score(
    logits_or_prob: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    threshold: float = 0.5,
    from_logits: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Hard Dice metric at a fixed abnormality-score threshold (default 0.5),
    per sample, averaged over the batch. ``from_logits`` (default
    ``True``, matching this module's ``forward()`` contract of returning
    raw logits) applies :func:`abnormality_score` before thresholding --
    pass ``from_logits=False`` if ``logits_or_prob`` is already a ``[0,
    1]`` score. (Thresholding a raw logit directly at 0.5 would be wrong:
    ``sigmoid(0) == 0.5``, not ``sigmoid(0.5)``.) A sample with no
    positive pixels in *both* prediction and target scores 1.0 (perfect
    empty-vs-empty match) rather than the ``0/0`` NaN a naive
    implementation would produce.
    """
    if logits_or_prob.shape != target.shape:
        raise ValueError(
            f"prediction shape {tuple(logits_or_prob.shape)} != target shape {tuple(target.shape)}"
        )
    score = abnormality_score(logits_or_prob) if from_logits else logits_or_prob
    pred = (score >= threshold).to(dtype=score.dtype)

    b = pred.shape[0]
    flat_p = pred.reshape(b, -1)
    flat_g = target.reshape(b, -1).to(dtype=flat_p.dtype)
    if valid_mask is not None:
        flat_m = valid_mask.reshape(b, -1).to(dtype=flat_p.dtype)
        flat_p = flat_p * flat_m
        flat_g = flat_g * flat_m

    intersection = (flat_p * flat_g).sum(dim=1)
    denom = flat_p.sum(dim=1) + flat_g.sum(dim=1)
    dice = torch.where(
        denom > 0, (2.0 * intersection) / denom.clamp_min(eps), torch.ones_like(denom)
    )
    return dice.mean()
