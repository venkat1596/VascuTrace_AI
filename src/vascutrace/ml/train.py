"""P6 training loop: seeded, mixed-precision, atomically-checkpointed,
resumable training for the Siamese PET/CT U-Net.

RESEARCH_PROTOTYPE_WARNING
---------------------------------------------------------------------------
Research prototype. Trained and evaluated using simulated vascular-like
abnormalities, not confirmed human post-angioplasty lesions.
---------------------------------------------------------------------------

Implementation notes
============================================================================
This module is the one place ``model.py``'s architecture and
``dataset.py``'s samples actually get trained. It owns exactly four
responsibilities: validate a run's configuration before touching any data
or GPU, run one AdamW + optional-AMP optimization loop with validation-only
early stopping on a positive-focused selection metric, checkpoint atomically
every epoch via ``checkpoint.py``, and resume that loop bit-for-bit from a
checkpoint. It introduces no new model code and no new metric-computation
code of its own -- ``dice_bce_loss``/``dice_score``/``abnormality_score``/
``build_model`` are consumed from ``model.py`` unmodified,
``focal_tversky_loss``/``combo_loss`` from ``losses.py`` unmodified, every
segmentation/detection statistic from ``metrics.py`` unmodified, and
training/validation examples come from ``dataset.py``'s
``SiameseCropDataset``/``frozen_validation_set``/``cache.py``'s
``CachedSampleDataset`` unmodified. This module's own job is orchestration
-- selecting among, and wiring together, those already-implemented pieces.

1. Fail before allocating (:class:`TrainConfig`)
   ------------------------------------------------------------------------
   ``TrainConfig`` is a frozen dataclass; all structural validation (batch
   size, epoch count, learning rate, split-overlap leakage, etc.) runs in
   ``__post_init__``, which Python guarantees executes before construction
   completes and *before* any caller code can use the object. A bad config
   therefore raises :class:`TrainConfigError` at the ``TrainConfig(...)``
   call site itself -- no model, dataset, or CUDA context has been touched
   yet, satisfying the implementation's "validate the config BEFORE allocating
   data/GPU" requirement structurally rather than by convention. The one
   check that is *not* a structural config problem -- "was ``cuda``
   requested on a machine without CUDA?" -- is deliberately not done here:
   it is an environment fact, not a config-shape fact, and is checked at
   the top of :func:`_execute` instead (see item 2).

2. No silent CPU fallback
   ------------------------------------------------------------------------
   If ``config.device == "cuda"`` and ``torch.cuda.is_available()`` is
   ``False``, :func:`_execute` raises :class:`CudaUnavailableError`
   immediately -- before building the model or dataset. Silently training
   on CPU when the caller explicitly asked for ``cuda`` would produce a
   checkpoint whose provenance (and wall-clock/AMP assumptions) silently
   disagree with what the caller believes happened; this project's own
   scientific-boundary discipline (never silently substitute one thing for
   another -- see ``geometry.py``'s fail-closed error taxonomy) applies
   here just as much as to a mismatched affine.

3. Mixed precision -- ``torch.amp.autocast`` + ``torch.amp.GradScaler``
   ------------------------------------------------------------------------
   When ``config.amp`` is ``True`` *and* ``config.device == "cuda"``
   (AMP is a CUDA-only optimization here; it is a documented no-op on CPU
   in this module, never silently "half-applied"), the forward pass and
   loss computation run inside ``torch.amp.autocast(device_type="cuda")``,
   and ``GradScaler`` scales the loss before ``backward()`` to keep small
   float16 gradients from underflowing to zero (Micikevicius, Narang,
   Alben, Diamos, Elsen, Garcia, Ginsburg, Houston, Kuchaiev, Venkatesh &
   Wu, 2018, *Mixed Precision Training*, ICLR, arXiv:1710.03740). The exact
   ``scale(loss).backward() -> [unscale_ -> clip] -> step(optimizer) ->
   update()`` sequence, and the specific requirement to call
   ``scaler.unscale_(optimizer)`` *before* gradient clipping (clipping an
   still-scaled gradient's norm against an unscaled threshold is invalid),
   follows PyTorch's own documented AMP recipe (verified against this
   project's torch 2.9.1 docs via WebFetch, not recalled:
   https://docs.pytorch.org/docs/2.9/notes/amp_examples.html, "Working
   with Unscaled Gradients"). When AMP is off (CPU, or ``amp=False``), the
   same ``GradScaler`` object is still constructed but with
   ``enabled=False`` -- its ``scale()``/``step()``/``update()`` calls
   become pass-throughs, so the training loop has exactly one code path
   for both cases rather than an AMP/non-AMP branch duplicated throughout
   the step function.

4. Gradient clipping
   ------------------------------------------------------------------------
   ``torch.nn.utils.clip_grad_norm_`` runs (by default, ``grad_clip_norm =
   1.0``) after every ``backward()`` -- "clip gradients to survive cliffs
   and exploding gradients" (Goodfellow, Bengio & Courville, 2016, *Deep
   Learning*, Ch. 8 "Optimization for Training Deep Models", "Cliffs and
   Exploding Gradients" / Key Takeaway 4: "Clip gradients to survive
   cliffs and exploding gradients"). Set ``TrainConfig(grad_clip_norm=
   None)`` to disable.

5. AdamW, not Adam-with-L2
   ------------------------------------------------------------------------
   The optimizer is ``torch.optim.AdamW`` (Loshchilov & Hutter, 2019,
   *Decoupled Weight Decay Regularization*, ICLR, arXiv:1711.05101):
   weight decay is applied as a direct multiplicative shrinkage of the
   parameters, decoupled from the (adaptively-scaled) gradient-based
   update, rather than folded into the gradient the way plain L2-in-Adam
   does -- the paper's central finding is that coupling the two makes
   Adam's effective per-parameter decay rate depend on that parameter's
   gradient history in a way that is not a *deliberate* regularization
   choice.

6. Validation-only early stopping
   ------------------------------------------------------------------------
   "Early stopping ... acts like L2 regularization ... but *automatically*
   selects the effective capacity. 'Free lunch' -- just costs validation
   monitoring" (Goodfellow, Bengio & Courville, 2016, Ch. 7
   "Regularization", "Early Stopping"; also this project's
   ``deep-learning-goodfellow`` skill, ch07 summary, Key Takeaway 3:
   "Early stopping is the cheapest effective regularizer -- always use
   it"). This module tracks improvement **only** on the frozen validation
   Dice (:func:`~src.vascutrace.ml.dataset.frozen_validation_set`) --
   training loss is logged but never gates a checkpoint or an early-stop
   decision, matching this implementation's "validation-only early stop" acceptance
   criterion and preventing a run from "early-stopping" on noisy training
   -loss fluctuations.

7. Every checkpointed epoch is the SAME code path as a resumed epoch
   ------------------------------------------------------------------------
   :func:`train` and :func:`resume` both call the single private
   :func:`_execute`, which differs only in whether it seeds fresh RNG
   streams (``resume_payload is None``) or restores them from a checkpoint
   (``resume_payload is not None``) before entering an *identical* epoch
   loop. This is what makes resume-equivalence provable rather than
   merely hoped-for: there is no second, resume-specific training loop
   implementation that could silently drift out of sync with the
   fresh-start one (see ``tests/test_ml_train.py``'s
   ``TestResumeEquivalence``).

8. Leakage-safety is enforced structurally, twice
   ------------------------------------------------------------------------
   :class:`TrainConfig.__post_init__` rejects any config whose
   ``train_bundle_dirs``/``val_bundle_dirs`` overlap. This is on top of
   (not a replacement for) ``dataset.py``'s own leakage-safety-by
   -construction guarantee (a ``SiameseCropDataset`` only ever reads the
   bundle directories its caller explicitly passed) -- two independent
   layers, so a caller error at the config-assembly stage is caught before
   a single sample is built, not just prevented at the dataset layer.

9. Precomputed synthetic-cache wiring (``src.vascutrace.ml.cache``)
   ------------------------------------------------------------------------
   On-the-fly training became infeasible once ``dataset.py``'s
   ``DatasetConfig.supersample`` default was corrected from ``1`` to the
   accuracy-mandated ``5`` (dataset.py's own module docstring): a positive
   sample now costs P3's own lesion-simulation time directly (tens of
   seconds on a real bundle). ``TrainConfig.train_cache_dir``/
   ``val_cache_dir`` (both optional, both-or-neither) point ``_execute``
   at a :class:`~src.vascutrace.ml.cache.CachedSampleDataset` instead of
   the on-the-fly ``SiameseCropDataset``/``frozen_validation_set`` path for
   that split -- see ``cache.py``'s own module docstring for how the cache
   is built (``precompute_synthetic_cache``) and why its index resolution
   is guaranteed drift-proof against ``dataset.py``'s own logic. Leakage
   -safety for a cache pairing is checked in ``__post_init__`` (before any
   allocation) by comparing the two caches' own ``manifest.json``
   -recorded bundle identities via
   :func:`~src.vascutrace.ml.cache.cache_bundle_identities` -- the same
   "disjoint bundle sets, checked before allocation" discipline item 8
   already establishes for the on-the-fly path, extended to cover the
   cache path too. :func:`compute_split_hash` is cache-aware for the same
   reason: :func:`resume`'s compatibility check (module docstring, item 7)
   must still catch "the caller pointed this resume at a different cache
   than the original run used," not just a different on-the-fly bundle
   -dir list.

10. Configurable training loss (``TrainConfig.loss``) -- fixing the
    collapse-prone LOSS SHAPE half of a measured failure
    ------------------------------------------------------------------------
    This project's own evaluation measured a trained checkpoint with an
    honest positive-only Dice of 0.487 despite a much higher (0.740)
    blended validation Dice at selection time -- literature (Sudre et al.
    2017; Kervadec et al. 2019) documents plain Dice/BCE loss as unstable
    and collapse-prone under exactly this class/foreground imbalance.
    ``TrainConfig.loss`` selects among three already-implemented, unmodified
    loss functions -- ``"dice_bce"`` (``model.py``'s original
    :func:`~src.vascutrace.ml.model.dice_bce_loss`, kept as the default
    -equivalent fallback), ``"focal_tversky"``, and ``"combo"`` (both from
    ``losses.py`` -- see that module's docstring for the full Tversky
    -index/focal-exponent derivation and the confirmed alpha=FN-weight=0.7
    /beta=FP-weight=0.3 convention) -- dispatched through one small
    ``dict[str, Callable]`` (:data:`_LOSS_FUNCTIONS`) so every option shares
    the exact same ``(logits, target, valid_mask=...)`` calling convention;
    no per-loss special-casing anywhere in the training step. Default:
    ``"combo"`` (Focal Tversky's recall-driving under-segmentation penalty
    plus BCE's early-training stability -- see ``losses.py``'s module
    docstring, item 4).

11. Positive-focused validation + checkpoint selection -- fixing the
    "empty-reference" CHECKPOINT-SELECTION half of the same failure
    ------------------------------------------------------------------------
    A blended validation Dice averaged over a validation set that is
    mostly-healthy-slices trivially rewards a checkpoint that predicts
    empty everywhere, because it scores a vacuous ``1.0`` on every healthy
    slice regardless of what it does on the (rarer) lesion-bearing ones --
    the "empty-reference" pitfall (Reinke, Tizabi, Sudre et al., 2024,
    *Understanding Metric-Related Pitfalls in Image Analysis Validation*,
    Nature Methods). :func:`_run_validation` now computes, from the SAME
    per-batch forward pass that already produced the existing blended-Dice
    number (no extra model calls), a full :class:`ValidationMetrics`: the
    blended Dice (kept, logged, for continuity -- never used for selection
    any more), the per-POSITIVE-case mean Dice/IoU, the aggregate per-lesion
    detection precision/recall/F1 (:func:`~src.vascutrace.ml.metrics.
    lesion_component_confusion` summed over every positive validation
    sample, fed to :func:`~src.vascutrace.ml.metrics.
    precision_recall_f_beta`), the negative-sample fully-clean rate (the
    fraction of healthy validation samples with ZERO false-activation
    connected components -- :func:`~src.vascutrace.ml.metrics.
    false_positive_components`), and two composite selection candidates:
    ``dice_x_clean = mean_positive_dice * negative_clean_rate`` (this
    project's own ``evaluate.py`` module's own recommended checkpoint
    -selection statistic -- see that module's docstring, item 7, and its
    ``_stat_selection_metric``, reproduced here identically over the SAME
    formula/metrics-primitive calls, not re-derived by guesswork) and
    ``det_f1_gated_dice`` (mean Dice restricted to positive samples where
    at least one lesion was actually detected --
    :attr:`~src.vascutrace.ml.metrics.DetectionCounts.tp` ``> 0`` --
    excluding undetected-lesion samples from the average entirely rather
    than letting their already-low Dice merely pull it down).
    ``TrainConfig.selection_metric`` picks which of
    ``{"blended_dice", "mean_positive_dice", "dice_x_clean",
    "det_f1_gated_dice"}`` gates BOTH early stopping and ``best.pt``
    checkpoint selection; default ``"dice_x_clean"`` -- this project's own
    ``evaluate.py`` module's explicitly-recommended checkpoint-selection
    suggestion (that module's docstring, item 7), the concretely-already
    -measured statistic (0.434 on the referenced checkpoint). The
    ``"det_f1_gated_dice"`` option is also fully implemented and selectable,
    but it remains an independently unvalidated alternative. The default
    therefore stays with the previously evaluated statistic. A
    ``nan``-producing aggregate (e.g.
    zero positive or zero negative samples in a tiny/misconfigured
    validation set) is mapped to ``0.0`` for SELECTION purposes only
    (:func:`_finite_or`) -- deliberately more conservative than
    ``evaluate.py``'s own reporting-context ``nan`` convention, because a
    ``nan`` selection metric is a correctness hazard here: once
    ``best_val_metric`` itself becomes ``nan``, every subsequent
    ``value > best_val_metric`` comparison is ``False`` under IEEE754,
    permanently disabling checkpoint selection for the rest of the run.

12. Cosine LR schedule with linear warmup -- a pure function of
    ``global_step``, so resume-equivalence needs no new checkpoint state
    ------------------------------------------------------------------------
    ``TrainConfig.lr_schedule`` (``"cosine"`` default, or ``"none"``) with
    ``TrainConfig.warmup_steps`` (default ``0``, opt-in) is now safe to
    enable for long runs because item 11 fixed the checkpoint-selection
    metric that a long, previously-uncorrected run was measured to reward
    collapse under. :func:`_lr_at_step` computes the learning rate for a
    given ``global_step`` as a PURE function of ``(global_step, base_lr,
    total_steps, warmup_steps, schedule)`` -- linear ramp
    ``0 -> base_lr`` over ``[0, warmup_steps)``, then a cosine decay
    ``base_lr -> 0`` over ``[warmup_steps, total_steps)`` -- and every
    training step sets ``optimizer.param_groups[*]["lr"]`` from this
    function immediately before ``zero_grad()``, rather than instantiating
    a stateful ``torch.optim.lr_scheduler`` object. This is a deliberate
    design choice for resume-equivalence, not a missed opportunity to use
    a standard scheduler class: a stateful scheduler object owns its OWN
    internal step counter, which would need to be an ADDITIONAL field
    captured in :class:`~src.vascutrace.ml.checkpoint.CheckpointPayload`
    and restored in exact lockstep with ``global_step`` on every resume --
    a second, redundant piece of state that could silently drift out of
    sync with ``global_step`` if a future change ever touched one without
    the other. Making the schedule a pure function of ``global_step``
    (which :class:`~src.vascutrace.ml.checkpoint.CheckpointPayload`
    ALREADY checkpoints and restores exactly -- see checkpoint.py's module
    docstring) means the LR trajectory is bit-for-bit resume-equivalent
    with ZERO new checkpoint fields and zero risk of a second counter
    drifting from the first (``tests/test_ml_train.py``'s
    ``TestSchedulerResumeEquivalence`` directly asserts the resumed run's
    per-step learning rate matches the continuous run's, in addition to
    ``TestResumeEquivalence``'s pre-existing weight/RNG-stream checks --
    kept on ``lr_schedule="none"`` since that test's own point is general
    resume mechanics, independent of any one loss/schedule choice).
    ``total_steps`` is itself a deterministic function of
    ``config.max_epochs`` and the training dataset's length/``batch_size``/
    ``limit_train_batches`` -- recomputed identically by both a fresh
    :func:`train` call and a :func:`resume` call given the same config,
    exactly like ``split_hash``/``config_hash`` already are (module
    docstring, item 9) -- so it, too, needs no separate checkpoint field IN
    THE COMMON CASE (resuming with the SAME ``max_epochs`` the run
    started with, to continue an interrupted run to its original target).
    ``TrainConfig.lr_schedule_total_steps`` (optional, ``None`` by default)
    exists for the less-common but legitimate case where a resume call
    intentionally uses a DIFFERENT (typically larger) ``max_epochs`` than
    the original run -- a genuine gap found and fixed during this implementation's
    own resume-equivalence verification: without a pinned override,
    ``total_steps`` would silently be recomputed from whatever
    ``max_epochs`` each individual call happens to use, reshaping the
    cosine curve's horizon out from under a resume that only meant to
    EXTEND training, not intentionally re-plan its schedule. Set this field
    explicitly (to the SAME value across every call) when that is the
    desired, deliberate behavior.

13. Train-time geometric + intensity augmentation -- config-gated,
    EXPLORATORY (``TrainConfig.augment``, default ``False``)
    ------------------------------------------------------------------------
    ``dataset.py``'s simulated-source sampler varies source parameters
    (radius, uptake multiplier, and blur; its own module docstring, item 3)
    but does not perturb the same rendered sample
    geometrically or photometrically the way standard image augmentation
    does (Goodfellow, Bengio & Courville, 2016, Ch. 7 "Regularization",
    "Dataset Augmentation": "for many classification problems... the
    operation of injecting noise... has been used by many" -- the
    project's own ``deep-learning-goodfellow`` skill, ch07 summary, Key
    Takeaway: "Dataset augmentation -- very effective for images"). When
    ``config.augment`` is ``True``, :func:`_augment_batch` applies, to
    every TRAINING batch only (never validation -- see below), ONE shared
    random affine (rotation in ``[-augment_rotation_deg,
    augment_rotation_deg]``, isotropic scale in ``[1 -
    augment_scale_delta, 1 + augment_scale_delta]``, translation in
    ``[-augment_translate_px, augment_translate_px]`` pixels) to
    ``left_view``/``right_view``/``pet_diff``/``target_mask``/
    ``valid_mask`` TOGETHER via ``torch.nn.functional.affine_grid`` +
    ``grid_sample`` (images/``pet_diff`` bilinear, ``target_mask``/
    ``valid_mask`` nearest -- a discrete 0/1 mask warped bilinearly would
    invent fractional "soft" boundary pixels the ground truth never had),
    so the five tensors -- which share one in-plane ``(H, W)`` pixel grid
    by the tensor contract (``tensor_schema.py``) -- stay pixel-aligned
    after the warp. ``padding_mode="zeros"`` on ``valid_mask`` itself is
    what keeps "valid_mask semantics correct after the transform": any
    pixel rotated/translated/scaled in from outside the original crop (or
    that was already invalid) becomes ``0`` in the WARPED valid_mask, and
    every downstream loss/metric call already masks by ``valid_mask`` --
    no separate FOV-recomputation logic is needed. Deliberately excludes
    horizontal flip (this implementation's own explicit instruction): ``left_view``/
    ``right_view`` are already each other's physical mirror through the
    subject's own fitted sagittal plane (``dataset.py``'s module docstring,
    item 2); an independent per-sample flip would silently redefine which
    tensor "means" left vs. right, breaking the Siamese pair's semantics.
    PET/CT intensity jitter (small per-sample gain + bias, separately
    ranged for PET vs. CT via ``augment_pet_gain_delta``/
    ``augment_pet_bias``/``augment_ct_gain_delta``/``augment_ct_bias``) is
    applied with the SAME sampled ``(gain, bias)`` to ``left_view`` and
    ``right_view``'s PET (respectively CT) channels -- which makes
    ``pet_diff``'s own jitter exact rather than independently-drawn-and
    -therefore-inconsistent: since ``pet_diff = left_pet - right_pet``
    (both already network-normalized, ``dataset.py``'s module docstring,
    item 2), ``gain*(left_pet+bias) - gain*(right_pet+bias) =
    gain*(left_pet-right_pet)`` -- the additive bias cancels algebraically,
    so ``pet_diff`` only needs the SAME multiplicative ``gain`` reapplied,
    not a third, independently-sampled jitter that could silently decouple
    it from what ``left_view``/``right_view`` actually show. This module
    never augments the VALIDATION split (``_run_validation`` takes no
    augmentation parameter and this flag is never read there) -- the
    implementation's "hard-mask threshold-0.5 eval UNCHANGED" requirement, and
    standard practice (Goodfellow, Bengio & Courville, 2016, Ch. 11, "get a
    reproducible, understood baseline" before optimizing further): a
    validation split that itself changed sample-to-sample would make
    checkpoint selection nondeterministic and epoch-to-epoch comparisons
    meaningless. When ``config.augment`` is ``False`` (the default),
    :func:`_augment_batch` is never called and the training loop consumes
    zero extra random draws. Generated CPU tests compare implicit defaults
    with explicit-off settings and require equal raw model state and
    validation records.

14. Weight EMA -- config-gated, EXPLORATORY (``TrainConfig.ema_decay``,
    default ``None`` = off)
    ------------------------------------------------------------------------
    Polyak-averaged ("EMA") weights are a cheap variance-reduction step
    over the tail of an optimization trajectory (Polyak & Juditsky, 1992,
    *Acceleration of Stochastic Approximation by Averaging*, SIAM J.
    Control Optim. 30(4); operationalized for deep nets as "Mean Teacher"
    -style EMA, Tarvainen & Valpola, 2017, arXiv:1703.01780) -- exactly the
    kind of averaging that may reduce snapshot variance. When
    ``config.ema_decay`` is set (e.g. ``0.999``), :func:`_execute`
    maintains a SECOND model instance, ``ema_model`` (same architecture,
    ``build_model(config.model_config)``), initialized from the live
    model's own starting weights and updated in-place after every
    optimizer step via :func:`_update_ema`:
    ``ema_w <- decay*ema_w + (1-decay)*current_w``, applied to every
    tensor in ``state_dict()`` uniformly -- correct here specifically
    because ``SiameseBilateralUNet`` normalizes with GroupNorm exclusively
    (``model.py``'s module docstring, item 4), which carries no
    running-mean/-var BUFFER the way BatchNorm would (a buffer EMA-average
    would need a different, non-Polyak update rule); every ``state_dict()``
    entry here is a plain learned float tensor, so uniform EMA-averaging is
    exact, not an approximation glossing over a buffer/parameter
    distinction that does not exist in this architecture. Each epoch, if
    EMA is enabled, ``ema_model`` is ALSO run through
    :func:`_run_validation` (the identical function, identical metrics,
    used for the raw model) and its own independent best-tracking state
    (``best_ema_val_metric``/``best_ema_epoch``) gates two SEPARATE
    checkpoint files, ``last_ema.pt``/``best_ema.pt``, written via the SAME
    ``save_checkpoint``/``CheckpointPayload`` machinery as the raw
    ``last.pt``/``best.pt`` -- so ``evaluate.py`` can evaluate an EMA
    checkpoint with ZERO code changes of its own (same schema, same
    ``load_checkpoint`` -> ``build_model`` -> ``load_state_dict`` path).
    Deliberate, flagged scope limits: (a) early stopping and the PRIMARY
    ``best.pt``/``last.pt`` selection remain driven by the RAW model's
    validation exactly as before item 14 -- EMA never influences when this
    run stops or what the primary checkpoint is, keeping this item purely
    additive; (b) the EMA shadow is NOT captured in ``CheckpointPayload``'s
    frozen schema (``checkpoint.py``, unmodified by this item) and is
    therefore not restorable across a :func:`resume` call. Resume fails
    closed for an EMA-enabled config until that state is checkpointed; it
    never silently re-initializes the averaging trajectory. When
    ``config.ema_decay``
    is ``None`` (the default), no ``ema_model`` is built, no EMA update
    runs, and no ``*_ema.pt`` file is written -- the baseline path is
    untouched by this item.

15. Secondary checkpoint-selection tracking -- config-gated, EXPLORATORY
    (``TrainConfig.secondary_selection_metric``, default ``None`` = off)
    ------------------------------------------------------------------------
    Item 11's ``dice_x_clean`` selection policy is a healthy-clean-rate
    -gated metric: this run's own background measured it keeping a
    checkpoint can differ from one selected on positive simulated-source
    overlap because ``dice_x_clean`` also rewards not activating on
    negative samples. Rather than silently swapping the
    default selection policy (a "no silent algorithm substitution" HALT
    concern -- this project's own engineering requirements), this item adds
    a SECOND, independent best-tracking loop, gated by
    ``config.secondary_selection_metric`` (one of the same
    ``_SELECTION_METRIC_NAMES`` item 11 already validates), that tracks
    and saves ``best_<secondary_selection_metric>.pt`` (and, when EMA is
    also enabled, ``best_ema_<secondary_selection_metric>.pt``) ALONGSIDE
    the existing ``dice_x_clean``-gated ``best.pt`` -- never replacing it.
    This makes the tradeoff between the two policies directly inspectable
    (two checkpoints from the SAME run, evaluable with the SAME
    ``evaluate.py`` path) rather than requiring a second full training run
    to recover the alternative. Every epoch's raw
    ``mean_positive_iou``/``mean_positive_dice``/``negative_clean_rate``/
    ``dice_x_clean`` was ALREADY logged unconditionally to
    ``metrics.jsonl``'s ``"validation"`` event by item 11 (see that item's
    ``_append_metrics_line`` call) -- this item adds no new per-epoch
    LOGGING, only the second checkpoint-selection/save policy. When
    ``config.secondary_selection_metric`` is ``None`` (the default), no
    extra file is ever written -- the baseline path is untouched by this
    item.

16. Online hard-negative mining -- config-gated, EXPLORATORY
    (``TrainConfig.hard_negative_mining``, default ``False``)
    ------------------------------------------------------------------------
    This is a sampling/emphasis mechanism: it changes which cached examples
    the optimizer draws more often without changing the per-sample loss
    formula. Its effect on overlap, component detection, and negative
    activation must be measured; the mechanism alone makes no outcome or
    causal claim.

    Mechanism -- **online hard-negative mining at the SAMPLE (not pixel)
    granularity**, adapted from OHEM (Shrivastava, Gupta & Girshick, 2016,
    *Training Region-based Object Detectors with Online Hard Example
    Mining*, CVPR (oral), arXiv:1604.03540): OHEM ranks candidate examples
    by their OWN loss each iteration and keeps only the hardest fraction for
    the gradient step. This module's analogue ranks TRAINING-SET NEGATIVE
    (target-empty) samples by an epoch-accumulated per-sample
    false-activation score and OVERSAMPLES the hardest fraction into
    subsequent epochs' mini-batches -- chosen over a PIXEL-level OHEM
    variant (keep only the top-k hardest background pixels per sample, in
    ``losses.py``) for two concrete reasons specific to this failure mode:
    (a) the target estimand is activation on entire negative slices (the
    negative_clean_rate metric, item 11, is itself a per-SAMPLE statistic --
    "did this whole healthy slice stay clean" -- not a per-pixel one), so a
    sample-level fix targets the SAME granularity the failure was measured
    at; (b) a pixel-level variant inside a positive sample's background
    region would ALSO reshape the positive/negative pixel-count balance
    ``losses.py``'s already-tuned Tversky alpha/beta convention depends on
    (that module's docstring, item 4) -- entangling this experiment with a
    second, unintended loss-shape change -- whereas sample-level
    oversampling changes only how OFTEN an already-existing, already
    -correctly-shaped training example is drawn, never what its own loss
    computation looks like once drawn (``loss_fn`` itself, ``_LOSS_FUNCTIONS``,
    is completely unmodified by this item).

    Concretely, when ``config.hard_negative_mining`` is ``True``
    (structurally REQUIRES ``config.train_cache_dir`` -- see
    ``TrainConfig.__post_init__`` -- because this mechanism needs a stable,
    index-addressable dataset AND a known positive/negative split point,
    both of which only :class:`~src.vascutrace.ml.cache.CachedSampleDataset`
    provides: ``cache.py``'s own ``precompute_synthetic_cache`` writes every
    POSITIVE sample at indices ``[0, manifest.total_positive)`` and every
    NEGATIVE sample at ``[manifest.total_positive, total_samples)`` -- see
    that module's own task-assembly loop -- so which cache index is a
    negative is known with zero extra I/O, not by loading every sample to
    inspect its target mask):

    - The first ``max(1, config.hard_negative_warmup_epochs)`` epochs use
      ORDINARY uniform-random sampling (:func:`_build_hard_negative_train_loader`
      with ``weights=None`` falls back to plain ``shuffle=True``, over the
      SAME index-wrapped dataset/collate as the mining path, purely so every
      negative index gets an initial observed score before any mining
      decision is made -- mining epoch 0 with zero prior observations would
      have nothing to rank). This floor is a hard ``max(1, ...)``, not just
      the configured value, because epoch 0 categorically cannot have any
      prior-epoch score to mine from.
    - During every training step, :func:`_per_sample_clipped_negative_score`
      computes,
      from the SAME ``logits`` tensor the training step already produced
      (``.detach()`` -- no extra model call, no gradient contribution of its
      own), a per-sample false-activation score for every NEGATIVE sample in
      the batch: a score-space approximation to the mean binary
      cross-entropy against an all-zero target. It clamps ``1-p`` to
      ``1e-6``, so the per-pixel contribution is capped at
      ``-log(1e-6)`` and saturated high logits can tie. This historical
      v6 scoring rule is retained for v6/v7b comparability; it is not the
      numerically stable, uncapped logits-space BCE. The score tracks
      false activation on a target-empty training sample -- the quantity item 11's
      ``negative_clean_rate``/``dice_x_clean`` selection metrics already
      measure at validation time, now tracked per-sample on the TRAIN split.
      Each observed score updates a per-index exponential moving average
      (``config.hard_negative_score_momentum``, default 0.5 -- the same
      Polyak-averaging idea item 14 already uses for weights, applied here
      to a scalar) rather than being overwritten, so a negative index that
      is not drawn in a particular mining epoch (``WeightedRandomSampler``
      draws WITH replacement once mining is active, so per-epoch coverage of
      every index is no longer guaranteed the way plain ``shuffle=True``'s
      is) keeps its most-recent smoothed estimate rather than reverting to
      "unseen."
    - At the end of every epoch, the hardest ``config.hard_negative_fraction``
      (default 0.35) of ALL OBSERVED negative indices (by that EMA score,
      descending) are marked "hard"; a per-index sampling-weight vector is
      built -- ``config.hard_negative_oversample_weight`` (default 3.0) for
      hard negatives, ``1.0`` for every other index (easy negatives AND
      ALL positives). Individual positive weights stay unchanged, but the
      total negative sampling mass increases whenever hard-negative
      weights exceed 1.0; this implementation is therefore NOT class-mass
      preserving. A future fixed-class-mass sampler would be a distinct
      algorithm, not an interpretation of this one. The weights are
      installed as next epoch's
      ``WeightedRandomSampler`` weights (``num_samples=len(train_dataset)``,
      matching plain ``shuffle=True``'s per-epoch example count, so the
      cosine LR schedule's already-computed ``total_steps`` -- item 12,
      a pure function of dataset length/batch_size -- stays valid even
      though the SAMPLING distribution per index has changed).
    - Every epoch, a ``"hard_negative_mining"`` event is appended to
      ``metrics.jsonl`` (:func:`_append_metrics_line`, the same sink item 11
      already uses) recording ``n_negative_total``/``n_negative_seen``,
      ``n_hard_negatives_mined``, the realized mined fraction, the
      configured target fraction/weight, and ``mean_negative_score``
      (epoch-wide average observed false-activation score across ALL
      negatives, the trend line to watch for a downward slope as training
      suppresses false activations) plus the same average split into
      ``mean_hard_negative_score``/``mean_easy_negative_score`` and the
      score threshold used that epoch, so the mining mechanism's own effect
      is directly inspectable from the log without re-running anything.

    Deliberate, flagged scope limits (same discipline as item 14's EMA
    -shadow flag): (a) restricted to the cache path only (on-the-fly
    ``SiameseCropDataset`` has no stable positive/negative index-split
    -point equivalent to ``manifest.total_positive`` without an extra,
    unbuilt bookkeeping pass -- out of this implementation's scope, and every real
    training config in this project already uses the cache path); (b) the
    per-index score/weight state is NOT part of :class:`~src.vascutrace.ml.
    checkpoint.CheckpointPayload`'s frozen schema and is therefore not
    restorable across :func:`resume`; resume fails closed for a hard
    -negative-mining config instead of silently resetting scores, weights,
    or warmup state. When
    ``config.hard_negative_mining`` is ``False`` (the default), none of
    :func:`_IndexedDataset`/:func:`_collate_indexed_samples`/
    :func:`_build_hard_negative_train_loader`/
    :func:`_per_sample_clipped_negative_score`
    is ever constructed or called, and the pre-built ``train_loader`` (the
    exact ``_build_train_loader`` call every prior item already used) is
    reused unchanged for every epoch -- the baseline (``siamese_v4_big``
    -/``siamese_v5exp``-equivalent) path therefore skips the mining
    machinery entirely.

17. Soft-target experiment -- config-gated, EXPLORATORY, honest-hard-mask
    -eval-preserving (``TrainConfig.soft_target``, default ``False``)
    ------------------------------------------------------------------------
    ``dataset.py``'s own module docstring, item 9, propagates a new
    ``Sample.source_fraction`` field -- the simulator's sharp, pre-blur
    fractional capsule occupancy -- that the pipeline previously discarded
    at the ``occupancy >= 0.5`` binarization producing ``target_mask``.
    This item lets TRAINING consume that continuous field directly, using
    a loss that is actually PROPER on it (``losses.py``'s own module
    docstring, item 5: the existing ``combo`` loss's Tversky machinery is
    not proper on a fractional target, so it is not an acceptable route
    here).

    Three routing rules, enforced structurally, not by convention:

    - **Training target.** ``config.soft_target=True`` routes the training
      step's target tensor to ``batch["source_fraction"]`` instead of
      ``batch["target_mask"]`` (the ``target = (... if config.soft_target
      else ...)`` line in the main training loop, immediately after
      ``batch["pet_diff"]`` is read). ``config.soft_target=False`` keeps
      the hard target path.
    - **Loss.** ``TrainConfig.__post_init__`` refuses ``soft_target=True``
      combined with any ``loss`` outside ``_SOFT_TARGET_COMPATIBLE_
      LOSS_NAMES`` (``{"soft_bce", "soft_combo"}``, ``losses.py``'s new,
      additive :func:`~src.vascutrace.ml.losses.soft_bce_loss`/
      :func:`~src.vascutrace.ml.losses.soft_combo_loss`) -- this is the
      structural guard against the exact "improper loss on a soft target"
      failure mode this item exists to avoid, not a docstring-only
      warning.
    - **Cache.** ``TrainConfig.__post_init__`` also refuses
      ``soft_target=True`` unless ``config.train_cache_dir``'s manifest
      declares ``has_source_fraction=True`` (``cache.py``'s module
      docstring, item 7) -- a cache built before that item (e.g.
      ``p6_cache``/``p6_cache_big``) would otherwise silently supply
      ``cache.py``'s own zero-filled fallback array, which is safe for
      every ``soft_target=False`` reader but would be a silent
      correctness bug for a ``soft_target=True`` one.

    **Validation, checkpoint selection, and every reported metric ALWAYS
    consume ``target_mask`` regardless of ``config.soft_target``** -- the
    hard-mask evaluation protocol this project has used throughout
    continues under this item. This is enforced structurally,
    not merely documented: :func:`_iter_val_batches` always calls
    :func:`_collate_samples` with its default ``include_source_fraction=
    False``, so ``"source_fraction"`` cannot be a KEY in any batch dict
    :func:`_run_validation` ever sees (that function also carries an
    explicit ``assert "source_fraction" not in batch`` as a defense-in
    -depth spy check, not the sole guarantee); ``evaluate.py`` reads
    ``Sample.target_mask`` directly and never constructs or touches
    ``Sample.source_fraction`` at all. Only the TRAINING loader
    (:func:`_build_train_loader`/:func:`_build_hard_negative_train_loader`
    -- orthogonal to item 16's hard-negative mining, which may be combined
    with this item) ever requests ``include_source_fraction=True``, and
    only when ``config.soft_target`` is ``True``.

    Train-time augmentation (item 13) also needed one targeted correction:
    :func:`_augment_batch`'s existing ``mode="nearest"`` choice for the
    target tensor was deliberately chosen because a discrete ``{0,1}``
    mask warped bilinearly would invent fractional values the label never
    had -- exactly backwards for ``source_fraction``, which IS already a
    genuinely continuous field. The training loop now passes
    ``target_interp="bilinear"`` to :func:`_augment_batch` only when
    ``config.soft_target`` is ``True``; every hard-target caller keeps the
    original ``"nearest"`` default unchanged.

    ``config.soft_target``/``config.loss`` are both included in
    :func:`compute_config_hash` and :func:`_hyperparams_dict`; resume
    compatibility verifies both the stored hash and the explicit
    hyperparameter map. EMA, hard-negative-mining, and secondary-selector
    state are not represented by the frozen checkpoint schema, so
    :func:`resume` rejects runs with any of those features enabled rather
    than silently reinitializing their state. No
    :class:`~src.vascutrace.ml.model.ModelConfig`, inference
    state-dict, or checkpoint-payload-schema change is required or made by
    this item -- ``source_fraction`` never becomes a network INPUT or
    OUTPUT channel; it only changes which tensor the loss function is
    compared against during the training step.

    ``configs/train_siamese_v7b.yaml`` is the HNM-preserving exploratory
    configuration for this mechanism. ``train_siamese_v7soft.yaml`` is a
    superseded ablation. Neither config is certifiable or authorizes a run.

18. Boundary-local auxiliary loss (A/B/C experiment) --
    config-gated, EXPLORATORY, hard-mask-eval-preserving
    (``TrainConfig.lambda_boundary``/``boundary_aux_target``, defaults
    ``0.0``/``"none"``)
    ------------------------------------------------------------------------
    Implements the frozen boundary-loss contract: the PRIMARY objective stays the
    shipped hard-mask ``combo_loss`` (item 4/10 -- unchanged, unlike item
    17's soft-target REPLACEMENT), and a separately-normalized boundary
    -local auxiliary term (``losses.py``'s new, additive
    :func:`~src.vascutrace.ml.losses.boundary_auxiliary_loss`, that
    module's own docstring item 6) is added on top, scaled by
    ``lambda_boundary``:

        total_loss = combo_loss(logits, target_mask, valid_mask)
                     + lambda_boundary * aux_term(target_mode)

    Three routing rules, enforced structurally, exactly mirroring item 17's
    own discipline:

    - **Arm selection.** ``boundary_aux_target in {"none", "hard",
      "fraction"}`` selects, respectively: arm A (aux off, the existing
      hard-combo path, byte-identical to every pre-item-18 config), arm B
      (aux target = hard ``target_mask``), arm C (aux target =
      ``source_fraction``). ``lambda_boundary`` multiplies the aux term
      only -- it never touches the primary ``combo_loss`` weight.
    - **Mutual exclusion with item 17.** ``TrainConfig.__post_init__``
      refuses ``boundary_aux_target != "none"`` combined with
      ``soft_target=True`` -- the boundary experiment Sec 5's ``L_hard`` is literally "the
      shipped ``combo_loss(logits, target_mask, valid_mask)``"; routing the
      TRAINING target to ``source_fraction`` (item 17) would silently
      change what the "hard" half of this sum actually is. This guard is
      NOT explicitly named in the plan's own text (the plan's A/B/C arms
      never combine with item 17 in the first place) -- it is added here
      defensively, to keep the plan's ``L_hard`` definition true by
      construction rather than merely by caller convention.
    - **Cache.** Exactly item 17's own check, reused verbatim
      (``cache_has_source_fraction``): ``boundary_aux_target != "none"``
      requires ``train_cache_dir`` set and its manifest to declare
      ``has_source_fraction=True``.

    **Validation, checkpoint selection, and every reported metric ALWAYS
    consume ``target_mask`` only, exactly as item 17 already established**
    -- this item does not touch ``_run_validation``/``_iter_val_batches``/
    ``_collate_samples`` in any way beyond widening the SAME
    ``include_source_fraction`` gate item 17 already built (now
    ``config.soft_target or config.boundary_aux_target != "none"``, TRAINING
    loader only) -- the mutual-exclusion guard above means this widening
    can never let ``source_fraction`` reach validation: ``_iter_val_batches``
    still always calls ``_collate_samples`` with its hard-coded
    ``include_source_fraction=False`` default, so item 17's own spy
    assertion in ``_run_validation`` (``assert "source_fraction" not in
    batch``) remains the same, unweakened guarantee.

    **Augmentation alignment.** ``config.augment=True`` (every
    real A/B/C config carries this, inherited from v6exp/v7b) applies one
    shared random affine to ``left``/``right``/``diff``/``target``/``valid``
    together (item 13). Before this item, ``source_fraction`` was never
    passed through that warp at all when ``soft_target=False`` (item 17
    only ever augmented ``source_fraction`` when it WAS the training
    target) -- augmenting the model's input/hard-target crop while leaving
    the boundary auxiliary's own ``source_fraction`` map un-warped would
    silently misalign ``W``/``G`` against the augmented logits/target_mask
    every augmented step, corrupting the auxiliary loss without raising an
    error. :func:`_augment_batch` therefore gained an OPTIONAL
    ``source_fraction`` keyword (default ``None``, backward-compatible --
    every existing caller/test that omits it gets the ORIGINAL 5-tuple
    return unchanged, consuming the SAME RNG draws): when supplied, the
    SAME shared ``grid`` this call already computed additionally warps
    ``source_fraction`` with bilinear interpolation (item 17's own
    continuous-field rationale) and the function returns a 6-tuple instead.
    The training step passes ``source_fraction`` only when
    ``config.boundary_aux_target != "none"``.

    **Logging (also closes the Phase-1 logging gap this project's own
    2026-07-18 diagnostic flagged -- no prior per-step BCE-vs-FTL/aux
    gradient-magnitude visibility existed).** Every ``"train_step"`` record
    now additionally carries ``lambda_boundary`` (config value, always),
    ``L_hard``/``L_aux`` (raw scalars, PRE-``lambda_boundary`` -- arm A logs
    ``L_hard == train_loss``, ``L_aux == 0.0``), ``boundary_count``/
    ``boundary_fraction`` (from
    :class:`~src.vascutrace.ml.losses.BoundaryAuxiliaryLoss`, ``0.0`` for
    arm A), and, ONLY at a logging step (``(global_step) %
    log_every_n_steps == 0``) AND only when the aux term is active (cost
    -guarded -- an extra pair of ``torch.autograd.grad`` calls every single
    step would be needlessly expensive), ``hard_grad_norm``/
    ``aux_grad_norm``: the L2 norm of the RAW (pre-``lambda_boundary``, pre
    -clip) gradient of ``L_hard``/``L_aux`` respectively with respect to
    every trainable model parameter, computed via two separate
    ``torch.autograd.grad(..., retain_graph=True)`` calls BEFORE the real
    ``scaler.scale(loss).backward()`` (mirrors the boundary experiment Sec 5's own
    production-gradient protocol wording, "Compute hard and auxiliary
    gradients in separate torch.autograd.grad calls before clipping",
    generalized here to the whole trainable parameter set rather than that
    protocol's specific head/decoder subset, which is reserved for the
    SEPARATE, standalone ``scripts/p3_lambda_probe.py`` -- this per-step log
    is a cheap ONLINE diagnostic, not that offline protocol itself). Both
    are ``None`` (JSON ``null``) when not computed, never a poisoned/NaN
    float (``_safe_grad_l2_norm``'s own finite guard). ``"validation"``/
    ``"validation_ema"`` records additionally carry ``lambda_boundary``
    (the static config scalar only -- never ``L_aux``/``boundary_count``/
    ``boundary_fraction``, which would require ``source_fraction`` and are
    therefore structurally impossible to compute on the validation path
    without violating the hard-mask-only evaluation invariant above).

    ``lambda_boundary``/``boundary_aux_target`` are both included in
    :func:`compute_config_hash` and :func:`_hyperparams_dict` (so distinct
    A/B/C configs hash distinctly and resume compatibility rejects an arm
    swap) and in :func:`_write_manifest`. No :class:`~src.vascutrace.ml.
    model.ModelConfig`, inference state-dict, or checkpoint-payload-schema
    change is required or made by this item -- exactly item 17's own
    closing statement, for the same reason (the auxiliary never becomes a
    network input or output channel, only an additional loss term).

    ``configs/train_siamese_p3_A.yaml``/``_B.yaml``/``_C.yaml`` are the
    frozen A/B/C configs for this experiment family. The entire family is
    EXPLORATORY -- NOT CERTIFIABLE; none of these configs authorizes a run.

19. Tversky FN/FP rebalance -- config-gated, EXPLORATORY, ONE-KNOB area
    lever (Phase 4 area-loss lever; ``TrainConfig.tversky_fn_weight``/
    ``tversky_fp_weight``, defaults ``0.7``/``0.3``)
    ------------------------------------------------------------------------
    Diagnosis this item targets: the project's own L0 error-taxonomy work
    found the v6 IoU gap AREA-dominated, with over-segmentation the largest
    bucket -- yet the shipped ``"combo"``/``"focal_tversky"`` loss (item 10)
    uses ``losses.py``'s own FN-tilted Tversky operating point
    (``alpha``/FN-weight ``= 0.7 > beta``/FP-weight ``= 0.3``, Abraham & Khan
    2019's reported choice, adopted here specifically to fight the DIFFERENT,
    previously-measured under-segmentation failure documented in item 10's
    own diagnosis), which structurally rewards over-prediction on every
    training step. This item exposes exactly that one knob at the
    ``TrainConfig`` layer -- it does not touch ``losses.py`` at all:
    :func:`~src.vascutrace.ml.losses.focal_tversky_loss` and :func:`~src.
    vascutrace.ml.losses.combo_loss` already accept ``alpha``/``beta`` as
    ordinary keyword parameters defaulting to ``0.7``/``0.3`` (that module's
    own docstring, item 1) -- both functions are BYTE-FOR-BYTE UNCHANGED by
    this item.

    Mechanism: when ``config.loss`` is ``"combo"`` or ``"focal_tversky"``
    (:data:`_TVERSKY_WEIGHTED_LOSS_NAMES`) AND ``tversky_fn_weight``/
    ``tversky_fp_weight`` actually differ from ``losses.py``'s own defaults
    (:data:`_TVERSKY_ALPHA_DEFAULT`/:data:`_TVERSKY_BETA_DEFAULT`, ``0.7``/
    ``0.3``), ``loss_fn`` is built via ``functools.partial(_LOSS_FUNCTIONS[
    config.loss], alpha=config.tversky_fn_weight, beta=config.
    tversky_fp_weight)`` instead of the bare dict lookup -- the training
    step's own ``loss_fn(logits, target, valid_mask=valid)`` call site
    (item 10's established convention) is UNCHANGED source text either way;
    ``functools.partial`` only pre-binds two keyword arguments the callee
    already accepted. Any other ``config.loss`` value (``"dice_bce"``,
    ``"soft_bce"``, ``"soft_combo"``) ignores these two fields entirely --
    none of those functions take an ``alpha``/``beta`` Tversky-style pair,
    so ``config.tversky_fn_weight``/``tversky_fp_weight`` are structurally
    inert whenever ``config.loss`` is one of them.

    Byte-for-byte default-path identity -- BOTH the numeric result AND the
    exact callable/signature dispatched through ``_LOSS_FUNCTIONS``:
    ``tversky_fn_weight=0.7``, ``tversky_fp_weight=0.3`` (the defaults)
    reproduce ``losses.py``'s OWN ``alpha=0.7``/``beta=0.3`` defaults
    exactly, so the "at defaults, skip functools.partial entirely" branch
    above dispatches ``_LOSS_FUNCTIONS[config.loss]`` with the identical
    ``(logits, target, valid_mask=...)`` call every pre-item-19 config
    (including ``v6exp.yaml``) already made -- not merely a numerically
    -equal partial-wrapped call, the SAME bare callable, no extra kwargs at
    all. This is deliberately stronger than "numerically identical": it
    also keeps every pre-item-19 test that monkeypatches
    ``_LOSS_FUNCTIONS[...]`` with a test double expecting exactly that
    narrower ``(logits, target, valid_mask=None)`` signature working
    unchanged. Numeric identity at the defaults is verified by
    ``tests/test_ml_tversky_rebalance.py::TestTverskyKnobDefaultUnchanged``
    (``loss_fn is combo_loss`` at 0.7/0.3 and ``torch.equal`` output), and
    the override path (0.5/0.5 genuinely reaching ``combo_loss``) by
    ``TestTverskyKnobReachesLoss`` in the same module.

    Validation: ``TrainConfig.__post_init__`` requires both fields finite
    and strictly ``> 0`` (a Tversky index with a zero or negative FN/FP
    weight is either degenerate or ill-defined -- not merely a bad
    hyperparameter choice) -- fail-before-allocating, matching this
    project's own item-1 discipline. Both fields are included in
    :func:`compute_config_hash`, :func:`_hyperparams_dict`, and
    :func:`_write_manifest`, so a rebalanced run hashes distinctly from
    v6exp and a resume call rejects a silent FN/FP-weight swap exactly as
    item 18 already established for ``lambda_boundary``/
    ``boundary_aux_target``.

    ``configs/train_siamese_p4l3_dice.yaml`` clones ``v6exp.yaml`` EXACTLY
    and changes ONLY ``tversky_fn_weight: 0.5``/``tversky_fp_weight: 0.5``
    -- the symmetric (Dice-equivalent -- ``losses.py``'s own item 1 states
    ``alpha=beta=0.5`` recovers plain Dice from the Tversky index) choice,
    i.e. the overlap/IoU-aligned operating point, as the direct one-knob
    area-rebalance counterpart to v6exp's FN-tilted default. EXPLORATORY --
    NOT CERTIFIABLE; this config does not authorize a run.

20. Constrained-floor IoU checkpoint selector -- config-gated, EXPLORATORY
    (Track A / P4 of the "path to IoU 0.70" plan;
    ``TrainConfig.constrained_iou_selection``, default ``False`` = off)
    ------------------------------------------------------------------------
    Diagnosis this item targets: item 11's ``dice_x_clean`` primary
    selector and item 15's raw ``mean_positive_iou`` secondary selector
    are each single-objective -- ``dice_x_clean`` early-stops on a
    clean-weighted metric that can systematically discard a slightly
    -dirtier, higher-IoU epoch (the Phase 4 early-stop confound), while an
    UNCONSTRAINED ``mean_positive_iou`` secondary can save an epoch that
    fails the project's own precision/F1 legality floor (v6exp's raw IoU
    peak 0.607 at epoch 95 is exactly this failure: component precision
    0.893 < the 0.901 floor, so it is not a reportable operating point).
    This item adds a THIRD, independent best-tracking loop that performs
    literal CONSTRAINED optimization -- maximize ``mean_positive_iou``
    subject to ``detection_precision >= constrained_iou_min_precision``
    AND ``detection_f1 >= constrained_iou_min_f1`` AND
    ``negative_clean_rate >= constrained_iou_min_clean`` -- so every run
    gated by this flag captures its own best LEGAL IoU checkpoint without a
    second training run.

    Mechanism: gated by ``config.constrained_iou_selection`` (default
    ``False``). When ``True``, each epoch's ALREADY-COMPUTED
    ``ValidationMetrics`` (the SAME object item 11's primary/item 15's
    secondary selectors read -- no recomputation, no extra model forward
    pass, no change to ``_run_validation``'s eval semantics or its
    hard-mask-only invariant) is checked against the three floors above. An
    epoch that clears all three floors is "legal"; among legal epochs only,
    the one with the highest ``mean_positive_iou`` is kept, and its weights
    are saved to ``best_constrained_iou.pt`` (and, when EMA is also
    enabled, ``best_ema_constrained_iou.pt``, gated by the EMA model's OWN
    ``ValidationMetrics`` from the same epoch -- never the raw model's).
    An epoch that fails any floor contributes nothing: best-so-far state is
    left unchanged and no checkpoint write happens for that epoch. This is
    a FOURTH parallel checkpoint file (alongside ``best.pt``,
    ``best_<secondary_selection_metric>.pt``, and their ``_ema`` variants)
    -- it never replaces or influences ``best.pt``, item 15's secondary
    checkpoint, ``best_val_metric``, ``best_epoch``, or
    ``epochs_without_improvement`` (early stopping stays gated on
    ``config.selection_metric`` exactly as before this item, per this
    item's own "keep the primary path unchanged by default" constraint).
    Every epoch's legality/improvement decision is logged unconditionally
    to ``metrics.jsonl`` as a ``"constrained_iou_selection"`` (and, when EMA
    is on, ``"constrained_iou_selection_ema"``) event -- including
    ``"legal": False`` epochs, so the full floor-crossing history is
    inspectable without re-running. If NO epoch across the whole run ever
    qualifies, ``best_constrained_iou.pt``/``best_ema_constrained_iou.pt``
    are never written (:class:`TrainResult`'s corresponding
    ``best_constrained_iou_epoch``/``best_ema_constrained_iou_epoch``
    fields stay ``None``) and a single explicit
    ``"constrained_iou_selection_summary"``/
    ``"constrained_iou_selection_ema_summary"`` event with
    ``"qualified": False`` is appended at the end of the run so that "no
    candidate" is never silently indistinguishable from "flag was off."

    Floors: ``constrained_iou_min_precision`` (default ``0.901``),
    ``constrained_iou_min_f1`` (default ``0.859``) reproduce this project's
    own reported baseline component-precision/F1 floor exactly;
    ``constrained_iou_min_clean`` (default ``0.0``) is OFF by default (no
    clean-rate constraint) and may be set e.g. ``0.70`` to also require a
    minimum negative clean rate. ``TrainConfig.__post_init__`` requires all
    three finite and in ``[0, 1]`` -- fail-before-allocating, matching this
    project's own item-1 discipline.

    Resume: like item 14's EMA and item 15's secondary selector, this
    item's best-tracking state (``best_constrained_iou_val``/``_epoch`` and
    their EMA counterparts) is NOT reconstructed from ``metrics.jsonl`` on
    resume -- doing so correctly would need a second, floor-aware variant
    of :func:`_restore_resume_selection_state`, out of scope here. Rather
    than silently under- or over-restoring that state,
    ``config.constrained_iou_selection=True`` is added to :func:`resume`'s
    existing ``unsupported_state`` list (alongside ``ema_decay``,
    ``hard_negative_mining``, and ``secondary_selection_metric`` -- see
    item 15) and rejected with :class:`CheckpointCompatibilityError`,
    exactly the established pattern for checkpoint-external tracking state
    this module already uses. All four fields are included in
    :func:`compute_config_hash`, :func:`_hyperparams_dict`, and
    :func:`_write_manifest`, so a constrained-selector-enabled run hashes
    distinctly from one without it (a fresh, non-resumed run with this flag
    on is always fully supported).

21. Deep supervision -- train-only multi-scale aux heads (B2 lever
    B2/L4; ``TrainConfig.deep_supervision``, default ``False`` = off)
    ------------------------------------------------------------------------
    The frozen specification includes two corrections this implementation follows
    exactly (not the spec's original 3-head/nearest-downsample draft):
    TWO aux heads at decoder scales ``{2, 4}`` (the spec's x8 head is
    dropped -- marginal target support even under max-pool), and hard
    -mask downsampling by MAX-POOL (``F.max_pool2d`` -- "any positive pixel
    in the block -> 1"), not nearest/strided sampling, because nearest
    -style downsampling vanishes 24% of this dataset's lesions at x4 and
    79% at x8 (median lesion ~18px), while max-pool preserves all 78 val
    positives at every scale.

    Diagnosis this item targets: ``model.py``'s single full-res objective
    gives one gradient averaged over every lesion size; the project's own
    L0 diagnosis found the residual IoU gap is bidirectional (under- AND
    over-segmentation) and size-correlated (Spearman +0.38-0.49 with
    lesion radius/area) -- a symptom a single-scale loss cannot address by
    reweighting alone (Phase 4's own zero-sum Tversky-rebalance proof:
    over-seg down implies under/miss up). Deep supervision gives the
    SHARED decoder an extra, per-scale gradient at each aux head's own
    resolution, without changing the loss family or the eval head.

    Mechanism: ``model.py``'s ``SiameseBilateralUNet.forward`` grows a
    ``return_aux: bool = False`` parameter; this training step (ONLY this
    step, ONLY when ``config.deep_supervision`` is True) is the sole
    caller ever passing ``return_aux=True``. The two aux logit maps
    (``aux_logits_by_scale[2]``: ``[B,1,72,40]``; ``[4]``: ``[B,1,36,20]``)
    are supervised by the SAME, byte-for-byte-unchanged
    :func:`~src.vascutrace.ml.losses.combo_loss` the main term already
    uses -- same ``tversky_weight``/``bce_weight``/``alpha``/``beta``/
    ``gamma`` (the training step's existing ``loss_fn`` partial, reused
    verbatim for every aux scale, so a Tversky-rebalanced run (item 19)
    rebalances its aux terms identically to its main term, with zero extra
    plumbing) -- against a per-scale downsampled target/valid pair:

    ::

        target_i = F.max_pool2d(target_mask, kernel_size=scale_i)   # hard,
                                                                      # {0,1}
        valid_i  = 1 - F.max_pool2d(1 - valid_mask, kernel_size=scale_i)
                   # min-pool: a downsampled pixel is valid only if its
                   # WHOLE block is valid -- the conservative direction,
                   # chosen because valid_mask marks in-FOV pixels and an
                   # aux-scale pixel should never be counted valid on the
                   # strength of only PART of its receptive block actually
                   # being in-FOV.
        L_aux    = sum_i weight_i * loss_fn(aux_logits_by_scale[i], target_i,
                                             valid_mask=valid_i)
        total    = loss_fn(logits, target, valid_mask=valid) + L_aux

    ``deep_supervision_scales``/``deep_supervision_weights`` (defaults
    ``(2, 4)``/``(0.5, 0.25)``, element-wise aligned) are validated
    (equal length, scales a subset of the model's two supported aux
    points ``{2, 4}``, weights finite and ``> 0``) and drive both the
    downsample factor AND the per-scale weight -- no scale/weight pair is
    silently dropped or reordered. ``boundary_auxiliary_loss`` (item 18)
    is NEVER imported/called from this path -- deep supervision's aux
    term is ``combo_loss`` on a downsampled HARD region mask, not a
    sparse fractional-boundary band; the two mechanisms are orthogonal
    and this module keeps them so (no interaction validated or assumed).

    Model wiring: ``TrainConfig.__post_init__`` derives
    ``config.model_config.deep_supervision`` FROM ``config.deep_supervision``
    (via ``dataclasses.replace`` -- ``model_config`` is frozen) so a config
    file only ever needs to set the one top-level flag; the model always
    ends up with the aux heads it needs and none it doesn't. Because
    ``model_signature()`` folds ``ModelConfig.deep_supervision`` into the
    architecture hash ONLY when True (``model.py`` item 8a),
    ``config.deep_supervision=False`` (the default) yields the exact
    pre-B2 ``model_signature()`` string -- every existing v6/B0 checkpoint
    stays loadable, byte-identical, with this item entirely off.

    Aux heads are structurally TRAIN-ONLY: ``_run_validation`` and
    ``evaluate.py`` both call ``model(left, right, diff)`` with no
    ``return_aux`` argument (defaults ``False``) -- neither code path was
    touched by this item, so the hard-mask-only, threshold-0.5 eval
    invariant this project has always enforced stays structurally intact;
    there is no code path by which an aux logit or a downsampled mask can
    reach a reported metric.

    ``deep_supervision``/``deep_supervision_scales``/
    ``deep_supervision_weights`` are included in :func:`compute_config_hash`
    and :func:`_hyperparams_dict` (and therefore :func:`_write_manifest`),
    so a deep-supervision-enabled run hashes distinctly and
    :func:`resume` rejects a config whose deep-supervision fields (or
    whose derived ``model_signature()``) differ from the checkpoint's own
    -- the existing generic hyperparams-dict/config-hash/model-signature
    comparisons in :func:`resume` already cover this; no NEW
    ``unsupported_state`` entry is needed (unlike item 20's constrained
    selector, deep supervision introduces no checkpoint-external
    best-tracking state to reconstruct -- it only changes what one
    training step's forward/loss calls look like).

22. B3 -- additive soft-DML term on top of the B2 base
    (``TrainConfig.soft_term_enabled``, default ``False`` = off)
    ------------------------------------------------------------------------
    The frozen specification supersedes an earlier draft that had proposed replacing
    item 21's hard aux terms with soft multi-scale ones. The frozen B3 design
    pins the mechanism to the governing intent -- "soft DML on the B2
    base" -- literally: item 21's hard main term AND its hard deep-sup aux
    terms are RETAINED byte-identical; this item only ADDS one new,
    full-resolution soft term on top:

    ::

        total = combo(target_mask_full, logits_full, valid_full)              # item 4/21 hard main, UNCHANGED
              + sum_i w_i * combo(maxpool(target_mask, i), aux_logits_i,
                                   minpool(valid_mask, i))                     # item 21 hard deep-sup aux, UNCHANGED, i in {2, 4}
              + soft_term_weight * soft_combo_loss(logits_full,
                                                     source_fraction_full,
                                                     valid_full)               # B3 ADDITIVE soft term, FULL-RES ONLY

    ``soft_combo_loss`` (masked soft BCE + the ``DML_2`` soft Dice
    -semimetric -- losses.py module docstring, item 5) is reused
    COMPLETELY UNMODIFIED. This is deliberately NOT edge/boundary-BCE (item
    18's ``boundary_auxiliary_loss``, a different, orthogonal mechanism
    never touched by this item) and NOT a target replacement (that is item
    17's ``soft_target``, a mutually exclusive, different experiment -- see
    below). Multi-scale soft supervision (an avg-pooled ``source_fraction``
    at each deep-sup aux scale) is explicitly OUT OF SCOPE for this item --
    deferred to a future "B3.1" -- the soft term here is full-resolution
    only, matching the formula above exactly.

    **Critical distinction from item 17 (``soft_target``).** ``soft_target
    =True`` REPLACES the training step's ``target`` tensor with
    ``source_fraction`` and is mutually exclusive with ``deep_supervision``
    (item 21's own ``__post_init__`` check). ``soft_term_enabled=True`` is a
    different, ADDITIVE mechanism: it changes nothing about what
    ``target``/``hard_loss``/the deep-sup aux terms compute -- it only adds
    one new term to the running ``loss`` sum -- and is therefore fully
    COMPATIBLE with ``deep_supervision=True`` (the B3-on-B2-base
    configuration this item exists for). ``TrainConfig.__post_init__``
    refuses ``soft_term_enabled=True`` combined with ``soft_target=True``
    (combining the additive term with the target-replacement path is a
    different, unspecified experiment -- same "refuse silent combination"
    discipline items 18/21 already established for their own neighbors).

    **Mechanism.** When ``config.soft_term_enabled`` is ``True``, the
    training step fetches ``batch["source_fraction"]`` (via the SAME
    ``include_source_fraction`` training-collate gate items 17/18 already
    built -- ``_build_train_loader``/``_build_hard_negative_train_loader``'s
    collate now requests it whenever ``config.soft_target or
    config.boundary_aux_target != "none" or config.soft_term_enabled``),
    computes ``soft_combo_loss(logits, source_fraction, valid_mask=valid)``
    at the model's full-resolution output (no pooling), and adds
    ``soft_term_weight * that value`` to the loss the hard main term (and,
    when ``deep_supervision=True``, the hard deep-sup aux terms) already
    produced. When both item 18's boundary auxiliary and this item are
    enabled together (an unspecified, untested combination this item does
    not forbid), the two mechanisms share ONE fetched-and-augmented
    ``source_fraction`` tensor rather than two independent ones -- fetching
    it twice would, under ``config.augment=True``, warp it through two
    INDEPENDENT random grids (``_augment_batch`` draws its own randomness
    per call), silently misaligning one mechanism's ``source_fraction``
    against the other's; sharing one tensor/one warp avoids that entirely.

    ``soft_term_weight`` (``beta`` in the frozen B3 design's notation) is set by
    the OUTCOME-BLIND gradient-balance probe this item ships alongside
    (``scripts/b3_grad_balance_probe.py``): ``beta* = 0.5 / r_median``,
    where ``r = |g_soft| / |g_hard|`` is measured at FIXED init on real
    train batches (``g_hard`` = grad of the full B2 total, ``g_soft`` =
    grad of the raw ``soft_combo_loss`` term, both w.r.t. the SAME shared
    decoder+head parameter subset), so that ``beta* * r_median = 0.5`` by
    construction -- accepted iff that scaled median lands in ``[0.2,
    0.8]``. This item's OWN default (``soft_term_weight=0.0``) is an inert
    placeholder, never a claimed-good operating point -- a caller MUST run
    the probe and set ``soft_term_weight`` from its printed ``beta*``
    before a real training run (this item does not, and structurally
    cannot, run that probe itself).

    **Drift monitor (also ONLINE, unlike the probe's fixed-init-only
    view).** The init-only probe cannot see the top risk the frozen B3 design
    itself names -- *late* soft-term gradient dominance emerging over the
    course of training as the hard term's own gradient magnitude decays
    while the soft term's does not. So, at every ``is_logging_step``
    (mirrors item 18's own cost-guarded pattern EXACTLY -- an extra pair of
    ``torch.autograd.grad(..., retain_graph=True)`` calls every single step
    would be needlessly expensive) AND only when ``config.soft_term_enabled``,
    this item computes ``hard_total_grad_norm``/``soft_grad_norm``: the L2
    norm of the RAW (pre-``soft_term_weight``, pre-clip) gradient of the
    pre-B3 running ``loss`` (item 4's hard main, plus item 18's boundary
    aux when also active, plus item 21's hard deep-sup aux when also
    active -- i.e. literally "everything already added to ``loss`` before
    this item's own term") and of the raw ``soft_combo_loss`` term
    respectively, BOTH restricted to the shared decoder+head parameter
    subset (:func:`_shared_decoder_head_params` -- the SAME
    ``up4``/``up3``/``up2``/``up1``/``diff_stem``/``diff_fuse``/``head``
    attribute set ``scripts/b3_grad_balance_probe.py`` uses offline, so the
    online and offline ratios are directly comparable numbers, NOT item
    18's own whole-trainable-parameter-set convention). Both
    ``torch.autograd.grad`` calls run BEFORE the real
    ``scaler.scale(loss).backward()`` (mirrors item 18's own ordering
    exactly -- ``retain_graph=True`` keeps the graph intact for the real
    backward pass that follows). ``soft_hard_grad_ratio =
    soft_grad_norm / hard_total_grad_norm`` (``None``/JSON ``null`` when
    either norm is unavailable or ``hard_total_grad_norm`` is exactly
    ``0.0`` -- never a poisoned/``inf`` value). ``"train_step"`` records
    additionally carry ``soft_term_enabled``/``soft_term_weight`` (config
    values, always), ``L_soft`` (the raw, pre-weight ``soft_combo_loss``
    scalar; ``0.0`` when this item is off, matching item 18's own
    always-present-schema convention), and
    ``soft_grad_norm``/``hard_total_grad_norm``/``soft_hard_grad_ratio``
    (``None`` except at a qualifying logging step). This is a
    TRAIN-DIAGNOSTIC log only -- it never gates, clips, or otherwise alters
    the actual optimizer step.

    **Train-only; validation invariant untouched.** ``_run_validation``
    calls ``model(left, right, diff)`` with no ``source_fraction`` and its
    own pre-existing spy assertion (``assert "source_fraction" not in
    batch``) is structurally unaffected by this item -- ``_iter_val_batches``
    always uses ``_collate_samples``'s ``include_source_fraction=False``
    default, so this item adds no new path by which ``source_fraction``
    could reach validation/checkpoint-selection/any reported metric.

    **Default-off byte-identical to B2.** ``soft_term_enabled=False`` (the
    default) means: the collate gate above is unchanged from item
    18's own condition whenever ``soft_term_enabled`` is omitted, no
    ``source_fraction`` fetch happens beyond what items 17/18 already
    trigger, ``hard_total_loss`` is computed but never used beyond being
    the same ``loss`` value the pre-item-22 code already produced, and the
    training step adds no extra term to ``loss`` -- numerically identical
    to a pre-item-22 B2 run, one training step verified via direct tensor
    comparison (this implementation's own verification).

    ``soft_term_enabled``/``soft_term_weight`` are included in
    :func:`compute_config_hash`, :func:`_hyperparams_dict`, and
    :func:`_write_manifest`, so a B3-enabled run hashes distinctly and
    :func:`resume` rejects a config whose B3 fields differ from the
    checkpoint's own -- the existing generic hyperparams-dict/config-hash
    comparisons in :func:`resume` already cover this; no NEW
    ``unsupported_state`` entry is needed (same reasoning as item 21 --
    this item introduces no checkpoint-external best-tracking state, only
    an additional loss term and an online diagnostic log).
============================================================================
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.vascutrace.data.contract import CROP_SCHEMA_VERSION
from src.vascutrace.geometry import RESEARCH_PROTOTYPE_WARNING
from src.vascutrace.ml.cache import (
    CacheSchemaError,
    CachedSampleDataset,
    cache_bundle_identities,
    cache_has_source_fraction,
    read_cache_manifest,
)
from src.vascutrace.ml.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointPayload,
    capture_rng_state,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
)
from src.vascutrace.ml.dataset import (
    DatasetConfig,
    Sample,
    SiameseCropDataset,
    frozen_validation_set,
)
from src.vascutrace.ml.losses import (
    BoundaryAuxiliaryLoss,
    boundary_auxiliary_loss,
    combo_loss,
    focal_tversky_loss,
    soft_bce_loss,
    soft_combo_loss,
)
from src.vascutrace.ml.metrics import (
    DEFAULT_SCORE_THRESHOLD,
    dice as metric_dice,
    false_positive_components,
    iou_jaccard,
    lesion_component_confusion,
    precision_recall_f_beta,
)
from src.vascutrace.ml.model import (
    ModelConfig,
    abnormality_score,
    build_model,
    dice_bce_loss,
    dice_score,
    model_signature,
)
from src.vascutrace.ml.tensor_schema import (
    CT_CHANNEL_SLICE,
    PET_CHANNEL_SLICE,
    TENSOR_SCHEMA_VERSION,
)

__all__ = [
    "RESEARCH_PROTOTYPE_WARNING",
    "TRAIN_MODULE_VERSION",
    "TrainConfig",
    "TrainConfigError",
    "CudaUnavailableError",
    "NonFiniteLossError",
    "CudaOutOfMemoryError",
    "CheckpointCompatibilityError",
    "TrainResult",
    "ValidationMetrics",
    "seed_everything",
    "worker_init_fn",
    "discover_bundle_dirs",
    "compute_split_hash",
    "compute_config_hash",
    "train",
    "resume",
]

TRAIN_MODULE_VERSION = "p6-train-v1"

# Dispatch table for TrainConfig.loss -- see module docstring, item 10.
# Every entry shares the exact (logits, target, valid_mask=...) calling
# convention (model.py's dice_bce_loss and losses.py's focal_tversky_loss/
# combo_loss all match it), so the training step never special-cases by
# loss name.
_LOSS_FUNCTIONS: dict[str, Any] = {
    "dice_bce": dice_bce_loss,
    "focal_tversky": focal_tversky_loss,
    "combo": combo_loss,
    # Soft-target-experiment losses -- see module docstring, item 17, and
    # losses.py's own module docstring, item 5. Proper on a continuous
    # (soft) target, unlike "combo" (item 1's Tversky machinery is
    # provably improper on soft targets -- losses.py module docstring,
    # item 5). Usable on a hard {0,1} target too (BCE-with-logits does not
    # special-case its input).
    "soft_bce": soft_bce_loss,
    "soft_combo": soft_combo_loss,
}
_LOSS_NAMES: frozenset[str] = frozenset(_LOSS_FUNCTIONS)
# The subset of _LOSS_NAMES this project asserts is soft-label-proper --
# see module docstring, item 17: TrainConfig.__post_init__ requires
# config.loss to be one of these whenever config.soft_target is True,
# specifically to prevent the exact failure mode this experiment exists to
# avoid (silently training combo_loss's improper Tversky machinery on a
# fractional target).
_SOFT_TARGET_COMPATIBLE_LOSS_NAMES: frozenset[str] = frozenset(
    {"soft_bce", "soft_combo"}
)

# The subset of _LOSS_NAMES whose loss function accepts a Tversky-style
# alpha (FN weight) / beta (FP weight) pair -- see module docstring, item
# 19 (Phase 4 area-loss lever). loss_fn construction, below, threads
# config.tversky_fn_weight/tversky_fp_weight into exactly these two via
# functools.partial; every other config.loss value ignores both fields.
_TVERSKY_WEIGHTED_LOSS_NAMES: frozenset[str] = frozenset({"combo", "focal_tversky"})

# Matches losses.py's own focal_tversky_loss/combo_loss alpha (FN weight) /
# beta (FP weight) defaults exactly (that module's docstring, item 1) --
# kept as an explicit constant pair here (not re-derived via
# inspect.signature) so any future change to either module's default is a
# visible, single-source-of-truth edit. loss_fn construction, below, only
# threads config.tversky_fn_weight/tversky_fp_weight through
# functools.partial when they DIFFER from these defaults -- see module
# docstring, item 19, for why: it keeps _LOSS_FUNCTIONS[config.loss] called
# with the exact original (logits, target, valid_mask=...) signature (no
# extra kwargs at all) whenever the knob is left at its default, preserving
# every pre-item-19 monkeypatched test double's assumed calling contract,
# not merely the numeric loss value.
_TVERSKY_ALPHA_DEFAULT = 0.7
_TVERSKY_BETA_DEFAULT = 0.3

# TrainConfig.boundary_aux_target options -- see module docstring, item 18
# (the boundary experiment A/B/C lineage). "none" = arm A (aux off); "hard" = arm B; -
# "fraction" = arm C.
_BOUNDARY_AUX_TARGET_NAMES: frozenset[str] = frozenset({"none", "hard", "fraction"})

# TrainConfig.deep_supervision_scales supported values -- see module
# docstring, item 21. Exactly the two decoder attach points model.py's
# SiameseBilateralUNet.forward exposes under return_aux=True
# (aux_logits_by_scale keys {2, 4}); the spec's original x8 head was
# dropped by the Root correction this item follows.
_DEEP_SUPERVISION_SUPPORTED_SCALES: frozenset[int] = frozenset({2, 4})

# TrainConfig.selection_metric / secondary_selection_metric options -- see
# module docstring, items 11 and 15. "mean_positive_iou" was added by item
# 15 specifically so config.secondary_selection_metric can track the raw
# (un-clean-rate-gated) positive IoU alongside the primary
# "dice_x_clean"-gated best.pt -- it is a valid *primary* selection_metric
# too (the validation/mapping below make no primary/secondary distinction),
# though this project's own recommended default primary policy remains
# "dice_x_clean" (item 11).
_SELECTION_METRIC_NAMES: tuple[str, ...] = (
    "blended_dice",
    "mean_positive_dice",
    "mean_positive_iou",
    "dice_x_clean",
    "det_f1_gated_dice",
)
_DEFAULT_SELECTION_METRIC = "dice_x_clean"

# TrainConfig.lr_schedule options -- see module docstring, item 12.
_LR_SCHEDULES: tuple[str, ...] = ("cosine", "none")


# ---------------------------------------------------------------------------
# Typed errors -- see module docstring, items 1-2, and the implementation's HALT
# taxonomy (no silent fallback, no silent retry).
# ---------------------------------------------------------------------------


class TrainConfigError(ValueError):
    """Raised by ``TrainConfig.__post_init__`` for a structurally invalid
    config -- see module docstring, item 1.
    """


class CudaUnavailableError(RuntimeError):
    """Raised when ``TrainConfig.device == "cuda"`` but
    ``torch.cuda.is_available()`` is ``False`` -- see module docstring,
    item 2. VascuTrace never silently trains on CPU instead.
    """


class NonFiniteLossError(RuntimeError):
    """Raised when a training-step loss is NaN/Inf. VascuTrace never
    silently continues (e.g. by skipping the step) on a broken run.
    """


class CudaOutOfMemoryError(RuntimeError):
    """Raised (message only, wrapping ``torch.OutOfMemoryError``) on a CUDA
    OOM. VascuTrace never silently retries with a smaller batch size or
    falls back to CPU.
    """


class CheckpointCompatibilityError(RuntimeError):
    """Raised by :func:`resume` when the checkpoint's schema versions,
    model signature, or split/config hash do not match the caller-supplied
    :class:`TrainConfig` -- see ``checkpoint.py``'s module docstring,
    item 4.
    """


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Frozen, validated-at-construction training run configuration. See
    module docstring, item 1.
    """

    train_bundle_dirs: tuple[Path, ...]
    val_bundle_dirs: tuple[Path, ...]
    run_root: Path

    seed: int = 0
    val_seed: int | None = None  # defaults to `seed` if unset
    batch_size: int = 2
    max_epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 5
    device: str = "cpu"
    amp: bool = False
    num_workers: int = 0
    grad_clip_norm: float | None = 1.0

    train_positive_fraction: float = 0.5
    val_positive_fraction: float = 0.5

    limit_train_batches: int | None = None
    limit_val_batches: int | None = None
    log_every_n_steps: int = 10

    # Precomputed synthetic-sample cache (see cache.py). When set, train()
    # uses CachedSampleDataset instead of the on-the-fly SiameseCropDataset
    # for that split -- see module docstring, item 9. Both, or neither,
    # must be set (a mixed cache/on-the-fly train/val pairing has no
    # leakage-safety specification of its own -- see __post_init__ below).
    train_cache_dir: Path | None = None
    val_cache_dir: Path | None = None

    # Loss selection -- see module docstring, item 10.
    loss: str = "combo"

    # Tversky FN/FP rebalance -- see module docstring, item 19 (Phase 4
    # area-loss lever). Threaded (via functools.partial at loss_fn construction
    # time, below) into "combo"/"focal_tversky" only -- every other
    # config.loss value ignores both fields. Defaults (0.7/0.3) reproduce
    # losses.py's own focal_tversky_loss/combo_loss alpha/beta defaults
    # exactly, so a config that never sets these two keys trains a
    # numerically identical loss to every pre-item-19 run.
    tversky_fn_weight: float = 0.7
    tversky_fp_weight: float = 0.3

    # Soft-target experiment -- see module docstring, item 17. False
    # (default) = the existing hard-target path (v4_big/v5exp/v6exp
    # -equivalent). True routes the TRAINING
    # step's target tensor to Sample.source_fraction (a fractional field;
    # see dataset.py's module docstring, item 9) instead of target_mask --
    # VALIDATION/checkpoint-selection/every reported metric always use
    # target_mask regardless of this flag (see __post_init__'s
    # soft-loss-compatibility check, below, and _run_validation's own spy
    # assertion).
    soft_target: bool = False

    # Positive-focused checkpoint-selection metric -- see module docstring,
    # item 11.
    selection_metric: str = _DEFAULT_SELECTION_METRIC

    # Cosine LR schedule with linear warmup -- see module docstring, item 12.
    lr_schedule: str = "cosine"
    warmup_steps: int = 0
    # Optional override for the schedule's total step count -- see module
    # docstring, item 12, "Pinning total_steps across a max_epochs-extending
    # resume". When None (the common case: resume() called with the SAME
    # max_epochs as the original run, to continue an interrupted run to its
    # original target), total_steps is auto-computed from config.max_epochs
    # and is therefore automatically resume-stable. Set this explicitly if a
    # resume call uses a DIFFERENT max_epochs than the run started with (an
    # intentional "extend the schedule" resume) and the caller wants the
    # cosine schedule's total horizon to stay fixed across that boundary.
    lr_schedule_total_steps: int | None = None

    # Train-time augmentation -- see module docstring, item 13. All defaults
    # reproduce the pre-item-13 code path (augment=False -> _augment_batch
    # is never called, zero extra RNG draws, TRAINING SPLIT ONLY -- the
    # validation path never augments regardless of this flag).
    augment: bool = False
    augment_rotation_deg: float = 10.0
    augment_translate_px: float = 6.0
    augment_scale_delta: float = 0.10
    augment_pet_gain_delta: float = 0.10
    augment_pet_bias: float = 0.05
    augment_ct_gain_delta: float = 0.10
    augment_ct_bias: float = 0.05

    # Weight EMA -- see module docstring, item 14. None (default) = off;
    # e.g. 0.999 maintains an ema_model shadow, validated/checkpointed
    # (last_ema.pt/best_ema.pt) alongside (never instead of) the raw
    # last.pt/best.pt.
    ema_decay: float | None = None

    # Secondary checkpoint-selection tracking -- see module docstring, item
    # 15. None (default) = off; one of _SELECTION_METRIC_NAMES tracks and
    # saves best_<name>.pt (and best_ema_<name>.pt when EMA is also on)
    # ALONGSIDE (never replacing) the primary selection_metric-gated
    # best.pt.
    secondary_selection_metric: str | None = None

    # Online hard-negative mining -- see module docstring, item 16. False
    # (default) = off; the baseline path skips all mining machinery when
    # this is off. Requires train_cache_dir (see
    # __post_init__) -- the mechanism needs CachedSampleDataset's stable
    # index addressing and known positive/negative split point.
    hard_negative_mining: bool = False
    hard_negative_fraction: float = 0.35
    hard_negative_oversample_weight: float = 3.0
    hard_negative_warmup_epochs: int = 1
    hard_negative_score_momentum: float = 0.5

    # Boundary-local auxiliary loss (the boundary experiment A/B/C lineage) -- see module
    # docstring, item 18. "none" (default) = arm A: aux off, byte-identical
    # to the pre-item-18 hard-combo path. lambda_boundary=0.0 (default) is
    # inert regardless of boundary_aux_target.
    lambda_boundary: float = 0.0
    boundary_aux_target: str = "none"

    # Constrained-floor IoU checkpoint selector (Track A / P4 of the "path
    # to IoU 0.70" plan) -- see module docstring, item 20. False (default)
    # = off, byte-identical to the pre-item-20 path: no extra validation
    # comparisons beyond the three range checks below, no extra checkpoint
    # writes, no extra required log events.
    constrained_iou_selection: bool = False
    constrained_iou_min_precision: float = 0.901
    constrained_iou_min_f1: float = 0.859
    constrained_iou_min_clean: float = 0.0

    # Deep supervision -- train-only multi-scale aux heads (Phase 4 lever
    # B2/L4) -- see module docstring, item 21. False (default) = off,
    # byte-identical to the pre-item-21 path: model_config.deep_supervision
    # is derived False, no aux heads exist, the training step's loss is
    # exactly loss_fn(logits, target, valid_mask=valid) with no extra term.
    deep_supervision: bool = False
    deep_supervision_scales: tuple[int, ...] = (2, 4)
    deep_supervision_weights: tuple[float, ...] = (0.5, 0.25)

    # Phase 4 lever B3 -- additive soft-DML term on top of the B2 base
    # (frozen B3 design; module docstring, item 22). False (default) = off,
    # byte-identical to the pre-item-22 path: no source_fraction fetch
    # beyond what items 17/18 already trigger, no extra loss term, no
    # extra grad-norm diagnostic. ADDITIVE (unlike item 17's soft_target,
    # which REPLACES the training target): the hard main term and, when
    # deep_supervision=True, the hard deep-sup aux terms are computed
    # exactly as before; this item only ADDS soft_term_weight *
    # soft_combo_loss(logits, source_fraction, valid_mask) on top -- see
    # __post_init__ for the soft_target mutual-exclusion check.
    # soft_term_weight (0.0 default) is an inert placeholder -- a real run
    # MUST set it from scripts/b3_grad_balance_probe.py's own printed
    # beta*, not from this default.
    soft_term_enabled: bool = False
    soft_term_weight: float = 0.0

    model_config: ModelConfig = field(default_factory=ModelConfig)
    dataset_config: DatasetConfig = field(default_factory=DatasetConfig)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "train_bundle_dirs", tuple(Path(p) for p in self.train_bundle_dirs)
        )
        object.__setattr__(
            self, "val_bundle_dirs", tuple(Path(p) for p in self.val_bundle_dirs)
        )
        object.__setattr__(self, "run_root", Path(self.run_root))
        object.__setattr__(
            self,
            "train_cache_dir",
            Path(self.train_cache_dir) if self.train_cache_dir is not None else None,
        )
        object.__setattr__(
            self,
            "val_cache_dir",
            Path(self.val_cache_dir) if self.val_cache_dir is not None else None,
        )
        if self.val_seed is None:
            object.__setattr__(self, "val_seed", self.seed)
        # Deep supervision -- module docstring, item 21. Normalize to tuples
        # (a YAML-loaded config passes plain lists) BEFORE validation/hashing
        # below, matching train_bundle_dirs/val_bundle_dirs's own
        # normalize-first pattern above -- compute_config_hash/
        # _hyperparams_dict use repr(), which must be stable regardless of
        # whether the caller passed a list or a tuple.
        object.__setattr__(
            self, "deep_supervision_scales", tuple(self.deep_supervision_scales)
        )
        object.__setattr__(
            self, "deep_supervision_weights", tuple(self.deep_supervision_weights)
        )
        # Derive model_config.deep_supervision FROM config.deep_supervision
        # (module docstring, item 21, "Model wiring") -- ModelConfig is
        # frozen, so dataclasses.replace() builds a new instance rather than
        # mutating a possibly-shared one. This runs UNCONDITIONALLY (not
        # just when True) so a caller-supplied model_config.deep_supervision
        # can never silently disagree with the top-level flag -- e.g.
        # config.deep_supervision=False always yields
        # model_config.deep_supervision=False, which is exactly what
        # T1's byte-identical-to-v6 invariant requires regardless of what
        # the caller happened to pass for model_config.
        if self.model_config.deep_supervision != self.deep_supervision:
            object.__setattr__(
                self,
                "model_config",
                replace(self.model_config, deep_supervision=self.deep_supervision),
            )

        errors: list[str] = []

        # train_bundle_dirs/val_bundle_dirs are only required when the
        # corresponding *_cache_dir is unset (an on-the-fly split needs
        # them; a cache-only split gets its bundle identities from the
        # cache's own manifest.json instead -- see compute_split_hash).
        if self.train_cache_dir is None and not self.train_bundle_dirs:
            errors.append(
                "train_bundle_dirs must be non-empty when train_cache_dir is not set"
            )
        if self.val_cache_dir is None and not self.val_bundle_dirs:
            errors.append(
                "val_bundle_dirs must be non-empty when val_cache_dir is not set"
            )
        overlap = set(self.train_bundle_dirs) & set(self.val_bundle_dirs)
        if overlap:
            errors.append(
                "train_bundle_dirs and val_bundle_dirs must not overlap "
                f"(leakage-safety); shared: {sorted(str(p) for p in overlap)}"
            )

        if (self.train_cache_dir is None) != (self.val_cache_dir is None):
            errors.append(
                "train_cache_dir and val_cache_dir must both be set or both "
                "be left None -- a mixed cache/on-the-fly pairing has no "
                "leakage-safety check of its own"
            )
        elif self.train_cache_dir is not None and self.val_cache_dir is not None:
            try:
                train_cache_bundles = cache_bundle_identities(self.train_cache_dir)
                val_cache_bundles = cache_bundle_identities(self.val_cache_dir)
            except CacheSchemaError as exc:
                errors.append(f"cache manifest error: {exc}")
            else:
                cache_overlap = train_cache_bundles & val_cache_bundles
                if cache_overlap:
                    errors.append(
                        "train_cache_dir and val_cache_dir manifests share "
                        f"bundle(s) (leakage-safety); shared count: "
                        f"{len(cache_overlap)}"
                    )

        if self.batch_size <= 0:
            errors.append(f"batch_size must be > 0, got {self.batch_size}")
        if self.max_epochs <= 0:
            errors.append(f"max_epochs must be > 0, got {self.max_epochs}")
        if not (np.isfinite(self.lr) and self.lr > 0):
            errors.append(f"lr must be finite and > 0, got {self.lr}")
        if not (np.isfinite(self.weight_decay) and self.weight_decay >= 0):
            errors.append(
                f"weight_decay must be finite and >= 0, got {self.weight_decay}"
            )
        if self.early_stop_patience <= 0:
            errors.append(
                f"early_stop_patience must be > 0, got {self.early_stop_patience}"
            )
        if self.device not in ("cpu", "cuda"):
            errors.append(f'device must be "cpu" or "cuda", got {self.device!r}')
        if self.num_workers < 0:
            errors.append(f"num_workers must be >= 0, got {self.num_workers}")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            errors.append(
                f"grad_clip_norm must be None or > 0, got {self.grad_clip_norm}"
            )
        if not (0.0 <= self.train_positive_fraction <= 1.0):
            errors.append(
                "train_positive_fraction must be in [0, 1], got "
                f"{self.train_positive_fraction}"
            )
        if not (0.0 <= self.val_positive_fraction <= 1.0):
            errors.append(
                "val_positive_fraction must be in [0, 1], got "
                f"{self.val_positive_fraction}"
            )
        if self.limit_train_batches is not None and self.limit_train_batches <= 0:
            errors.append(
                "limit_train_batches must be None or > 0, got "
                f"{self.limit_train_batches}"
            )
        if self.limit_val_batches is not None and self.limit_val_batches <= 0:
            errors.append(
                f"limit_val_batches must be None or > 0, got {self.limit_val_batches}"
            )
        if self.log_every_n_steps <= 0:
            errors.append(
                f"log_every_n_steps must be > 0, got {self.log_every_n_steps}"
            )
        if self.loss not in _LOSS_NAMES:
            errors.append(
                f"loss must be one of {sorted(_LOSS_NAMES)}, got {self.loss!r}"
            )
        # Tversky FN/FP rebalance -- see module docstring, item 19. A zero
        # or negative FN/FP weight is degenerate/ill-defined for the
        # Tversky index (losses.py module docstring, item 1), not merely an
        # unusual hyperparameter choice -- fail before allocating.
        if not (np.isfinite(self.tversky_fn_weight) and self.tversky_fn_weight > 0.0):
            errors.append(
                "tversky_fn_weight must be finite and > 0, got "
                f"{self.tversky_fn_weight}"
            )
        if not (np.isfinite(self.tversky_fp_weight) and self.tversky_fp_weight > 0.0):
            errors.append(
                "tversky_fp_weight must be finite and > 0, got "
                f"{self.tversky_fp_weight}"
            )
        # Soft-target experiment -- see module docstring, item 17. Both
        # checks are cheap, structural, fail-BEFORE-allocating checks
        # (matching this project's own "fail before allocating" discipline
        # -- module docstring, item 1) that exist specifically to prevent
        # the exact failure mode this experiment is designed to avoid:
        # silently training combo_loss's improper Tversky machinery on a
        # fractional target, or silently training soft_target=True on a
        # cache that can only supply zero-filled fallback values (cache.py
        # module docstring, item 7).
        if self.soft_target and self.loss not in _SOFT_TARGET_COMPATIBLE_LOSS_NAMES:
            errors.append(
                "soft_target=True requires a soft-label-proper loss "
                f"(one of {sorted(_SOFT_TARGET_COMPATIBLE_LOSS_NAMES)}); "
                f"got loss={self.loss!r} -- combo_loss/focal_tversky_loss's "
                "Tversky machinery is provably improper on a fractional "
                "target (losses.py module docstring, item 5) and refusing "
                "this combination is the whole point of this check"
            )
        if self.soft_target and self.train_cache_dir is None:
            errors.append(
                "soft_target=True requires train_cache_dir to be set -- "
                "on-the-fly SiameseCropDataset sampling is not wired to "
                "this experiment's soft-loader path (see module "
                "docstring, item 17)"
            )
        elif self.soft_target and self.train_cache_dir is not None:
            try:
                has_soft = cache_has_source_fraction(self.train_cache_dir)
            except CacheSchemaError as exc:
                errors.append(
                    f"soft_target=True: could not read train_cache_dir manifest: {exc}"
                )
            else:
                if not has_soft:
                    errors.append(
                        "soft_target=True requires train_cache_dir to be a "
                        "cache built with source_fraction propagation "
                        "(manifest has_source_fraction=True); got a cache "
                        "that would otherwise silently supply zero-filled "
                        "fallback values (cache.py module docstring, item "
                        f"7) at {self.train_cache_dir}"
                    )
        if self.selection_metric not in _SELECTION_METRIC_NAMES:
            errors.append(
                f"selection_metric must be one of {list(_SELECTION_METRIC_NAMES)}, "
                f"got {self.selection_metric!r}"
            )
        if self.lr_schedule not in _LR_SCHEDULES:
            errors.append(
                f"lr_schedule must be one of {list(_LR_SCHEDULES)}, "
                f"got {self.lr_schedule!r}"
            )
        if self.warmup_steps < 0:
            errors.append(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if (
            self.lr_schedule_total_steps is not None
            and self.lr_schedule_total_steps <= 0
        ):
            errors.append(
                "lr_schedule_total_steps must be None or > 0, got "
                f"{self.lr_schedule_total_steps}"
            )

        if self.augment_rotation_deg < 0:
            errors.append(
                f"augment_rotation_deg must be >= 0, got {self.augment_rotation_deg}"
            )
        if self.augment_translate_px < 0:
            errors.append(
                f"augment_translate_px must be >= 0, got {self.augment_translate_px}"
            )
        if not (0.0 <= self.augment_scale_delta < 1.0):
            errors.append(
                f"augment_scale_delta must be in [0, 1), got {self.augment_scale_delta}"
            )
        if not (0.0 <= self.augment_pet_gain_delta < 1.0):
            errors.append(
                "augment_pet_gain_delta must be in [0, 1), got "
                f"{self.augment_pet_gain_delta}"
            )
        if self.augment_pet_bias < 0:
            errors.append(f"augment_pet_bias must be >= 0, got {self.augment_pet_bias}")
        if not (0.0 <= self.augment_ct_gain_delta < 1.0):
            errors.append(
                "augment_ct_gain_delta must be in [0, 1), got "
                f"{self.augment_ct_gain_delta}"
            )
        if self.augment_ct_bias < 0:
            errors.append(f"augment_ct_bias must be >= 0, got {self.augment_ct_bias}")
        if self.ema_decay is not None and not (0.0 < self.ema_decay < 1.0):
            errors.append(f"ema_decay must be None or in (0, 1), got {self.ema_decay}")
        if (
            self.secondary_selection_metric is not None
            and self.secondary_selection_metric not in _SELECTION_METRIC_NAMES
        ):
            errors.append(
                "secondary_selection_metric must be None or one of "
                f"{list(_SELECTION_METRIC_NAMES)}, got "
                f"{self.secondary_selection_metric!r}"
            )

        # Online hard-negative mining -- see module docstring, item 16.
        if self.hard_negative_mining and self.train_cache_dir is None:
            errors.append(
                "hard_negative_mining requires train_cache_dir to be set -- "
                "the mechanism needs CachedSampleDataset's stable index "
                "addressing and known positive/negative split point "
                "(manifest.total_positive); see module docstring, item 16"
            )
        if not (0.0 < self.hard_negative_fraction <= 1.0):
            errors.append(
                "hard_negative_fraction must be in (0, 1], got "
                f"{self.hard_negative_fraction}"
            )
        if not (
            np.isfinite(self.hard_negative_oversample_weight)
            and self.hard_negative_oversample_weight >= 1.0
        ):
            errors.append(
                "hard_negative_oversample_weight must be finite and >= 1.0, "
                f"got {self.hard_negative_oversample_weight}"
            )
        if self.hard_negative_warmup_epochs < 0:
            errors.append(
                "hard_negative_warmup_epochs must be >= 0, got "
                f"{self.hard_negative_warmup_epochs}"
            )
        if not (0.0 < self.hard_negative_score_momentum < 1.0):
            errors.append(
                "hard_negative_score_momentum must be in (0, 1), got "
                f"{self.hard_negative_score_momentum}"
            )

        # Boundary-local auxiliary loss (the boundary experiment A/B/C lineage) -- see
        # module docstring, item 18. Same fail-BEFORE-allocating discipline
        # item 17 already established for the closely related soft-target
        # experiment.
        if self.boundary_aux_target not in _BOUNDARY_AUX_TARGET_NAMES:
            errors.append(
                "boundary_aux_target must be one of "
                f"{sorted(_BOUNDARY_AUX_TARGET_NAMES)}, got "
                f"{self.boundary_aux_target!r}"
            )
        if not (np.isfinite(self.lambda_boundary) and self.lambda_boundary >= 0.0):
            errors.append(
                f"lambda_boundary must be finite and >= 0, got {self.lambda_boundary}"
            )
        if self.boundary_aux_target != "none":
            # Mutual exclusion with item 17 -- see module docstring, item
            # 18, "Mutual exclusion with item 17": the boundary experiment Sec 5's L_hard is
            # literally the shipped combo_loss(logits, target_mask,
            # valid_mask); soft_target=True would silently change what the
            # training step's own "target" variable is.
            if self.soft_target:
                errors.append(
                    "boundary_aux_target != 'none' requires soft_target=False "
                    "-- the boundary experiment Sec 5's L_hard term is the shipped hard-mask "
                    "combo_loss(logits, target_mask, valid_mask); combining "
                    "this auxiliary with the soft-target training path "
                    "(item 17) is a different, unspecified experiment"
                )
            if self.train_cache_dir is None:
                errors.append(
                    "boundary_aux_target != 'none' requires train_cache_dir "
                    "to be set -- the auxiliary needs Sample.source_fraction, "
                    "which only the cache-backed training loader can supply "
                    "(see module docstring, item 18)"
                )
            else:
                try:
                    has_soft = cache_has_source_fraction(self.train_cache_dir)
                except CacheSchemaError as exc:
                    errors.append(
                        "boundary_aux_target != 'none': could not read "
                        f"train_cache_dir manifest: {exc}"
                    )
                else:
                    if not has_soft:
                        errors.append(
                            "boundary_aux_target != 'none' requires "
                            "train_cache_dir to be a cache built with "
                            "source_fraction propagation (manifest "
                            "has_source_fraction=True); got a cache that "
                            "would otherwise silently supply zero-filled "
                            "fallback values (cache.py module docstring, "
                            f"item 7) at {self.train_cache_dir}"
                        )

        # Constrained-floor IoU checkpoint selector -- see module
        # docstring, item 20 (Track A / P4). Same fail-BEFORE-allocating
        # discipline as every other config-gated mechanism above; a floor
        # outside [0, 1] is meaningless for a precision/F1/clean-rate
        # comparison.
        if not (
            np.isfinite(self.constrained_iou_min_precision)
            and 0.0 <= self.constrained_iou_min_precision <= 1.0
        ):
            errors.append(
                "constrained_iou_min_precision must be finite and in "
                f"[0, 1], got {self.constrained_iou_min_precision}"
            )
        if not (
            np.isfinite(self.constrained_iou_min_f1)
            and 0.0 <= self.constrained_iou_min_f1 <= 1.0
        ):
            errors.append(
                "constrained_iou_min_f1 must be finite and in [0, 1], got "
                f"{self.constrained_iou_min_f1}"
            )
        if not (
            np.isfinite(self.constrained_iou_min_clean)
            and 0.0 <= self.constrained_iou_min_clean <= 1.0
        ):
            errors.append(
                "constrained_iou_min_clean must be finite and in [0, 1], "
                f"got {self.constrained_iou_min_clean}"
            )

        # Deep supervision -- module docstring, item 21. Same
        # fail-BEFORE-allocating discipline as every other config-gated
        # mechanism above. Scales/weights are validated even when
        # deep_supervision=False (a caller who sets a bad scale/weight pair
        # but forgets to also flip the flag should still get a clear error,
        # not a silently-inert misconfiguration).
        if len(self.deep_supervision_scales) != len(self.deep_supervision_weights):
            errors.append(
                "deep_supervision_scales and deep_supervision_weights must "
                f"have equal length, got {len(self.deep_supervision_scales)} "
                f"scales and {len(self.deep_supervision_weights)} weights"
            )
        unsupported_scales = (
            set(self.deep_supervision_scales) - _DEEP_SUPERVISION_SUPPORTED_SCALES
        )
        if unsupported_scales:
            errors.append(
                "deep_supervision_scales must be a subset of "
                f"{sorted(_DEEP_SUPERVISION_SUPPORTED_SCALES)}, got "
                f"unsupported value(s) {sorted(unsupported_scales)}"
            )
        for weight in self.deep_supervision_weights:
            if not (np.isfinite(weight) and weight > 0.0):
                errors.append(
                    "deep_supervision_weights entries must be finite and > "
                    f"0, got {self.deep_supervision_weights!r}"
                )
                break
        if self.deep_supervision and not self.deep_supervision_scales:
            errors.append(
                "deep_supervision=True requires a non-empty deep_supervision_scales"
            )
        if self.deep_supervision and self.soft_target:
            errors.append(
                "deep_supervision=True requires soft_target=False -- the "
                "aux terms max-pool-downsample the HARD target_mask (module "
                "docstring, item 21); combining deep supervision with the "
                "soft-target training path (item 17) is a different, "
                "unspecified experiment"
            )

        # Phase 4 lever B3 -- additive soft-DML term (frozen B3 design; module
        # docstring, item 22). Same fail-before-allocating discipline as
        # every other config-gated mechanism above.
        if not (np.isfinite(self.soft_term_weight) and self.soft_term_weight >= 0.0):
            errors.append(
                f"soft_term_weight must be finite and >= 0, got {self.soft_term_weight}"
            )
        if self.soft_term_enabled and self.soft_target:
            errors.append(
                "soft_term_enabled=True requires soft_target=False -- "
                "soft_target REPLACES the training target (item 17); "
                "soft_term_enabled ADDS a full-resolution soft term on top "
                "of the existing hard target (item 22, B3 frozen B3 design); "
                "combining the two is a different, unspecified experiment"
            )
        if self.soft_term_enabled and self.train_cache_dir is None:
            errors.append(
                "soft_term_enabled=True requires train_cache_dir to be set "
                "-- the additive soft term needs Sample.source_fraction, "
                "which only the cache-backed training loader can supply "
                "(see module docstring, item 22)"
            )
        elif self.soft_term_enabled and self.train_cache_dir is not None:
            try:
                has_soft = cache_has_source_fraction(self.train_cache_dir)
            except CacheSchemaError as exc:
                errors.append(
                    "soft_term_enabled=True: could not read train_cache_dir "
                    f"manifest: {exc}"
                )
            else:
                if not has_soft:
                    errors.append(
                        "soft_term_enabled=True requires train_cache_dir to "
                        "be a cache built with source_fraction propagation "
                        "(manifest has_source_fraction=True); got a cache "
                        "that would otherwise silently supply zero-filled "
                        "fallback values (cache.py module docstring, item "
                        f"7) at {self.train_cache_dir}"
                    )

        if errors:
            raise TrainConfigError("; ".join(errors))


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Summary of one :func:`train`/:func:`resume` call."""

    final_epoch: int
    global_step: int
    best_epoch: int | None
    best_val_metric: float | None
    last_checkpoint_path: Path
    best_checkpoint_path: Path | None
    stopped_early: bool
    metrics_log_path: Path
    manifest_path: Path

    # Secondary selection tracking -- see module docstring, item 15. All
    # None when config.secondary_selection_metric is None (the default).
    best_secondary_metric_name: str | None = None
    best_secondary_epoch: int | None = None
    best_secondary_val_metric: float | None = None
    best_secondary_checkpoint_path: Path | None = None

    # Weight EMA -- see module docstring, item 14. ema_enabled mirrors
    # config.ema_decay is not None; the rest are None when EMA is off.
    ema_enabled: bool = False
    best_ema_epoch: int | None = None
    best_ema_val_metric: float | None = None
    best_ema_checkpoint_path: Path | None = None
    best_ema_secondary_epoch: int | None = None
    best_ema_secondary_val_metric: float | None = None
    best_ema_secondary_checkpoint_path: Path | None = None

    # Constrained-floor IoU checkpoint selector -- see module docstring,
    # item 20. All None when config.constrained_iou_selection is False
    # (the default) or when it was True but no epoch ever cleared the
    # legality floors (see item 20's "qualified": False summary event).
    best_constrained_iou_epoch: int | None = None
    best_constrained_iou_val_metric: float | None = None
    best_constrained_iou_checkpoint_path: Path | None = None
    best_ema_constrained_iou_epoch: int | None = None
    best_ema_constrained_iou_val_metric: float | None = None
    best_ema_constrained_iou_checkpoint_path: Path | None = None


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed every global RNG stream (python ``random``, numpy legacy
    global, torch CPU, torch CUDA) and set cuDNN to its deterministic
    -friendly mode. Deliberately conservative (disables cuDNN's
    autotuning/``benchmark`` mode) -- a reproducible baseline before
    optimizing for raw speed (Goodfellow, Bengio & Courville, 2016, Ch. 11
    "Practical Methodology": debug by first getting a reproducible,
    understood baseline).
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """``DataLoader(worker_init_fn=...)``: re-seed each worker process's
    ``random``/``numpy`` global state from torch's own per-worker
    ``initial_seed()`` (already derived deterministically by torch from the
    main process's base seed + ``worker_id``), so worker-local library code
    that happens to touch the legacy global RNGs is deterministic too. Note
    ``dataset.py``'s own sample generation never depends on this -- every
    :class:`~src.vascutrace.ml.dataset.Sample` is already a pure function
    of its index via a per-sample ``numpy.random.default_rng`` seed, so
    this hook exists for defense-in-depth, not because the dataset needs
    it.
    """
    import random

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Bundle discovery (metadata-only filesystem globbing; never reads array
# contents -- see ``cli.py``'s ``doctor``/``dry-run`` for callers).
# ---------------------------------------------------------------------------


def discover_bundle_dirs(data_root: Path) -> tuple[Path, ...]:
    """Every ``<data_root>/*/*/`` directory that looks like a saved
    :class:`~src.vascutrace.data.contract.CropBundle` (has both
    ``bundle.json`` and ``bundle.npz``), sorted for determinism. Metadata
    -only filesystem globbing -- this never opens or reads array contents,
    so it does not violate "consume P2 bundles via the dataset, don't read
    Data/ directly" (``data/processed/p2/crops/...`` is this project's own
    P2-produced, gitignored processed directory, not raw ``Data/``).
    """
    data_root = Path(data_root)
    if not data_root.is_dir():
        return ()
    found = [
        candidate
        for candidate in sorted(data_root.glob("*/*"))
        if (candidate / "bundle.json").is_file()
        and (candidate / "bundle.npz").is_file()
    ]
    return tuple(found)


# ---------------------------------------------------------------------------
# Hashing -- "hash the split subject lists + config", never store raw.
# See checkpoint.py's module docstring, item 3.
# ---------------------------------------------------------------------------


def _bundle_identity(bundle_dir: Path) -> str:
    """The trailing ``<subject>/<session>`` path components only (matches
    ``data.contract.bundle_directory``'s own layout) -- NOT the full
    filesystem path, which could vary across machines.
    """
    parts = Path(bundle_dir).parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else str(bundle_dir)


def _hash_repr(payload: Any) -> str:
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()[:16]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_training_identity(cache_dir: Path) -> tuple[Any, ...]:
    """Non-reversible input to the cache-aware training split digest.

    Bundle identities alone cannot distinguish two generated caches made
    from the same subjects with different simulation configuration, seed,
    counts, soft-target support, or persisted sample inventory. Bind all
    trajectory-relevant manifest fields and every persisted sample's exact
    byte digest. Raw identifiers are used only inside the one-way digest and
    are never logged.
    """
    cache_dir = Path(cache_dir)
    manifest = read_cache_manifest(cache_dir)
    per_bundle_counts = tuple(
        sorted(
            (
                identity,
                counts["positive"],
                counts["negative"],
            )
            for identity, counts in manifest["per_bundle_counts"].items()
        )
    )
    sample_inventory = tuple(
        (path.name, _sha256_file(path))
        for path in sorted(cache_dir.glob("sample_*.npz"))
    )
    return (
        "cache",
        manifest["cache_schema_version"],
        manifest["tensor_schema_version"],
        manifest["crop_schema_version"],
        manifest["dataset_builder_version"],
        manifest["config_hash"],
        manifest["seed"],
        tuple(sorted(manifest["bundle_identities"])),
        tuple(sorted(manifest["excluded_qc_bundle_identities"])),
        per_bundle_counts,
        manifest["total_positive"],
        manifest["total_negative"],
        manifest["total_samples"],
        bool(manifest.get("has_source_fraction", False)),
        sample_inventory,
    )


def compute_split_hash(config: TrainConfig) -> str:
    """One-way digest of the exact train/validation data identities. Never
    reversible to the raw subject list from the hash alone.

    Cache-aware: when ``config.train_cache_dir``/``config.val_cache_dir``
    are set, the bundle identities come from that cache's own
    ``manifest.json`` and validated file inventory rather than only the
    bundle identities. This distinguishes caches generated from the same
    subjects with different seeds, simulator configuration, counts, or
    soft-target support and lets :func:`resume` reject cache substitution.
    """
    if config.train_cache_dir is not None:
        train_identity = _cache_training_identity(config.train_cache_dir)
    else:
        train_identity = (
            "bundles",
            tuple(sorted(_bundle_identity(d) for d in config.train_bundle_dirs)),
        )
    if config.val_cache_dir is not None:
        val_identity = _cache_training_identity(config.val_cache_dir)
    else:
        val_identity = (
            "bundles",
            tuple(sorted(_bundle_identity(d) for d in config.val_bundle_dirs)),
        )
    return _hash_repr((train_identity, val_identity))


def compute_config_hash(config: TrainConfig) -> str:
    """Digest of the architecture/training hyperparameters that determine
    whether a checkpoint is compatible with a resume config. Deliberately
    excludes ``run_root``/``device`` (a local path and a runtime placement
    choice, not a training-trajectory-defining hyperparameter -- a CPU
    -started smoke run may be legitimately resumed on a CUDA machine or
    vice versa without that alone being flagged incompatible).
    """
    fields = (
        config.batch_size,
        config.max_epochs,
        config.lr,
        config.weight_decay,
        config.early_stop_patience,
        config.amp,
        config.num_workers,
        config.grad_clip_norm,
        config.train_positive_fraction,
        config.val_positive_fraction,
        config.seed,
        config.val_seed,
        config.loss,
        config.tversky_fn_weight,
        config.tversky_fp_weight,
        config.soft_target,
        config.selection_metric,
        config.lr_schedule,
        config.warmup_steps,
        config.augment,
        config.augment_rotation_deg,
        config.augment_translate_px,
        config.augment_scale_delta,
        config.augment_pet_gain_delta,
        config.augment_pet_bias,
        config.augment_ct_gain_delta,
        config.augment_ct_bias,
        config.ema_decay,
        config.secondary_selection_metric,
        config.hard_negative_mining,
        config.hard_negative_fraction,
        config.hard_negative_oversample_weight,
        config.hard_negative_warmup_epochs,
        config.hard_negative_score_momentum,
        config.lambda_boundary,
        config.boundary_aux_target,
        config.constrained_iou_selection,
        config.constrained_iou_min_precision,
        config.constrained_iou_min_f1,
        config.constrained_iou_min_clean,
        config.deep_supervision,
        config.deep_supervision_scales,
        config.deep_supervision_weights,
        config.soft_term_enabled,
        config.soft_term_weight,
        model_signature(config.model_config),
        repr(config.dataset_config),
    )
    return _hash_repr(fields)


def _hyperparams_dict(config: TrainConfig) -> dict[str, Any]:
    """Non-identifying hyperparameters only -- no bundle dirs, no
    ``run_root`` (a local path) -- see checkpoint.py's module docstring,
    item 3.
    """
    return {
        "batch_size": config.batch_size,
        "max_epochs": config.max_epochs,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "early_stop_patience": config.early_stop_patience,
        "amp": config.amp,
        "num_workers": config.num_workers,
        "grad_clip_norm": config.grad_clip_norm,
        "train_positive_fraction": config.train_positive_fraction,
        "val_positive_fraction": config.val_positive_fraction,
        "seed": config.seed,
        "val_seed": config.val_seed,
        "limit_train_batches": config.limit_train_batches,
        "limit_val_batches": config.limit_val_batches,
        "log_every_n_steps": config.log_every_n_steps,
        "loss": config.loss,
        "tversky_fn_weight": config.tversky_fn_weight,
        "tversky_fp_weight": config.tversky_fp_weight,
        "soft_target": config.soft_target,
        "selection_metric": config.selection_metric,
        "lr_schedule": config.lr_schedule,
        "warmup_steps": config.warmup_steps,
        "lr_schedule_total_steps": config.lr_schedule_total_steps,
        "augment": config.augment,
        "augment_rotation_deg": config.augment_rotation_deg,
        "augment_translate_px": config.augment_translate_px,
        "augment_scale_delta": config.augment_scale_delta,
        "augment_pet_gain_delta": config.augment_pet_gain_delta,
        "augment_pet_bias": config.augment_pet_bias,
        "augment_ct_gain_delta": config.augment_ct_gain_delta,
        "augment_ct_bias": config.augment_ct_bias,
        "ema_decay": config.ema_decay,
        "secondary_selection_metric": config.secondary_selection_metric,
        "hard_negative_mining": config.hard_negative_mining,
        "hard_negative_fraction": config.hard_negative_fraction,
        "hard_negative_oversample_weight": config.hard_negative_oversample_weight,
        "hard_negative_warmup_epochs": config.hard_negative_warmup_epochs,
        "hard_negative_score_momentum": config.hard_negative_score_momentum,
        "lambda_boundary": config.lambda_boundary,
        "boundary_aux_target": config.boundary_aux_target,
        "constrained_iou_selection": config.constrained_iou_selection,
        "constrained_iou_min_precision": config.constrained_iou_min_precision,
        "constrained_iou_min_f1": config.constrained_iou_min_f1,
        "constrained_iou_min_clean": config.constrained_iou_min_clean,
        "deep_supervision": config.deep_supervision,
        "deep_supervision_scales": config.deep_supervision_scales,
        "deep_supervision_weights": config.deep_supervision_weights,
        "soft_term_enabled": config.soft_term_enabled,
        "soft_term_weight": config.soft_term_weight,
    }


# ---------------------------------------------------------------------------
# Batching (a small custom collate -- Sample.meta is a plain dict the
# default collate can't stack).
# ---------------------------------------------------------------------------


def _collate_samples(
    samples: list[Sample], *, include_source_fraction: bool = False
) -> dict[str, Any]:
    """``include_source_fraction`` (module docstring, item 17) defaults
    ``False`` -- the baseline batch dict shape is unchanged bit-for-bit.
    :func:`_iter_val_batches` (validation) NEVER passes ``True``, so
    ``"source_fraction"`` structurally cannot appear in a validation batch
    -- the spy property module docstring item 17 promises is enforced HERE,
    not merely by convention at each call site.
    """
    batch: dict[str, Any] = {
        "left_view": torch.stack([s.left_view for s in samples]),
        "right_view": torch.stack([s.right_view for s in samples]),
        "pet_diff": torch.stack([s.pet_diff for s in samples]),
        "target_mask": torch.stack([s.target_mask for s in samples]),
        "valid_mask": torch.stack([s.valid_mask for s in samples]),
        "raw_pet": torch.stack([s.raw_pet for s in samples]),
        "meta": [s.meta for s in samples],
    }
    if include_source_fraction:
        batch["source_fraction"] = torch.stack([s.source_fraction for s in samples])
    return batch


def _iter_val_batches(samples: list[Sample] | CachedSampleDataset, batch_size: int):
    """Batch an indexable, ``len()``-able collection of :class:`Sample`.
    Deliberately uses explicit index access (``samples[j]``), not Python
    slicing (``samples[i:i+n]``) -- a plain ``list[Sample]``
    (``frozen_validation_set``'s eager output) supports slicing, but
    :class:`~src.vascutrace.ml.cache.CachedSampleDataset` (a
    ``torch.utils.data.Dataset``) only supports single-index
    ``__getitem__``, so this must work for both uniformly.
    """
    n = len(samples)
    for i in range(0, n, batch_size):
        batch = [samples[j] for j in range(i, min(i + batch_size, n))]
        yield _collate_samples(batch)


def _build_train_loader(
    dataset: SiameseCropDataset | CachedSampleDataset,
    config: TrainConfig,
    generator: torch.Generator,
) -> DataLoader:
    # module docstring, items 17/18/22: only the TRAINING loader ever
    # requests source_fraction, and only when soft_target=True (item 17) OR
    # boundary_aux_target != "none" (item 18) OR soft_term_enabled=True
    # (item 22, B3's additive term -- COMPATIBLE with either of the other
    # two conditions, unlike soft_target/boundary_aux_target's own mutual
    # exclusion; any condition alone is sufficient here).
    # functools.partial of a module-level function stays picklable
    # (required for num_workers > 0 -- see dataset.py's own module
    # docstring, item 7, on picklability).
    collate = (
        functools.partial(_collate_samples, include_source_fraction=True)
        if (
            config.soft_target
            or config.boundary_aux_target != "none"
            or config.soft_term_enabled
        )
        else _collate_samples
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        worker_init_fn=worker_init_fn if config.num_workers > 0 else None,
        collate_fn=collate,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Train-time augmentation -- config-gated (module docstring, item 13).
# TRAINING SPLIT ONLY; never called from _run_validation.
# ---------------------------------------------------------------------------


def _uniform(n: int, low: float, high: float, device: torch.device) -> torch.Tensor:
    """``n`` iid draws from ``U(low, high)`` on ``device`` from the
    already-seeded global torch RNG stream (CPU stream if ``device`` is
    CPU, CUDA stream if ``device`` is CUDA -- both seeded together by
    :func:`seed_everything`). ``low == high`` (a caller-configured zero
    -width augmentation range, e.g. ``augment_translate_px=0``) is handled
    explicitly rather than calling ``torch.Tensor.uniform_`` with equal
    bounds, which is not guaranteed well-defined.
    """
    if low >= high:
        return torch.full((n,), low, device=device, dtype=torch.float32)
    return torch.empty(n, device=device, dtype=torch.float32).uniform_(low, high)


def _affine_theta(
    angle_deg: torch.Tensor,
    scale: torch.Tensor,
    tx_norm: torch.Tensor,
    ty_norm: torch.Tensor,
) -> torch.Tensor:
    """Batched ``[B, 2, 3]`` ``theta`` for ``torch.nn.functional.
    affine_grid``/``grid_sample`` -- see module docstring, item 13. Builds
    the INVERSE of the forward "rotate by ``angle_deg``, scale by
    ``scale``, then translate by ``(tx_norm, ty_norm)``" transform
    analytically (rotation-plus-uniform-scale is a scaled orthogonal 2x2
    matrix ``A = scale * R(angle)``, so ``A^-1 = (1/scale) * R(-angle)`` --
    no per-sample matrix solve needed), because ``affine_grid``/
    ``grid_sample``'s own documented convention is that ``theta`` maps
    OUTPUT normalized coordinates to the INPUT coordinates to sample from
    (verified against this project's torch 2.9.1 docs via WebFetch, not
    recalled: https://docs.pytorch.org/docs/2.9/generated/torch.nn.
    functional.affine_grid.html). Deliberately no flip term -- see module
    docstring, item 13, for why a horizontal flip is excluded from this
    augmentation set entirely.
    """
    angle_rad = angle_deg * (math.pi / 180.0)
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)
    inv_scale = 1.0 / scale
    r00 = inv_scale * cos_a
    r01 = inv_scale * sin_a
    r10 = -inv_scale * sin_a
    r11 = inv_scale * cos_a
    tx_out = -(r00 * tx_norm + r01 * ty_norm)
    ty_out = -(r10 * tx_norm + r11 * ty_norm)
    return torch.stack(
        [
            torch.stack([r00, r01, tx_out], dim=-1),
            torch.stack([r10, r11, ty_out], dim=-1),
        ],
        dim=-2,
    )


@torch.no_grad()
def _augment_batch(
    left: torch.Tensor,
    right: torch.Tensor,
    diff: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    config: TrainConfig,
    target_interp: str = "nearest",
    source_fraction: torch.Tensor | None = None,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
):
    """One shared random affine (rotation + isotropic scale + translation,
    no flip) applied to all five already-device tensors TOGETHER, plus a
    PET/CT intensity gain+bias jitter -- see module docstring, item 13, for
    the full derivation (including why ``pet_diff`` needs only the shared
    ``pet_gain`` re-applied, not an independent jitter). Caller (the
    training step in :func:`_execute`) only invokes this when
    ``config.augment`` is ``True``; :func:`_run_validation` never does.

    ``target_interp`` (module docstring, item 17) defaults ``"nearest"``
    -- item 13's ORIGINAL rationale ("a discrete 0/1 ground-truth mask
    warped bilinearly would invent fractional 'soft' boundary pixels the
    label never had") is exactly backwards when ``target`` is instead
    ``source_fraction`` (``config.soft_target=True``): that field is
    ALREADY a genuinely continuous field (dataset.py module docstring,
    item 9), so bilinear resampling is the standard, correct choice there
    -- nearest-neighbor would instead needlessly discard its sub-pixel
    smoothness. The training step (:func:`_execute`) passes
    ``target_interp="bilinear"`` only when ``config.soft_target`` is
    ``True``; every existing hard-target caller is unaffected (default
    unchanged).

    ``source_fraction`` (module docstring, item 18) is an OPTIONAL sixth
    tensor, ``None`` by default -- every existing caller/test that omits it
    gets the ORIGINAL 5-tuple return, byte-identical (same RNG draws, same
    grid, same everything; a ``None`` keyword consumes no extra randomness).
    When the item-18 boundary-auxiliary training step supplies it (only
    when ``config.boundary_aux_target != "none"``), the SAME shared
    ``grid`` this call already computes for ``target``/``valid`` also warps
    ``source_fraction`` with bilinear interpolation (it is already a
    genuinely continuous field -- the same rationale ``target_interp=
    "bilinear"`` uses above), and the function returns a 6-tuple with the
    warped ``source_fraction`` appended. This exists because, before this
    item, augmenting the input crop/hard target while leaving the boundary
    auxiliary's own ``source_fraction`` map un-warped would silently
    misalign the auxiliary's support/target against the augmented logits
    every augmented step -- a correctness gap the boundary experiment Sec 5's own text does
    not explicitly address, closed here rather than left implicit.
    """
    device = left.device
    batch_size = left.shape[0]
    h, w = left.shape[-2], left.shape[-1]

    angle_deg = _uniform(
        batch_size, -config.augment_rotation_deg, config.augment_rotation_deg, device
    )
    scale = _uniform(
        batch_size,
        1.0 - config.augment_scale_delta,
        1.0 + config.augment_scale_delta,
        device,
    )
    tx_px = _uniform(
        batch_size, -config.augment_translate_px, config.augment_translate_px, device
    )
    ty_px = _uniform(
        batch_size, -config.augment_translate_px, config.augment_translate_px, device
    )
    tx_norm = tx_px / (w / 2.0)
    ty_norm = ty_px / (h / 2.0)

    theta = _affine_theta(angle_deg, scale, tx_norm, ty_norm)
    grid = F.affine_grid(theta, size=(batch_size, 1, h, w), align_corners=False)

    left_g = F.grid_sample(
        left, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    right_g = F.grid_sample(
        right, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    # Nearest interpolation for a discrete 0/1 ground-truth mask (module
    # docstring, item 13) -- bilinear for the continuous source_fraction
    # target (item 17); see target_interp's own docstring above. valid_mask
    # is always nearest (it is always a {0,1}-ish mask regardless of
    # config.soft_target).
    target_g = F.grid_sample(
        target, grid, mode=target_interp, padding_mode="zeros", align_corners=False
    )
    valid_g = F.grid_sample(
        valid, grid, mode="nearest", padding_mode="zeros", align_corners=False
    )

    pet_gain = _uniform(
        batch_size,
        1.0 - config.augment_pet_gain_delta,
        1.0 + config.augment_pet_gain_delta,
        device,
    ).view(-1, 1, 1, 1)
    pet_bias = _uniform(
        batch_size, -config.augment_pet_bias, config.augment_pet_bias, device
    ).view(-1, 1, 1, 1)
    ct_gain = _uniform(
        batch_size,
        1.0 - config.augment_ct_gain_delta,
        1.0 + config.augment_ct_gain_delta,
        device,
    ).view(-1, 1, 1, 1)
    ct_bias = _uniform(
        batch_size, -config.augment_ct_bias, config.augment_ct_bias, device
    ).view(-1, 1, 1, 1)

    left_g[:, PET_CHANNEL_SLICE] = left_g[:, PET_CHANNEL_SLICE] * pet_gain + pet_bias
    right_g[:, PET_CHANNEL_SLICE] = right_g[:, PET_CHANNEL_SLICE] * pet_gain + pet_bias
    left_g[:, CT_CHANNEL_SLICE] = left_g[:, CT_CHANNEL_SLICE] * ct_gain + ct_bias
    right_g[:, CT_CHANNEL_SLICE] = right_g[:, CT_CHANNEL_SLICE] * ct_gain + ct_bias
    # Safety clamp: bounds the jittered tensors near their pre-jitter
    # network-normalized ranges (PET clip(0,10)/10 -> [0, 1]; CT/pet_diff
    # clip(-1000,1000)/1000 and left_pet-right_pet -> [-1, 1] --
    # tensor_schema.py's own formulas) so a worst-case sampled gain/bias
    # combination cannot hand the network an unbounded input. Does not
    # touch target_mask/valid_mask (no intensity jitter applied to either;
    # already exactly 0/1 after the nearest-interpolation geometric warp).
    left_g[:, PET_CHANNEL_SLICE].clamp_(0.0, 1.5)
    right_g[:, PET_CHANNEL_SLICE].clamp_(0.0, 1.5)
    left_g[:, CT_CHANNEL_SLICE].clamp_(-1.5, 1.5)
    right_g[:, CT_CHANNEL_SLICE].clamp_(-1.5, 1.5)
    # ``pet_diff`` is a derived input, so recompute it after the view-wise
    # clamps. The shared bias cancels only before clipping; independently
    # clamping the two views can otherwise break the declared identity at the
    # normalized PET boundary.
    diff_g = left_g[:, PET_CHANNEL_SLICE] - right_g[:, PET_CHANNEL_SLICE]

    if source_fraction is None:
        return left_g, right_g, diff_g, target_g, valid_g

    # module docstring, item 18 -- SAME grid, bilinear (continuous field).
    source_fraction_g = F.grid_sample(
        source_fraction,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return left_g, right_g, diff_g, target_g, valid_g, source_fraction_g


# ---------------------------------------------------------------------------
# Weight EMA -- config-gated (module docstring, item 14).
# ---------------------------------------------------------------------------


@torch.no_grad()
def _update_ema(
    ema_model: torch.nn.Module, model: torch.nn.Module, decay: float
) -> None:
    """In-place ``ema <- decay*ema + (1-decay)*current`` for every tensor in
    ``state_dict()`` -- see module docstring, item 14, for why uniform
    EMA-averaging over the WHOLE state dict (no buffer/parameter
    special-casing) is exact for this architecture (GroupNorm only, no
    BatchNorm running-mean/-var buffer). ``state_dict()`` tensors share
    storage with the live module's own parameters/buffers, so this in
    -place update mutates ``ema_model`` directly -- the standard basis for
    ``torch.optim.swa_utils.AveragedModel``'s own update rule.
    """
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()
    for key, ema_tensor in ema_state.items():
        ema_tensor.mul_(decay).add_(model_state[key], alpha=1.0 - decay)


# ---------------------------------------------------------------------------
# Online hard-negative mining -- config-gated, see module docstring, item
# 16. TRAINING SPLIT ONLY; requires CachedSampleDataset (train_cache_dir)
# for stable index addressing and a known positive/negative split point.
# None of this is constructed/called when config.hard_negative_mining is
# False -- the baseline train_loader/collate path is untouched.
# ---------------------------------------------------------------------------


class _IndexedDataset(torch.utils.data.Dataset):
    """Wraps ``base`` so ``__getitem__`` also returns the ORIGINAL dataset
    index alongside the :class:`~src.vascutrace.ml.dataset.Sample` -- see
    module docstring, item 16. Only constructed when
    ``config.hard_negative_mining`` is ``True``.
    """

    def __init__(self, base: CachedSampleDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[int, Sample]:
        return index, self.base[index]


def _collate_indexed_samples(
    items: list[tuple[int, Sample]], *, include_source_fraction: bool = False
) -> dict[str, Any]:
    """Same as :func:`_collate_samples`, plus a ``"_dataset_index"`` long
    tensor (``[B]``) carrying each sample's original
    :class:`_IndexedDataset` index through the batch -- see module
    docstring, item 16. ``include_source_fraction`` forwards to
    :func:`_collate_samples` unchanged -- see module docstring, item 17
    (hard-negative-mining and the soft-target experiment are orthogonal;
    this loader must support both active at once).
    """
    indices = [idx for idx, _ in items]
    batch = _collate_samples(
        [sample for _, sample in items],
        include_source_fraction=include_source_fraction,
    )
    batch["_dataset_index"] = torch.tensor(indices, dtype=torch.long)
    return batch


def _build_hard_negative_train_loader(
    indexed_dataset: _IndexedDataset,
    config: TrainConfig,
    generator: torch.Generator,
    weights: torch.Tensor | None,
) -> DataLoader:
    """DataLoader for one epoch of hard-negative-mining training -- see
    module docstring, item 16. ``weights is None`` (the ``max(1,
    config.hard_negative_warmup_epochs)``-epoch warmup, and any epoch before
    a single negative score has been observed) falls back to plain
    ``shuffle=True`` -- the SAME per-epoch sampling distribution
    :func:`_build_train_loader` uses, just routed through the index
    -carrying collate so this epoch's own scores can be recorded. Once
    ``weights`` is set, sampling is ``WeightedRandomSampler(..., num_samples
    =len(indexed_dataset), replacement=True)`` so the per-epoch example
    count -- and therefore the cosine LR schedule's already-computed
    ``total_steps`` (item 12) -- stays the same as the uniform path even
    though the per-index draw PROBABILITY has changed.
    """
    # module docstring, items 17/18/22 -- same rationale as
    # _build_train_loader.
    collate = functools.partial(
        _collate_indexed_samples,
        include_source_fraction=(
            config.soft_target
            or config.boundary_aux_target != "none"
            or config.soft_term_enabled
        ),
    )
    if weights is None:
        return DataLoader(
            indexed_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=config.num_workers,
            worker_init_fn=worker_init_fn if config.num_workers > 0 else None,
            collate_fn=collate,
            drop_last=False,
        )
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(indexed_dataset),
        replacement=True,
        generator=generator,
    )
    return DataLoader(
        indexed_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        worker_init_fn=worker_init_fn if config.num_workers > 0 else None,
        collate_fn=collate,
        drop_last=False,
    )


@torch.no_grad()
def _per_sample_clipped_negative_score(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Historical capped per-sample false-activation ranking score.

    See module docstring, item 16. Returns ``(is_negative, neg_score)``, both shape
    ``[B]``. ``is_negative[i]`` is ``True`` iff sample ``i`` has no positive
    pixel in its (valid-masked) target -- the SAME target-content
    convention :func:`_run_validation` already uses (module docstring, item
    11), not ``meta["positive"]``. ``neg_score[i]`` is the mean
    ``-log(clamp(1-sigmoid(logit), min=1e-6))`` over that sample's valid
    pixels. It approximates zero-target BCE but is capped at
    ``-log(1e-6)`` per pixel, so saturated high logits can tie. Computed
    from ``logits`` the training step's own forward pass already produced
    (callers pass ``logits.detach()`` -- no extra model call, no gradient of
    its own). Entries where ``is_negative[i]`` is ``False`` carry a
    meaningless ``neg_score[i]`` -- callers gate on ``is_negative`` before
    reading it.
    """
    target_flat = target.reshape(target.shape[0], -1)
    valid_flat = valid.reshape(valid.shape[0], -1)
    target_bin = (target_flat >= DEFAULT_SCORE_THRESHOLD) & (valid_flat >= 0.5)
    is_negative = ~target_bin.any(dim=1)

    probs = torch.sigmoid(logits.float()).reshape(logits.shape[0], -1)
    eps = 1e-6
    score_map = -torch.log((1.0 - probs).clamp(min=eps))
    valid_f = valid_flat.float()
    numer = (score_map * valid_f).sum(dim=1)
    denom = valid_f.sum(dim=1).clamp(min=1.0)
    neg_score = numer / denom
    return is_negative, neg_score


# ---------------------------------------------------------------------------
# Manifest + metrics log
# ---------------------------------------------------------------------------


def _write_manifest(
    run_root: Path, config: TrainConfig, split_hash: str, config_hash: str
) -> Path:
    manifest = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "tensor_schema_version": TENSOR_SCHEMA_VERSION,
        "crop_schema_version": CROP_SCHEMA_VERSION,
        "train_module_version": TRAIN_MODULE_VERSION,
        "model_signature": model_signature(config.model_config),
        "torch_version": torch.__version__,
        "device": config.device,
        "amp": config.amp,
        "seed": config.seed,
        "val_seed": config.val_seed,
        "loss": config.loss,
        "tversky_fn_weight": config.tversky_fn_weight,
        "tversky_fp_weight": config.tversky_fp_weight,
        "soft_target": config.soft_target,
        "selection_metric": config.selection_metric,
        "lr_schedule": config.lr_schedule,
        "warmup_steps": config.warmup_steps,
        "augment": config.augment,
        "ema_decay": config.ema_decay,
        "secondary_selection_metric": config.secondary_selection_metric,
        "hard_negative_mining": config.hard_negative_mining,
        "lambda_boundary": config.lambda_boundary,
        "boundary_aux_target": config.boundary_aux_target,
        "constrained_iou_selection": config.constrained_iou_selection,
        "constrained_iou_min_precision": config.constrained_iou_min_precision,
        "constrained_iou_min_f1": config.constrained_iou_min_f1,
        "constrained_iou_min_clean": config.constrained_iou_min_clean,
        "deep_supervision": config.deep_supervision,
        "deep_supervision_scales": config.deep_supervision_scales,
        "deep_supervision_weights": config.deep_supervision_weights,
        "soft_term_enabled": config.soft_term_enabled,
        "soft_term_weight": config.soft_term_weight,
        "split_hash": split_hash,
        "config_hash": config_hash,
        "calibration_status": "uncalibrated",
        "research_prototype_warning": RESEARCH_PROTOTYPE_WARNING,
        "created_at": datetime.now(UTC).isoformat(),
    }
    path = run_root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def _append_metrics_line(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(record, allow_nan=False) + "\n")


# ---------------------------------------------------------------------------
# Validation -- see module docstring, item 11.
# ---------------------------------------------------------------------------


def _finite_or(value: float, default: float) -> float:
    """``value`` if finite, else ``default``. Deliberately more
    conservative than ``metrics.py``/``evaluate.py``'s own ``nan``
    -on-undefined convention for any value that can feed a SELECTION
    decision -- see module docstring, item 11, for why a ``nan`` selection
    metric is a correctness hazard (it permanently disables "improved"
    comparisons for the rest of the run once ``best_val_metric`` itself
    becomes ``nan``).
    """
    return value if math.isfinite(value) else default


@dataclass(frozen=True, slots=True)
class ValidationMetrics:
    """Every validation-time statistic this module computes for one epoch
    -- see module docstring, item 11. ``blended_dice`` is the original,
    pre-specification metric (kept for continuity/logging, never used for
    selection any more). All ``metrics.py``-derived aggregates are
    NaN-sanitized to ``0.0`` (see :func:`_finite_or`) since every field
    here can feed :func:`_select_metric_value`.
    """

    blended_dice: float
    mean_positive_dice: float
    mean_positive_iou: float
    detection_precision: float
    detection_recall: float
    detection_f1: float
    negative_clean_rate: float
    dice_x_clean: float
    det_f1_gated_dice: float
    n_positive: int
    n_negative: int


def _select_metric_value(metrics: ValidationMetrics, name: str) -> float:
    """The named field of ``metrics`` -- ``name`` is already validated to
    be one of :data:`_SELECTION_METRIC_NAMES` by ``TrainConfig.
    __post_init__`` before this is ever called.
    """
    mapping = {
        "blended_dice": metrics.blended_dice,
        "mean_positive_dice": metrics.mean_positive_dice,
        "mean_positive_iou": metrics.mean_positive_iou,
        "dice_x_clean": metrics.dice_x_clean,
        "det_f1_gated_dice": metrics.det_f1_gated_dice,
    }
    return mapping[name]


@torch.no_grad()
def _run_validation(
    model: torch.nn.Module,
    val_samples: list[Sample] | CachedSampleDataset,
    config: TrainConfig,
    device: torch.device,
    use_amp: bool,
) -> ValidationMetrics:
    """One epoch's full validation pass -- ONE forward pass per batch
    produces both the existing blended-Dice number and every
    ``metrics.py``-derived positive-focused statistic (module docstring,
    item 11); no extra model calls. A sample's POSITIVE/NEGATIVE bucket is
    decided from its ACTUAL target content (``target >= threshold``, valid
    -region-masked), matching ``evaluate.py``'s own documented policy (that
    module's docstring, item 2) -- not from ``meta["positive"]``, which
    ``dataset.py``'s own module docstring (item 6) documents as a field
    that has previously disagreed with actual target content due to a
    since-fixed bug.
    """
    was_training = model.training
    model.eval()

    blended_total = 0.0
    blended_count = 0

    positive_dices: list[float] = []
    positive_ious: list[float] = []
    positive_detected_dices: list[float] = []
    lesion_tp_sum = 0
    lesion_fp_sum = 0
    lesion_fn_sum = 0
    negative_clean_flags: list[bool] = []

    for batch_idx, batch in enumerate(
        _iter_val_batches(val_samples, config.batch_size)
    ):
        # Spy assertion (module docstring, item 17): _iter_val_batches ->
        # _collate_samples always defaults include_source_fraction=False,
        # so this key structurally cannot be present -- this assert is a
        # defense-in-depth re-check, not the sole guarantee, that
        # source_fraction is unreachable from the evaluation path
        # regardless of config.soft_target.
        assert "source_fraction" not in batch, (
            "source_fraction must never reach the validation path "
            "(hard-mask-only evaluation invariant -- module docstring, "
            "item 17)"
        )
        if (
            config.limit_val_batches is not None
            and batch_idx >= config.limit_val_batches
        ):
            break
        left = batch["left_view"].to(device)
        right = batch["right_view"].to(device)
        diff = batch["pet_diff"].to(device)
        target = batch["target_mask"].to(device)
        valid = batch["valid_mask"].to(device)
        with torch.amp.autocast(device_type=config.device, enabled=use_amp):
            logits = model(left, right, diff)

        blended = dice_score(logits, target, valid_mask=valid, from_logits=True)
        n = left.shape[0]
        blended_total += float(blended) * n
        blended_count += n

        score_np = abnormality_score(logits).detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        valid_np = valid.detach().cpu().numpy()
        for i in range(n):
            p_i = score_np[i, 0]
            t_i = target_np[i, 0]
            v_i = valid_np[i, 0]
            target_bin = (t_i >= DEFAULT_SCORE_THRESHOLD) & (v_i >= 0.5)
            if target_bin.any():
                d = metric_dice(p_i, t_i, valid_mask=v_i)
                iou = iou_jaccard(p_i, t_i, valid_mask=v_i)
                counts = lesion_component_confusion(p_i, t_i, valid_mask=v_i)
                positive_dices.append(d)
                positive_ious.append(iou)
                lesion_tp_sum += counts.tp
                lesion_fp_sum += counts.fp
                lesion_fn_sum += counts.fn
                if counts.tp > 0:
                    positive_detected_dices.append(d)
            else:
                fp_components = false_positive_components(p_i, valid_mask=v_i)
                negative_clean_flags.append(fp_components == 0)

    if was_training:
        model.train()

    blended_dice = blended_total / blended_count if blended_count > 0 else 0.0
    mean_positive_dice = float(np.mean(positive_dices)) if positive_dices else 0.0
    mean_positive_iou = float(np.mean(positive_ious)) if positive_ious else 0.0
    detection_stats = precision_recall_f_beta(
        lesion_tp_sum, lesion_fp_sum, lesion_fn_sum, beta=1.0
    )
    negative_clean_rate = (
        float(np.mean(negative_clean_flags)) if negative_clean_flags else 0.0
    )
    dice_x_clean = mean_positive_dice * negative_clean_rate
    det_f1_gated_dice = (
        float(np.mean(positive_detected_dices)) if positive_detected_dices else 0.0
    )

    return ValidationMetrics(
        blended_dice=blended_dice,
        mean_positive_dice=mean_positive_dice,
        mean_positive_iou=mean_positive_iou,
        detection_precision=_finite_or(detection_stats.precision, 0.0),
        detection_recall=_finite_or(detection_stats.recall, 0.0),
        detection_f1=_finite_or(detection_stats.f_beta, 0.0),
        negative_clean_rate=negative_clean_rate,
        dice_x_clean=dice_x_clean,
        det_f1_gated_dice=det_f1_gated_dice,
        n_positive=len(positive_dices),
        n_negative=len(negative_clean_flags),
    )


# ---------------------------------------------------------------------------
# Cosine LR schedule with linear warmup -- see module docstring, item 12.
# A pure function of global_step; no scheduler object, no new checkpoint
# state.
# ---------------------------------------------------------------------------


def _steps_per_epoch(dataset_len: int, batch_size: int, limit: int | None) -> int:
    """The number of optimizer steps one epoch actually takes -- ceiling
    division (matching ``DataLoader(..., drop_last=False)``'s own final
    -partial-batch behavior), bounded by ``limit_train_batches`` if set.
    """
    full = -(-dataset_len // batch_size)  # ceil division
    return full if limit is None else min(full, limit)


def _lr_at_step(
    step: int,
    *,
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
    schedule: str,
) -> float:
    """The learning rate for ``step`` (0-indexed, the step ABOUT to be
    taken) -- a pure function of its five arguments, no hidden state. See
    module docstring, item 12, for why this design, rather than a stateful
    ``torch.optim.lr_scheduler`` object, is what makes resume-equivalence
    provable with zero new checkpoint fields.
    """
    if schedule == "none":
        return base_lr
    if schedule != "cosine":
        raise ValueError(
            f'unknown lr_schedule {schedule!r}; expected "cosine" or "none"'
        )

    if warmup_steps > 0 and step < warmup_steps:
        # Linear ramp 0 -> base_lr over [0, warmup_steps).
        return base_lr * float(step + 1) / float(warmup_steps)

    # Cosine decay base_lr -> 0 over [warmup_steps, total_steps).
    decay_span = max(1, total_steps - warmup_steps)
    progress = min(1.0, float(step - warmup_steps) / float(decay_span))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _safe_grad_l2_norm(grads: tuple[torch.Tensor | None, ...]) -> float | None:
    """L2 norm of a tuple of (possibly ``None``) per-parameter gradients,
    as returned by ``torch.autograd.grad(..., allow_unused=True)`` -- see
    module docstring, item 18. Mirrors the boundary experiment Sec 5's own "gradient norm =
    L2 norm of concatenated, unscaled gradients" definition, applied here
    to whichever parameter set the caller passed in (the whole trainable
    model, for this per-step online log -- see item 18's own note on why
    that differs from the offline production-gradient probe's head/decoder
    subset).

    Returns ``None`` -- never ``NaN``/``Inf`` -- when every entry is
    ``None`` (nothing in the parameter set was connected to that loss's
    graph) or when the computed norm is itself non-finite: callers write
    this straight into ``metrics.jsonl`` (``allow_nan=False``), so a bad
    value must become a structured null, not a poisoned float.
    """
    present = [g for g in grads if g is not None]
    if not present:
        return None
    total_sq = 0.0
    for g in present:
        total_sq += float(torch.sum(g.detach().to(torch.float64) ** 2).item())
    norm = math.sqrt(total_sq)
    return norm if math.isfinite(norm) else None


def _max_pool_hard_target(target_mask: torch.Tensor, scale: int) -> torch.Tensor:
    """Deep-supervision aux target at one decoder scale -- module docstring,
    item 21, Root correction (empirical downsample-sparsity check,
    2026-07-19): ``F.max_pool2d(target_mask, kernel_size=scale)`` -- "any
    positive pixel in the block -> 1". ``target_mask`` is guaranteed exactly
    ``{0, 1}``-valued here (``TrainConfig.__post_init__`` rejects
    ``deep_supervision=True`` combined with ``soft_target=True``, and
    ``_augment_batch``'s ``target_interp="nearest"`` path -- the only path
    reachable when ``soft_target=False`` -- never invents a fractional
    value), and max-pooling a ``{0, 1}`` tensor can only ever produce
    another ``{0, 1}`` tensor (the max of a set of zeros and ones is itself
    0 or 1) -- so the aux target stays exactly binary at every scale, never
    an invented fractional/soft value (forbidden by the spec: a
    bilinear-downsampled hard mask would silently turn the aux term into an
    improper soft-target loss).
    """
    return F.max_pool2d(target_mask, kernel_size=scale)


def _min_pool_valid_mask(valid_mask: torch.Tensor, scale: int) -> torch.Tensor:
    """Deep-supervision aux ``valid_mask`` at one decoder scale -- module
    docstring, item 21: conservative min-pool, "a downsampled pixel is
    valid only if its WHOLE block is valid." Implemented via the identity
    ``min-pool(v) = 1 - max-pool(1 - v)`` (both ``v`` and the pooled result
    are exactly ``{0, 1}``-valued -- same binariness argument as
    :func:`_max_pool_hard_target` -- so this is an exact min, not an
    approximation): a block containing ANY invalid (``0``) pixel has at
    least one ``1`` in ``1 - v``, so ``max_pool2d(1 - v)`` is ``1`` there
    and the downsampled valid pixel becomes ``0``; only a block that is
    ALL-valid (every ``1 - v`` entry ``0``) yields a downsampled ``1``. This
    is the conservative direction -- ``valid_mask`` marks in-FOV pixels, and
    an aux-scale pixel should never be counted valid on the strength of only
    part of its receptive block actually being in-FOV.
    """
    return 1.0 - F.max_pool2d(1.0 - valid_mask, kernel_size=scale)


# TrainConfig.soft_term_enabled drift-monitor -- module docstring, item 22.
# The exact attribute names of SiameseBilateralUNet's decoder + final 1x1
# head (model.py), matching scripts/b3_grad_balance_probe.py's own offline
# subset (itself following scripts/p3_lambda_probe.py's _DECODER_ATTRS
# precedent, plus "head") -- the frozen B3 design's own "shared decoder+head"
# wording, so the online per-step ratio and the offline fixed-init probe's
# ratio are directly comparable numbers.
_SHARED_DECODER_HEAD_ATTRS: tuple[str, ...] = (
    "up4",
    "up3",
    "up2",
    "up1",
    "diff_stem",
    "diff_fuse",
    "head",
)


def _shared_decoder_head_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Trainable parameters of the shared decoder + final head -- module
    docstring, item 22, "shared decoder+head params". Deliberately
    NARROWER than item 18's own online grad-norm diagnostic (which spans
    the WHOLE trainable parameter set) -- the frozen B3 design's drift-monitor
    wording is explicit about the decoder+head subset, matching
    ``scripts/b3_grad_balance_probe.py``'s own denominator exactly.
    """
    params: list[torch.nn.Parameter] = []
    for attr in _SHARED_DECODER_HEAD_ATTRS:
        params.extend(p for p in getattr(model, attr).parameters() if p.requires_grad)
    return params


# ---------------------------------------------------------------------------
# Resume compatibility
# ---------------------------------------------------------------------------


def _verify_resume_compatibility(
    payload: CheckpointPayload, config: TrainConfig, split_hash: str
) -> None:
    mismatches: list[str] = []
    if payload.checkpoint_schema_version != CHECKPOINT_SCHEMA_VERSION:
        mismatches.append(
            "checkpoint_schema_version: checkpoint="
            f"{payload.checkpoint_schema_version!r} current={CHECKPOINT_SCHEMA_VERSION!r}"
        )
    if payload.tensor_schema_version != TENSOR_SCHEMA_VERSION:
        mismatches.append(
            f"tensor_schema_version: checkpoint={payload.tensor_schema_version!r} "
            f"current={TENSOR_SCHEMA_VERSION!r}"
        )
    if payload.crop_schema_version != CROP_SCHEMA_VERSION:
        mismatches.append(
            f"crop_schema_version: checkpoint={payload.crop_schema_version!r} "
            f"current={CROP_SCHEMA_VERSION!r}"
        )
    expected_signature = model_signature(config.model_config)
    if payload.model_signature != expected_signature:
        mismatches.append(
            f"model_signature: checkpoint={payload.model_signature!r} "
            f"config={expected_signature!r}"
        )
    if payload.split_hash != split_hash:
        mismatches.append(
            f"split_hash: checkpoint={payload.split_hash!r} config={split_hash!r} "
            "(resume config's train/val bundle dirs differ from the original run's)"
        )
    # max_epochs is the sole supported resume-time change: callers may
    # extend a run, with lr_schedule_total_steps pinned when needed. Bind
    # every other trajectory-affecting field to the checkpoint's original
    # value. Reconstruct the original hash using the checkpoint's stored
    # max_epochs so the hash itself is enforced rather than merely
    # recorded as provenance.
    checkpoint_max_epochs = payload.hyperparams.get("max_epochs")
    if not isinstance(checkpoint_max_epochs, int):
        mismatches.append(
            "hyperparams.max_epochs: checkpoint is missing a valid integer value"
        )
    else:
        expected_config_hash = compute_config_hash(
            replace(config, max_epochs=checkpoint_max_epochs)
        )
        if payload.config_hash != expected_config_hash:
            mismatches.append(
                f"config_hash: checkpoint={payload.config_hash!r} "
                f"current-compatible={expected_config_hash!r}"
            )

    current_hyperparams = _hyperparams_dict(config)
    for name, current_value in current_hyperparams.items():
        if name in {"max_epochs", "log_every_n_steps"}:
            continue
        if name not in payload.hyperparams:
            mismatches.append(f"hyperparams.{name}: missing from checkpoint")
        elif payload.hyperparams[name] != current_value:
            mismatches.append(
                f"hyperparams.{name}: checkpoint={payload.hyperparams[name]!r} "
                f"current={current_value!r}"
            )

    unsupported_state: list[str] = []
    if config.ema_decay is not None:
        unsupported_state.append("ema_decay")
    if config.hard_negative_mining:
        unsupported_state.append("hard_negative_mining")
    if config.secondary_selection_metric is not None:
        unsupported_state.append("secondary_selection_metric")
    if config.constrained_iou_selection:
        unsupported_state.append("constrained_iou_selection")
    if unsupported_state:
        mismatches.append(
            "resume is unsupported while checkpoint-external state is enabled: "
            + ", ".join(unsupported_state)
        )
    if payload.best_val_metric_name != config.selection_metric:
        mismatches.append(
            "selection_metric: checkpoint tracked best_val_metric under "
            f"{payload.best_val_metric_name!r} but the resume config's "
            f"selection_metric is {config.selection_metric!r} -- these must "
            "match for best_val_metric comparisons to stay meaningful across "
            "the resume boundary (module docstring, item 11)"
        )
    if mismatches:
        raise CheckpointCompatibilityError(
            "checkpoint is incompatible with the supplied TrainConfig: "
            + "; ".join(mismatches)
        )


def _restore_resume_selection_state(
    metrics_path: Path,
    payload: CheckpointPayload,
    selection_metric: str,
) -> tuple[int | None, int]:
    """Reconstruct best epoch and early-stop counter from the durable log.

    ``CheckpointPayload`` stores the best value but not the epoch that set it
    or the consecutive non-improvement count. Guessing either at a resume
    boundary changes stopping behavior. The append-only validation log has
    exactly the required history, so resume accepts it only when epochs
    ``0..payload.epoch`` form one complete, selector-consistent sequence whose
    reconstructed best agrees with the checkpoint.
    """
    if not metrics_path.is_file():
        raise CheckpointCompatibilityError(
            f"cannot restore resume selection state: missing {metrics_path.name}"
        )

    validation_by_epoch: dict[int, float] = {}
    try:
        lines = metrics_path.read_text().splitlines()
    except OSError as exc:
        raise CheckpointCompatibilityError(
            f"cannot restore resume selection state: unreadable {metrics_path.name}"
        ) from exc

    for line_number, line in enumerate(lines, start=1):
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CheckpointCompatibilityError(
                f"cannot restore resume selection state: invalid JSON at "
                f"{metrics_path.name}:{line_number}"
            ) from exc
        if not isinstance(record, dict) or record.get("event") != "validation":
            continue
        epoch = record.get("epoch")
        metric_name = record.get("selection_metric_name")
        metric_value = record.get("selection_metric_value")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise CheckpointCompatibilityError(
                f"cannot restore resume selection state: invalid validation epoch "
                f"at {metrics_path.name}:{line_number}"
            )
        if epoch > payload.epoch:
            raise CheckpointCompatibilityError(
                "cannot restore resume selection state: metrics log contains "
                f"validation epoch {epoch} beyond checkpoint epoch {payload.epoch}"
            )
        if epoch in validation_by_epoch:
            raise CheckpointCompatibilityError(
                "cannot restore resume selection state: duplicate validation "
                f"record for epoch {epoch}"
            )
        if metric_name != selection_metric:
            raise CheckpointCompatibilityError(
                "cannot restore resume selection state: validation selector "
                f"{metric_name!r} at epoch {epoch} != {selection_metric!r}"
            )
        if (
            not isinstance(metric_value, (int, float))
            or isinstance(metric_value, bool)
            or not math.isfinite(float(metric_value))
        ):
            raise CheckpointCompatibilityError(
                "cannot restore resume selection state: non-finite/non-numeric "
                f"selection value at epoch {epoch}"
            )
        validation_by_epoch[epoch] = float(metric_value)

    expected_epochs = set(range(payload.epoch + 1))
    if set(validation_by_epoch) != expected_epochs:
        missing = sorted(expected_epochs - validation_by_epoch.keys())
        raise CheckpointCompatibilityError(
            "cannot restore resume selection state: validation history must "
            f"contain exactly epochs 0..{payload.epoch}; missing={missing}"
        )

    best_value: float | None = None
    best_epoch: int | None = None
    epochs_without_improvement = 0
    for epoch in range(payload.epoch + 1):
        value = validation_by_epoch[epoch]
        if best_value is None or value > best_value:
            best_value = value
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

    if payload.best_val_metric is None or best_value is None:
        raise CheckpointCompatibilityError(
            "cannot restore resume selection state: checkpoint/log best metric "
            "is missing"
        )
    if not math.isclose(
        float(payload.best_val_metric), best_value, rel_tol=0.0, abs_tol=1e-12
    ):
        raise CheckpointCompatibilityError(
            "cannot restore resume selection state: checkpoint best_val_metric "
            f"{payload.best_val_metric!r} != reconstructed {best_value!r}"
        )
    return best_epoch, epochs_without_improvement


# ---------------------------------------------------------------------------
# The shared execution path -- see module docstring, item 7.
# ---------------------------------------------------------------------------


def _execute(
    config: TrainConfig, resume_payload: CheckpointPayload | None
) -> TrainResult:
    if config.device == "cuda" and not torch.cuda.is_available():
        raise CudaUnavailableError(
            'TrainConfig.device == "cuda" but torch.cuda.is_available() is '
            "False. VascuTrace never silently falls back to CPU -- fix the "
            "device request or the environment."
        )

    run_root = config.run_root
    run_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    use_amp = bool(config.amp and config.device == "cuda")

    model = build_model(config.model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler(
        device="cuda" if config.device == "cuda" else "cpu", enabled=use_amp
    )

    # Weight EMA -- see module docstring, item 14. None when
    # config.ema_decay is None (the default); the baseline path never
    # allocates this second model. Initialized from the (freshly-built or
    # resumed) raw model's OWN starting weights below, once model_state_dict
    # is final for this call (after the resume load, if any).
    ema_model: torch.nn.Module | None = None
    if config.ema_decay is not None:
        ema_model = build_model(config.model_config).to(device)

    # Cache-aware dataset construction -- see module docstring, item 9.
    # CachedSampleDataset (src.vascutrace.ml.cache) is used whenever the
    # caller set the corresponding *_cache_dir; the on-the-fly
    # SiameseCropDataset/frozen_validation_set path remains the fallback
    # when it is unset. TrainConfig.__post_init__ already proved (before
    # this function was ever entered) that a cache pairing's manifests are
    # leakage-safe (disjoint bundle sets).
    train_dataset: SiameseCropDataset | CachedSampleDataset
    val_samples: list[Sample] | CachedSampleDataset
    if config.train_cache_dir is not None:
        train_dataset = CachedSampleDataset(config.train_cache_dir)
    else:
        train_dataset = SiameseCropDataset(
            config.train_bundle_dirs,
            seed=config.seed,
            positive_fraction=config.train_positive_fraction,
            config=config.dataset_config,
        )
    if config.val_cache_dir is not None:
        val_samples = CachedSampleDataset(config.val_cache_dir)
    else:
        val_samples = frozen_validation_set(
            config.val_bundle_dirs,
            seed=config.val_seed,
            positive_fraction=config.val_positive_fraction,
            config=config.dataset_config,
        )

    split_hash = compute_split_hash(config)
    config_hash = compute_config_hash(config)
    metrics_path = run_root / "metrics.jsonl"

    if resume_payload is not None:
        _verify_resume_compatibility(resume_payload, config, split_hash)
        model.load_state_dict(resume_payload.model_state_dict)
        optimizer.load_state_dict(resume_payload.optimizer_state_dict)
        scaler.load_state_dict(resume_payload.scaler_state_dict)
        generator = restore_rng(resume_payload)
        start_epoch = resume_payload.epoch + 1
        global_step = resume_payload.global_step
        best_val_metric: float | None = resume_payload.best_val_metric
        best_epoch, epochs_without_improvement = _restore_resume_selection_state(
            metrics_path,
            resume_payload,
            config.selection_metric,
        )
        manifest_path = run_root / "manifest.json"
        if not manifest_path.is_file():
            manifest_path = _write_manifest(run_root, config, split_hash, config_hash)
    else:
        seed_everything(config.seed)
        generator = torch.Generator()
        generator.manual_seed(config.seed)
        start_epoch = 0
        global_step = 0
        best_val_metric = None
        best_epoch = None
        epochs_without_improvement = 0
        manifest_path = _write_manifest(run_root, config, split_hash, config_hash)

    if ema_model is not None:
        # Initialize the EMA shadow from the fresh raw model. Resume with
        # EMA enabled is rejected above because CheckpointPayload does not
        # store the shadow trajectory.
        ema_model.load_state_dict(model.state_dict())

    # Online hard-negative mining -- see module docstring, item 16. When
    # config.hard_negative_mining is False (the default), train_loader is
    # built exactly as every prior item already did and reused unchanged
    # for every epoch -- the baseline path below never constructs
    # _IndexedDataset/WeightedRandomSampler at all.
    train_loader: DataLoader | None = None
    hard_neg_indexed_dataset: _IndexedDataset | None = None
    hard_neg_scores: np.ndarray | None = None
    hard_neg_seen: np.ndarray | None = None
    hard_neg_weights: torch.Tensor | None = None
    hard_neg_negative_start = 0
    if config.hard_negative_mining:
        # __post_init__ already proved train_cache_dir is set, and the
        # branch above therefore built train_dataset as a
        # CachedSampleDataset -- see module docstring, item 16.
        assert isinstance(train_dataset, CachedSampleDataset)
        hard_neg_indexed_dataset = _IndexedDataset(train_dataset)
        n_train = len(train_dataset)
        # CachedSampleDataset.manifest is the plain dict read_cache_manifest
        # returns (cache.py), not a CacheManifest dataclass instance -- see
        # module docstring, item 16.
        hard_neg_negative_start = train_dataset.manifest["total_positive"]
        hard_neg_scores = np.zeros(n_train, dtype=np.float64)
        hard_neg_seen = np.zeros(n_train, dtype=bool)
    else:
        train_loader = _build_train_loader(train_dataset, config, generator)

    # A pure function of (max_epochs, dataset length, batch_size,
    # limit_train_batches) -- recomputed identically by train() and
    # resume() given the same config, exactly like split_hash/config_hash
    # already are (module docstring, items 9 and 12) -- UNLESS the caller
    # pins config.lr_schedule_total_steps explicitly (needed only when a
    # resume call intentionally uses a DIFFERENT max_epochs than the
    # original run -- see that field's own docstring). Drives the cosine
    # LR schedule; needs no checkpoint field of its own.
    if config.lr_schedule_total_steps is not None:
        total_steps = config.lr_schedule_total_steps
    else:
        steps_per_epoch = _steps_per_epoch(
            len(train_dataset), config.batch_size, config.limit_train_batches
        )
        total_steps = max(1, config.max_epochs * steps_per_epoch)

    # Tversky FN/FP rebalance -- module docstring, item 19 (Phase 4 Lever
    # L3). functools.partial pre-binds alpha/beta ONLY when the config
    # actually overrides losses.py's own defaults (_TVERSKY_ALPHA_DEFAULT/
    # _TVERSKY_BETA_DEFAULT, above) -- at the defaults, _LOSS_FUNCTIONS[
    # config.loss] is called with the EXACT original (logits, target,
    # valid_mask=...) signature, no extra kwargs at all, so every
    # pre-item-19 monkeypatched test double (assuming that narrower
    # signature) keeps working unchanged. The training step's own
    # `loss_fn(logits, target, valid_mask=valid)` call site (item 10) is
    # unchanged source text either way. Every non-Tversky config.loss value
    # takes the bare dict lookup exactly as before this item existed.
    tversky_overridden = (
        config.tversky_fn_weight != _TVERSKY_ALPHA_DEFAULT
        or config.tversky_fp_weight != _TVERSKY_BETA_DEFAULT
    )
    if config.loss in _TVERSKY_WEIGHTED_LOSS_NAMES and tversky_overridden:
        loss_fn = functools.partial(
            _LOSS_FUNCTIONS[config.loss],
            alpha=config.tversky_fn_weight,
            beta=config.tversky_fp_weight,
        )
    else:
        loss_fn = _LOSS_FUNCTIONS[config.loss]

    last_path = run_root / "last.pt"
    best_path = run_root / "best.pt"

    # Secondary selection tracking -- see module docstring, item 15. All
    # inert (never read/written) when config.secondary_selection_metric is
    # None.
    best_secondary_path = (
        run_root / f"best_{config.secondary_selection_metric}.pt"
        if config.secondary_selection_metric is not None
        else None
    )
    best_secondary_val_metric: float | None = None
    best_secondary_epoch: int | None = None

    # Constrained-floor IoU checkpoint selector -- see module docstring,
    # item 20 (Track A / P4). Independent best-tracking state; never
    # influences best.pt/best_<secondary>.pt above or early stopping. All
    # inert (never read/written) when config.constrained_iou_selection is
    # False.
    best_constrained_iou_path = (
        run_root / "best_constrained_iou.pt"
        if config.constrained_iou_selection
        else None
    )
    best_constrained_iou_val: float | None = None
    best_constrained_iou_epoch: int | None = None

    # Weight EMA tracking -- see module docstring, item 14. All inert when
    # ema_model is None.
    last_ema_path = run_root / "last_ema.pt"
    best_ema_path = run_root / "best_ema.pt"
    best_ema_secondary_path = (
        run_root / f"best_ema_{config.secondary_selection_metric}.pt"
        if config.secondary_selection_metric is not None
        else None
    )
    best_ema_val_metric: float | None = None
    best_ema_epoch: int | None = None
    best_ema_secondary_val_metric: float | None = None
    best_ema_secondary_epoch: int | None = None
    best_ema_constrained_iou_path = (
        run_root / "best_ema_constrained_iou.pt"
        if config.constrained_iou_selection
        else None
    )
    best_ema_constrained_iou_val: float | None = None
    best_ema_constrained_iou_epoch: int | None = None

    stopped_early = epochs_without_improvement >= config.early_stop_patience
    final_epoch = start_epoch - 1

    epoch_range = () if stopped_early else range(start_epoch, config.max_epochs)
    for epoch in epoch_range:
        model.train()
        # Online hard-negative mining -- see module docstring, item 16.
        # config.hard_negative_mining=False (the default) always takes the
        # `else` branch, reusing the SAME pre-built train_loader object
        # every epoch exactly as before this item existed.
        hard_neg_sampling_mode_this_epoch = "n/a"
        if config.hard_negative_mining:
            epoch_loader = _build_hard_negative_train_loader(
                hard_neg_indexed_dataset, config, generator, hard_neg_weights
            )
            # Captured BEFORE this epoch's own end-of-epoch weight update
            # below, so the log line for THIS epoch reports what sampling
            # this epoch actually trained under, not what the NEXT epoch
            # will use -- see module docstring, item 16.
            hard_neg_sampling_mode_this_epoch = (
                "mining" if hard_neg_weights is not None else "uniform"
            )
        else:
            epoch_loader = train_loader
        for batch_idx, batch in enumerate(epoch_loader):
            if (
                config.limit_train_batches is not None
                and batch_idx >= config.limit_train_batches
            ):
                break

            # Set the LR for the step about to be taken -- a pure function
            # of global_step (module docstring, item 12); no scheduler
            # object, no extra checkpoint state.
            current_lr = _lr_at_step(
                global_step,
                base_lr=config.lr,
                total_steps=total_steps,
                warmup_steps=config.warmup_steps,
                schedule=config.lr_schedule,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            left = batch["left_view"].to(device, non_blocking=True)
            right = batch["right_view"].to(device, non_blocking=True)
            diff = batch["pet_diff"].to(device, non_blocking=True)
            # Soft-target experiment -- module docstring, item 17.
            # config.soft_target=False (default) always selects target_mask.
            # TrainConfig.__post_init__
            # already guarantees batch["source_fraction"] exists here
            # whenever config.soft_target is True (the train loader's own
            # collate_fn -- _build_train_loader/_build_hard_negative_
            # train_loader -- only requests it in that case).
            target = (
                batch["source_fraction"] if config.soft_target else batch["target_mask"]
            ).to(device, non_blocking=True)
            valid = batch["valid_mask"].to(device, non_blocking=True)

            # Boundary-local auxiliary loss (item 18) / B3 additive soft
            # term (item 22) both need source_fraction. Fetched into ONE
            # shared tensor rather than two independent ones: under
            # config.augment=True, _augment_batch draws its own random
            # affine grid per call, so fetching/warping source_fraction
            # twice would silently misalign one mechanism's copy against
            # the other's. config.boundary_aux_target != "none" implies
            # (__post_init__) soft_target=False, so `target` above is
            # always target_mask whenever the boundary auxiliary reads it;
            # the train loader's collate_fn already guarantees
            # batch["source_fraction"] exists whenever either condition is
            # taken (same include_source_fraction gate items 17/18/22
            # established).
            source_fraction_needed = (
                config.boundary_aux_target != "none" or config.soft_term_enabled
            )
            source_fraction_batch = (
                batch["source_fraction"].to(device, non_blocking=True)
                if source_fraction_needed
                else None
            )

            # Train-time augmentation -- see module docstring, item 13.
            # TRAINING SPLIT ONLY (never called from _run_validation); a
            # no-op (config.augment defaults False) leaves this whole block
            # unreached, consuming zero extra RNG draws.
            if config.augment:
                augmented = _augment_batch(
                    left,
                    right,
                    diff,
                    target,
                    valid,
                    config=config,
                    # item 17: bilinear for the continuous source_fraction
                    # target; nearest (unchanged) for the hard target_mask.
                    target_interp="bilinear" if config.soft_target else "nearest",
                    # items 18/22: warps source_fraction with the SAME
                    # shared grid whenever the boundary auxiliary and/or the
                    # B3 soft term is active, so its support/target stay
                    # aligned with the augmented logits/target_mask; a no-op
                    # (None in, None out) when neither is on.
                    source_fraction=source_fraction_batch,
                )
                if source_fraction_batch is not None:
                    (
                        left,
                        right,
                        diff,
                        target,
                        valid,
                        source_fraction_batch,
                    ) = augmented
                else:
                    left, right, diff, target, valid = augmented

            optimizer.zero_grad(set_to_none=True)
            boundary_result: BoundaryAuxiliaryLoss | None = None
            # Deep supervision -- module docstring, item 21. Per-scale raw
            # (pre-weight) aux combo_loss values, for logging only; stays
            # an empty dict (and deep_sup_loss_total stays 0.0) whenever
            # config.deep_supervision=False.
            deep_sup_aux_terms: dict[int, float] = {}
            deep_sup_loss_total = 0.0
            try:
                with torch.amp.autocast(device_type=config.device, enabled=use_amp):
                    # return_aux=True ONLY here, ONLY when
                    # config.deep_supervision -- the sole caller in this
                    # codebase that ever requests aux logits (module
                    # docstring, item 21, "aux heads are structurally
                    # TRAIN-ONLY"). Every other call site (_run_validation,
                    # evaluate.py) omits the argument and gets the
                    # pre-B2 bare-logits return unchanged.
                    if config.deep_supervision:
                        logits, aux_logits_by_scale = model(
                            left, right, diff, return_aux=True
                        )
                    else:
                        logits = model(left, right, diff)
                    # L_hard -- the boundary experiment Sec 5: "the shipped combo_loss(logits,
                    # target_mask, valid_mask)". Unchanged computation from
                    # every pre-item-18 config; this IS the whole loss when
                    # boundary_aux_target == "none" (arm A).
                    hard_loss = loss_fn(logits, target, valid_mask=valid)
                    if config.boundary_aux_target != "none":
                        boundary_result = boundary_auxiliary_loss(
                            logits,
                            source_fraction_batch,
                            target,
                            valid,
                            target_mode=config.boundary_aux_target,
                        )
                        loss = hard_loss + config.lambda_boundary * boundary_result.loss
                    else:
                        loss = hard_loss
                    # Deep-supervision aux term -- module docstring, item 21.
                    # `target`/`valid` are guaranteed the hard target_mask/
                    # valid_mask here (deep_supervision=True requires
                    # soft_target=False, TrainConfig.__post_init__), so the
                    # max-pool/min-pool downsamples below always stay exactly
                    # {0,1}-valued. `loss_fn` is the SAME (possibly Tversky
                    # -rebalanced) partial the main term already uses, called
                    # verbatim at each aux scale -- no new loss function, no
                    # edit to combo_loss/losses.py.
                    if config.deep_supervision:
                        deep_sup_loss = logits.new_zeros(())
                        for scale, weight in zip(
                            config.deep_supervision_scales,
                            config.deep_supervision_weights,
                            strict=True,
                        ):
                            aux_target = _max_pool_hard_target(target, scale)
                            aux_valid = _min_pool_valid_mask(valid, scale)
                            aux_term = loss_fn(
                                aux_logits_by_scale[scale],
                                aux_target,
                                valid_mask=aux_valid,
                            )
                            deep_sup_loss = deep_sup_loss + weight * aux_term
                            deep_sup_aux_terms[scale] = aux_term.detach().item()
                        loss = loss + deep_sup_loss
                        deep_sup_loss_total = deep_sup_loss.detach().item()
                    # `loss` here is exactly the B2 total (item 4's hard
                    # main term, plus item 18's boundary aux when also
                    # active, plus item 21's hard deep-sup aux when also
                    # active) -- the frozen B3 design's own "hard" denominator
                    # for B3's additive soft term and its drift monitor.
                    # Captured BEFORE the soft term is added below, so the
                    # module docstring, item 22's own grad-norm diagnostic
                    # (further down) can differentiate this EXACT tensor.
                    hard_total_loss = loss
                    # B3 additive soft-DML term, as specified in module
                    # docstring item 22:
                    # total = B2 total (unchanged above) + soft_term_weight
                    # * soft_combo_loss(logits, source_fraction, valid) --
                    # FULL-RESOLUTION ONLY (no per-scale pooling; multi
                    # -scale soft supervision is out of scope, deferred to
                    # a future B3.1). soft_combo_loss (losses.py) is reused
                    # completely unmodified.
                    soft_term_loss: torch.Tensor | None = None
                    if config.soft_term_enabled:
                        soft_term_loss = soft_combo_loss(
                            logits, source_fraction_batch, valid_mask=valid
                        )
                        loss = (
                            hard_total_loss + config.soft_term_weight * soft_term_loss
                        )
            except torch.OutOfMemoryError as exc:
                raise CudaOutOfMemoryError(
                    "CUDA out of memory during forward/loss computation "
                    f"(batch_size={config.batch_size}); VascuTrace never "
                    "silently retries with a smaller batch or falls back "
                    "to CPU -- reduce batch_size/model width or free VRAM."
                ) from exc

            if not torch.isfinite(loss):
                raise NonFiniteLossError(
                    f"non-finite training loss ({loss.detach().item()!r}) at "
                    f"epoch={epoch}, global_step={global_step}; aborting "
                    "rather than silently continuing a broken run."
                )

            # Boundary-aux gradient-norm diagnostic -- module docstring,
            # item 18. Cost-guarded: only at a logging step AND only when
            # the aux term is active (an extra pair of backward-graph
            # traversals every step would be needlessly expensive). MUST
            # run before scaler.scale(loss).backward() below -- that call
            # frees the graph unless retain_graph=True, so these two
            # torch.autograd.grad(..., retain_graph=True) calls run first,
            # leaving the graph intact for the real backward pass that
            # follows. Neither call touches .grad on any parameter (unlike
            # .backward()), so this cannot interfere with the optimizer
            # step that follows.
            hard_grad_norm: float | None = None
            aux_grad_norm: float | None = None
            is_logging_step = (global_step + 1) % config.log_every_n_steps == 0
            if boundary_result is not None and is_logging_step:
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                hard_grads = torch.autograd.grad(
                    hard_loss,
                    trainable_params,
                    retain_graph=True,
                    allow_unused=True,
                )
                hard_grad_norm = _safe_grad_l2_norm(hard_grads)
                aux_grads = torch.autograd.grad(
                    boundary_result.loss,
                    trainable_params,
                    retain_graph=True,
                    allow_unused=True,
                )
                aux_grad_norm = _safe_grad_l2_norm(aux_grads)

            # B3 soft-term drift monitor -- module docstring, item 22. Same
            # cost-guarded, before-backward, retain_graph=True pattern as
            # the boundary-aux diagnostic just above, but restricted to the
            # SHARED DECODER+HEAD parameter subset (not the whole trainable
            # parameter set item 18 uses) -- the frozen B3 design's own "shared
            # decoder+head" wording, chosen so this ONLINE per-step ratio is
            # directly comparable to scripts/b3_grad_balance_probe.py's own
            # OFFLINE fixed-init ratio. This is the mechanism that can see
            # LATE soft-dominant drift the init-only probe structurally
            # cannot (the frozen B3 design's own top-named risk).
            soft_grad_norm: float | None = None
            hard_total_grad_norm: float | None = None
            soft_hard_grad_ratio: float | None = None
            if config.soft_term_enabled and is_logging_step:
                shared_params = _shared_decoder_head_params(model)
                hard_total_grads = torch.autograd.grad(
                    hard_total_loss,
                    shared_params,
                    retain_graph=True,
                    allow_unused=True,
                )
                hard_total_grad_norm = _safe_grad_l2_norm(hard_total_grads)
                soft_grads = torch.autograd.grad(
                    soft_term_loss,
                    shared_params,
                    retain_graph=True,
                    allow_unused=True,
                )
                soft_grad_norm = _safe_grad_l2_norm(soft_grads)
                if (
                    hard_total_grad_norm is not None
                    and soft_grad_norm is not None
                    and hard_total_grad_norm > 0.0
                ):
                    soft_hard_grad_ratio = soft_grad_norm / hard_total_grad_norm

            try:
                scaler.scale(loss).backward()
                if config.grad_clip_norm is not None:
                    # unscale BEFORE clipping -- clipping a still-scaled
                    # gradient's norm against an unscaled threshold is
                    # invalid (see module docstring, item 3).
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_clip_norm
                    )
                scaler.step(optimizer)
                scaler.update()
            except torch.OutOfMemoryError as exc:
                raise CudaOutOfMemoryError(
                    "CUDA out of memory during backward/optimizer step "
                    f"(batch_size={config.batch_size}); VascuTrace never "
                    "silently retries with a smaller batch or falls back "
                    "to CPU -- reduce batch_size/model width or free VRAM."
                ) from exc

            if ema_model is not None:
                # See module docstring, item 14 -- updated every optimizer
                # step, immediately after the raw model's own step, so the
                # shadow always reflects "decay-weighted average through
                # the just-taken step."
                _update_ema(ema_model, model, config.ema_decay)

            if config.hard_negative_mining:
                # See module docstring, item 16 -- scored from the SAME
                # logits this step already produced (.detach(), no extra
                # model call, no gradient contribution). batch
                # ["_dataset_index"] is only present because epoch_loader
                # is a hard-negative-mining loader (_collate_indexed_samples)
                # whenever this branch runs.
                is_negative, neg_score = _per_sample_clipped_negative_score(
                    logits.detach(), target, valid
                )
                idx_np = batch["_dataset_index"].cpu().numpy()
                is_neg_np = is_negative.cpu().numpy()
                neg_score_np = neg_score.detach().cpu().numpy()
                momentum = config.hard_negative_score_momentum
                for b in range(idx_np.shape[0]):
                    if not is_neg_np[b]:
                        continue
                    di = int(idx_np[b])
                    value = float(neg_score_np[b])
                    if hard_neg_seen[di]:
                        hard_neg_scores[di] = (
                            momentum * hard_neg_scores[di] + (1.0 - momentum) * value
                        )
                    else:
                        hard_neg_scores[di] = value
                        hard_neg_seen[di] = True

            global_step += 1
            if global_step % config.log_every_n_steps == 0:
                # Boundary-local auxiliary loss diagnostics -- module
                # docstring, item 18. L_hard/L_aux are RAW (pre-
                # lambda_boundary) scalars; arm A (boundary_result is None)
                # logs L_hard == train_loss, L_aux/boundary_count/
                # boundary_fraction == 0.0, so every arm's record shares the
                # same schema. hard_grad_norm/aux_grad_norm were computed
                # above (before backward()) only when this was ALSO a
                # logging step at that point -- is_logging_step used the
                # pre-increment global_step, which is exactly this
                # post-increment global_step, so the two checks agree.
                _append_metrics_line(
                    metrics_path,
                    {
                        "event": "train_step",
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_loss": loss.detach().item(),
                        "lr": current_lr,
                        "lambda_boundary": config.lambda_boundary,
                        "L_hard": hard_loss.detach().item(),
                        "L_aux": (
                            boundary_result.loss.detach().item()
                            if boundary_result is not None
                            else 0.0
                        ),
                        "boundary_count": (
                            boundary_result.boundary_count.item()
                            if boundary_result is not None
                            else 0.0
                        ),
                        "boundary_fraction": (
                            boundary_result.boundary_fraction.item()
                            if boundary_result is not None
                            else 0.0
                        ),
                        "hard_grad_norm": hard_grad_norm,
                        "aux_grad_norm": aux_grad_norm,
                        # Deep supervision -- module docstring, item 21.
                        # deep_sup_loss_total/deep_sup_aux_terms are the RAW
                        # (already scale-weighted for the total, per-scale
                        # UNweighted for the breakdown) values from the loss
                        # computation above; both stay 0.0/{} whenever
                        # config.deep_supervision=False.
                        "deep_supervision": config.deep_supervision,
                        "L_deep_sup": deep_sup_loss_total,
                        "deep_sup_aux_terms": deep_sup_aux_terms,
                        # B3 additive soft-DML term -- module docstring,
                        # item 22. L_soft is the RAW (pre-soft_term_weight)
                        # soft_combo_loss scalar; 0.0 when this item is off,
                        # matching item 18's own always-present-schema
                        # convention. soft_grad_norm/hard_total_grad_norm/
                        # soft_hard_grad_ratio are None except at a
                        # qualifying logging step (the drift monitor).
                        "soft_term_enabled": config.soft_term_enabled,
                        "soft_term_weight": config.soft_term_weight,
                        "L_soft": (
                            soft_term_loss.detach().item()
                            if soft_term_loss is not None
                            else 0.0
                        ),
                        "soft_grad_norm": soft_grad_norm,
                        "hard_total_grad_norm": hard_total_grad_norm,
                        "soft_hard_grad_ratio": soft_hard_grad_ratio,
                    },
                )

        # Online hard-negative mining -- end-of-epoch bookkeeping: rank
        # this epoch's observed negative-sample scores, mark the hardest
        # config.hard_negative_fraction as "hard," and install their
        # oversampling weight for the NEXT epoch's loader -- see module
        # docstring, item 16. Logged unconditionally (the "negative-loss
        # trend" this implementation's own instruction asks for) whenever mining is
        # enabled, even during the uniform warmup epochs.
        # sampling_mode_this_epoch (captured above, BEFORE this block runs)
        # tells the reader what THIS epoch actually trained under;
        # mining_active_next_epoch tells them what the epoch about to start
        # will use -- deliberately two separate fields so "epoch 0's own
        # line says mining_active_next_epoch=True" is never misread as
        # "epoch 0 itself used mining" (it did not -- see the warmup floor
        # above).
        if config.hard_negative_mining:
            n_train_total = len(hard_neg_indexed_dataset)
            neg_indices = np.arange(hard_neg_negative_start, n_train_total)
            seen_mask = hard_neg_seen[neg_indices]
            n_negative_total = int(neg_indices.shape[0])
            n_negative_seen = int(seen_mask.sum())
            observed_neg_indices = neg_indices[seen_mask]
            observed_scores = hard_neg_scores[observed_neg_indices]

            mining_active_next_epoch = (epoch + 1) >= max(
                1, config.hard_negative_warmup_epochs
            )
            n_hard = 0
            threshold_score: float | None = None
            mean_hard_score: float | None = None
            mean_easy_score: float | None = None
            score_null_reasons: dict[str, str] = {}
            mean_negative_score = (
                float(np.mean(observed_scores)) if n_negative_seen > 0 else None
            )
            if n_negative_seen == 0:
                score_null_reasons["mean_negative_score"] = (
                    "no_negative_samples_observed"
                )
            if mining_active_next_epoch and n_negative_seen > 0:
                n_hard = max(
                    1, int(round(config.hard_negative_fraction * n_negative_seen))
                )
                n_hard = min(n_hard, n_negative_seen)
                order = np.argsort(observed_scores)[::-1]  # hardest (highest) first
                hard_local = order[:n_hard]
                easy_local = order[n_hard:]
                threshold_score = float(observed_scores[hard_local[-1]])
                mean_hard_score = float(np.mean(observed_scores[hard_local]))
                mean_easy_score = (
                    float(np.mean(observed_scores[easy_local]))
                    if easy_local.size > 0
                    else None
                )
                if easy_local.size == 0:
                    score_null_reasons["mean_easy_negative_score"] = (
                        "all_observed_negatives_selected_as_hard"
                    )
                weights_arr = np.ones(n_train_total, dtype=np.float64)
                weights_arr[observed_neg_indices[hard_local]] = (
                    config.hard_negative_oversample_weight
                )
                hard_neg_weights = torch.as_tensor(weights_arr, dtype=torch.double)
            else:
                # Still warming up (or no negative observed yet) -- next
                # epoch stays uniform (hard_neg_weights left None/unchanged
                # -- see _build_hard_negative_train_loader).
                hard_neg_weights = None
                partition_reason = (
                    "no_negative_samples_observed"
                    if n_negative_seen == 0
                    else "mining_warmup_not_complete"
                )
                score_null_reasons["mean_hard_negative_score"] = partition_reason
                score_null_reasons["mean_easy_negative_score"] = partition_reason
                score_null_reasons["hard_negative_score_threshold"] = partition_reason

            _append_metrics_line(
                metrics_path,
                {
                    "event": "hard_negative_mining",
                    "epoch": epoch,
                    "global_step": global_step,
                    "sampling_mode_this_epoch": hard_neg_sampling_mode_this_epoch,
                    "mining_active_next_epoch": bool(
                        mining_active_next_epoch and n_negative_seen > 0
                    ),
                    "n_negative_total": n_negative_total,
                    "n_negative_seen": n_negative_seen,
                    "n_hard_negatives_mined": int(n_hard),
                    "hard_negative_fraction_mined": (
                        float(n_hard) / n_negative_seen if n_negative_seen > 0 else 0.0
                    ),
                    "hard_negative_fraction_config": config.hard_negative_fraction,
                    "hard_negative_oversample_weight": config.hard_negative_oversample_weight,
                    "mean_negative_score": mean_negative_score,
                    "mean_hard_negative_score": mean_hard_score,
                    "mean_easy_negative_score": mean_easy_score,
                    "hard_negative_score_threshold": threshold_score,
                    "score_null_reasons": score_null_reasons,
                },
            )

        val_metrics = _run_validation(model, val_samples, config, device, use_amp)
        selected_value = _select_metric_value(val_metrics, config.selection_metric)
        _append_metrics_line(
            metrics_path,
            {
                "event": "validation",
                "epoch": epoch,
                "global_step": global_step,
                "blended_dice": val_metrics.blended_dice,
                "mean_positive_dice": val_metrics.mean_positive_dice,
                "mean_positive_iou": val_metrics.mean_positive_iou,
                "detection_precision": val_metrics.detection_precision,
                "detection_recall": val_metrics.detection_recall,
                "detection_f1": val_metrics.detection_f1,
                "negative_clean_rate": val_metrics.negative_clean_rate,
                "dice_x_clean": val_metrics.dice_x_clean,
                "det_f1_gated_dice": val_metrics.det_f1_gated_dice,
                "n_positive": val_metrics.n_positive,
                "n_negative": val_metrics.n_negative,
                "selection_metric_name": config.selection_metric,
                "selection_metric_value": selected_value,
                # module docstring, item 18: the static config scalar only
                # -- never L_aux/boundary_count/boundary_fraction, which
                # would require source_fraction and are therefore
                # structurally impossible to compute here without
                # violating the hard-mask-only evaluation invariant (see
                # _run_validation's own spy assertion).
                "lambda_boundary": config.lambda_boundary,
            },
        )

        # Update best-tracking state BEFORE building the checkpoint payload
        # (not after) -- a genuine bug was found and fixed here during this
        # implementation's own resume-equivalence verification: building last.pt's
        # payload with the PRE-this-epoch's-update best_val_metric, then
        # updating best_val_metric/best_epoch afterward only for best.pt
        # (via a separate replace(...)), created a systematic off-by-one
        # -epoch lag -- a resume boundary falling exactly on the epoch that
        # set the new best would reload a STALE (pre-update) best_val_metric
        # from last.pt and silently lose track of that epoch's own
        # improvement, corrupting best_epoch/best_val_metric across the
        # resume boundary even though model weights/RNG streams stayed
        # bit-identical. Updating first means last.pt and best.pt always
        # agree on "best-as-of-the-end-of-this-epoch," and resume's
        # `resume_payload.best_val_metric` is therefore always exactly the
        # value train()/resume() would have returned had this been the
        # final epoch -- see TestResumeEquivalence.
        improved = best_val_metric is None or selected_value > best_val_metric
        if improved:
            best_val_metric = selected_value
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        rng_state = capture_rng_state(generator)
        payload = CheckpointPayload(
            checkpoint_schema_version=CHECKPOINT_SCHEMA_VERSION,
            tensor_schema_version=TENSOR_SCHEMA_VERSION,
            crop_schema_version=CROP_SCHEMA_VERSION,
            model_signature=model_signature(config.model_config),
            model_config=config.model_config,
            dataset_config=config.dataset_config,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            scaler_state_dict=scaler.state_dict(),
            rng_state=rng_state,
            epoch=epoch,
            global_step=global_step,
            best_val_metric=best_val_metric,
            best_val_metric_name=config.selection_metric,
            hyperparams=_hyperparams_dict(config),
            split_hash=split_hash,
            config_hash=config_hash,
            calibration_status="uncalibrated",
            research_prototype_warning=RESEARCH_PROTOTYPE_WARNING,
            created_at=datetime.now(UTC).isoformat(),
        )
        save_checkpoint(last_path, payload)
        if improved:
            save_checkpoint(best_path, payload)

        # Secondary checkpoint-selection tracking -- see module docstring,
        # item 15. Independent best-tracking state; never influences
        # early stopping or the primary best.pt above. Reuses the SAME
        # model state at this epoch. The saved secondary payload replaces
        # the primary selector name/value with the selector that actually
        # chose this file, so checkpoint provenance is never misleading.
        if best_secondary_path is not None:
            secondary_value = _select_metric_value(
                val_metrics, config.secondary_selection_metric
            )
            secondary_improved = (
                best_secondary_val_metric is None
                or secondary_value > best_secondary_val_metric
            )
            if secondary_improved:
                best_secondary_val_metric = secondary_value
                best_secondary_epoch = epoch
                secondary_payload = replace(
                    payload,
                    best_val_metric=best_secondary_val_metric,
                    best_val_metric_name=config.secondary_selection_metric,
                )
                save_checkpoint(best_secondary_path, secondary_payload)

        # Constrained-floor IoU checkpoint selector -- see module
        # docstring, item 20. Uses the SAME val_metrics already computed
        # above (no recompute, no eval-semantics change). A "legal" epoch
        # clears all three configured floors; among legal epochs only, the
        # highest mean_positive_iou is kept. An epoch that fails a floor
        # leaves best_constrained_iou_val/_epoch unchanged -- if NO epoch
        # ever qualifies, best_constrained_iou_path is never written (see
        # the "qualified": False summary event appended after the loop).
        if best_constrained_iou_path is not None:
            constrained_legal = (
                val_metrics.detection_precision >= config.constrained_iou_min_precision
                and val_metrics.detection_f1 >= config.constrained_iou_min_f1
                and val_metrics.negative_clean_rate >= config.constrained_iou_min_clean
            )
            constrained_improved = constrained_legal and (
                best_constrained_iou_val is None
                or val_metrics.mean_positive_iou > best_constrained_iou_val
            )
            _append_metrics_line(
                metrics_path,
                {
                    "event": "constrained_iou_selection",
                    "epoch": epoch,
                    "global_step": global_step,
                    "mean_positive_iou": val_metrics.mean_positive_iou,
                    "detection_precision": val_metrics.detection_precision,
                    "detection_f1": val_metrics.detection_f1,
                    "negative_clean_rate": val_metrics.negative_clean_rate,
                    "constrained_iou_min_precision": (
                        config.constrained_iou_min_precision
                    ),
                    "constrained_iou_min_f1": config.constrained_iou_min_f1,
                    "constrained_iou_min_clean": config.constrained_iou_min_clean,
                    "legal": constrained_legal,
                    "improved": constrained_improved,
                },
            )
            if constrained_improved:
                best_constrained_iou_val = val_metrics.mean_positive_iou
                best_constrained_iou_epoch = epoch
                constrained_payload = replace(
                    payload,
                    best_val_metric=best_constrained_iou_val,
                    best_val_metric_name="mean_positive_iou",
                )
                save_checkpoint(best_constrained_iou_path, constrained_payload)

        # Weight EMA validation + checkpointing -- see module docstring,
        # item 14. Runs the IDENTICAL _run_validation function against
        # ema_model; gates two SEPARATE checkpoint files (last_ema.pt/
        # best_ema.pt, and best_ema_<secondary>.pt when secondary tracking
        # is also on) that never influence the raw model's own
        # best_val_metric/best_epoch/early-stopping state above.
        if ema_model is not None:
            ema_val_metrics = _run_validation(
                ema_model, val_samples, config, device, use_amp
            )
            ema_selected_value = _select_metric_value(
                ema_val_metrics, config.selection_metric
            )
            _append_metrics_line(
                metrics_path,
                {
                    "event": "validation_ema",
                    "epoch": epoch,
                    "global_step": global_step,
                    "blended_dice": ema_val_metrics.blended_dice,
                    "mean_positive_dice": ema_val_metrics.mean_positive_dice,
                    "mean_positive_iou": ema_val_metrics.mean_positive_iou,
                    "detection_precision": ema_val_metrics.detection_precision,
                    "detection_recall": ema_val_metrics.detection_recall,
                    "detection_f1": ema_val_metrics.detection_f1,
                    "negative_clean_rate": ema_val_metrics.negative_clean_rate,
                    "dice_x_clean": ema_val_metrics.dice_x_clean,
                    "det_f1_gated_dice": ema_val_metrics.det_f1_gated_dice,
                    "n_positive": ema_val_metrics.n_positive,
                    "n_negative": ema_val_metrics.n_negative,
                    "selection_metric_name": config.selection_metric,
                    "selection_metric_value": ema_selected_value,
                    "ema_decay": config.ema_decay,
                    # module docstring, item 18: see the "validation" event's
                    # own comment above -- static config scalar only.
                    "lambda_boundary": config.lambda_boundary,
                },
            )
            ema_improved = (
                best_ema_val_metric is None or ema_selected_value > best_ema_val_metric
            )
            if ema_improved:
                best_ema_val_metric = ema_selected_value
                best_ema_epoch = epoch

            # optimizer_state_dict/scaler_state_dict/rng_state below are the
            # RAW model's own current state, carried along only to satisfy
            # CheckpointPayload's schema (evaluate.py never reads them) --
            # module docstring, item 14, flags this is NOT a resumable EMA
            # trajectory.
            ema_payload = CheckpointPayload(
                checkpoint_schema_version=CHECKPOINT_SCHEMA_VERSION,
                tensor_schema_version=TENSOR_SCHEMA_VERSION,
                crop_schema_version=CROP_SCHEMA_VERSION,
                model_signature=model_signature(config.model_config),
                model_config=config.model_config,
                dataset_config=config.dataset_config,
                model_state_dict=ema_model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scaler_state_dict=scaler.state_dict(),
                rng_state=rng_state,
                epoch=epoch,
                global_step=global_step,
                best_val_metric=best_ema_val_metric,
                best_val_metric_name=config.selection_metric,
                hyperparams=_hyperparams_dict(config),
                split_hash=split_hash,
                config_hash=config_hash,
                calibration_status="uncalibrated",
                research_prototype_warning=RESEARCH_PROTOTYPE_WARNING,
                created_at=datetime.now(UTC).isoformat(),
            )
            save_checkpoint(last_ema_path, ema_payload)
            if ema_improved:
                save_checkpoint(best_ema_path, ema_payload)

            if best_ema_secondary_path is not None:
                ema_secondary_value = _select_metric_value(
                    ema_val_metrics, config.secondary_selection_metric
                )
                ema_secondary_improved = (
                    best_ema_secondary_val_metric is None
                    or ema_secondary_value > best_ema_secondary_val_metric
                )
                if ema_secondary_improved:
                    best_ema_secondary_val_metric = ema_secondary_value
                    best_ema_secondary_epoch = epoch
                    ema_secondary_payload = replace(
                        ema_payload,
                        best_val_metric=best_ema_secondary_val_metric,
                        best_val_metric_name=config.secondary_selection_metric,
                    )
                    save_checkpoint(best_ema_secondary_path, ema_secondary_payload)

            # Constrained-floor IoU checkpoint selector, EMA variant -- see
            # module docstring, item 20. Gated by the EMA model's OWN
            # ema_val_metrics from this same epoch (never the raw model's
            # val_metrics above) -- mirrors the raw-model block exactly.
            if best_ema_constrained_iou_path is not None:
                ema_constrained_legal = (
                    ema_val_metrics.detection_precision
                    >= config.constrained_iou_min_precision
                    and ema_val_metrics.detection_f1 >= config.constrained_iou_min_f1
                    and ema_val_metrics.negative_clean_rate
                    >= config.constrained_iou_min_clean
                )
                ema_constrained_improved = ema_constrained_legal and (
                    best_ema_constrained_iou_val is None
                    or ema_val_metrics.mean_positive_iou > best_ema_constrained_iou_val
                )
                _append_metrics_line(
                    metrics_path,
                    {
                        "event": "constrained_iou_selection_ema",
                        "epoch": epoch,
                        "global_step": global_step,
                        "mean_positive_iou": ema_val_metrics.mean_positive_iou,
                        "detection_precision": ema_val_metrics.detection_precision,
                        "detection_f1": ema_val_metrics.detection_f1,
                        "negative_clean_rate": ema_val_metrics.negative_clean_rate,
                        "constrained_iou_min_precision": (
                            config.constrained_iou_min_precision
                        ),
                        "constrained_iou_min_f1": config.constrained_iou_min_f1,
                        "constrained_iou_min_clean": (config.constrained_iou_min_clean),
                        "legal": ema_constrained_legal,
                        "improved": ema_constrained_improved,
                    },
                )
                if ema_constrained_improved:
                    best_ema_constrained_iou_val = ema_val_metrics.mean_positive_iou
                    best_ema_constrained_iou_epoch = epoch
                    ema_constrained_payload = replace(
                        ema_payload,
                        best_val_metric=best_ema_constrained_iou_val,
                        best_val_metric_name="mean_positive_iou",
                    )
                    save_checkpoint(
                        best_ema_constrained_iou_path, ema_constrained_payload
                    )

        final_epoch = epoch
        if epochs_without_improvement >= config.early_stop_patience:
            stopped_early = True
            break

    # Constrained-floor IoU checkpoint selector -- see module docstring,
    # item 20. If the flag was on but no epoch across the whole run ever
    # cleared the legality floors, best_constrained_iou.pt/
    # best_ema_constrained_iou.pt were never written above; log that
    # explicitly rather than leaving "no candidate" silently
    # indistinguishable from "flag was off."
    if best_constrained_iou_path is not None and best_constrained_iou_epoch is None:
        _append_metrics_line(
            metrics_path,
            {
                "event": "constrained_iou_selection_summary",
                "qualified": False,
                "message": (
                    "constrained_iou_selection was enabled but no epoch met "
                    "the legality floors (precision>="
                    f"{config.constrained_iou_min_precision}, f1>="
                    f"{config.constrained_iou_min_f1}, clean>="
                    f"{config.constrained_iou_min_clean}); "
                    "best_constrained_iou.pt was never written"
                ),
            },
        )
    if (
        best_ema_constrained_iou_path is not None
        and best_ema_constrained_iou_epoch is None
    ):
        _append_metrics_line(
            metrics_path,
            {
                "event": "constrained_iou_selection_ema_summary",
                "qualified": False,
                "message": (
                    "constrained_iou_selection was enabled (EMA) but no "
                    "epoch met the legality floors (precision>="
                    f"{config.constrained_iou_min_precision}, f1>="
                    f"{config.constrained_iou_min_f1}, clean>="
                    f"{config.constrained_iou_min_clean}); "
                    "best_ema_constrained_iou.pt was never written"
                ),
            },
        )

    return TrainResult(
        final_epoch=final_epoch,
        global_step=global_step,
        best_epoch=best_epoch,
        best_val_metric=best_val_metric,
        last_checkpoint_path=last_path,
        best_checkpoint_path=best_path if best_epoch is not None else None,
        stopped_early=stopped_early,
        metrics_log_path=metrics_path,
        manifest_path=manifest_path,
        best_secondary_metric_name=config.secondary_selection_metric,
        best_secondary_epoch=best_secondary_epoch,
        best_secondary_val_metric=best_secondary_val_metric,
        best_secondary_checkpoint_path=(
            best_secondary_path if best_secondary_epoch is not None else None
        ),
        ema_enabled=ema_model is not None,
        best_ema_epoch=best_ema_epoch,
        best_ema_val_metric=best_ema_val_metric,
        best_ema_checkpoint_path=(
            best_ema_path if best_ema_epoch is not None else None
        ),
        best_ema_secondary_epoch=best_ema_secondary_epoch,
        best_ema_secondary_val_metric=best_ema_secondary_val_metric,
        best_ema_secondary_checkpoint_path=(
            best_ema_secondary_path if best_ema_secondary_epoch is not None else None
        ),
        best_constrained_iou_epoch=best_constrained_iou_epoch,
        best_constrained_iou_val_metric=best_constrained_iou_val,
        best_constrained_iou_checkpoint_path=(
            best_constrained_iou_path
            if best_constrained_iou_epoch is not None
            else None
        ),
        best_ema_constrained_iou_epoch=best_ema_constrained_iou_epoch,
        best_ema_constrained_iou_val_metric=best_ema_constrained_iou_val,
        best_ema_constrained_iou_checkpoint_path=(
            best_ema_constrained_iou_path
            if best_ema_constrained_iou_epoch is not None
            else None
        ),
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def train(config: TrainConfig) -> TrainResult:
    """Run a fresh training run from ``config``. See module docstring."""
    return _execute(config, resume_payload=None)


def resume(run_root: Path, config: TrainConfig) -> TrainResult:
    """Resume the run at ``run_root`` (must contain a ``last.pt`` written by
    a previous :func:`train`/:func:`resume` call) and continue
    deterministically to ``config.max_epochs``.

    ``config`` must supply the same ``train_bundle_dirs``/
    ``val_bundle_dirs``/architecture/hyperparameters as the original run --
    see ``checkpoint.py``'s module docstring, item 4, for why this module
    cannot accept a bare ``run_root`` with no config: the checkpoint itself
    never stores the raw bundle-directory list (only its hash), by design.
    A mismatched ``config`` raises :class:`CheckpointCompatibilityError`
    rather than silently resuming with the wrong data or architecture.
    """
    run_root = Path(run_root)
    payload = load_checkpoint(run_root / "last.pt")
    return _execute(config, resume_payload=payload)
