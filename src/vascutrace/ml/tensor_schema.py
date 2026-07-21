"""Frozen 2.5D bilateral Siamese network tensor contract (Phase 6).

This is the single source of truth shared by the dataset builder
(``src.vascutrace.ml.dataset``) and the model (``src.vascutrace.ml.model``).
The SAME preprocessing/tensor implementation serves training and inference.

Upstream (P2, frozen): base-crop bundles on ``FIXED_CROP_SHAPE = (144, 80, 144)``
= ``(X R/L, Y A/P, Z S/I)``; the axial (adjacent-slice) axis is Z (axis 2, native
2.0 mm spacing); ``K = 5`` adjacent slices. Reflection swaps left<->right (the X
axis) through the session's fitted mid-sagittal plane (a physical Householder
mirror ``reflection_affine`` carried in each bundle) -- never ``np.flip``.

A 2.5D sample fixes an axial index ``z`` and takes the in-plane ``(X, Y) = (144, 80)``
plane with the K adjacent Z-slices ``[z-2 .. z+2]`` as channels. Both in-plane
dims are divisible by 16, so a 4-level U-Net downsamples cleanly
(144->72->36->18->9, 80->40->20->10->5).

The bilateral Siamese views:
- ``left_view``  : the crop itself (PET K-slices then CT K-slices) -> ``[2K, H, W]``
- ``right_view`` : the physically-reflected crop (PET K-slices then CT K-slices) -> ``[2K, H, W]``
A unilateral synthetic lesion breaks left/right symmetry, so it is visible in the
paired views and their difference. Target and valid masks are the CENTER slice.

Network normalization (applied by the dataset, identical at train and inference):
- PET: ``clip(pet_suvbw, 0, 10) / 10``            (raw SUVbw retained separately for quantification)
- CT : ``clip(ct_hu, -1000, 1000) / 1000``

Research prototype. Trained and evaluated using simulated vascular-like
abnormalities, not confirmed human post-angioplasty lesions.
"""

from src.vascutrace.data.contract import (
    ADJACENT_SLICE_COUNT_K,
    AXIAL_ADJACENT_SLICE_AXIS,
    CROP_SCHEMA_VERSION,
    FIXED_CROP_SHAPE,
)

TENSOR_SCHEMA_VERSION = "p6-tensor-v1"
UPSTREAM_CROP_SCHEMA_VERSION = CROP_SCHEMA_VERSION  # must match the consumed bundles

K: int = ADJACENT_SLICE_COUNT_K  # 5 adjacent axial slices
AXIAL_AXIS: int = AXIAL_ADJACENT_SLICE_AXIS  # 2 (Z, S/I)

# In-plane (H, W) of one axial slice = (X R/L, Y A/P).
IN_PLANE_HW: tuple[int, int] = (FIXED_CROP_SHAPE[0], FIXED_CROP_SHAPE[1])  # (144, 80)
NUM_AXIAL_SLICES: int = FIXED_CROP_SHAPE[AXIAL_AXIS]  # 144

# Valid center indices need K//2 context on each side.
HALF_K: int = K // 2  # 2
FIRST_CENTER_Z: int = HALF_K  # 2
LAST_CENTER_Z: int = NUM_AXIAL_SLICES - HALF_K - 1  # 141

# Network normalization (clip then scale). Raw SUVbw is NOT normalized here.
PET_CLIP: tuple[float, float] = (0.0, 10.0)
PET_SCALE: float = 10.0
CT_CLIP: tuple[float, float] = (-1000.0, 1000.0)
CT_SCALE: float = 1000.0

# Channel layout per Siamese view: PET K-slices followed by CT K-slices.
CHANNELS_PER_VIEW: int = 2 * K  # 10
PET_CHANNEL_SLICE = slice(0, K)  # channels [0:K]
CT_CHANNEL_SLICE = slice(K, 2 * K)  # channels [K:2K]

# Sample tensor shapes (torch float32), H, W = IN_PLANE_HW:
#   left_view   : [2K, H, W]
#   right_view  : [2K, H, W]
#   pet_diff    : [K,  H, W]   (network-normalized left PET minus right PET; decoder guide)
#   target_mask : [1,  H, W]   (center-slice ground-truth synthetic lesion; all-zero for healthy)
#   valid_mask  : [1,  H, W]   (center-slice valid-PET mask, uint8/float32)
LEFT_VIEW_SHAPE: tuple[int, int, int] = (
    CHANNELS_PER_VIEW,
    IN_PLANE_HW[0],
    IN_PLANE_HW[1],
)
RIGHT_VIEW_SHAPE: tuple[int, int, int] = (
    CHANNELS_PER_VIEW,
    IN_PLANE_HW[0],
    IN_PLANE_HW[1],
)
PET_DIFF_SHAPE: tuple[int, int, int] = (K, IN_PLANE_HW[0], IN_PLANE_HW[1])
TARGET_SHAPE: tuple[int, int, int] = (1, IN_PLANE_HW[0], IN_PLANE_HW[1])

__all__ = [
    "TENSOR_SCHEMA_VERSION",
    "UPSTREAM_CROP_SCHEMA_VERSION",
    "K",
    "AXIAL_AXIS",
    "IN_PLANE_HW",
    "NUM_AXIAL_SLICES",
    "HALF_K",
    "FIRST_CENTER_Z",
    "LAST_CENTER_Z",
    "PET_CLIP",
    "PET_SCALE",
    "CT_CLIP",
    "CT_SCALE",
    "CHANNELS_PER_VIEW",
    "PET_CHANNEL_SLICE",
    "CT_CHANNEL_SLICE",
    "LEFT_VIEW_SHAPE",
    "RIGHT_VIEW_SHAPE",
    "PET_DIFF_SHAPE",
    "TARGET_SHAPE",
]
