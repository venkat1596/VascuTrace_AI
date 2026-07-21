"""QUADRA_HC dataset discovery, label-semantics pin, and grid validation
(VascuTrace Phase 2 data pipeline).

RESEARCH_PROTOTYPE_WARNING
---------------------------------------------------------------------------
Research prototype. Trained and evaluated using simulated vascular-like
abnormalities, not confirmed human post-angioplasty lesions.
---------------------------------------------------------------------------

This module is the sole gateway between the on-disk QUADRA_HC archive layout
and every downstream Phase 2 stage (``split``, ``crops``). Every typed error
raised here carries a code plus only non-identifying detail (counts,
0-based ordinals, group names) -- never a raw filesystem path or subject
identifier in the exception *message* -- matching the geometry contract's
(P1) fail-closed, code-only error style. (Ordinals and group names are safe
to log/print anywhere; they are not participant identifiers.)

Expected layout (confirmed against the certified data-fitness note,
``docs/p2_data_fitness_2026-07-16.md``, Q1): ``<root>/Imaging Data/`` holds
48 subject directories, each with a ``Test/`` and ``Retest/`` session, each
session holding exactly 1 PET, 1 CT, and an 8-file ``Segmentations/`` folder
-- 48 x 2 x 10 = 960 NIfTI files total.

Highest-stakes check: the label-semantics pin
==============================================================================
QUADRA_HC's MOOSE segmentations carry no embedded pipeline-version marker in
any local file (confirmed on a full-cohort provenance pass, fitness note
Q2/Q6 "Provenance-pinning rule") -- the integer-to-anatomy mapping (iliac
left/right = 7/8 in the ``Cardiac`` group, femur left/right = 5/6 in the
``Peripheral-Bones`` group) is trusted only because every one of the 96
sessions was checked and agreed in that pass, not because any file declares
it. :func:`pin_label_semantics` re-verifies that agreement, every time the
pipeline runs, against every session it is given, and raises a typed
:class:`LabelPinError` closed on the first cohort-wide disagreement it
finds -- silently trusting a hardcoded mapping against unverified future
data (e.g. a re-extracted or updated archive copy) is exactly the risk the
audit flagged as highest stakes.
============================================================================
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import nibabel as nib
import numpy as np

from src.vascutrace.geometry import GeometryContractError, validate_nifti_grid

__all__ = [
    "RESEARCH_PROTOTYPE_WARNING",
    "SUBJECT_PREFIX",
    "SESSION_NAMES",
    "SEGMENTATION_GROUPS",
    "LABEL_SCHEME",
    "SUBJECT_COUNT",
    "SESSIONS_PER_SUBJECT",
    "SESSION_COUNT",
    "FILES_PER_SESSION",
    "NIFTI_COUNT",
    "SessionPaths",
    "DatasetManifest",
    "IngestErrorCode",
    "IngestError",
    "LabelPinSessionResult",
    "LabelPinReport",
    "LabelPinErrorCode",
    "LabelPinError",
    "GridValidationSessionResult",
    "GridValidationReport",
    "GridValidationErrorCode",
    "GridValidationError",
    "ProvenanceReport",
    "IngestResult",
    "discover_dataset",
    "pin_label_semantics",
    "validate_dataset_grids",
    "collect_provenance",
    "ingest_dataset",
]

RESEARCH_PROTOTYPE_WARNING = (
    "Research prototype. Trained and evaluated using simulated vascular-like "
    "abnormalities, not confirmed human post-angioplasty lesions."
)

# ---------------------------------------------------------------------------
# Layout / label-scheme constants (fitness note Q1/Q2/Q4)
# ---------------------------------------------------------------------------

SUBJECT_PREFIX = "QUADRA_HC_"
SESSION_NAMES: tuple[str, ...] = ("Test", "Retest")
SEGMENTATION_GROUPS: tuple[str, ...] = (
    "Body-Composition",
    "Cardiac",
    "Digestive",
    "Muscles",
    "Organs",
    "Peripheral-Bones",
    "Ribs",
    "Vertebrae",
)

# group -> {anatomy_name: integer_label}. Iliac artery labels live in the
# Cardiac group; femur labels in Peripheral-Bones. Frozen at ingestion (see
# module docstring).
LABEL_SCHEME: dict[str, dict[str, int]] = {
    "Cardiac": {"iliac_left": 7, "iliac_right": 8},
    "Peripheral-Bones": {"femur_left": 5, "femur_right": 6},
}

SUBJECT_COUNT = 48
SESSIONS_PER_SUBJECT = 2
SESSION_COUNT = SUBJECT_COUNT * SESSIONS_PER_SUBJECT
FILES_PER_SESSION = 2 + len(SEGMENTATION_GROUPS)  # PET + CT + 8 segmentation groups
NIFTI_COUNT = SESSION_COUNT * FILES_PER_SESSION


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SessionPaths:
    """Resolved on-disk paths for one subject/session. Local-only: this
    dataclass is never shipped to a tracked file or chat response (see the
    data-safety rules governing this module's callers).
    """

    subject: str
    session: str
    pet: Path
    ct: Path
    segmentations: dict[str, Path]

    def segmentation(self, group: str) -> Path:
        return self.segmentations[group]


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """The reconciled 48-subject/96-session/960-file inventory."""

    root: Path
    sessions: tuple[SessionPaths, ...]

    @property
    def subject_count(self) -> int:
        return len({s.subject for s in self.sessions})

    @property
    def session_count(self) -> int:
        return len(self.sessions)

    @property
    def nifti_count(self) -> int:
        return sum(2 + len(s.segmentations) for s in self.sessions)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class IngestErrorCode(StrEnum):
    """Typed, code-only reasons dataset discovery/reconciliation failed."""

    IMAGING_ROOT_NOT_FOUND = "IMAGING_ROOT_NOT_FOUND"
    SUBJECT_COUNT_MISMATCH = "SUBJECT_COUNT_MISMATCH"
    MISSING_SESSION_DIRECTORY = "MISSING_SESSION_DIRECTORY"
    MISSING_SESSION_FILE = "MISSING_SESSION_FILE"
    MISSING_SEGMENTATION_GROUP = "MISSING_SEGMENTATION_GROUP"
    SESSION_COUNT_MISMATCH = "SESSION_COUNT_MISMATCH"
    NIFTI_COUNT_MISMATCH = "NIFTI_COUNT_MISMATCH"


class IngestError(ValueError):
    """Raised on any dataset-discovery/reconciliation failure. ``detail``
    carries only non-identifying counts/ordinals/group names -- never a raw
    path or subject identifier.
    """

    def __init__(
        self, code: IngestErrorCode, detail: dict[str, object] | None = None
    ) -> None:
        super().__init__(code.value)
        self.code = code
        self.detail = detail or {}


def discover_dataset(root: Path) -> DatasetManifest:
    """Walk ``<root>/Imaging Data/`` and reconcile it against the expected
    48-subject/96-session/960-file inventory (fitness note Q1). Raises
    :class:`IngestError` closed on the first structural disagreement.
    """
    root = Path(root)
    imaging_root = root / "Imaging Data"
    if not imaging_root.is_dir():
        raise IngestError(IngestErrorCode.IMAGING_ROOT_NOT_FOUND)

    subject_dirs = sorted(
        p
        for p in imaging_root.iterdir()
        if p.is_dir() and p.name.startswith(SUBJECT_PREFIX)
    )
    if len(subject_dirs) != SUBJECT_COUNT:
        raise IngestError(
            IngestErrorCode.SUBJECT_COUNT_MISMATCH,
            {"found": len(subject_dirs), "expected": SUBJECT_COUNT},
        )

    sessions: list[SessionPaths] = []
    for subject_ordinal, subject_dir in enumerate(subject_dirs):
        subject = subject_dir.name
        for session_name in SESSION_NAMES:
            session_dir = subject_dir / session_name
            if not session_dir.is_dir():
                raise IngestError(
                    IngestErrorCode.MISSING_SESSION_DIRECTORY,
                    {"subject_ordinal": subject_ordinal, "session": session_name},
                )
            pet_path = session_dir / f"{subject}_{session_name}_PT-SUV.nii.gz"
            ct_path = session_dir / f"{subject}_{session_name}_CT-AC.nii.gz"
            if not pet_path.is_file() or not ct_path.is_file():
                raise IngestError(
                    IngestErrorCode.MISSING_SESSION_FILE,
                    {"subject_ordinal": subject_ordinal, "session": session_name},
                )
            seg_dir = session_dir / "Segmentations"
            segmentations: dict[str, Path] = {}
            for group in SEGMENTATION_GROUPS:
                seg_path = seg_dir / f"{subject}_{session_name}_{group}.nii.gz"
                if not seg_path.is_file():
                    raise IngestError(
                        IngestErrorCode.MISSING_SEGMENTATION_GROUP,
                        {
                            "subject_ordinal": subject_ordinal,
                            "session": session_name,
                            "group": group,
                        },
                    )
                segmentations[group] = seg_path
            sessions.append(
                SessionPaths(
                    subject=subject,
                    session=session_name,
                    pet=pet_path,
                    ct=ct_path,
                    segmentations=segmentations,
                )
            )

    manifest = DatasetManifest(root=root, sessions=tuple(sessions))
    if manifest.session_count != SESSION_COUNT:
        raise IngestError(
            IngestErrorCode.SESSION_COUNT_MISMATCH,
            {"found": manifest.session_count, "expected": SESSION_COUNT},
        )
    if manifest.nifti_count != NIFTI_COUNT:
        raise IngestError(
            IngestErrorCode.NIFTI_COUNT_MISMATCH,
            {"found": manifest.nifti_count, "expected": NIFTI_COUNT},
        )
    return manifest


# ---------------------------------------------------------------------------
# Label-semantics pin (highest-stakes check; see module docstring)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LabelPinSessionResult:
    session_ordinal: int
    passed: bool
    missing_labels: tuple[str, ...]
    non_integer_dtype_groups: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LabelPinReport:
    session_results: tuple[LabelPinSessionResult, ...]

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.session_results)

    @property
    def failing_ordinals(self) -> tuple[int, ...]:
        return tuple(r.session_ordinal for r in self.session_results if not r.passed)


class LabelPinErrorCode(StrEnum):
    LABEL_SEMANTICS_MISMATCH = "LABEL_SEMANTICS_MISMATCH"


class LabelPinError(ValueError):
    """Raised when any session disagrees with the frozen label-semantics
    mapping (missing label, or a non-integer-dtype segmentation array).
    Carries the full :class:`LabelPinReport` (0-based session ordinals plus
    anatomy/group name strings only -- never a subject identifier or path).
    """

    def __init__(self, code: LabelPinErrorCode, report: LabelPinReport) -> None:
        super().__init__(code.value)
        self.code = code
        self.report = report


def _load_array(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(path).dataobj)


def pin_label_semantics(
    manifest: DatasetManifest, *, sessions: Sequence[SessionPaths] | None = None
) -> LabelPinReport:
    """Verify :data:`LABEL_SCHEME` holds for every session in ``manifest``
    (or the caller-supplied ``sessions`` subset). Raises
    :class:`LabelPinError` if any session disagrees.
    """
    target_sessions = sessions if sessions is not None else manifest.sessions
    results: list[LabelPinSessionResult] = []
    for ordinal, session in enumerate(target_sessions):
        missing: list[str] = []
        bad_dtype: list[str] = []
        for group, names_by_label in LABEL_SCHEME.items():
            data = _load_array(session.segmentation(group))
            if not np.issubdtype(data.dtype, np.integer):
                bad_dtype.append(group)
                continue
            for name, label in names_by_label.items():
                if not np.any(data == label):
                    missing.append(name)
        results.append(
            LabelPinSessionResult(
                session_ordinal=ordinal,
                passed=not missing and not bad_dtype,
                missing_labels=tuple(missing),
                non_integer_dtype_groups=tuple(bad_dtype),
            )
        )
    report = LabelPinReport(session_results=tuple(results))
    if not report.all_passed:
        raise LabelPinError(LabelPinErrorCode.LABEL_SEMANTICS_MISMATCH, report)
    return report


# ---------------------------------------------------------------------------
# Grid validation (delegates entirely to geometry.validate_nifti_grid, P1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GridValidationSessionResult:
    session_ordinal: int
    pet_passed: bool
    ct_passed: bool
    pet_error_code: str | None
    ct_error_code: str | None


@dataclass(frozen=True, slots=True)
class GridValidationReport:
    session_results: tuple[GridValidationSessionResult, ...]

    @property
    def all_passed(self) -> bool:
        return all(r.pet_passed and r.ct_passed for r in self.session_results)

    @property
    def failing_ordinals(self) -> tuple[int, ...]:
        return tuple(
            r.session_ordinal
            for r in self.session_results
            if not (r.pet_passed and r.ct_passed)
        )


class GridValidationErrorCode(StrEnum):
    GRID_VALIDATION_FAILED = "GRID_VALIDATION_FAILED"


class GridValidationError(ValueError):
    def __init__(
        self, code: GridValidationErrorCode, report: GridValidationReport
    ) -> None:
        super().__init__(code.value)
        self.code = code
        self.report = report


def validate_dataset_grids(
    manifest: DatasetManifest, *, sessions: Sequence[SessionPaths] | None = None
) -> GridValidationReport:
    """Run :func:`src.vascutrace.geometry.validate_nifti_grid` (P1) over
    every session's PET and CT grid. Header-only (does not load pixel
    data), so this is cheap even across the full cohort. Raises
    :class:`GridValidationError` if any grid fails validation.
    """
    target_sessions = sessions if sessions is not None else manifest.sessions
    results: list[GridValidationSessionResult] = []
    for ordinal, session in enumerate(target_sessions):
        pet_ok, pet_code = True, None
        ct_ok, ct_code = True, None
        try:
            validate_nifti_grid(nib.load(session.pet))
        except GeometryContractError as exc:
            pet_ok, pet_code = False, exc.code.value
        try:
            validate_nifti_grid(nib.load(session.ct))
        except GeometryContractError as exc:
            ct_ok, ct_code = False, exc.code.value
        results.append(
            GridValidationSessionResult(ordinal, pet_ok, ct_ok, pet_code, ct_code)
        )
    report = GridValidationReport(tuple(results))
    if not report.all_passed:
        raise GridValidationError(
            GridValidationErrorCode.GRID_VALIDATION_FAILED, report
        )
    return report


# ---------------------------------------------------------------------------
# Provenance recording (documentary + per-session content hashes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProvenanceReport:
    """Aggregate, non-identifying provenance record.

    ``header_text_present_count`` counts sessions where any of the
    ``descrip``/``aux_file``/``intent_name`` NIfTI header text fields on
    either checked segmentation group were non-empty -- a full-cohort
    re-check of the fitness note's Q2 spot-sample finding (all empty).
    ``moose_version_documentary`` is never read from a local file (no local
    file carries it, per the fitness note); it is carried here purely as a
    documentary label sourced from this project's own research corpus, and
    is explicitly NOT re-verified against the external QUADRA technical
    descriptor in this module (network access is off for Phase 2 work).
    """

    sessions_checked: int
    header_text_present_count: int
    moose_version_documentary: str | None


_MOOSE_VERSION_DOCUMENTARY = (
    "3.0.13"  # documentary only; see ProvenanceReport docstring
)


def _header_text_nonempty(img: nib.Nifti1Image) -> bool:
    header = img.header
    for field_name in ("descrip", "aux_file", "intent_name"):
        raw = bytes(header[field_name]).rstrip(b"\x00").strip()
        if raw:
            return True
    return False


def collect_provenance(
    manifest: DatasetManifest, *, sessions: Sequence[SessionPaths] | None = None
) -> ProvenanceReport:
    """Header-only provenance scan (Cardiac + Peripheral-Bones groups, the
    two groups the label pin already touches) over ``sessions`` (or the
    full manifest). Never raises -- provenance is informational, not a gate.
    """
    target_sessions = sessions if sessions is not None else manifest.sessions
    present_count = 0
    for session in target_sessions:
        any_present = False
        for group in ("Cardiac", "Peripheral-Bones"):
            img = nib.load(session.segmentation(group))
            if _header_text_nonempty(img):
                any_present = True
        if any_present:
            present_count += 1
    return ProvenanceReport(
        sessions_checked=len(target_sessions),
        header_text_present_count=present_count,
        moose_version_documentary=_MOOSE_VERSION_DOCUMENTARY,
    )


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IngestResult:
    manifest: DatasetManifest
    label_pin_report: LabelPinReport
    grid_validation_report: GridValidationReport
    provenance_report: ProvenanceReport


def ingest_dataset(
    root: Path, *, sessions: Iterable[SessionPaths] | None = None
) -> IngestResult:
    """Run discovery, grid validation, label-semantics pin, and provenance
    collection, in that order, over ``root``. Raises the first typed error
    encountered. ``sessions`` (if given) restricts the label-pin/grid
    -validation/provenance passes to a subset of the discovered manifest
    (used for fast local smoke checks); discovery itself always reconciles
    the full on-disk tree.
    """
    manifest = discover_dataset(root)
    target = tuple(sessions) if sessions is not None else manifest.sessions
    grid_report = validate_dataset_grids(manifest, sessions=target)
    label_report = pin_label_semantics(manifest, sessions=target)
    provenance_report = collect_provenance(manifest, sessions=target)
    return IngestResult(
        manifest=manifest,
        label_pin_report=label_report,
        grid_validation_report=grid_report,
        provenance_report=provenance_report,
    )
