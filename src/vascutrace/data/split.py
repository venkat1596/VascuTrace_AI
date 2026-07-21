"""Deterministic, sex-stratified, subject-level cohort split (VascuTrace
Phase 2 data pipeline).

RESEARCH_PROTOTYPE_WARNING
---------------------------------------------------------------------------
Research prototype. Trained and evaluated using simulated vascular-like
abnormalities, not confirmed human post-angioplasty lesions.
---------------------------------------------------------------------------

Split policy (frozen, per this implementation and certified in
``docs/p2_data_fitness_2026-07-16.md``, Q7):

- Seed ``20260713`` (``SPLIT_SEED``), ``numpy.random.default_rng``.
- Sex-stratified, subject-level: 8 validation / 8 test / 32 train subjects,
  drawn via Hamilton (largest-remainder) apportionment of the val/test
  target counts proportional to each sex's share of the (remaining) pool,
  then random subject selection within each sex via the seeded generator.
- Subject-level, never session-level: a subject's ``Test`` and ``Retest``
  sessions always land in the same partition, by construction (the split
  never looks at session identity at all).
- :func:`leakage_proof` is a mechanical, zero-overlap check run at both the
  subject and session grain -- see its docstring.

This module never reads ``Data/`` itself except through
:func:`load_subject_sex_table` (a thin, isolated I/O boundary); every other
function operates on a plain ``{subject_id: sex}`` mapping, so the core
split/leakage-proof algorithm is fully testable with synthetic subject IDs
and needs no local dataset access.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "RESEARCH_PROTOTYPE_WARNING",
    "SPLIT_SEED",
    "SPLIT_SCHEMA_VERSION",
    "N_VAL_SUBJECTS",
    "N_TEST_SUBJECTS",
    "SubjectSplit",
    "LeakageProofResult",
    "load_subject_sex_table",
    "stratified_subject_split",
    "subject_partition_map",
    "leakage_proof",
    "assert_no_leakage",
    "save_subject_split",
    "load_subject_split",
]

RESEARCH_PROTOTYPE_WARNING = (
    "Research prototype. Trained and evaluated using simulated vascular-like "
    "abnormalities, not confirmed human post-angioplasty lesions."
)

SPLIT_SEED = 20260713
SPLIT_SCHEMA_VERSION = "p2-split-v1"
N_VAL_SUBJECTS = 8
N_TEST_SUBJECTS = 8


@dataclass(frozen=True, slots=True)
class SubjectSplit:
    schema_version: str
    seed: int
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]


# ---------------------------------------------------------------------------
# Demographics I/O (isolated boundary; not exercised by CPU-only tests)
# ---------------------------------------------------------------------------


def load_subject_sex_table(demographics_path: Path) -> dict[str, str]:
    """Read the QUADRA_HC demographics workbook and return
    ``{subject_id: sex}`` for all 48 subjects, with ``subject_id`` formatted
    as ``QUADRA_HC_{index:03d}`` to match the imaging-tree directory names.
    """
    frame = pd.read_excel(demographics_path, header=[0, 1])
    subject_col, sex_col = frame.columns[0], frame.columns[1]
    return {
        f"QUADRA_HC_{int(row[subject_col]):03d}": str(row[sex_col]).strip()
        for _, row in frame.iterrows()
    }


# ---------------------------------------------------------------------------
# Apportionment + split
# ---------------------------------------------------------------------------


def _largest_remainder_apportionment(
    pool_sizes: Mapping[str, int], target_total: int
) -> dict[str, int]:
    """Hamilton (largest-remainder) apportionment of ``target_total`` seats
    across the groups in ``pool_sizes``, proportional to each group's size.
    Deterministic: ties in the fractional remainder are broken by group-key
    sort order, not randomly.
    """
    total_pool = sum(pool_sizes.values())
    if total_pool == 0 or target_total == 0:
        return dict.fromkeys(pool_sizes, 0)
    raw = {key: target_total * size / total_pool for key, size in pool_sizes.items()}
    floors = {key: int(np.floor(value)) for key, value in raw.items()}
    remainder_needed = target_total - sum(floors.values())
    ranked = sorted(pool_sizes.keys(), key=lambda key: (-(raw[key] - floors[key]), key))
    for key in ranked[:remainder_needed]:
        floors[key] += 1
    return floors


def stratified_subject_split(
    subject_sex: Mapping[str, str],
    *,
    seed: int = SPLIT_SEED,
    n_val: int = N_VAL_SUBJECTS,
    n_test: int = N_TEST_SUBJECTS,
) -> SubjectSplit:
    """Deterministic, sex-stratified, subject-level split.

    Allocates ``n_val`` subjects (via largest-remainder apportionment
    proportional to each sex's share of the full pool), then ``n_test``
    subjects (proportional to each sex's share of the *remaining* pool
    after val is removed); every subject not selected for val or test is
    train. Subject selection within each sex/stage uses
    ``numpy.random.default_rng(seed)``, consumed in deterministic
    (sorted-subject-id) order, so the same ``(subject_sex, seed)`` pair
    always reproduces the identical split.
    """
    rng = np.random.default_rng(seed)
    by_sex: dict[str, list[str]] = defaultdict(list)
    for subject, sex in subject_sex.items():
        by_sex[sex].append(subject)
    for sex in by_sex:
        by_sex[sex] = sorted(by_sex[sex])

    def _draw(
        pool_by_sex: Mapping[str, list[str]], counts: Mapping[str, int]
    ) -> tuple[list[str], dict[str, list[str]]]:
        chosen: list[str] = []
        remainder: dict[str, list[str]] = {}
        for sex in sorted(pool_by_sex):
            subjects = pool_by_sex[sex]
            order = rng.permutation(len(subjects))
            take = counts.get(sex, 0)
            chosen.extend(subjects[int(i)] for i in order[:take])
            remainder[sex] = sorted(subjects[int(i)] for i in order[take:])
        return chosen, remainder

    val_counts = _largest_remainder_apportionment(
        {sex: len(subjects) for sex, subjects in by_sex.items()}, n_val
    )
    val, after_val = _draw(by_sex, val_counts)

    test_counts = _largest_remainder_apportionment(
        {sex: len(subjects) for sex, subjects in after_val.items()}, n_test
    )
    test, after_test = _draw(after_val, test_counts)

    train = sorted(subject for subjects in after_test.values() for subject in subjects)

    return SubjectSplit(
        schema_version=SPLIT_SCHEMA_VERSION,
        seed=seed,
        train=tuple(train),
        val=tuple(sorted(val)),
        test=tuple(sorted(test)),
    )


def subject_partition_map(split: SubjectSplit) -> dict[str, str]:
    """``{subject_id: "train" | "val" | "test"}`` for every subject in
    ``split``.
    """
    mapping: dict[str, str] = {}
    for subject in split.train:
        mapping[subject] = "train"
    for subject in split.val:
        mapping[subject] = "val"
    for subject in split.test:
        mapping[subject] = "test"
    return mapping


# ---------------------------------------------------------------------------
# Leakage proof
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LeakageProofResult:
    """Mechanical zero-overlap leakage proof result. ``passed`` is the
    single boolean gate; the other fields let a caller (or test) pinpoint
    exactly which check failed.
    """

    subject_level_disjoint: bool
    subject_level_full_coverage: bool
    session_level_no_subject_split: bool
    every_session_subject_known: bool
    passed: bool


def leakage_proof(
    split: SubjectSplit,
    *,
    all_subjects: Sequence[str] | None = None,
    sessions: Sequence[tuple[str, str]] | None = None,
) -> LeakageProofResult:
    """Mechanical, zero-overlap leakage proof, checked at both the subject
    and session grain.

    - **Subject level**: ``train``, ``val``, ``test`` are pairwise disjoint;
      if ``all_subjects`` is supplied, their union equals it exactly (full
      coverage, no subject silently dropped).
    - **Session level** (if ``sessions`` -- an iterable of
      ``(subject, session_name)`` pairs -- is supplied): every session's
      subject is a known split member, and no single subject's sessions are
      spread across more than one partition (the concrete leakage this
      implementation's split policy exists to prevent: a subject's Test and Retest
      sessions must always land in the same partition).
    """
    train_set, val_set, test_set = set(split.train), set(split.val), set(split.test)
    subject_level_disjoint = (
        not (train_set & val_set)
        and not (train_set & test_set)
        and not (val_set & test_set)
    )

    if all_subjects is not None:
        subject_level_full_coverage = (train_set | val_set | test_set) == set(
            all_subjects
        )
    else:
        subject_level_full_coverage = True

    every_session_subject_known = True
    session_level_no_subject_split = True
    if sessions is not None:
        partition_of = subject_partition_map(split)
        seen_partitions: dict[str, set[str]] = defaultdict(set)
        for subject, _session_name in sessions:
            if subject not in partition_of:
                every_session_subject_known = False
                continue
            seen_partitions[subject].add(partition_of[subject])
        session_level_no_subject_split = all(
            len(v) <= 1 for v in seen_partitions.values()
        )

    passed = (
        subject_level_disjoint
        and subject_level_full_coverage
        and session_level_no_subject_split
        and every_session_subject_known
    )
    return LeakageProofResult(
        subject_level_disjoint=subject_level_disjoint,
        subject_level_full_coverage=subject_level_full_coverage,
        session_level_no_subject_split=session_level_no_subject_split,
        every_session_subject_known=every_session_subject_known,
        passed=passed,
    )


def assert_no_leakage(result: LeakageProofResult) -> None:
    """Raise ``AssertionError`` if ``result.passed`` is ``False`` -- the
    pipeline's hard leakage gate.
    """
    if not result.passed:
        raise AssertionError(f"leakage proof failed: {result}")


# ---------------------------------------------------------------------------
# Persistence (writes to the caller-supplied, gitignored output root)
# ---------------------------------------------------------------------------


def save_subject_split(
    split: SubjectSplit,
    output_root: Path,
    *,
    sex_by_subject: Mapping[str, str] | None = None,
) -> Path:
    """Write ``split`` to ``<output_root>/subject_split.json``.
    ``output_root`` is caller-supplied and MUST be a gitignored local path
    -- this module does not enforce that, the caller does. Contains subject
    identifiers by design (a local, gitignored artifact) -- never write this
    payload to a tracked file or a chat response.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": split.schema_version,
        "seed": split.seed,
        "n_train": len(split.train),
        "n_val": len(split.val),
        "n_test": len(split.test),
        "train": list(split.train),
        "val": list(split.val),
        "test": list(split.test),
    }
    if sex_by_subject is not None:
        payload["sex_counts"] = {
            partition: dict(
                sorted(Counter(sex_by_subject[subject] for subject in subjects).items())
            )
            for partition, subjects in (
                ("train", split.train),
                ("val", split.val),
                ("test", split.test),
            )
        }
    path = output_root / "subject_split.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_subject_split(path: Path) -> SubjectSplit:
    payload = json.loads(Path(path).read_text())
    return SubjectSplit(
        schema_version=payload["schema_version"],
        seed=payload["seed"],
        train=tuple(payload["train"]),
        val=tuple(payload["val"]),
        test=tuple(payload["test"]),
    )
