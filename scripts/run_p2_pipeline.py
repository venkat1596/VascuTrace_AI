"""CLI entry point for the VascuTrace Phase 2 data pipeline.

Thin wrapper only -- all logic lives in ``src.vascutrace.data``. Never reads
or writes anything outside the caller-supplied ``--data-root`` (read-only)
and ``--output-root`` (a gitignored local path the caller is responsible for
choosing; this script does not enforce that it is gitignored).

Usage
-----
    uv run python scripts/run_p2_pipeline.py ingest --data-root Data/QUADRA_HC
    uv run python scripts/run_p2_pipeline.py split --data-root Data/QUADRA_HC --output-root data/processed/p2
    uv run python scripts/run_p2_pipeline.py crops --data-root Data/QUADRA_HC --output-root data/processed/p2 --limit 6

Every subcommand prints only aggregate counts/booleans -- never a subject
identifier or raw file path -- to stdout, matching this project's
aggregate-only data-safety discipline for anything that could end up in a
log or terminal transcript.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# This project's ``src/`` layout is not installed as a package (see
# ``pyproject.toml``, ``[tool.uv] package = false``); ``pytest`` gets the
# repo root on ``sys.path`` via ``pythonpath = ["."]``, but a plain
# ``python scripts/run_p2_pipeline.py`` invocation does not -- add it
# explicitly, matching ``tests/conftest.py``'s own approach.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vascutrace.data.crops import run_crop_pipeline
from src.vascutrace.data.ingest import ingest_dataset
from src.vascutrace.data.split import (
    leakage_proof,
    load_subject_sex_table,
    save_subject_split,
    stratified_subject_split,
)


def _cmd_ingest(args: argparse.Namespace) -> None:
    result = ingest_dataset(args.data_root)
    summary = {
        "subjects": result.manifest.subject_count,
        "sessions": result.manifest.session_count,
        "nifti_files": result.manifest.nifti_count,
        "label_pin_all_passed": result.label_pin_report.all_passed,
        "grid_validation_all_passed": result.grid_validation_report.all_passed,
        "provenance_header_text_present_count": (
            result.provenance_report.header_text_present_count
        ),
        "moose_version_documentary": result.provenance_report.moose_version_documentary,
    }
    print(json.dumps(summary, indent=2))


def _cmd_split(args: argparse.Namespace) -> None:
    demographics_path = args.data_root / "Demographics (All).xlsx"
    subject_sex = load_subject_sex_table(demographics_path)
    split = stratified_subject_split(subject_sex)
    result = ingest_dataset(args.data_root)
    sessions = [(s.subject, s.session) for s in result.manifest.sessions]
    proof = leakage_proof(split, all_subjects=list(subject_sex), sessions=sessions)
    path = save_subject_split(split, args.output_root, sex_by_subject=subject_sex)
    summary = {
        "n_train": len(split.train),
        "n_val": len(split.val),
        "n_test": len(split.test),
        "leakage_proof_passed": proof.passed,
        "output_path_relative_to_output_root": str(path.relative_to(args.output_root)),
    }
    print(json.dumps(summary, indent=2))


def _cmd_crops(args: argparse.Namespace) -> None:
    result = ingest_dataset(args.data_root)
    sessions = result.manifest.sessions
    if args.limit is not None:
        sessions = sessions[: args.limit]
    report = run_crop_pipeline(result.manifest, args.output_root, sessions=sessions)
    summary = {
        "sessions_built": len(report.outcomes),
        "qc_flagged_count": report.qc_flagged_count,
        "bbox_exceeded_count": report.bbox_exceeded_count,
        "mean_valid_coverage": report.mean_valid_coverage,
        "min_valid_coverage": report.min_valid_coverage,
    }
    print(json.dumps(summary, indent=2))


def _add_common_arguments(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--data-root",
        type=Path,
        default=Path("Data/QUADRA_HC"),
        help="Read-only dataset root.",
    )
    subparser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/p2"),
        help="Gitignored local output root (caller-verified).",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest", help="Discover + label-pin + grid-validate the full cohort."
    )
    _add_common_arguments(ingest_parser)

    split_parser = subparsers.add_parser(
        "split", help="Build and save the deterministic subject split."
    )
    _add_common_arguments(split_parser)

    crops_parser = subparsers.add_parser(
        "crops", help="Build a sample of crop bundles."
    )
    _add_common_arguments(crops_parser)
    crops_parser.add_argument(
        "--limit", type=int, default=None, help="Build only the first N sessions."
    )

    args = parser.parse_args()
    if args.command == "ingest":
        _cmd_ingest(args)
    elif args.command == "split":
        _cmd_split(args)
    elif args.command == "crops":
        _cmd_crops(args)


if __name__ == "__main__":
    main()
