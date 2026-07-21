"""Run and persist the VascuTrace agentic/product evaluation suite."""

import argparse
from pathlib import Path

from vascutrace.evaluation import run_evaluation_suite, write_evaluation_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation/summary.json"),
    )
    args = parser.parse_args()
    summary = run_evaluation_suite(args.output.parent)
    write_evaluation_summary(summary, args.output)
    print(
        f"Product evaluation: {summary.passed} passed, {summary.failed} failed; "
        f"report={args.output}"
    )
    raise SystemExit(0 if summary.all_passed else 1)


if __name__ == "__main__":
    main()
