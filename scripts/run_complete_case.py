import argparse
from pathlib import Path

from vascutrace.pipeline import run_complete_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/complete_case")
    )
    args = parser.parse_args()
    print(run_complete_case(args.output_dir))


if __name__ == "__main__":
    main()
