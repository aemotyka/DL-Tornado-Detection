"""Command-line inference for a frozen model bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from tornado_detection.inference import TornadoDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tornet-detect",
        description=(
            "Run the frozen offline tornado detector on one "
            "four-frame TorNet NetCDF file."
        ),
    )
    parser.add_argument(
        "--bundle",
        required=True,
        type=Path,
        help="Directory containing the release model bundle",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Four-frame TorNet NetCDF input file",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination JSON file",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device (default: auto)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    detector = TornadoDetector.from_bundle(
        arguments.bundle,
        device=arguments.device,
    )
    result = detector.predict_file(arguments.input)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result.to_dict(), indent=2) + "\n"
    )
    print(f"wrote: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
