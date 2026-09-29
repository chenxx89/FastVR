"""Single-config FastVR training command."""

from __future__ import annotations

import argparse

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FastVR")
    parser.add_argument("--config", required=True, help="Training YAML configuration.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    from fastvr.training.config import load_training_config
    from fastvr.training.runner import run

    run(load_training_config(args.config))


if __name__ == "__main__":
    main()
