"""Run from the repository root: python -m ctmot.proto.run_prediction --help."""

import argparse
import json
from pathlib import Path

import yaml

from .predictor import predictor_from_config


def main():
    parser = argparse.ArgumentParser(description="Query a 6D CV/CA state using RK45.")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent / "config" / "prediction_cv.yaml")
    parser.add_argument("--state", type=float, nargs=6, required=True,
                        metavar=("PX", "PY", "PZ", "VX", "VY", "VZ"))
    parser.add_argument("--state-time-us", type=int, required=True)
    parser.add_argument("--query-time-us", type=int, required=True)
    args = parser.parse_args()
    try:
        with args.config.open() as stream:
            predictor = predictor_from_config(yaml.safe_load(stream))
        result = predictor.predict(args.state, args.state_time_us, args.query_time_us)
        print(json.dumps(result.to_dict(), indent=2, allow_nan=False))
    except (OSError, TypeError, ValueError, RuntimeError, yaml.YAMLError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
