"""Train, evaluate"""
import argparse
import json
from pathlib import Path
if __package__:
    from .regression import read_data, experiment
else:
    from regression import read_data, experiment

TASK_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(TASK_DIR / "data/train.csv"))
    parser.add_argument("--output-dir", default=str(TASK_DIR / "models"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report, model = experiment(read_data(args.data, training=True), args.seed)
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory/"model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    (directory/"metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
