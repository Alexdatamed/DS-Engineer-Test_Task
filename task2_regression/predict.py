"""predict.py python script for model inference on test data"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
if __package__:
    from .regression import FEATURES, read_data, predict_sparse
else:
    from regression import FEATURES, read_data, predict_sparse

TASK_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(TASK_DIR / "data/hidden_test.csv"))
    parser.add_argument("--model", default=str(TASK_DIR / "models/model.json"))
    parser.add_argument("--output", default=str(TASK_DIR / "predictions.csv"))
    args = parser.parse_args()
    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    if model.get("format_version") != 1 or model.get("features") != FEATURES:
        raise ValueError("Unsupported model schema")
    predictions = predict_sparse(model, read_data(args.data).to_numpy(float))
    if not np.isfinite(predictions).all():
        raise ValueError("Non-finite predictions")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"target": predictions}).to_csv(output, index=False)
    print(f"Saved {len(predictions)} predictions to {output}")


if __name__ == "__main__":
    main()
