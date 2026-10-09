# DS Engineer Test Task

Python implementation of three tasks: island counting, tabular regression, and a unified MNIST classification interface.

## Setup

Python 3.11 or newer is required. Run commands from the repository root.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

For a CPU-only PyTorch installation, first run:
`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Repository structure

```text
ds-engineer-test-task/
├── task1_islands/
│   ├── solution.py
│   ├── test_solution.py
│   └── example.txt
├── task2_regression/
│   ├── data/                       # train.csv and hidden_test.csv
│   ├── notebooks/eda.ipynb
│   ├── regression.py               # shared training and inference logic
│   ├── train.py
│   ├── predict.py
│   ├── predictions.csv
│   ├── models/                     # final model.json and metrics.json
│   ├── reports/                    # metrics, environment and verification
│   └── test_regression.py
├── task3_mnist/
│   ├── classifier.py
│   ├── interfaces.py
│   ├── models/
│   │   ├── cnn.py
│   │   ├── random_forest.py
│   │   └── random_model.py
│   └── tests/test_classifier.py
├── scripts/execute_notebook.py
├── requirements.txt
└── README.md
```

## Task 1 — Counting islands

```bash
python task1_islands/solution.py < task1_islands/example.txt
```

PowerShell:
```powershell
Get-Content task1_islands/example.txt | python task1_islands/solution.py
```

Input is M N followed by M*N binary cells. Output is a single integer.
Land is connected horizontally or vertically; diagonal contact does not connect islands.
The three supplied examples produce 2, 3 and 2 respectively.
An iterative DFS avoids recursion depth limits and preserves the input matrix.
Time complexity is O(MN), which is optimal for reading the complete matrix;
additional space is O(MN). Invalid dimensions, cell counts and nonbinary cells are rejected.

## Task 2 — Tabular regression

```bash
jupyter lab task2_regression/notebooks/eda.ipynb
python scripts/execute_notebook.py
python task2_regression/train.py --seed 42
python task2_regression/predict.py
```

Paths have defaults relative to the script directory. Custom paths are supported:
```bash
python task2_regression/train.py --data task2_regression/data/train.csv --output-dir task2_regression/models
python task2_regression/predict.py --data task2_regression/data/hidden_test.csv --model task2_regression/models/model.json --output task2_regression/predictions.csv
```

The submitted model can be used directly for inference without retraining.
Model parameters are stored as numeric JSON rather than pickle.
The prediction file contains 10,000 rows and one column, target, in the original hidden-test order.

### Exploratory analysis and method

The training dataset contains 90,000 rows, 53 numeric features and a target.
There are no missing values or complete duplicate rows.
The executed notebook includes descriptive statistics, unique counts, correlations
with raw and squared features, a target histogram and scatter plots of feature 6
and its square against the target.

Rows are split into training, validation and holdout partitions (60/20/20, seed 42).
Compare a mean baseline, linear least squares and forward selection over 106 raw
and squared features. Scaling and feature selection use only training rows.
Choose the number of terms on validation, preferring simpler models when RMSE
improvements are below 1e-10. Refit on training plus validation for holdout evaluation,
then fit the final submission model on all labeled rows.

The selected terms are feature 6 squared and feature 7:
`target = feature_6**2 + feature_7` within numerical precision.

| Model | Partition | RMSE |
|---|---|---:|
| Mean baseline | Validation | 28.9164 |
| Linear regression | Validation | 28.9224 |
| Sparse quadratic regression | Holdout | approximately 3.20e-13 |

Exact metrics are in task2_regression/models/metrics.json. Small numerical differences
can occur across NumPy/BLAS versions. Runtime versions are recorded in
task2_regression/reports/environment.json.

### Limitations

A random split assumes independent rows; temporal or grouped observations require
an appropriate split. Prior full-dataset inspection revealed the target relationship,
so the holdout is not completely blind to the hypothesis. The training procedure
rediscovers the terms without hardcoding feature indices. The relationship appears
synthetic; feature provenance and inference-time availability are unknown.
Hidden-test RMSE cannot be computed because its target values are unavailable.

## Task 3 — MNIST classification interface

```python
import numpy as np
from task3_mnist.classifier import DigitClassifier

classifier = DigitClassifier("rand", seed=42)
digit = classifier.predict(np.zeros((28, 28, 1), dtype=np.uint8))
```

DigitClassificationInterface defines the backend contract. CNN, Random Forest and
random implementations are separated into model modules. DigitClassifier accepts
cnn, rf or rand and returns a Python integer in [0,9] for an image of shape (28,28,1).
The public input convention is finite real pixels in [0,255], normalized internally
to [0,1]. Backends must use this same normalization convention.

- CNN receives a (28,28,1) image, converted internally to PyTorch (1,1,28,28).
  Inference uses eval mode and inference_mode; output must contain ten finite logits.
- Random Forest receives a flattened 784-element vector.
- Random model receives the central 10x10 crop, image[9:19,9:19,0].

For RF, pass a fitted sklearn RandomForestClassifier as backend:
`DigitClassifier("rf", backend=fitted_rf)`.
For CNN, create task3_mnist.models.cnn.build_cnn(), load trained weights with
load_state_dict(), and pass the model as backend:
`DigitClassifier("cnn", backend=trained_cnn)`.

MNIST training and trained weights are outside the requested scope.
Both the facade and backend training methods raise NotImplementedError.
Tests with a small fitted RF and an untrained CNN verify adaptation and inference
contracts; they do not establish MNIST classification accuracy.

## Verification

```bash
python -m unittest discover -s task1_islands -t . -v
python -m unittest discover -s task2_regression -t . -v
python -m unittest discover -s task3_mnist/tests -t . -v
# Run all tasks:
python -m unittest discover -s . -v
```

Checks cover the supplied island examples, diagonal connectivity, empty and invalid
matrices, input preservation, large components, out-of-sample nonlinear regression,
MNIST image validation, center cropping, RF adaptation and CNN inference.
The executed notebook and reports provide the experiment results. GitHub Actions
runs the unit tests on pushes and pull requests.
