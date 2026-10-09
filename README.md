# DS Engineer Test Task

## Task 1: Counting islands

Run in PowerShell:
Get-Content task1_islands/example.txt | python task1_islands/solution.py

Run tests:
python -m unittest discover -s task1_islands -t . -v

## Task 2: Tabular regression

Install dependencies:
python -m pip install -r requirements.txt

Explore the data:
jupyter lab task2_regression/notebooks/eda.ipynb

Train:
python task2_regression/train.py

Predict using the saved model:
python task2_regression/predict.py

Run regression tests:
python -m unittest discover -s task2_regression -t . -v

The model selects feature 6 squared and feature 7.
Holdout RMSE is approximately 3.20e-13.
Prior full-dataset inspection suggested this relationship; the holdout estimate is not fully blind.
predictions.csv contains 10000 target predictions in original hidden-test order.
