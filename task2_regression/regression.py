"""Regression on the tabular data"""
import numpy as np
import pandas as pd

FEATURES = [str(i) for i in range(53)]


def read_data(path, training=False):
    frame = pd.read_csv(path)
    expected = FEATURES + (["target"] if training else [])
    if set(frame.columns) != set(expected) or len(frame.columns) != len(expected):
        raise ValueError(f"Expected exactly these columns: {expected}")
    frame = frame[expected].apply(pd.to_numeric, errors="raise")
    if frame.empty or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Data must be nonempty, numeric and finite")
    return frame


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.asarray(actual)-predicted)**2)))


def split_indices(n, seed=42):
    if n < 10:
        raise ValueError("At least 10 rows required")
    order = np.random.default_rng(seed).permutation(n)
    return order[:int(.6*n)], order[int(.6*n):int(.8*n)], order[int(.8*n):]


def expand(x):
    return np.column_stack([x, x*x])


def fit_sparse(x, y, terms):
    """Forward selection on normalized linear and square features.

    All means, scales and selected terms are learned only from supplied rows.
    Refit selected coefficients jointly after each forward selection step.
    """
    expanded = expand(x)
    mean = expanded.mean(axis=0)
    scale = expanded.std(axis=0)
    scale[scale == 0] = 1
    normalized = (expanded-mean)/scale
    residual = y-y.mean()
    selected = []
    for _ in range(terms):
        scores = np.abs(normalized.T @ residual)
        scores[selected] = -np.inf
        selected.append(int(np.argmax(scores)))
        design = np.column_stack([np.ones(len(x)), normalized[:, selected]])
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = y-design @ coef
    return {"selected": selected, "mean": mean.tolist(), "scale": scale.tolist(),
            "coef": coef.tolist(), "features": FEATURES, "format_version": 1}


def predict_sparse(model, x):
    selected = model["selected"]
    normalized = (expand(x)-np.array(model["mean"]))/np.array(model["scale"])
    return np.column_stack([np.ones(len(x)), normalized[:, selected]]) @ np.array(model["coef"])


def term_names(model):
    return [FEATURES[i] if i < 53 else f"{FEATURES[i-53]}^2" for i in model["selected"]]


def experiment(frame, seed=42):
    """Choose complexity on validation; report final holdout exactly once."""
    x, y = frame[FEATURES].to_numpy(float), frame.target.to_numpy(float)
    train, valid, test = split_indices(len(frame), seed)
    baseline = np.full(len(valid), y[train].mean())
    raw_design = np.column_stack([np.ones(len(train)), x[train]])
    linear_coef = np.linalg.lstsq(raw_design, y[train], rcond=None)[0]
    results = {"mean_validation_rmse": rmse(y[valid], baseline),
               "linear_validation_rmse": rmse(y[valid], np.column_stack([np.ones(len(valid)), x[valid]]) @ linear_coef)}
    candidates = []
    for terms in (1, 2, 3, 5):
        model = fit_sparse(x[train], y[train], terms)
        score = rmse(y[valid], predict_sparse(model, x[valid]))
        candidates.append((score, terms))
    # Numerical differences below 1e-10 do not justify additional complexity.
    best_score = min(score for score, _ in candidates)
    terms = min(k for score, k in candidates if score <= best_score + 1e-10)
    development = np.concatenate([train, valid])
    evaluated = fit_sparse(x[development], y[development], terms)
    results.update({"seed": seed, "split_sizes": [len(train), len(valid), len(test)],
                    "candidates": [{"terms": k, "validation_rmse": s} for s, k in candidates],
                    "selected_terms": term_names(evaluated), "terms": terms,
                    "holdout_rmse": rmse(y[test], predict_sparse(evaluated, x[test])),
                    "holdout_max_absolute_error": float(np.max(np.abs(y[test]-predict_sparse(evaluated, x[test]))))})
    return results, fit_sparse(x, y, terms)
