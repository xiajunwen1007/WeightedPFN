"""WeightedPFN on data with missing values: drop incomplete rows, weight complete rows by 1 / P(complete).

Simulated X+Y mechanism: whether a row is complete depends on x0 and on the outcome y; incomplete rows lose x3 and x4.
Run from this folder:  python example.py            (GPU if available, else CPU; ~1 min on CPU)
"""
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from weightedpfn import WeightedPFNClassifier, WeightedPFNRegressor

warnings.filterwarnings("ignore", message="Running on CPU")  # tabpfn: CPU with >1000 rows may be slow


def simulate(n, rng):
    X = rng.normal(size=(n, 5))
    y = np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 3] + 0.8 * X[:, 4] + rng.normal(size=n)
    return X, y


def inverse_probability_weights(always_observed, complete, folds=5):
    """1 / P(complete | always-observed variables), cross-fitted so no row is scored by a model that saw it."""
    pi = cross_val_predict(LogisticRegression(max_iter=1000), always_observed, complete, cv=folds, method="predict_proba")[:, 1]
    return 1.0 / np.clip(pi, 1e-4, 1.0)


rng = np.random.default_rng(0)
X, y = simulate(3000, rng)
X_test, y_test = simulate(2000, rng)  # target population: everyone, complete covariates

# Missingness: a row is complete with a probability that depends on x0 and y (X+Y mechanism).
p_complete = 1 / (1 + np.exp(-(-0.5 + 1.0 * X[:, 0] - 1.5 * (y - y.mean()) / y.std())))
complete = rng.random(len(y)) < p_complete
X_observed = X.copy()
X_observed[~complete, 3:] = np.nan  # x3, x4 missing in incomplete rows; x0, x1, x2 and y always observed
print(f"{complete.sum()} of {len(y)} rows are complete")

# Weights for the complete rows, estimated from the always-observed variables (x0, x1, x2, y).
always_observed = np.column_stack([X_observed[:, :3], y])
weights = inverse_probability_weights(always_observed, complete)[complete]
X_context, y_context = X_observed[complete], y[complete]

# Regression
for name, w in (("unweighted", None), ("weighted", weights)):
    model = WeightedPFNRegressor(device="auto").fit(X_context, y_context, sample_weight=w)
    rmse = np.sqrt(np.mean((model.predict(X_test) - y_test) ** 2))
    print(f"regression  {name:<10}  device={model.device_}  RMSE={rmse:.4f}")

# Classification: is y above the population median?
threshold = np.median(y)
for name, w in (("unweighted", None), ("weighted", weights)):
    model = WeightedPFNClassifier(device="auto").fit(X_context, (y_context > threshold).astype(int), sample_weight=w)
    accuracy = np.mean(model.predict(X_test) == (y_test > threshold))
    print(f"classification {name:<10}  device={model.device_}  accuracy={accuracy:.4f}")
