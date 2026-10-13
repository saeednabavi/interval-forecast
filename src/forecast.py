"""Point forecast plus split-conformal prediction intervals."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


def sample(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    depth = rng.uniform(20, 180, n)
    throughput = rng.uniform(10, 80, n)
    reagent = rng.uniform(0.2, 2.5, n)
    ore = rng.integers(0, 3, n)
    ore_oh = np.eye(3)[ore]
    grade = (
        0.8
        + 0.012 * depth
        - 0.00003 * depth ** 2
        + 0.004 * throughput
        + 0.35 * reagent
        + ore_oh @ np.array([0.0, 0.4, -0.25])
        + rng.normal(0, 0.18, n)
    )
    features = np.column_stack([depth, throughput, reagent, ore_oh])
    return features, grade


def conformal_quantile(residuals: np.ndarray, coverage: float) -> float:
    n = len(residuals)
    level = np.ceil((n + 1) * coverage) / n
    level = min(float(level), 1.0)
    return float(np.quantile(residuals, level, method="higher"))


class IntervalModel:
    def __init__(self, coverage: float = 0.9) -> None:
        self.coverage = coverage
        self.regressor = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.08,
            random_state=7,
        )
        self.radius = 0.0

    def fit(self, features: np.ndarray, target: np.ndarray, cal_fraction: float = 0.3) -> None:
        cut = int(len(target) * (1 - cal_fraction))
        self.regressor.fit(features[:cut], target[:cut])
        residual = np.abs(target[cut:] - self.regressor.predict(features[cut:]))
        self.radius = conformal_quantile(residual, self.coverage)

    def predict_interval(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        center = self.regressor.predict(features)
        return center, center - self.radius, center + self.radius


def interval_report(target: np.ndarray, low: np.ndarray, high: np.ndarray, nominal: float) -> dict:
    covered = (target >= low) & (target <= high)
    return {
        "nominal_coverage": nominal,
        "empirical_coverage": float(covered.mean()),
        "mean_width": float(np.mean(high - low)),
        "n_test": int(len(target)),
    }
