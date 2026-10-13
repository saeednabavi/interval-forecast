"""Train a grade model and check how often the 90% interval covers the truth."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from forecast import IntervalModel, interval_report, sample  # noqa: E402

REPORT = ROOT / "models" / "metrics.json"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(7)
    features, target = sample(1000, rng)
    train_end = 750
    model = IntervalModel(coverage=0.9)
    model.fit(features[:train_end], target[:train_end])
    center, low, high = model.predict_interval(features[train_end:])
    report = interval_report(target[train_end:], low, high, model.coverage)
    report["mae"] = float(mean_absolute_error(target[train_end:], center))
    report["interval_radius"] = model.radius
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"test MAE:            {report['mae']:.3f}")
    print(f"nominal coverage:    {report['nominal_coverage']:.2f}")
    print(f"empirical coverage:  {report['empirical_coverage']:.3f}")
    print(f"mean interval width: {report['mean_width']:.3f}")


if __name__ == "__main__":
    main()
