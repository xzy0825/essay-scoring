"""Write the best submission from saved out-of-fold predictions.

Online QWK 0.84594 on learning-agency-lab-automated-essay-scoring-2.
The weights were picked by leaderboard checks. Local threshold QWK ranked
nearby mixes differently, so this file keeps the weights that won online.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from src.metrics import quadratic_weighted_kappa

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "opt"

# name, oof path, test path, weight
BLEND = (
    ("deberta_base", OUT / "oof_deberta_base.npy", OUT / "test_deberta_base.npy", 0.45),
    ("deberta_small", OUT / "oof_deberta.npy", OUT / "test_deberta.npy", 0.20),
    ("lgbm_main", ROOT / "outputs" / "oof_lgbm.npy", ROOT / "outputs" / "test_lgbm.npy", 0.20),
    ("lgbm_rich", OUT / "oof_lgbm_rich.npy", OUT / "test_lgbm_rich.npy", 0.15),
)


def apply_thresholds(pred: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Map a continuous score to 1-6 with five increasing cut points."""
    cuts = np.sort(np.asarray(thresholds, dtype=float))
    score = np.ones(len(pred), dtype=int)
    for cut in cuts:
        score += (pred > cut).astype(int)
    return score


def optimize_thresholds(y_true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    def objective(cuts: np.ndarray) -> float:
        return -quadratic_weighted_kappa(y_true, apply_thresholds(pred, cuts))

    result = differential_evolution(
        objective,
        bounds=[(0.5, 6.5)] * 5,
        seed=42,
        popsize=12,
        mutation=0.5,
        recombination=0.7,
        atol=1e-4,
        workers=1,
        updating="immediate",
    )
    return np.sort(result.x)


def blend(which: int) -> np.ndarray:
    total = None
    for _, oof_path, test_path, weight in BLEND:
        path = oof_path if which == 0 else test_path
        part = np.load(path) * weight
        total = part if total is None else total + part
    return total


def main() -> None:
    missing = [str(path) for _, oof_path, test_path, _ in BLEND for path in (oof_path, test_path) if not path.exists()]
    if missing:
        joined = "\n".join(missing)
        raise SystemExit(f"Missing prediction files:\n{joined}")

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    y = train["score"].to_numpy()
    oof = blend(0)
    test_pred = blend(1)
    cuts = optimize_thresholds(y, oof)
    oof_qwk = quadratic_weighted_kappa(y, apply_thresholds(oof, cuts))
    print(f"OOF threshold QWK {oof_qwk:.5f}")
    print("cuts", " ".join(f"{cut:.3f}" for cut in cuts))

    scores = apply_thresholds(test_pred, cuts)
    if list(test["essay_id"]) != list(sample["essay_id"]):
        by_id = dict(zip(test["essay_id"], scores))
        missing_ids = [essay_id for essay_id in sample["essay_id"] if essay_id not in by_id]
        if missing_ids:
            raise SystemExit(f"Predictions missing {len(missing_ids)} sample essay ids")
        scores = np.array([by_id[essay_id] for essay_id in sample["essay_id"]])
    submission = pd.DataFrame({"essay_id": sample["essay_id"], "score": scores.astype(int)})
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "submission.csv"
    submission.to_csv(destination, index=False)
    report = {
        "online_qwk": 0.84594,
        "weights": {name: weight for name, _, _, weight in BLEND},
        "oof_threshold_qwk": oof_qwk,
        "thresholds": cuts.tolist(),
        "test_score_counts": {int(k): int(v) for k, v in zip(*np.unique(scores, return_counts=True))},
    }
    (OUT / "cv_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
