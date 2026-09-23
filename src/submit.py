"""Pick the OOF-best score mapping and write submission.csv."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from src.metrics import as_score, quadratic_weighted_kappa

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"


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


def evaluate(name: str, y_true: np.ndarray, pred: np.ndarray) -> dict:
    rounded = quadratic_weighted_kappa(y_true, pred)
    cuts = optimize_thresholds(y_true, pred)
    tuned = quadratic_weighted_kappa(y_true, apply_thresholds(pred, cuts))
    print(f"{name} round {rounded:.5f} threshold {tuned:.5f} cuts {np.round(cuts, 3).tolist()}")
    return {
        "name": name,
        "round_qwk": rounded,
        "threshold_qwk": tuned,
        "thresholds": cuts.tolist(),
        "qwk": max(rounded, tuned),
        "use_thresholds": tuned >= rounded,
    }


def scores_for(pred: np.ndarray, result: dict) -> np.ndarray:
    if result["use_thresholds"]:
        return apply_thresholds(pred, np.asarray(result["thresholds"]))
    return as_score(pred)


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    y = train["score"].to_numpy()
    candidates: list[tuple[str, np.ndarray, np.ndarray]] = []

    lgbm_oof = OUT / "oof_lgbm.npy"
    deberta_oof = OUT / "oof_deberta.npy"
    if lgbm_oof.exists():
        candidates.append(("lgbm", np.load(lgbm_oof), np.load(OUT / "test_lgbm.npy")))
    if deberta_oof.exists():
        candidates.append(("deberta", np.load(deberta_oof), np.load(OUT / "test_deberta.npy")))
    if len(candidates) == 2:
        candidates.append(
            (
                "blend",
                0.5 * (candidates[0][1] + candidates[1][1]),
                0.5 * (candidates[0][2] + candidates[1][2]),
            )
        )
    if not candidates:
        raise SystemExit("No out-of-fold predictions found in outputs/.")

    results = [evaluate(name, y, pred) for name, pred, _ in candidates]
    best = max(results, key=lambda item: item["qwk"])
    chosen = next(item for item in candidates if item[0] == best["name"])
    pred_scores = scores_for(chosen[2], best)
    by_id = dict(zip(test["essay_id"], pred_scores))
    missing = [essay_id for essay_id in sample["essay_id"] if essay_id not in by_id]
    if missing:
        raise SystemExit(f"Predictions missing {len(missing)} sample essay ids")
    submission = pd.DataFrame(
        {
            "essay_id": sample["essay_id"],
            "score": [int(by_id[essay_id]) for essay_id in sample["essay_id"]],
        }
    )
    submission.to_csv(OUT / "submission.csv", index=False)
    report = {"selected": best, "candidates": results}
    (OUT / "cv_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"selected {best['name']} QWK {best['qwk']:.5f}")
    print(f"wrote {OUT / 'submission.csv'}")


if __name__ == "__main__":
    main()
