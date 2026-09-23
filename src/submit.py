"""Pick the OOF-best score mapping and write submission.csv."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from src.metrics import quadratic_weighted_kappa
from src.prompts import assign_prompts
from src.splits import HOLDOUT_FOLD

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "opt"
MIN_PROMPT_ROWS = 200


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


def fit_prompt_thresholds(y_true: np.ndarray, pred: np.ndarray, prompts: np.ndarray) -> dict:
    global_cuts = optimize_thresholds(y_true, pred)
    per_prompt = {}
    for name in sorted(set(prompts.tolist())):
        if name == "unknown":
            continue
        mask = prompts == name
        if int(mask.sum()) < MIN_PROMPT_ROWS or len(np.unique(y_true[mask])) < 3:
            continue
        per_prompt[name] = optimize_thresholds(y_true[mask], pred[mask])
    return {"global": global_cuts.tolist(), "per_prompt": {key: value.tolist() for key, value in per_prompt.items()}}


def apply_prompt_thresholds(pred: np.ndarray, prompts: np.ndarray, spec: dict) -> np.ndarray:
    scores = apply_thresholds(pred, np.asarray(spec["global"]))
    for name, cuts in spec["per_prompt"].items():
        mask = prompts == name
        if mask.any():
            scores[mask] = apply_thresholds(pred[mask], np.asarray(cuts))
    return scores


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    y = train["score"].to_numpy()
    fold_ids = np.load(OUT / "fold_ids.npy")
    calibrate = fold_ids != HOLDOUT_FOLD
    holdout = fold_ids == HOLDOUT_FOLD
    train_prompts = assign_prompts(train["full_text"])
    test_prompts = assign_prompts(test["full_text"])
    print("train prompts", dict(zip(*np.unique(train_prompts, return_counts=True))))

    models = {}
    if (OUT / "oof_lgbm.npy").exists():
        models["lgbm"] = (np.load(OUT / "oof_lgbm.npy"), np.load(OUT / "test_lgbm.npy"))
    if (OUT / "oof_deberta.npy").exists():
        models["deberta"] = (np.load(OUT / "oof_deberta.npy"), np.load(OUT / "test_deberta.npy"))
    if not models:
        raise SystemExit("No out-of-fold predictions found in outputs/opt/.")

    weight_grid = [1.0] if "lgbm" not in models or "deberta" not in models else [i / 10 for i in range(11)]
    best_weight = 1.0
    best_cal = -1.0
    for weight in weight_grid:
        pred = blend_prediction(models, weight, which=0)
        score = quadratic_weighted_kappa(y[calibrate], pred[calibrate])
        print(f"calibration round weight {weight:.1f} QWK {score:.5f}")
        if score > best_cal:
            best_cal = score
            best_weight = weight

    oof_pred = blend_prediction(models, best_weight, which=0)
    test_pred = blend_prediction(models, best_weight, which=1)
    spec = fit_prompt_thresholds(y[calibrate], oof_pred[calibrate], train_prompts[calibrate])
    holdout_scores = apply_prompt_thresholds(oof_pred[holdout], train_prompts[holdout], spec)
    holdout_qwk = quadratic_weighted_kappa(y[holdout], holdout_scores)
    rounded_holdout = quadratic_weighted_kappa(y[holdout], oof_pred[holdout])
    print(f"holdout round {rounded_holdout:.5f} prompt-threshold {holdout_qwk:.5f}")
    print(f"blend weight on deberta {best_weight:.1f} prompts {list(spec['per_prompt'])}")

    pred_scores = apply_prompt_thresholds(test_pred, test_prompts, spec)
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
    report = {
        "deberta_weight": best_weight,
        "calibration_round_qwk": best_cal,
        "holdout_round_qwk": rounded_holdout,
        "holdout_prompt_qwk": holdout_qwk,
        "thresholds": spec,
    }
    (OUT / "cv_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUT / 'submission.csv'}")


def blend_prediction(models: dict, deberta_weight: float, which: int) -> np.ndarray:
    if "lgbm" not in models:
        return models["deberta"][which]
    if "deberta" not in models:
        return models["lgbm"][which]
    lgbm, deberta = models["lgbm"][which], models["deberta"][which]
    return deberta_weight * deberta + (1.0 - deberta_weight) * lgbm


if __name__ == "__main__":
    main()
