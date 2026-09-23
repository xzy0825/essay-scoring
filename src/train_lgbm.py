"""Stratified 5-fold TF-IDF + LightGBM regression baseline."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

from src.metrics import quadratic_weighted_kappa

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
SEED = 42
N_SPLITS = 5


def hand_features(texts: pd.Series) -> np.ndarray:
    rows = []
    for text in texts:
        words = text.split()
        sentences = [part for part in re.split(r"[.!?]+", text) if part.strip()]
        paragraphs = [part for part in text.split("\n") if part.strip()]
        avg_word = float(np.mean([len(word) for word in words])) if words else 0.0
        rows.append([len(text), len(words), len(sentences), len(paragraphs), avg_word])
    return np.asarray(rows, dtype=np.float32)


def make_vectorizers() -> tuple[TfidfVectorizer, TfidfVectorizer]:
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=20_000,
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=10_000,
        min_df=2,
        sublinear_tf=True,
    )
    return word, char


def transform(
    word: TfidfVectorizer,
    char: TfidfVectorizer,
    texts: pd.Series,
    *,
    fit: bool,
) -> sparse.csr_matrix:
    if fit:
        x_word = word.fit_transform(texts)
        x_char = char.fit_transform(texts)
    else:
        x_word = word.transform(texts)
        x_char = char.transform(texts)
    return sparse.hstack(
        [x_word, x_char, sparse.csr_matrix(hand_features(texts))],
        format="csr",
        dtype=np.float32,
    )


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["score"].to_numpy(dtype=np.float32)
    oof = np.zeros(len(train), dtype=np.float32)
    test_pred = np.zeros(len(test), dtype=np.float32)
    fold_scores: list[float] = []

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(train, y.astype(int)), start=1):
        word, char = make_vectorizers()
        x_tr = transform(word, char, train.loc[tr_idx, "full_text"], fit=True)
        x_va = transform(word, char, train.loc[va_idx, "full_text"], fit=False)
        x_te = transform(word, char, test["full_text"], fit=False)
        model = lgb.LGBMRegressor(
            objective="regression",
            learning_rate=0.05,
            n_estimators=3000,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=-1,
            force_col_wise=True,
            verbosity=-1,
        )
        model.fit(
            x_tr,
            y[tr_idx],
            eval_X=x_va,
            eval_y=y[va_idx],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va_idx] = model.predict(x_va)
        test_pred += model.predict(x_te) / N_SPLITS
        score = quadratic_weighted_kappa(y[va_idx], oof[va_idx])
        fold_scores.append(score)
        print(f"fold {fold} QWK {score:.5f} best_iter {model.best_iteration_}")

    oof_score = quadratic_weighted_kappa(y, oof)
    print(f"OOF QWK {oof_score:.5f}")
    np.save(OUT / "oof_lgbm.npy", oof)
    np.save(OUT / "test_lgbm.npy", test_pred)
    report = {
        "model": "tfidf_lightgbm",
        "n_splits": N_SPLITS,
        "seed": SEED,
        "fold_qwk_rounded": fold_scores,
        "oof_qwk_rounded": oof_score,
        "seconds": round(time.time() - started, 1),
    }
    (OUT / "lgbm_cv.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    submission = pd.DataFrame(
        {
            "essay_id": test["essay_id"],
            "score": np.clip(np.rint(test_pred), 1, 6).astype(int),
        }
    )
    submission.to_csv(OUT / "submission.csv", index=False)
    print(f"wrote {OUT / 'submission.csv'} in {report['seconds']}s")


if __name__ == "__main__":
    main()
