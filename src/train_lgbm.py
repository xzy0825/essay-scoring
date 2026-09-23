"""Stratified 5-fold TF-IDF + LightGBM with spelling, connectives, and prompt.

Writes outputs/opt/oof_lgbm_rich.npy. The best blend also uses the earlier
LightGBM in outputs/oof_lgbm.npy, trained on the main branch with basic length
features and early stopping on the outer fold.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.metrics import quadratic_weighted_kappa
from src.prompts import RULES, assign_prompts
from src.splits import inner_fit_eval, outer_folds

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "opt"
SEED = 42
N_SPLITS = 5


_TOKEN = re.compile(r"[A-Za-z]+")
_REPEAT = re.compile(r"(.)\1\1")
_PROMPT_NAMES = [name for name, _ in RULES] + ["unknown"]
_PHRASES = (
    "for example",
    "for instance",
    "in addition",
    "in conclusion",
    "on the other hand",
    "as a result",
    "in contrast",
    "for this reason",
)
_WORD_GROUPS = (
    ("however", "although", "though", "despite", "whereas", "nevertheless", "nonetheless"),
    ("because", "therefore", "thus", "hence", "consequently"),
    ("furthermore", "moreover", "additionally"),
    ("finally", "overall", "lastly"),
)
_VOCAB: set[str] | None = None


def _vocab() -> set[str]:
    global _VOCAB
    if _VOCAB is None:
        raw = Path("/usr/share/dict/words").read_text(encoding="utf-8", errors="ignore").split()
        words = {word.lower() for word in raw if word.isalpha()}
        for _, keys in RULES:
            for key in keys:
                words.update(_TOKEN.findall(key.lower()))
        _VOCAB = words
    return _VOCAB


def hand_features(texts: pd.Series) -> np.ndarray:
    vocab = _vocab()
    prompts = assign_prompts(texts)
    rows = []
    for text, prompt in zip(texts, prompts):
        words = text.split()
        tokens = _TOKEN.findall(text.lower())
        n_tok = max(len(tokens), 1)
        sentences = [part for part in re.split(r"[.!?]+", text) if part.strip()]
        paragraphs = [part for part in text.split("\n") if part.strip()]
        para_lens = [len(part.split()) for part in paragraphs] or [0]
        n_sent = max(len(sentences), 1)
        n_para = max(len(paragraphs), 1)
        counts = Counter(tokens)
        lowered = text.lower()
        oov = sum(1 for token in tokens if len(token) > 2 and token not in vocab)
        repeated = sum(1 for token in tokens if _REPEAT.search(token))
        phrase_hits = sum(lowered.count(phrase) for phrase in _PHRASES)
        group_rates = [sum(counts[word] for word in group) / n_tok for group in _WORD_GROUPS]
        avg_word = float(np.mean([len(word) for word in words])) if words else 0.0
        row = [
            len(text),
            len(words),
            len(sentences),
            len(paragraphs),
            avg_word,
            oov / n_tok,
            repeated / n_tok,
            text.count("?") / n_sent,
            text.count(",") / n_tok,
            len(counts) / n_tok,
            float(np.std(para_lens)),
            sum(1 for length in para_lens if length < 20) / n_para,
            float(np.mean(para_lens)),
            phrase_hits / n_tok,
            *group_rates,
        ]
        row.extend(float(prompt == name) for name in _PROMPT_NAMES)
        rows.append(row)
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

    def make_model(n_estimators: int) -> lgb.LGBMRegressor:
        return lgb.LGBMRegressor(
            objective="regression",
            learning_rate=0.05,
            n_estimators=n_estimators,
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

    for fold, (tr_idx, va_idx) in enumerate(outer_folds(y, N_SPLITS, SEED)):
        fit_idx, early_idx = inner_fit_eval(tr_idx, y, seed=SEED)
        word, char = make_vectorizers()
        x_fit = transform(word, char, train.loc[fit_idx, "full_text"], fit=True)
        x_early = transform(word, char, train.loc[early_idx, "full_text"], fit=False)
        probe = make_model(3000)
        probe.fit(
            x_fit,
            y[fit_idx],
            eval_X=x_early,
            eval_y=y[early_idx],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        n_trees = max(int(probe.best_iteration_), 50)
        word, char = make_vectorizers()
        x_tr = transform(word, char, train.loc[tr_idx, "full_text"], fit=True)
        x_va = transform(word, char, train.loc[va_idx, "full_text"], fit=False)
        x_te = transform(word, char, test["full_text"], fit=False)
        model = make_model(n_trees)
        model.fit(x_tr, y[tr_idx])
        oof[va_idx] = model.predict(x_va)
        test_pred += model.predict(x_te) / N_SPLITS
        score = quadratic_weighted_kappa(y[va_idx], oof[va_idx])
        fold_scores.append(score)
        print(f"fold {fold + 1} QWK {score:.5f} trees {n_trees}")

    oof_score = quadratic_weighted_kappa(y, oof)
    print(f"OOF QWK {oof_score:.5f}")
    np.save(OUT / "oof_lgbm_rich.npy", oof)
    np.save(OUT / "test_lgbm_rich.npy", test_pred)
    report = {
        "model": "tfidf_lightgbm_rich",
        "n_splits": N_SPLITS,
        "seed": SEED,
        "fold_qwk_rounded": fold_scores,
        "oof_qwk_rounded": oof_score,
        "seconds": round(time.time() - started, 1),
    }
    (OUT / "lgbm_rich_cv.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote rich LightGBM predictions in {report['seconds']}s")


if __name__ == "__main__":
    main()
