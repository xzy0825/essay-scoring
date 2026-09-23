"""Average a tail-window prediction into DeBERTa-base scores for truncated essays.

The fold models were trained on the first 1024 tokens. This pass did not change
the best submission: 13 test essays moved in continuous score and stayed inside
the same threshold bin. On the 112 truncated training essays, the head-only
score was better. Kept so the negative result can be rerun.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.splits import inner_fit_eval, outer_folds
from src.train_deberta import (
    MAX_LENGTH,
    EssayRegressor,
    _loader,
    length_features,
    materialize_model,
    predict,
    standardize,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CKPT = ROOT / "outputs" / "opt" / "deberta_base"
OUT = ROOT / "outputs" / "opt"


def tail_ids(tokenizer, text: str) -> list[int] | None:
    body = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    room = MAX_LENGTH - 2
    if len(body) <= room:
        return None
    return [tokenizer.cls_token_id, *body[-room:], tokenizer.sep_token_id]


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("Tail prediction expects a CUDA GPU.")
    started = time.time()
    device = torch.device("cuda")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["score"].to_numpy(dtype=np.float32)
    train_texts = train["full_text"].tolist()
    test_texts = test["full_text"].tolist()
    model_dir = materialize_model("microsoft/deberta-v3-base")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, fix_mistral_regex=False)
    train_tail = [tail_ids(tokenizer, text) for text in train_texts]
    test_tail = [tail_ids(tokenizer, text) for text in test_texts]
    train_long = np.array([ids is not None for ids in train_tail])
    test_long = np.array([ids is not None for ids in test_tail])
    print(f"truncated train {int(train_long.sum())} test {int(test_long.sum())}")

    raw_train = length_features(train_texts)
    raw_test = length_features(test_texts)
    oof_tail = np.zeros(len(train), dtype=np.float32)
    test_tail_sum = np.zeros(len(test), dtype=np.float32)
    folds = outer_folds(y)
    for fold, (tr_idx, va_idx) in enumerate(folds):
        fit_idx, early_idx = inner_fit_eval(tr_idx, y)
        _, _, outer_features, test_features = standardize(
            raw_train[fit_idx],
            raw_train[early_idx],
            raw_train[va_idx],
            raw_test,
        )
        model = EssayRegressor(model_dir).to(device)
        state = torch.load(CKPT / f"fold{fold}.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        pad_id = tokenizer.pad_token_id

        va_keep = train_long[va_idx]
        if va_keep.any():
            local = np.flatnonzero(va_keep)
            chosen = va_idx[local]
            loader = _loader(
                [train_tail[i] for i in chosen],
                outer_features[local],
                None,
                pad_id,
                2,
                False,
            )
            oof_tail[chosen] = predict(model, loader, device)

        te_local = np.flatnonzero(test_long)
        loader = _loader(
            [test_tail[i] for i in te_local],
            test_features[te_local],
            None,
            pad_id,
            2,
            False,
        )
        pred = np.zeros(len(test), dtype=np.float32)
        pred[te_local] = predict(model, loader, device)
        test_tail_sum += pred
        del model
        torch.cuda.empty_cache()
        print(f"fold {fold + 1} tail rows {int(va_keep.sum())}", flush=True)

    head_oof = np.load(OUT / "oof_deberta_base.npy")
    head_test = np.load(OUT / "test_deberta_base.npy")
    oof = head_oof.copy()
    test_pred = head_test.copy()
    oof[train_long] = 0.5 * head_oof[train_long] + 0.5 * oof_tail[train_long]
    test_pred[test_long] = 0.5 * head_test[test_long] + 0.5 * (test_tail_sum[test_long] / len(folds))
    np.save(OUT / "oof_deberta_base_tail.npy", oof)
    np.save(OUT / "test_deberta_base_tail.npy", test_pred)
    np.save(OUT / "train_long.npy", train_long)
    print(f"wrote tail-averaged base predictions in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
