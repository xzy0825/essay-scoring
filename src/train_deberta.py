"""5-fold DeBERTa-v3-small regression for essay scores."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from src.metrics import quadratic_weighted_kappa
from src.splits import inner_fit_eval, outer_folds

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / os.environ.get("AES_OUT", "outputs/opt/deberta")
STEM = os.environ.get("AES_STEM", "deberta")
SEED = 42
N_SPLITS = 5
MODEL_NAME = os.environ.get("AES_MODEL", "microsoft/deberta-v3-small")
MAX_LENGTH = int(os.environ.get("AES_MAX_LENGTH", "1024"))
EPOCHS = 3
BATCH_SIZE = int(os.environ.get("AES_BATCH", "2"))
GRAD_ACCUM = int(os.environ.get("AES_GRAD_ACCUM", "8"))
LEARNING_RATE = 2e-5
PREDICT_BATCH = int(os.environ.get("AES_PREDICT_BATCH", "4"))
N_LENGTH_FEATURES = 4
CHECKPOINT = os.environ.get("AES_CHECKPOINT", "0") == "1"


def materialize_model(model_name: str) -> str:
    """Return a local model directory that contains safetensors weights.

    deberta-v3-small is published as pytorch_model.bin. Transformers refuses
    torch.load on torch 2.5, so convert the checkpoint once and reuse it.
    """
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_file

    folder = Path(
        snapshot_download(
            model_name,
            ignore_patterns=["tf_model.h5", "*.h5", "*.msgpack", "flax_model*", "rust_model*"],
        )
    )
    safetensors_path = folder / "model.safetensors"
    if not safetensors_path.exists():
        print("converting pytorch_model.bin to safetensors")
        state = torch.load(folder / "pytorch_model.bin", map_location="cpu", weights_only=True)
        save_file({key: value.contiguous() for key, value in state.items()}, safetensors_path)
    return str(folder)


def length_features(texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        words = text.split()
        sentences = [part for part in re.split(r"[.!?]+", text) if part.strip()]
        paragraphs = [part for part in text.split("\n") if part.strip()]
        rows.append([len(text), len(words), len(sentences), max(len(paragraphs), 1)])
    return np.log1p(np.asarray(rows, dtype=np.float32))


def standardize(reference: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    center = reference.mean(axis=0)
    scale = np.clip(reference.std(axis=0), 1e-6, None)
    return [((array - center) / scale).astype(np.float32) for array in (reference, *others)]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EssayDataset(Dataset):
    def __init__(self, input_ids: list[list[int]], features: np.ndarray, labels: np.ndarray | None = None):
        self.input_ids = input_ids
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "input_ids": torch.tensor(self.input_ids[index], dtype=torch.long),
            "features": torch.tensor(self.features[index], dtype=torch.float),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.float)
        return item


class EssayRegressor(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name,
            use_safetensors=True,
            local_files_only=True,
        ).float()
        if CHECKPOINT:
            self.backbone.gradient_checkpointing_enable()
        hidden = self.backbone.config.hidden_size
        self.dropout = torch.nn.Dropout(0.1)
        self.head = torch.nn.Linear(hidden + N_LENGTH_FEATURES, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = hidden.float()
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        pooled = torch.cat([pooled, features.float()], dim=1)
        return self.head(self.dropout(pooled)).squeeze(-1)


def collate(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    padded = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=pad_id,
    )
    out = {
        "input_ids": padded,
        "attention_mask": (padded != pad_id).long(),
        "features": torch.stack([item["features"] for item in batch]),
    }
    if "labels" in batch[0]:
        out["labels"] = torch.stack([item["labels"] for item in batch])
    return out


def encode(tokenizer, texts: list[str]) -> list[list[int]]:
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        add_special_tokens=True,
    )
    return encoded["input_ids"]


def predict(model, loader, device) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            feats = batch["features"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(ids, mask, feats)
            preds.append(pred.float().cpu().numpy())
    return np.concatenate(preds)


def train_one_fold(
    model: EssayRegressor,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    y_valid: np.ndarray,
    device: torch.device,
    fold_dir: Path,
    grad_accum: int,
) -> EssayRegressor:
    no_decay = {"bias", "LayerNorm.weight"}
    groups = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(k in n for k in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(groups, lr=LEARNING_RATE)
    updates_per_epoch = max(1, (len(train_loader) + grad_accum - 1) // grad_accum)
    total_steps = updates_per_epoch * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.06 * total_steps)),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda")
    best_qwk = -1.0
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        seen = 0
        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch}", leave=False), start=1):
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            feats = batch["features"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(ids, mask, feats)
                loss = torch.nn.functional.mse_loss(pred, labels) / grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * grad_accum * len(labels)
            seen += len(labels)
            if step % grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        val_pred = predict(model, valid_loader, device)
        qwk = quadratic_weighted_kappa(y_valid, val_pred)
        print(f"  epoch {epoch} train_mse {running / max(seen, 1):.4f} val_qwk {qwk:.5f}")
        if qwk >= best_qwk:
            best_qwk = qwk
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(best_state, fold_dir)
    return model


def _loader(ids, features, labels, pad_id: int, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        EssayDataset(ids, features, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda batch: collate(batch, pad_id),
    )


def run_fold(
    batch_size: int,
    grad_accum: int,
    model_dir: str,
    tokenizer,
    train_ids,
    train_features,
    early_ids,
    early_features,
    y_tr,
    y_early,
    outer_ids,
    outer_features,
    test_ids,
    test_features,
    device,
    fold_path: Path,
):
    pad_id = tokenizer.pad_token_id
    model = None
    try:
        model = EssayRegressor(model_dir).to(device)
        train_loader = _loader(train_ids, train_features, y_tr, pad_id, batch_size, True)
        early_loader = _loader(early_ids, early_features, None, pad_id, PREDICT_BATCH, False)
        outer_loader = _loader(outer_ids, outer_features, None, pad_id, PREDICT_BATCH, False)
        test_loader = _loader(test_ids, test_features, None, pad_id, PREDICT_BATCH, False)
        model = train_one_fold(model, train_loader, early_loader, y_early, device, fold_path, grad_accum)
        return predict(model, outer_loader, device), predict(model, test_loader, device)
    except RuntimeError as exc:
        if batch_size > 1 and "out of memory" in str(exc).lower():
            print(f"CUDA OOM at batch {batch_size}, retrying this fold at batch 1")
            del model
            torch.cuda.empty_cache()
            return run_fold(
                1,
                grad_accum * batch_size,
                model_dir,
                tokenizer,
                train_ids,
                train_features,
                early_ids,
                early_features,
                y_tr,
                y_early,
                outer_ids,
                outer_features,
                test_ids,
                test_features,
                device,
                fold_path,
            )
        raise


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("DeBERTa training expects the RTX 3080. CUDA is not available.")
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["score"].to_numpy(dtype=np.float32)
    model_dir = materialize_model(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, fix_mistral_regex=False)
    print("tokenizing essays")
    train_texts = train["full_text"].tolist()
    test_texts = test["full_text"].tolist()
    train_ids = encode(tokenizer, train_texts)
    test_ids = encode(tokenizer, test_texts)
    raw_train_features = length_features(train_texts)
    raw_test_features = length_features(test_texts)
    lengths = [len(ids) for ids in train_ids]
    capped = sum(length >= MAX_LENGTH for length in lengths)
    print(
        f"token length min/median/max {min(lengths)} {sorted(lengths)[len(lengths)//2]} {max(lengths)}"
        f" capped {capped}"
    )

    oof = np.zeros(len(train), dtype=np.float32)
    test_pred = np.zeros(len(test), dtype=np.float32)
    fold_ids = np.full(len(train), -1, dtype=np.int8)
    fold_scores: list[float | None] = [None] * N_SPLITS
    progress_path = OUT / "progress.json"
    partial_ready = (OUT / "oof_partial.npy").exists() and (OUT / "test_partial.npy").exists()
    if progress_path.exists() and partial_ready:
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        same_setup = (
            saved.get("model") == MODEL_NAME
            and saved.get("max_length") == MAX_LENGTH
            and saved.get("epochs") == EPOCHS
            and saved.get("batch_size") == BATCH_SIZE
        )
        if same_setup:
            oof = np.load(OUT / "oof_partial.npy")
            test_pred = np.load(OUT / "test_partial.npy")
            fold_scores = saved["fold_qwk_rounded"]
            print("resuming completed folds", [i + 1 for i, score in enumerate(fold_scores) if score is not None])
        else:
            print("checkpoint config changed, training from scratch")

    folds = outer_folds(y, N_SPLITS, SEED)
    for fold, (_, va_idx) in enumerate(folds):
        fold_ids[va_idx] = fold
    started = time.time()
    opt_dir = OUT.parent
    for fold, (tr_idx, va_idx) in enumerate(folds):
        if fold_scores[fold] is not None:
            continue
        print(f"fold {fold + 1}/{N_SPLITS}")
        fit_idx, early_idx = inner_fit_eval(tr_idx, y, seed=SEED)
        fit_features, early_features, outer_features, test_features = standardize(
            raw_train_features[fit_idx],
            raw_train_features[early_idx],
            raw_train_features[va_idx],
            raw_test_features,
        )
        oof_fold, test_fold = run_fold(
            BATCH_SIZE,
            GRAD_ACCUM,
            model_dir,
            tokenizer,
            [train_ids[i] for i in fit_idx],
            fit_features,
            [train_ids[i] for i in early_idx],
            early_features,
            y[fit_idx],
            y[early_idx],
            [train_ids[i] for i in va_idx],
            outer_features,
            test_ids,
            test_features,
            device,
            OUT / f"fold{fold}.pt",
        )
        oof[va_idx] = oof_fold
        test_pred += test_fold / N_SPLITS
        score = quadratic_weighted_kappa(y[va_idx], oof_fold)
        fold_scores[fold] = score
        print(f"fold {fold + 1} QWK {score:.5f}")
        np.save(OUT / "oof_partial.npy", oof)
        np.save(OUT / "test_partial.npy", test_pred)
        progress_path.write_text(
            json.dumps(
                {
                    "model": MODEL_NAME,
                    "max_length": MAX_LENGTH,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "fold_qwk_rounded": fold_scores,
                }
            ),
            encoding="utf-8",
        )
        torch.cuda.empty_cache()

    oof_score = quadratic_weighted_kappa(y, oof)
    print(f"OOF QWK {oof_score:.5f}")
    np.save(opt_dir / f"oof_{STEM}.npy", oof)
    np.save(opt_dir / f"test_{STEM}.npy", test_pred)
    np.save(opt_dir / "fold_ids.npy", fold_ids)
    report = {
        "model": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "learning_rate": LEARNING_RATE,
        "n_splits": N_SPLITS,
        "seed": SEED,
        "fold_qwk_rounded": fold_scores,
        "oof_qwk_rounded": oof_score,
        "capped_essays": capped,
        "seconds": round(time.time() - started, 1),
    }
    (opt_dir / f"{STEM}_cv.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"finished in {report['seconds']}s")


if __name__ == "__main__":
    main()
