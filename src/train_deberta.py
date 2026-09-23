"""5-fold DeBERTa-v3-small regression for essay scores."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from src.metrics import quadratic_weighted_kappa

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "deberta"
SEED = 42
N_SPLITS = 5
MODEL_NAME = "microsoft/deberta-v3-small"
MAX_LENGTH = 512
EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 4
LEARNING_RATE = 2e-5
PREDICT_BATCH = 16


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


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EssayDataset(Dataset):
    def __init__(self, input_ids: list[list[int]], labels: np.ndarray | None = None):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {"input_ids": torch.tensor(self.input_ids[index], dtype=torch.long)}
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
        hidden = self.backbone.config.hidden_size
        self.dropout = torch.nn.Dropout(0.1)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = hidden.float()
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return self.head(self.dropout(pooled)).squeeze(-1)


def collate(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    padded = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=pad_id,
    )
    out = {"input_ids": padded, "attention_mask": (padded != pad_id).long()}
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
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(ids, mask)
            preds.append(pred.float().cpu().numpy())
    return np.concatenate(preds)


def train_one_fold(
    model: EssayRegressor,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    y_valid: np.ndarray,
    device: torch.device,
    fold_dir: Path,
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
    updates_per_epoch = max(1, (len(train_loader) + GRAD_ACCUM - 1) // GRAD_ACCUM)
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
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(ids, mask)
                loss = torch.nn.functional.mse_loss(pred, labels) / GRAD_ACCUM
            scaler.scale(loss).backward()
            running += loss.item() * GRAD_ACCUM * len(labels)
            seen += len(labels)
            if step % GRAD_ACCUM == 0 or step == len(train_loader):
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


def run_fold(batch_size: int, model_dir: str, tokenizer, train_ids, valid_ids, y_tr, y_va, test_ids, device, fold_path: Path):
    pad_id = tokenizer.pad_token_id
    model = None
    try:
        model = EssayRegressor(model_dir).to(device)
        train_loader = DataLoader(
            EssayDataset(train_ids, y_tr),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=lambda batch: collate(batch, pad_id),
        )
        valid_loader = DataLoader(
            EssayDataset(valid_ids),
            batch_size=PREDICT_BATCH,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=lambda batch: collate(batch, pad_id),
        )
        test_loader = DataLoader(
            EssayDataset(test_ids),
            batch_size=PREDICT_BATCH,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=lambda batch: collate(batch, pad_id),
        )
        model = train_one_fold(model, train_loader, valid_loader, y_va, device, fold_path)
        oof = predict(model, valid_loader, device)
        test_pred = predict(model, test_loader, device)
        return oof, test_pred
    except RuntimeError as exc:
        if batch_size > 2 and "out of memory" in str(exc).lower():
            print(f"CUDA OOM at batch {batch_size}, retrying this fold at batch 2")
            del model
            torch.cuda.empty_cache()
            return run_fold(2, model_dir, tokenizer, train_ids, valid_ids, y_tr, y_va, test_ids, device, fold_path)
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
    train_ids = encode(tokenizer, train["full_text"].tolist())
    test_ids = encode(tokenizer, test["full_text"].tolist())
    lengths = [len(ids) for ids in train_ids]
    print(f"token length min/median/max {min(lengths)} {sorted(lengths)[len(lengths)//2]} {max(lengths)}")

    oof = np.zeros(len(train), dtype=np.float32)
    test_pred = np.zeros(len(test), dtype=np.float32)
    fold_scores: list[float | None] = [None] * N_SPLITS
    progress_path = OUT / "progress.json"
    partial_ready = (OUT / "oof_partial.npy").exists() and (OUT / "test_partial.npy").exists()
    if progress_path.exists() and partial_ready:
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        if saved.get("model") == MODEL_NAME and saved.get("max_length") == MAX_LENGTH and saved.get("epochs") == EPOCHS:
            oof = np.load(OUT / "oof_partial.npy")
            test_pred = np.load(OUT / "test_partial.npy")
            fold_scores = saved["fold_qwk_rounded"]
            print("resuming completed folds", [i + 1 for i, s in enumerate(fold_scores) if s is not None])
        else:
            print("checkpoint config changed, training from scratch")

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    started = time.time()
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(train, y.astype(int))):
        if fold_scores[fold] is not None:
            continue
        print(f"fold {fold + 1}/{N_SPLITS}")
        fold_path = OUT / f"fold{fold}.pt"
        oof_fold, test_fold = run_fold(
            BATCH_SIZE,
            model_dir,
            tokenizer,
            [train_ids[i] for i in tr_idx],
            [train_ids[i] for i in va_idx],
            y[tr_idx],
            y[va_idx],
            test_ids,
            device,
            fold_path,
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
                    "fold_qwk_rounded": fold_scores,
                }
            ),
            encoding="utf-8",
        )
        torch.cuda.empty_cache()

    oof_score = quadratic_weighted_kappa(y, oof)
    print(f"OOF QWK {oof_score:.5f}")
    np.save(ROOT / "outputs" / "oof_deberta.npy", oof)
    np.save(ROOT / "outputs" / "test_deberta.npy", test_pred)
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
        "seconds": round(time.time() - started, 1),
    }
    (ROOT / "outputs" / "deberta_cv.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"finished in {report['seconds']}s")


if __name__ == "__main__":
    main()
