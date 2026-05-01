"""
Load a HuggingFace causal LM with hidden states, mean-pool per layer, optionally truncate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_model_and_tokenizer(model_name: str, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    model.to(device)
    return tokenizer, model, device


@torch.inference_mode()
def hidden_states_mean_pooled(
    tokenizer,
    model,
    device: str,
    texts: list[str],
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    """
    Returns array of shape (n_samples, n_layers + 1, hidden_dim).
    Layer 0 is embeddings; last is final transformer layer output before LM head.
    """
    all_pooled: list[torch.Tensor] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="encode"):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        # hidden_states: tuple length n_layers+1, each (batch, seq, hidden)
        hs = torch.stack(outputs.hidden_states, dim=0)  # (L+1, B, S, H)
        mask = inputs["attention_mask"].float()  # (B, S)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        # Mean pool over sequence, masked
        hs = hs * mask.unsqueeze(0).unsqueeze(-1)
        pooled = hs.sum(dim=2) / denom.unsqueeze(0)  # (L+1, B, H)
        all_pooled.append(pooled.permute(1, 0, 2).cpu())  # (B, L+1, H)
    return torch.cat(all_pooled, dim=0).numpy()


def extract_for_split(
    csv_path: Path,
    out_path: Path,
    model_name: str,
    max_length: int,
    batch_size: int,
    device: str | None,
) -> None:
    df = pd.read_csv(csv_path)
    texts = df["text"].astype(str).tolist()
    tokenizer, model, dev = load_model_and_tokenizer(model_name, device=device)
    feats = hidden_states_mean_pooled(
        tokenizer, model, dev, texts, max_length=max_length, batch_size=batch_size
    )
    labels = df["label"].values.astype(np.int64)
    np.savez_compressed(
        out_path,
        hidden_states=feats,
        labels=labels,
        texts=np.array(texts, dtype=object),
    )
    meta = {
        "model_name": model_name,
        "max_length": max_length,
        "n_samples": len(texts),
        "n_layers_plus_one": int(feats.shape[1]),
        "hidden_dim": int(feats.shape[2]),
        "csv": str(csv_path),
    }
    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--out-dir", type=Path, default=Path("features/cache"))
    ap.add_argument("--model-name", default="gpt2-medium")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        csv_path = args.data_dir / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        out_path = args.out_dir / f"{split}_{args.model_name.replace('/', '_')}.npz"
        extract_for_split(
            csv_path,
            out_path,
            args.model_name,
            args.max_length,
            args.batch_size,
            args.device,
        )


if __name__ == "__main__":
    main()
