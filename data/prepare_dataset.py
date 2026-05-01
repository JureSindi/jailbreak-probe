from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def _load_jailbreak_dataset(preferred: str) -> pd.DataFrame:
    """Load jailbreak prompts; try preferred source then public fallbacks."""
    candidates = [preferred, "yukiyounai/AdvBench", "mlabonne/harmful_behaviors"]
    seen = set()
    last_err = None
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        try:
            ds = load_dataset(name, split="train")
            rows = ds.to_pandas()
            if "prompt" in rows.columns:
                text_col = "prompt"
            elif "text" in rows.columns:
                text_col = "text"
            elif "goal" in rows.columns:
                text_col = "goal"
            else:
                text_col = rows.columns[0]
            out = pd.DataFrame({"text": rows[text_col].astype(str)})
            out = out.drop_duplicates(subset=["text"]).reset_index(drop=True)
            out["source"] = name
            return out
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"Could not load jailbreak data. Last error: {last_err}")


def _load_alpaca_benign() -> pd.DataFrame:
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    df = ds.to_pandas()
    # Use instruction as the user-facing prompt; include input when present (standard Alpaca format).
    texts = []
    for _, row in df.iterrows():
        ins = str(row.get("instruction", "")).strip()
        inp = str(row.get("input", "")).strip()
        if inp:
            t = f"{ins}\n{inp}".strip()
        else:
            t = ins
        if t:
            texts.append(t)
    out = pd.DataFrame({"text": texts})
    out = out.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return out


def _assign_families(texts: list[str], n_clusters: int, seed: int) -> np.ndarray:
    """Cluster prompts into families for group-wise splitting."""
    if len(texts) <= 1:
        return np.zeros(len(texts), dtype=np.int64)
    n_clusters = max(2, min(n_clusters, len(texts) // 2))
    vectorizer = TfidfVectorizer(
        max_features=4096,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )
    X = vectorizer.fit_transform(texts)
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=256,
        n_init="auto",
    )
    return km.fit_predict(X)


def _split_sizes(n: int, train_f: float, val_f: float) -> tuple[int, int, int]:
    nt = int(round(train_f * n))
    nv = int(round(val_f * n))
    nte = n - nt - nv
    return nt, nv, nte


def _group_split_greedy(
    df: pd.DataFrame,
    group_col: str,
    train_f: float,
    val_f: float,
    test_f: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Assign whole families to train/val/test while approximating target per-class sizes.
    Greedy: each family goes to the split with the largest remaining quota.
    """
    rng = np.random.default_rng(seed)
    families = df[group_col].unique().tolist()
    rng.shuffle(families)
    n = len(df)
    nt, nv, nte = _split_sizes(n, train_f, val_f)
    targets = {"train": nt, "val": nv, "test": nte}
    counts = {"train": 0, "val": 0, "test": 0}
    buckets: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    for fam in families:
        idx = df.index[df[group_col] == fam].tolist()
        size = len(idx)
        # Prefer split with largest (target - current) deficit; tie-break random.
        def score(split: str) -> float:
            return targets[split] - counts[split]

        order = sorted(("train", "val", "test"), key=score, reverse=True)
        best = order[0]
        if size > 0 and all(counts[s] + size > targets[s] for s in order):
            # If this family cannot fit any bucket without huge overflow, use best score anyway.
            pass
        buckets[best].extend(idx)
        counts[best] += size

    train_df = df.loc[buckets["train"]].reset_index(drop=True)
    val_df = df.loc[buckets["val"]].reset_index(drop=True)
    test_df = df.loc[buckets["test"]].reset_index(drop=True)
    return train_df, val_df, test_df


def prepare(
    out_dir: Path,
    n_per_class: int,
    jb_dataset: str,
    n_jailbreak_families: int,
    n_benign_families: int,
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    jb_raw = _load_jailbreak_dataset(jb_dataset)
    benign_raw = _load_alpaca_benign()

    jb_texts = jb_raw["text"].tolist()
    if len(jb_texts) > n_per_class:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(jb_texts), size=n_per_class, replace=False)
        jb_texts = [jb_texts[i] for i in sorted(pick)]

    benign_texts = benign_raw["text"].tolist()
    if len(benign_texts) > n_per_class:
        rng = np.random.default_rng(seed + 1)
        pick = rng.choice(len(benign_texts), size=n_per_class, replace=False)
        benign_texts = [benign_texts[i] for i in sorted(pick)]

    jb_fam = _assign_families(jb_texts, n_jailbreak_families, seed)
    bn_fam = _assign_families(benign_texts, n_benign_families, seed + 7)

    jb_df = pd.DataFrame(
        {
            "text": jb_texts,
            "label": 1,
            "family_id": [f"jb_{i}" for i in jb_fam],
        }
    )
    bn_df = pd.DataFrame(
        {
            "text": benign_texts,
            "label": 0,
            "family_id": [f"bn_{i}" for i in bn_fam],
        }
    )
    # Split classes separately so each fold stays ~50/50 jailbreak vs benign (group split is not label-stratified).
    jb_train, jb_val, jb_test = _group_split_greedy(
        jb_df, "family_id", 0.7, 0.15, 0.15, seed=seed
    )
    bn_train, bn_val, bn_test = _group_split_greedy(
        bn_df, "family_id", 0.7, 0.15, 0.15, seed=seed + 99
    )
    train_df = pd.concat([jb_train, bn_train], ignore_index=True)
    val_df = pd.concat([jb_val, bn_val], ignore_index=True)
    test_df = pd.concat([jb_test, bn_test], ignore_index=True)
    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    meta = {
        "jailbreak_source": jb_raw["source"].iloc[0],
        "n_per_class": n_per_class,
        "n_jailbreak_families": n_jailbreak_families,
        "n_benign_families": n_benign_families,
        "seed": seed,
        "splits": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "label_counts": {
            "train": train_df["label"].value_counts().to_dict(),
            "val": val_df["label"].value_counts().to_dict(),
            "test": test_df["label"].value_counts().to_dict(),
        },
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Wrote:", out_dir / "train.csv", out_dir / "val.csv", out_dir / "test.csv")
    print(json.dumps(meta, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare balanced AdvBench + Alpaca splits.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "processed",
    )
    p.add_argument("--n-per-class", type=int, default=500)
    p.add_argument(
        "--jailbreak-dataset",
        default="walledai/AdvBench",
        help="Try this first; falls back to yukiyounai/AdvBench if gated/unavailable.",
    )
    p.add_argument("--n-jailbreak-families", type=int, default=40)
    p.add_argument("--n-benign-families", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    prepare(
        args.out_dir,
        args.n_per_class,
        args.jailbreak_dataset,
        args.n_jailbreak_families,
        args.n_benign_families,
        args.seed,
    )


if __name__ == "__main__":
    main()
