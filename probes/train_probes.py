"""
Train linear probes: per-layer, final-layer baseline, and multi-layer concatenation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def load_npz(path: Path):
    z = np.load(path, allow_pickle=True)
    hs = z["hidden_states"]  # (N, L+1, H)
    y = z["labels"]
    return hs, y


def train_final_layer(hs_train, y_train, hs_val, y_val, C: float):
    X_tr = hs_train[:, -1, :]
    X_va = hs_val[:, -1, :]
    clf = LogisticRegression(max_iter=2000, C=C, solver="lbfgs", random_state=42)
    clf.fit(X_tr, y_train)
    prob = clf.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_val, prob)
    return clf, float(auc)


def train_single_layer_sweep(hs_train, y_train, hs_val, y_val, C: float):
    n_layers = hs_train.shape[1]
    models = []
    aucs = []
    for k in range(n_layers):
        clf = LogisticRegression(max_iter=2000, C=C, solver="lbfgs", random_state=42)
        clf.fit(hs_train[:, k, :], y_train)
        prob = clf.predict_proba(hs_val[:, k, :])[:, 1]
        auc = roc_auc_score(y_val, prob)
        models.append(clf)
        aucs.append(float(auc))
    return models, aucs


def train_multi_layer_concat(
    hs_train,
    y_train,
    hs_val,
    y_val,
    C: float,
    every_k: int = 1,
):
    idx = list(range(0, hs_train.shape[1], every_k))
    X_tr = np.concatenate([hs_train[:, i, :] for i in idx], axis=1)
    X_va = np.concatenate([hs_val[:, i, :] for i in idx], axis=1)
    clf = LogisticRegression(max_iter=3000, C=C, solver="saga", random_state=42)
    clf.fit(X_tr, y_train)
    prob = clf.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_val, prob)
    return clf, idx, float(auc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=Path, default=Path("features/cache"))
    ap.add_argument("--model-name", default="gpt2-medium")
    ap.add_argument("--out-dir", type=Path, default=Path("probes/artifacts"))
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--multi-every", type=int, default=1, help="Concatenate every Nth layer (1=all).")
    args = ap.parse_args()

    tag = args.model_name.replace("/", "_")
    train_path = args.features_dir / f"train_{tag}.npz"
    val_path = args.features_dir / f"val_{tag}.npz"
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    hs_tr, y_tr = load_npz(train_path)
    hs_va, y_va = load_npz(val_path)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    single_models, layer_aucs = train_single_layer_sweep(hs_tr, y_tr, hs_va, y_va, args.C)
    for k, m in enumerate(single_models):
        joblib.dump(m, args.out_dir / f"probe_layer_{k:02d}.joblib")

    final_clf, final_auc = train_final_layer(hs_tr, y_tr, hs_va, y_va, args.C)
    joblib.dump(final_clf, args.out_dir / "probe_final_layer.joblib")

    multi_clf, layer_idx, multi_auc = train_multi_layer_concat(
        hs_tr, y_tr, hs_va, y_va, args.C, every_k=args.multi_every
    )
    joblib.dump(multi_clf, args.out_dir / "probe_multi_concat.joblib")

    summary = {
        "model_name": args.model_name,
        "C": args.C,
        "multi_every": args.multi_every,
        "multi_layer_indices": layer_idx,
        "val_auroc_final_layer": final_auc,
        "val_auroc_multi_concat": multi_auc,
        "val_auroc_per_layer": layer_aucs,
        "best_single_layer_idx": int(np.argmax(layer_aucs)),
        "best_single_layer_auroc": float(np.max(layer_aucs)),
    }
    with open(args.out_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
