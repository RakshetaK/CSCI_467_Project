"""Softmax-regression baseline: hyperparameter search, final training, test evaluation.

Run from the repo root:  python3 -m src.run_softmax
"""
import itertools
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from src import data, metrics
from src.softmax import SoftmaxRegression

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "softmax"

L2_GRID = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
LR_GRID = [0.5, 1.0, 2.0]
EPOCHS = 10000
SEED = 0

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834"]
SEQUENTIAL = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
              "#256abf", "#184f95", "#0d366b"]


def search(X_tr, y_tr, X_val, y_val):
    """Train one model per (l2, lr) and score it on the validation set."""
    grid = []
    for l2, lr in itertools.product(L2_GRID, LR_GRID):
        model = SoftmaxRegression(lr=lr, l2=l2, epochs=EPOCHS).fit(X_tr, y_tr)
        grid.append({"l2": l2, "lr": lr, "epochs_run": model.n_epochs_run,
                     "train_acc": metrics.accuracy(y_tr, model.predict(X_tr)),
                     "val_acc": metrics.accuracy(y_val, model.predict(X_val)),
                     "val_loss": float(model.loss(X_val, y_val))})
    best = max(grid, key=lambda r: (r["val_acc"], -r["val_loss"]))
    return grid, best


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.6), facecolor=SURFACE)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL)
    # Color off-diagonal errors on their own scale; the diagonal would wash them out.
    off = cm.astype(float).copy()
    np.fill_diagonal(off, np.nan)
    ax.imshow(off, cmap=cmap, vmin=0, vmax=max(np.nanmax(off), 1))
    vmax = max(np.nanmax(off), 1)
    for i in range(10):
        for j in range(10):
            if i == j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#f0efec"))
                color, weight = TEXT_PRIMARY, "bold"
            else:
                color = "white" if cm[i, j] > 0.55 * vmax else TEXT_PRIMARY
                weight = "normal"
            if cm[i, j] or i == j:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color=color, fontweight=weight)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left")
    style_axes(ax)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_loss(history, title, path):
    fig, ax = plt.subplots(figsize=(5.6, 3.4), facecolor=SURFACE)
    for (key, label), color in zip([("train_loss", "Train"), ("val_loss", "Validation")], SERIES):
        ys = history[key]
        ax.plot(range(1, len(ys) + 1), ys, color=color, linewidth=2, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Epoch (full-batch gradient step)")
    ax.set_ylabel("Cross-entropy + L2")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def run(name):
    X_full, y_full, w_full = data.load(name, "tra")
    X_te, y_te, w_te = data.load(name, "tes")
    # Rescale raw integer features to [0, 1] so one learning rate suits both datasets.
    X_full = X_full / data.FEATURE_MAX[name]
    X_te = X_te / data.FEATURE_MAX[name]

    train_idx, val_idx = data.train_val_split(name, seed=SEED)
    X_tr, y_tr = X_full[train_idx], y_full[train_idx]
    X_val, y_val = X_full[val_idx], y_full[val_idx]

    grid, best = search(X_tr, y_tr, X_val, y_val)
    print(f"[{name}] best: l2={best['l2']}, lr={best['lr']}, val_acc={best['val_acc']:.4f}")

    # Loss curves for the chosen setting, tracked on the held-out validation set.
    curve_model = SoftmaxRegression(lr=best["lr"], l2=best["l2"], epochs=EPOCHS)
    curve_model.fit(X_tr, y_tr, X_val, y_val)
    monotone = bool(np.all(np.diff(curve_model.history["train_loss"]) <= 1e-12))

    # Final model: retrain on the whole official training file, then test once.
    start = time.perf_counter()
    model = SoftmaxRegression(lr=best["lr"], l2=best["l2"], epochs=EPOCHS).fit(X_full, y_full)
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    y_pred = model.predict(X_te)
    predict_seconds = time.perf_counter() - start

    cm = metrics.confusion_matrix(y_te, y_pred)
    result = {
        "dataset": name,
        "n_train": len(train_idx), "n_val": len(val_idx),
        "n_final_train": len(y_full), "n_test": len(y_te),
        "hyperparameter_grid": grid,
        "chosen": {"l2": best["l2"], "lr": best["lr"], "max_epochs": EPOCHS,
                   "epochs_run_final": model.n_epochs_run},
        "train_loss_monotone": monotone,
        "val_acc": best["val_acc"],
        "train_acc": metrics.accuracy(y_full, model.predict(X_full)),
        "test_acc": metrics.accuracy(y_te, y_pred),
        "test_per_class_acc": metrics.per_class_accuracy(y_te, y_pred).tolist(),
        "test_confusion_matrix": cm.tolist(),
        "top_confusions": metrics.top_confusions(cm, k=6),
        "n_params": model.n_params,
        "param_bytes": model.W.nbytes + model.b.nbytes,
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
    }
    if w_te is not None:
        per_writer = metrics.per_writer_accuracy(y_te, y_pred, w_te)
        result["test_per_writer_acc"] = per_writer
        result["val_writers"] = sorted(int(w) for w in np.unique(w_full[val_idx]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{name}.json").write_text(json.dumps(result, indent=2))
    pretty = {"optdigits": "OptDigits", "pendigits": "PenDigits"}[name]
    plot_confusion(cm, f"{pretty} softmax: test confusion matrix",
                   OUT_DIR / f"{name}_confusion.png")
    plot_loss(curve_model.history, f"{pretty} softmax: loss (l2={best['l2']}, lr={best['lr']})",
              OUT_DIR / f"{name}_loss.png")
    print(f"[{name}] test_acc={result['test_acc']:.4f}  train {train_seconds:.2f}s")
    return result


if __name__ == "__main__":
    for name in ["optdigits", "pendigits"]:
        run(name)
