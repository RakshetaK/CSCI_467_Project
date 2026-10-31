"""Loading and splitting for the UCI OptDigits and PenDigits datasets."""
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "dataset_467"
OPT_DIR = ROOT / "optical+recognition+of+handwritten+digits"
PEN_DIR = ROOT / "pen+based+recognition+of+handwritten+digits"

# Max raw feature value, used to rescale inputs to [0, 1].
FEATURE_MAX = {"optdigits": 16.0, "pendigits": 100.0}

# optdigits.tra is the concatenation of the authors' original
# training (1934), validation (946) and writer-dependent test (943) files.
OPT_ORIG_TRAIN_SIZE = 1934
OPT_ORIG_VAL_SIZE = 946

N_PEN_VAL_WRITERS = 6


def _read_csv(path):
    data = np.loadtxt(path, delimiter=",", dtype=np.int64)
    return data[:, :-1].astype(np.float64), data[:, -1]


def _read_compressed(path):
    # The .Z files use Unix `compress` (LZW), which Python's stdlib can't read.
    return subprocess.run(["zcat"], stdin=open(path, "rb"), capture_output=True,
                          check=True, text=True).stdout


def load_pen_writers(split):
    """Writer ID for each PenDigits row, parsed from the original UNIPEN file.

    Each sample has a line `.COMMENT <digit> <writer> <n>`, in the same order
    as the rows of the processed pendigits.{tra,tes} files.
    """
    text = _read_compressed(PEN_DIR / f"pendigits-orig.{split}.Z")
    return np.array([int(line.split()[2]) for line in text.splitlines()
                     if line.startswith(".COMMENT")])


def load(name, split):
    """Return (X, y, writers) for name in {optdigits, pendigits}, split in {tra, tes}.

    writers is None for OptDigits, whose files don't record writer identity.
    """
    if name == "optdigits":
        X, y = _read_csv(OPT_DIR / f"optdigits.{split}")
        return X, y, None
    if name == "pendigits":
        X, y = _read_csv(PEN_DIR / f"pendigits.{split}")
        writers = load_pen_writers(split)
        assert len(writers) == len(y)
        return X, y, writers
    raise ValueError(name)


def train_val_split(name, seed=0):
    """Split the official training file into train and validation index arrays.

    PenDigits: hold out N_PEN_VAL_WRITERS whole writers (writer-independent).
    OptDigits: no writer IDs exist, so use the authors' original
    training / validation files (writer-dependent validation).
    """
    X, y, writers = load(name, "tra")
    if name == "optdigits":
        train_idx = np.arange(OPT_ORIG_TRAIN_SIZE)
        val_idx = np.arange(OPT_ORIG_TRAIN_SIZE, OPT_ORIG_TRAIN_SIZE + OPT_ORIG_VAL_SIZE)
        return train_idx, val_idx
    rng = np.random.default_rng(seed)
    val_writers = rng.choice(np.unique(writers), N_PEN_VAL_WRITERS, replace=False)
    is_val = np.isin(writers, val_writers)
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def describe():
    for name in ["optdigits", "pendigits"]:
        X_tr, y_tr, w_tr = load(name, "tra")
        X_te, y_te, w_te = load(name, "tes")
        train_idx, val_idx = train_val_split(name)
        print(f"== {name}: {X_tr.shape[1]} features, "
              f"values {int(X_tr.min())}..{int(X_tr.max())}")
        print(f"  official train file: {len(y_tr)}   test file: {len(y_te)}")
        print(f"  our split -> train {len(train_idx)}, val {len(val_idx)}, test {len(y_te)}")
        if w_tr is not None:
            val_writers = sorted(int(w) for w in set(w_tr[val_idx]))
            print(f"  writers: train file {len(set(w_tr))}, test file {len(set(w_te))}; "
                  f"val writers {val_writers}")
            counts = np.bincount(w_tr)[1:]
            print(f"  samples per training writer: min {counts.min()}, max {counts.max()}")
        rows = {"train": y_tr[train_idx], "val": y_tr[val_idx],
                "train file": y_tr, "test": y_te}
        print("  class      " + " ".join(f"{k:>5}" for k in range(10)))
        for label, ys in rows.items():
            print(f"  {label:<10} " + " ".join(f"{c:>5}" for c in np.bincount(ys, minlength=10)))


if __name__ == "__main__":
    describe()
