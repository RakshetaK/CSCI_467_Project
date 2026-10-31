"""Evaluation metrics shared by all models."""
import numpy as np


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, n_classes=10):
    """cm[i, j] = number of examples with true class i predicted as class j."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_accuracy(y_true, y_pred, n_classes=10):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    return np.diag(cm) / cm.sum(axis=1)


def per_writer_accuracy(y_true, y_pred, writers):
    return {int(w): accuracy(y_true[writers == w], y_pred[writers == w])
            for w in np.unique(writers)}


def top_confusions(cm, k=5):
    """The k most frequent (true, predicted, count) off-diagonal errors."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.argsort(off, axis=None)[::-1][:k]
    return [(int(i), int(j), int(off[i, j])) for i, j in
            zip(*np.unravel_index(order, off.shape)) if off[i, j] > 0]
