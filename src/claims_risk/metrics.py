import numpy as np


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true, dtype="float64") - np.asarray(y_pred, dtype="float64"))))


def rmse(y_true, y_pred):
    d = np.asarray(y_true, dtype="float64") - np.asarray(y_pred, dtype="float64")
    return float(np.sqrt(np.mean(d * d)))


def gamma_deviance(y_true, y_pred):
    eps = 1e-6
    y_true = np.clip(np.asarray(y_true, dtype="float64"), eps, None)
    y_pred = np.clip(np.asarray(y_pred, dtype="float64"), eps, None)
    return float(2 * np.mean((y_true - y_pred) / y_pred - np.log(y_true / y_pred)))


def pinball_loss(y_true, y_pred, alpha):
    diff = np.asarray(y_true, dtype="float64") - np.asarray(y_pred, dtype="float64")
    loss = np.maximum(alpha * diff, (alpha - 1.0) * diff)
    return float(np.mean(loss))


def _average_ranks(a):
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    pos = np.arange(1, len(a) + 1, dtype="float64")
    if len(a) == 1:
        return np.ones(1)
    change = np.concatenate([[True], sa[1:] != sa[:-1]])
    group_id = np.cumsum(change) - 1
    counts = np.bincount(group_id)
    sums = np.bincount(group_id, weights=pos)
    avg = sums / counts
    ranks = np.empty(len(a), dtype="float64")
    ranks[order] = avg[group_id]
    return ranks


def normalized_gini(y_true, y_pred):
    """Normalized Gini = 2*AUC - 1 for discriminating positive-loss rows.

    Implemented with rank statistics only (no sklearn) so that it can run in
    the same process that has loaded LightGBM's OpenMP runtime.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype="float64")
    pos = y_true > 0
    npos = int(pos.sum())
    nneg = len(y_true) - npos
    if npos == 0 or nneg == 0:
        return 0.0
    ranks = _average_ranks(y_pred)
    auc = (ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
    return float(2 * auc - 1)


def calibration_stats(y_true, y_pred):
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    if len(y_true) < 2:
        return 1.0, 0.0
    slope, intercept = np.polyfit(y_pred, y_true, 1)
    return float(slope), float(intercept)
