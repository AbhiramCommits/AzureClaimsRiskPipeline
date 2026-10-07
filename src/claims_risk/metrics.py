import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def gamma_deviance(y_true, y_pred):
    eps = 1e-6
    y_true = np.clip(y_true, eps, None)
    y_pred = np.clip(y_pred, eps, None)
    return float(2 * np.mean((y_true - y_pred) / y_pred - np.log(y_true / y_pred)))

def pinball_loss(y_true, y_pred, alpha):
    diff = y_true - y_pred
    loss = np.maximum(alpha * diff, (alpha - 1.0) * diff)
    return float(np.mean(loss))

def normalized_gini(y_true, y_pred):
    try:
        from sklearn.metrics import roc_auc_score
        # Using Gini derived from ROC-AUC for binary or ranked loss concentration
        # Gini = 2 * AUC - 1
        binary_true = (y_true > 0).astype(int)
        if len(np.unique(binary_true)) < 2:
            return 0.0
        auc = roc_auc_score(binary_true, y_pred)
        return float(2 * auc - 1)
    except Exception:
        return 0.0

def calibration_stats(y_true, y_pred):
    # Regression of actual on predicted: actual = slope * predicted + intercept
    if len(y_true) < 2:
        return 1.0, 0.0
    slope, intercept = np.polyfit(y_pred, y_true, 1)
    return float(slope), float(intercept)
