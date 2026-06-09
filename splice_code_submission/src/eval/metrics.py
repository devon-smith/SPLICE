# evaluation metrics for cut-continuity scoring
# task is class-imbalanced (~7.5% positive), so AUPRC is the headline, AUROC alongside

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                              f1_score, precision_recall_curve, precision_score,
                              recall_score, roc_auc_score)


def ranking_metrics(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "positive_rate": float(y_true.mean()),
        "n": int(len(y_true)),
    }


# hard-decision metrics at a fixed threshold (score >= thr -> positive)
def threshold_metrics(y_true, scores, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# threshold that maximises F1, swept over the precision-recall curve
def best_f1_threshold(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    prec, rec, thr = precision_recall_curve(y_true, scores)
    if len(thr) == 0:
        return 0.5
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def evaluate(y_true, scores, threshold):
    return {**ranking_metrics(y_true, scores), **threshold_metrics(y_true, scores, threshold)}
