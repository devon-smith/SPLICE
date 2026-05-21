"""Unit tests for evaluation metrics (src.eval.metrics)."""

import numpy as np

from src.eval.metrics import best_f1_threshold, evaluate, ranking_metrics, threshold_metrics


def test_ranking_metrics_perfect_separation():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    m = ranking_metrics(y, scores)
    assert m["auroc"] == 1.0
    assert m["auprc"] == 1.0
    assert m["n"] == 4
    assert m["positive_rate"] == 0.5


def test_ranking_metrics_random_is_half():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    assert ranking_metrics(y, scores)["auroc"] == 0.5


def test_threshold_metrics_known_confusion():
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.4, 0.8, 0.1])  # at thr 0.5: pred = [1,0,1,0]
    m = threshold_metrics(y, scores, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["accuracy"] == 0.5
    assert abs(m["precision"] - 0.5) < 1e-9
    assert abs(m["recall"] - 0.5) < 1e-9


def test_best_f1_threshold_separates():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr = best_f1_threshold(y, scores)
    m = threshold_metrics(y, scores, thr)
    assert m["f1"] == 1.0  # perfectly separable -> F1 = 1 at the chosen threshold


def test_evaluate_merges_both():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.2, 0.7, 0.3, 0.9])
    m = evaluate(y, scores, 0.5)
    assert {"auroc", "auprc", "f1", "precision", "recall", "accuracy", "threshold"} <= set(m)


def test_threshold_metrics_no_positives_predicted():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    m = threshold_metrics(y, scores, 0.9)  # nothing crosses -> all predicted negative
    assert m["f1"] == 0.0
    assert m["tp"] == 0
