"""Evaluation metrics for the MedQA multi-agent system."""

from typing import Sequence

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar


def accuracy(predictions: Sequence[str], gold: Sequence[str]) -> float:
    correct = sum(p == g for p, g in zip(predictions, gold))
    return correct / len(gold)


def invalid_response_rate(predictions: Sequence[str], valid_options=("A", "B", "C", "D")) -> float:
    invalid = sum(p not in valid_options for p in predictions)
    return invalid / len(predictions)


def accuracy_gain(acc_v3: float, acc_v0: float) -> float:
    return acc_v3 - acc_v0


def bootstrap_ci(predictions: Sequence[str], gold: Sequence[str], n_boot: int = 1000, ci: float = 0.95):
    """Bootstrap confidence interval for accuracy. Returns (lower, upper)."""
    n = len(gold)
    correct = np.array([p == g for p, g in zip(predictions, gold)], dtype=float)
    boot_accs = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_accs.append(correct[idx].mean())
    lower = np.percentile(boot_accs, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_accs, (1 + ci) / 2 * 100)
    return lower, upper


def mcnemar_test(preds_a: Sequence[str], preds_b: Sequence[str], gold: Sequence[str]) -> float:
    """McNemar test comparing two systems' correctness patterns. Returns the p-value."""
    a_correct = np.array([p == g for p, g in zip(preds_a, gold)])
    b_correct = np.array([p == g for p, g in zip(preds_b, gold)])
    both_correct = int(np.sum(a_correct & b_correct))
    a_only = int(np.sum(a_correct & ~b_correct))
    b_only = int(np.sum(~a_correct & b_correct))
    both_wrong = int(np.sum(~a_correct & ~b_correct))
    table = [[both_correct, a_only], [b_only, both_wrong]]
    result = mcnemar(table, exact=True)
    return result.pvalue

