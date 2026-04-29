from __future__ import annotations

import numpy as np


def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = recommended[:k]
    return len(set(top_k) & relevant) / k


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    return len(set(top_k) & relevant) / len(relevant)


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    gains = np.array([1.0 if item in relevant else 0.0 for item in recommended[:k]])
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains * discounts))
    ideal_len = min(len(relevant), k)
    ideal_dcg = float(np.sum(discounts[:ideal_len]))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def average_precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for index, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / index
    return score / min(len(relevant), k)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.maximum(np.abs(y_true), 1.0)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sum(np.abs(y_true - y_pred)) / max(np.sum(np.abs(y_true)), 1.0))


def catalog_coverage(recommendations_by_user: dict[str, list[str]], catalog_size: int) -> float:
    recommended_items = {item for items in recommendations_by_user.values() for item in items}
    return len(recommended_items) / max(catalog_size, 1)


def category_diversity(recommended_categories: list[str]) -> float:
    if not recommended_categories:
        return 0.0
    return len(set(recommended_categories)) / len(recommended_categories)
