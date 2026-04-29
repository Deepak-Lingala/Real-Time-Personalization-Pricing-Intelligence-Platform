from ecommerce_intelligence.metrics import average_precision_at_k, ndcg_at_k, precision_at_k, recall_at_k


def test_ranking_metrics_with_relevant_hits() -> None:
    recommended = ["P1", "P2", "P3", "P4"]
    relevant = {"P2", "P4", "P9"}

    assert precision_at_k(recommended, relevant, 4) == 0.5
    assert recall_at_k(recommended, relevant, 4) == 2 / 3
    assert 0 < ndcg_at_k(recommended, relevant, 4) <= 1
    assert 0 < average_precision_at_k(recommended, relevant, 4) <= 1


def test_ranking_metrics_without_relevant_items() -> None:
    recommended = ["P1", "P2"]
    relevant: set[str] = set()

    assert precision_at_k(recommended, relevant, 2) == 0
    assert recall_at_k(recommended, relevant, 2) == 0
    assert ndcg_at_k(recommended, relevant, 2) == 0
    assert average_precision_at_k(recommended, relevant, 2) == 0

