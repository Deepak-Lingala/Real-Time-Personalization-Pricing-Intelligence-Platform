import pytest

from ecommerce_intelligence.data_generator import SyntheticEcommerceGenerator
from ecommerce_intelligence.features import FeatureEngineer
from ecommerce_intelligence.preprocessing import EventPreprocessor
from ecommerce_intelligence.recommenders import (
    LearningToRankReranker,
    TwoStageRecommendationSystem,
    TwoTowerRetrievalModel,
)


def test_two_stage_recommender_returns_ranked_outputs() -> None:
    bundle = SyntheticEcommerceGenerator(seed=7).generate_all(
        n_users=35,
        n_products=30,
        n_events=350,
        start_date="2025-01-01",
        days=21,
    )
    events = EventPreprocessor().clean_events(bundle.events)
    features = FeatureEngineer()
    user_features = features.build_user_features(events)
    product_features = features.build_product_features(events, bundle.product_catalog)

    model = TwoStageRecommendationSystem(random_state=7, retrieval_top_k=20, retrieval_backend="sklearn").fit(
        events,
        bundle.product_catalog,
        user_features,
        product_features,
    )
    recs = model.recommend(user_features["user_id"].iloc[0], k=5)

    assert len(recs) <= 5
    assert recs[0].product_id.startswith("P")
    assert 0 <= recs[0].predicted_purchase_probability <= 1


def test_torch_two_tower_backend_trains_when_torch_is_available() -> None:
    pytest.importorskip("torch")
    bundle = SyntheticEcommerceGenerator(seed=11).generate_all(
        n_users=20,
        n_products=18,
        n_events=160,
        start_date="2025-01-01",
        days=10,
    )
    events = EventPreprocessor().clean_events(bundle.events)

    model = TwoTowerRetrievalModel(
        embedding_dim=8,
        backend="torch",
        random_state=11,
        epochs=1,
        batch_size=64,
        torch_device="cpu",
    ).fit(events, bundle.product_catalog)
    candidates = model.retrieve(events["user_id"].iloc[0], top_k=5)

    assert model.backend_used == "torch"
    assert model.training_device == "cpu"
    assert model.training_loss_history
    assert len(candidates) == 5


def test_retrieval_and_ranking_inference_contracts() -> None:
    bundle = SyntheticEcommerceGenerator(seed=19).generate_all(
        n_users=32,
        n_products=28,
        n_events=320,
        start_date="2025-01-01",
        days=18,
    )
    events = EventPreprocessor().clean_events(bundle.events)
    features = FeatureEngineer()
    user_features = features.build_user_features(events)
    product_features = features.build_product_features(events, bundle.product_catalog)

    retrieval = TwoTowerRetrievalModel(backend="sklearn", random_state=19).fit(
        events,
        bundle.product_catalog,
    )
    candidates = retrieval.retrieve(str(user_features["user_id"].iloc[0]), top_k=12)
    reranker = LearningToRankReranker(random_state=19, backend="sklearn").fit(
        events,
        product_features,
        user_features,
        retrieval,
    )
    scored = reranker.score(candidates, str(user_features["user_id"].iloc[0]), user_features)

    assert retrieval.backend_used == "sklearn"
    assert reranker.backend_used == "sklearn_gradient_boosting"
    assert len(candidates) == 12
    assert scored["ranking_score"].between(0, 1).all()
