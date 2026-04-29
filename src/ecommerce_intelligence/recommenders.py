from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import normalize

from ecommerce_intelligence.features import FeatureEngineer
from ecommerce_intelligence.metrics import (
    average_precision_at_k,
    catalog_coverage,
    category_diversity,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


@dataclass(frozen=True)
class CompanyRecommendation:
    product_id: str
    product_category: str
    brand: str
    retrieval_score: float
    ranking_score: float
    product_score: float
    recommendation_reason: str
    category_match: bool
    predicted_purchase_probability: float


def _vector_normalize(values: np.ndarray) -> np.ndarray:
    finite = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    minimum = finite.min()
    maximum = finite.max()
    if np.isclose(maximum, minimum):
        return np.zeros_like(finite, dtype=float)
    return (finite - minimum) / (maximum - minimum)


class TwoTowerRetrievalModel:
    """Two-tower retrieval model with a real PyTorch training backend.

    `backend="auto"` uses PyTorch when it is installed, and trains on CUDA automatically when a
    Colab GPU is available. `backend="sklearn"` keeps CI/local smoke tests lightweight.
    """

    def __init__(
        self,
        embedding_dim: int = 48,
        backend: str = "auto",
        random_state: int = 42,
        epochs: int = 4,
        batch_size: int = 1024,
        learning_rate: float = 1e-3,
        negative_samples: int = 2,
        torch_device: str | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.backend = backend
        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.negative_samples = negative_samples
        self.torch_device = torch_device
        self.feature_engineer = FeatureEngineer()
        self.user_ids: list[str] = []
        self.product_ids: list[str] = []
        self.user_index: dict[str, int] = {}
        self.product_index: dict[str, int] = {}
        self.user_embeddings: np.ndarray | None = None
        self.product_embeddings: np.ndarray | None = None
        self.catalog: pd.DataFrame | None = None
        self.user_seen: dict[str, set[str]] = {}
        self.backend_used: str | None = None
        self.training_device: str | None = None
        self.training_loss_history: list[float] = []

    def fit(self, events: pd.DataFrame, catalog: pd.DataFrame) -> "TwoTowerRetrievalModel":
        self.catalog = catalog.reset_index(drop=True).copy()
        self.product_ids = self.catalog["product_id"].astype(str).tolist()
        self.product_index = {product_id: index for index, product_id in enumerate(self.product_ids)}
        self.user_seen = events.groupby("user_id")["product_id"].apply(set).to_dict()

        interaction_matrix = self.feature_engineer.build_interaction_matrix(events)
        self.user_ids = interaction_matrix.index.astype(str).tolist()
        self.user_index = {user_id: index for index, user_id in enumerate(self.user_ids)}
        aligned = interaction_matrix.reindex(columns=self.product_ids, fill_value=0)
        values = aligned.to_numpy(dtype=float)

        if self.backend in {"auto", "torch"}:
            try:
                self._fit_torch(events)
                self.backend_used = "torch"
                return self
            except ImportError:
                if self.backend == "torch":
                    raise
            except RuntimeError:
                if self.backend == "torch":
                    raise

        self._fit_sklearn(values)
        self.backend_used = "sklearn"
        return self

    def _fit_sklearn(self, values: np.ndarray) -> None:
        """CPU fallback for environments where PyTorch is unavailable."""

        components = max(2, min(self.embedding_dim, min(values.shape) - 1)) if min(values.shape) > 2 else 2
        if min(values.shape) > 2:
            svd = TruncatedSVD(n_components=components, random_state=self.random_state)
            users = svd.fit_transform(values)
            products = svd.components_.T
        else:
            users = np.zeros((len(self.user_ids), components), dtype=float)
            products = np.zeros((len(self.product_ids), components), dtype=float)

        self.user_embeddings = normalize(users)
        self.product_embeddings = normalize(products)

    def _fit_torch(self, events: pd.DataFrame) -> None:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        if not self.user_ids or not self.product_ids:
            raise RuntimeError("Cannot train PyTorch two-tower model without users and products")

        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        device = self.torch_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.training_device = device
        product_attributes = torch.tensor(self._product_attribute_matrix(), dtype=torch.float32).to(device)

        positive = events.loc[
            events["user_id"].isin(self.user_index) & events["product_id"].isin(self.product_index),
            ["user_id", "product_id", "event_type"],
        ].copy()
        if positive.empty:
            raise RuntimeError("Cannot train PyTorch two-tower model without interaction pairs")

        event_weight = {"view": 0.35, "search": 0.45, "click": 0.65, "add_to_cart": 0.9, "purchase": 1.0}
        positive["sample_weight"] = positive["event_type"].map(event_weight).fillna(0.5)
        positive = positive.sample(
            n=min(len(positive), 80000),
            weights="sample_weight",
            replace=len(positive) < min(len(positive), 80000),
            random_state=self.random_state,
        )

        user_indices = [self.user_index[str(user_id)] for user_id in positive["user_id"]]
        product_indices = [self.product_index[str(product_id)] for product_id in positive["product_id"]]
        labels = [1.0] * len(user_indices)

        positive_seen = {
            self.user_index[user_id]: {self.product_index[item] for item in items if item in self.product_index}
            for user_id, items in self.user_seen.items()
            if user_id in self.user_index
        }
        all_products = np.arange(len(self.product_ids))
        for user_index, _product_index in list(zip(user_indices, product_indices)):
            seen = positive_seen.get(user_index, set())
            candidate_pool = np.setdiff1d(all_products, list(seen), assume_unique=False)
            if len(candidate_pool) == 0:
                candidate_pool = all_products
            negatives = rng.choice(candidate_pool, size=self.negative_samples, replace=len(candidate_pool) < self.negative_samples)
            for negative_index in negatives:
                user_indices.append(user_index)
                product_indices.append(int(negative_index))
                labels.append(0.0)

        dataset = TensorDataset(
            torch.tensor(user_indices, dtype=torch.long),
            torch.tensor(product_indices, dtype=torch.long),
            torch.tensor(labels, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        class TorchTwoTower(nn.Module):
            def __init__(
                self,
                n_users: int,
                n_products: int,
                embedding_dim: int,
                product_attribute_dim: int,
            ) -> None:
                super().__init__()
                self.user_embedding = nn.Embedding(n_users, embedding_dim)
                self.product_embedding = nn.Embedding(n_products, embedding_dim)
                self.user_tower = nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.ReLU(),
                    nn.Linear(embedding_dim, embedding_dim),
                )
                self.product_tower = nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.ReLU(),
                    nn.Linear(embedding_dim, embedding_dim),
                )
                self.product_attribute_projection = nn.Sequential(
                    nn.Linear(product_attribute_dim, embedding_dim),
                    nn.ReLU(),
                    nn.Linear(embedding_dim, embedding_dim),
                )

            def forward(self, user_idx, product_idx):
                user_vector = nn.functional.normalize(self.user_tower(self.user_embedding(user_idx)), dim=1)
                product_input = self.product_embedding(product_idx) + self.product_attribute_projection(
                    product_attributes[product_idx]
                )
                product_vector = nn.functional.normalize(self.product_tower(product_input), dim=1)
                return (user_vector * product_vector).sum(dim=1)

            def export_embeddings(self):
                user_idx = torch.arange(self.user_embedding.num_embeddings, device=self.user_embedding.weight.device)
                product_idx = torch.arange(self.product_embedding.num_embeddings, device=self.product_embedding.weight.device)
                with torch.no_grad():
                    users = nn.functional.normalize(self.user_tower(self.user_embedding(user_idx)), dim=1)
                    product_input = self.product_embedding(product_idx) + self.product_attribute_projection(
                        product_attributes[product_idx]
                    )
                    products = nn.functional.normalize(self.product_tower(product_input), dim=1)
                return users.cpu().numpy(), products.cpu().numpy()

        model = TorchTwoTower(
            len(self.user_ids),
            len(self.product_ids),
            self.embedding_dim,
            product_attributes.shape[1],
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss()

        self.training_loss_history = []
        for _epoch in range(self.epochs):
            epoch_losses = []
            model.train()
            for batch_users, batch_products, batch_labels in loader:
                batch_users = batch_users.to(device)
                batch_products = batch_products.to(device)
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad()
                logits = model(batch_users, batch_products)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            self.training_loss_history.append(round(float(np.mean(epoch_losses)), 5) if epoch_losses else 0.0)

        model.eval()
        users, products = model.export_embeddings()
        self.user_embeddings = users
        self.product_embeddings = products

    def _product_attribute_matrix(self) -> np.ndarray:
        if self.catalog is None:
            return np.zeros((len(self.product_ids), 1), dtype=np.float32)

        frame = self.catalog.set_index("product_id").reindex(self.product_ids).reset_index()
        if "category" not in frame and "product_category" in frame:
            frame["category"] = frame["product_category"]

        numeric = pd.DataFrame(
            {
                "log_price": np.log1p(pd.to_numeric(frame.get("price", 0), errors="coerce").fillna(0)),
                "margin": pd.to_numeric(frame.get("margin", 0), errors="coerce").fillna(0),
                "rating": pd.to_numeric(frame.get("rating", 0), errors="coerce").fillna(0) / 5,
                "review_count": np.log1p(
                    pd.to_numeric(frame.get("review_count", 0), errors="coerce").fillna(0)
                )
                / 10,
                "inventory": np.log1p(
                    pd.to_numeric(frame.get("inventory", 0), errors="coerce").fillna(0)
                )
                / 10,
            }
        )
        numeric = (numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, 1)
        category_brand = pd.get_dummies(
            frame[["category", "brand"]].fillna("unknown").astype(str),
            columns=["category", "brand"],
            dtype=float,
        )
        image_source = (
            frame["image_feature_vector"]
            if "image_feature_vector" in frame
            else pd.Series([[]] * len(frame), index=frame.index)
        )
        image_vectors = np.vstack([self._parse_vector(value, expected_dim=8) for value in image_source])
        matrix = np.hstack([numeric.to_numpy(dtype=float), category_brand.to_numpy(dtype=float), image_vectors])
        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    @staticmethod
    def _parse_vector(value: object, expected_dim: int) -> np.ndarray:
        parsed: list[float]
        if isinstance(value, str):
            try:
                parsed = list(json.loads(value))
            except json.JSONDecodeError:
                parsed = []
        elif isinstance(value, (list, tuple, np.ndarray)):
            parsed = list(value)
        else:
            parsed = []

        vector = np.asarray(parsed, dtype=float).reshape(-1)
        if len(vector) < expected_dim:
            vector = np.pad(vector, (0, expected_dim - len(vector)))
        return vector[:expected_dim]

    def retrieve(self, user_id: str, top_k: int = 100, remove_seen: bool = True) -> pd.DataFrame:
        if self.user_embeddings is None or self.product_embeddings is None or self.catalog is None:
            raise RuntimeError("Retrieval model has not been fitted")

        if user_id in self.user_index:
            user_vector = self.user_embeddings[self.user_index[user_id]]
        else:
            user_vector = self.product_embeddings.mean(axis=0)
        scores = self.product_embeddings @ user_vector
        scores = _vector_normalize(scores)

        if remove_seen:
            for product_id in self.user_seen.get(user_id, set()):
                if product_id in self.product_index:
                    scores[self.product_index[product_id]] = -np.inf

        indices = np.argsort(scores)[::-1][:top_k]
        candidates = self.catalog.iloc[indices].copy()
        candidates["retrieval_score"] = [round(float(scores[index]), 5) for index in indices]
        return candidates.reset_index(drop=True)


class LearningToRankReranker:
    """LTR-style reranker for retrieved candidates.

    `backend="auto"` uses XGBoost Ranker or LightGBM Ranker when installed. The sklearn fallback
    keeps local smoke tests lightweight while preserving the same ranking feature contract.
    """

    NUMERIC_FEATURES = [
        "retrieval_score",
        "price",
        "margin",
        "rating",
        "review_count",
        "inventory_level",
        "demand_score",
        "conversion_rate",
        "ctr",
        "category_match",
    ]
    CATEGORICAL_FEATURES = ["category", "brand"]

    def __init__(self, random_state: int = 42, backend: str = "auto") -> None:
        self.random_state = random_state
        self.backend = backend
        self.backend_used: str | None = None
        self.pipeline: Pipeline | None = None

    def fit(
        self,
        events: pd.DataFrame,
        product_features: pd.DataFrame,
        user_features: pd.DataFrame,
        retrieval_model: TwoTowerRetrievalModel,
    ) -> "LearningToRankReranker":
        training = events.sample(n=min(len(events), 25000), random_state=self.random_state).copy()
        product_frame = product_features.rename(columns={"inventory": "inventory_level"}).copy()
        training = training.merge(product_frame, on="product_id", how="left", suffixes=("", "_product"))
        dominant = user_features.set_index("user_id")["dominant_category"].to_dict()
        training["category_match"] = [
            int(dominant.get(user_id) == category)
            for user_id, category in zip(training["user_id"], training["product_category"])
        ]
        training["retrieval_score"] = training.apply(
            lambda row: self._lookup_retrieval_score(retrieval_model, str(row["user_id"]), str(row["product_id"])),
            axis=1,
        )
        training = self._normalize_columns(training)
        training = training.dropna(subset=self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES + ["purchase_label"])
        if training["purchase_label"].nunique() < 2 and len(training) > 1:
            training.loc[training.index[0], "purchase_label"] = 1 - int(training["purchase_label"].iloc[0])
        training = training.sort_values("user_id").reset_index(drop=True)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.CATEGORICAL_FEATURES),
            ]
        )
        model, backend_used, uses_group = self._select_model()
        self.pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        features = training[self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES]
        labels = training["purchase_label"].astype(int)
        if uses_group:
            groups = training.groupby("user_id", sort=False).size().to_numpy()
            self.pipeline.fit(features, labels, model__group=groups)
        else:
            self.pipeline.fit(features, labels)
        self.backend_used = backend_used
        return self

    def score(self, candidates: pd.DataFrame, user_id: str, user_features: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None:
            raise RuntimeError("Reranker has not been fitted")
        frame = candidates.copy()
        dominant = user_features.set_index("user_id")["dominant_category"].to_dict()
        frame["category_match"] = (frame["category"] == dominant.get(user_id)).astype(int)
        frame = self._normalize_columns(frame)
        model_input = frame[self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES]
        backend_used = getattr(self, "backend_used", "sklearn_gradient_boosting")
        if backend_used in {"xgboost_ranker", "lightgbm_ranker"}:
            raw_scores = np.asarray(self.pipeline.predict(model_input), dtype=float)
            frame["ranking_score"] = 1 / (1 + np.exp(-raw_scores))
        else:
            frame["ranking_score"] = self.pipeline.predict_proba(model_input)[:, 1]
        return frame

    def _select_model(self):
        if self.backend in {"auto", "xgboost"}:
            try:
                from xgboost import XGBRanker

                return (
                    XGBRanker(
                        objective="rank:pairwise",
                        n_estimators=120,
                        learning_rate=0.06,
                        max_depth=4,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        tree_method="hist",
                        random_state=self.random_state,
                    ),
                    "xgboost_ranker",
                    True,
                )
            except ImportError:
                if self.backend == "xgboost":
                    raise

        if self.backend in {"auto", "lightgbm"}:
            try:
                from lightgbm import LGBMRanker

                return (
                    LGBMRanker(
                        objective="lambdarank",
                        n_estimators=160,
                        learning_rate=0.05,
                        num_leaves=31,
                        random_state=self.random_state,
                        verbose=-1,
                    ),
                    "lightgbm_ranker",
                    True,
                )
            except ImportError:
                if self.backend == "lightgbm":
                    raise

        return GradientBoostingClassifier(random_state=self.random_state), "sklearn_gradient_boosting", False

    @staticmethod
    def _lookup_retrieval_score(model: TwoTowerRetrievalModel, user_id: str, product_id: str) -> float:
        if model.user_embeddings is None or model.product_embeddings is None:
            return 0.0
        if user_id not in model.user_index or product_id not in model.product_index:
            return 0.0
        score = float(model.user_embeddings[model.user_index[user_id]] @ model.product_embeddings[model.product_index[product_id]])
        return score

    @classmethod
    def _normalize_columns(cls, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        if "product_category" in result and "category" not in result:
            result["category"] = result["product_category"]
        if "product_price" in result and "price" not in result:
            result["price"] = result["product_price"]
        if "product_margin" in result and "margin" not in result:
            result["margin"] = result["product_margin"]
        if "inventory" in result and "inventory_level" not in result:
            result["inventory_level"] = result["inventory"]
        for column in cls.NUMERIC_FEATURES:
            if column not in result:
                result[column] = 0.0
            result[column] = pd.to_numeric(result[column], errors="coerce").replace([np.inf, -np.inf], 0).fillna(0)
        for column in cls.CATEGORICAL_FEATURES:
            if column not in result:
                result[column] = "unknown"
        return result


class TwoStageRecommendationSystem:
    """Company-style candidate retrieval plus LTR reranking recommendation system."""

    def __init__(
        self,
        random_state: int = 42,
        retrieval_top_k: int = 100,
        retrieval_backend: str = "auto",
        ranking_backend: str = "auto",
        retrieval_epochs: int = 4,
        retrieval_batch_size: int = 1024,
    ) -> None:
        self.random_state = random_state
        self.retrieval_top_k = retrieval_top_k
        self.retrieval_model = TwoTowerRetrievalModel(
            random_state=random_state,
            backend=retrieval_backend,
            epochs=retrieval_epochs,
            batch_size=retrieval_batch_size,
        )
        self.reranker = LearningToRankReranker(random_state=random_state, backend=ranking_backend)
        self.user_features: pd.DataFrame | None = None
        self.product_features: pd.DataFrame | None = None

    def fit(self, events: pd.DataFrame, catalog: pd.DataFrame, user_features: pd.DataFrame, product_features: pd.DataFrame) -> "TwoStageRecommendationSystem":
        self.user_features = user_features.copy()
        self.product_features = product_features.copy()
        self.retrieval_model.fit(events, catalog)
        self.reranker.fit(events, product_features, user_features, self.retrieval_model)
        return self

    def recommend(self, user_id: str, k: int = 10) -> list[CompanyRecommendation]:
        if self.user_features is None or self.product_features is None:
            raise RuntimeError("Two-stage recommendation system has not been fitted")
        candidates = self.retrieval_model.retrieve(user_id, top_k=self.retrieval_top_k)
        enriched = candidates.merge(
            self.product_features.drop(columns=[column for column in ["category", "brand"] if column in self.product_features], errors="ignore"),
            on="product_id",
            how="left",
            suffixes=("", "_features"),
        )
        enriched = LearningToRankReranker._normalize_columns(enriched)
        filtered = self._apply_business_filters(enriched)
        if filtered.empty:
            filtered = enriched.head(k).copy()
        ranked = self.reranker.score(filtered, user_id, self.user_features)
        ranked["product_score"] = (
            0.52 * ranked["ranking_score"]
            + 0.28 * ranked["retrieval_score"].replace(-np.inf, 0)
            + 0.12 * ranked["margin"].fillna(0)
            + 0.08 * (ranked["inventory_level"].fillna(0) > 0).astype(float)
        )
        ranked = ranked.sort_values("product_score", ascending=False).head(k)
        dominant = self.user_features.set_index("user_id")["dominant_category"].to_dict().get(user_id)
        output = []
        for row in ranked.itertuples(index=False):
            category = str(getattr(row, "category"))
            category_match = bool(category == dominant)
            output.append(
                CompanyRecommendation(
                    product_id=str(getattr(row, "product_id")),
                    product_category=category,
                    brand=str(getattr(row, "brand")),
                    retrieval_score=round(float(getattr(row, "retrieval_score")), 4),
                    ranking_score=round(float(getattr(row, "ranking_score")), 4),
                    product_score=round(float(getattr(row, "product_score")), 4),
                    recommendation_reason=self._reason(category_match, float(getattr(row, "retrieval_score")), float(getattr(row, "ranking_score"))),
                    category_match=category_match,
                    predicted_purchase_probability=round(float(getattr(row, "ranking_score")), 4),
                )
            )
        return output

    def evaluate(self, train_events: pd.DataFrame, test_events: pd.DataFrame, catalog: pd.DataFrame, user_features: pd.DataFrame, product_features: pd.DataFrame, k: int = 10) -> dict[str, float]:
        self.fit(train_events, catalog, user_features, product_features)
        relevant_by_user = (
            test_events.loc[test_events["purchase_label"] == 1]
            .groupby("user_id")["product_id"]
            .apply(lambda values: set(values.tolist()))
            .to_dict()
        )
        final_recs: dict[str, list[str]] = {}
        retrieval_recs: dict[str, list[str]] = {}
        categories: list[str] = []
        rows = []
        cold_rows = []
        user_event_counts = train_events.groupby("user_id").size().to_dict()

        for user_id, relevant in relevant_by_user.items():
            retrieved = self.retrieval_model.retrieve(user_id, top_k=100)["product_id"].astype(str).tolist()
            recommendations = self.recommend(user_id, k=k)
            recommended_ids = [item.product_id for item in recommendations]
            categories.extend(item.product_category for item in recommendations)
            final_recs[user_id] = recommended_ids
            retrieval_recs[user_id] = retrieved
            metric_row = {
                "retrieval_recall_at_100": recall_at_k(retrieved, relevant, 100),
                "precision_at_k": precision_at_k(recommended_ids, relevant, k),
                "recall_at_k": recall_at_k(recommended_ids, relevant, k),
                "ndcg_at_k": ndcg_at_k(recommended_ids, relevant, k),
                "map_at_k": average_precision_at_k(recommended_ids, relevant, k),
            }
            rows.append(metric_row)
            if user_event_counts.get(user_id, 0) <= 3:
                cold_rows.append(metric_row)

        metrics = {key: 0.0 for key in ["retrieval_recall_at_100", "precision_at_k", "recall_at_k", "ndcg_at_k", "map_at_k"]}
        if rows:
            frame = pd.DataFrame(rows)
            metrics.update({column: round(float(frame[column].mean()), 4) for column in frame.columns})
        metrics["catalog_coverage"] = round(catalog_coverage(final_recs, len(catalog)), 4)
        metrics["diversity"] = round(category_diversity(categories), 4)
        metrics["cold_start_performance"] = round(float(pd.DataFrame(cold_rows)["recall_at_k"].mean()), 4) if cold_rows else 0.0
        metrics["simulated_ctr_lift"] = round(metrics["precision_at_k"] * 0.42 + metrics["diversity"] * 0.08, 4)
        return metrics

    @staticmethod
    def _apply_business_filters(candidates: pd.DataFrame) -> pd.DataFrame:
        frame = candidates.copy()
        margin = frame["margin"].fillna(0)
        inventory = frame["inventory_level"].fillna(0)
        return frame.loc[(inventory > 0) & (margin >= 0.12)].copy()

    @staticmethod
    def _reason(category_match: bool, retrieval_score: float, ranking_score: float) -> str:
        if category_match:
            return "Recommended because it matches the user's category affinity"
        if ranking_score > retrieval_score:
            return "Recommended because ranking features predict purchase intent"
        return "Recommended because the two-tower retrieval model found similar behavior"
