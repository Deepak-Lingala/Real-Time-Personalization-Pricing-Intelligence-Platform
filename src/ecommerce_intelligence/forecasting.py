from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ecommerce_intelligence.metrics import mape, wape


@dataclass(frozen=True)
class ForecastResult:
    product_id: str
    horizon_days: int
    forecast: list[dict]


class DemandForecaster:
    """Global product-level forecaster using lag, calendar, promotion, and seasonality features.

    `backend="auto"` uses LightGBM when available and falls back to sklearn for portable runs.
    """

    FEATURES = [
        "day_index",
        "day_of_week",
        "month",
        "seasonality_factor",
        "holiday_flag",
        "promotion_flag",
        "lag_7",
        "rolling_7",
        "category",
    ]

    def __init__(self, random_state: int = 42, backend: str = "auto") -> None:
        self.random_state = random_state
        self.backend = backend
        self.backend_used: str | None = None
        self.pipeline: Pipeline | None = None
        self.history: pd.DataFrame | None = None
        self.start_date: pd.Timestamp | None = None

    def fit(self, demand: pd.DataFrame) -> "DemandForecaster":
        frame = self._make_supervised(demand)
        numeric = [column for column in self.FEATURES if column != "category"]
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["category"]),
            ]
        )
        model, backend_used = self._select_model()
        self.pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        self.pipeline.fit(frame[self.FEATURES], frame["daily_sales"])
        self.backend_used = backend_used
        self.history = frame
        self.start_date = pd.to_datetime(frame["date"]).min()
        return self

    def forecast(self, product_id: str, horizon_days: int = 30) -> ForecastResult:
        if self.pipeline is None or self.history is None or self.start_date is None:
            raise RuntimeError("Demand forecaster has not been fitted")
        product_history = self.history.loc[self.history["product_id"] == product_id].sort_values("date")
        if product_history.empty:
            raise ValueError(f"No demand history found for {product_id}")

        category = str(product_history["category"].iloc[-1])
        last_date = pd.to_datetime(product_history["date"]).max()
        rolling_values = product_history["daily_sales"].tail(14).astype(float).tolist()
        rows = []
        for step in range(1, horizon_days + 1):
            date = last_date + pd.Timedelta(days=step)
            lag_7 = rolling_values[-7] if len(rolling_values) >= 7 else float(np.mean(rolling_values))
            rolling_7 = float(np.mean(rolling_values[-7:]))
            feature_row = pd.DataFrame(
                [
                    {
                        "date": date.date().isoformat(),
                        "product_id": product_id,
                        "category": category,
                        "day_index": int((date - self.start_date).days),
                        "day_of_week": int(date.dayofweek),
                        "month": int(date.month),
                        "seasonality_factor": self._seasonality_factor(date),
                        "holiday_flag": int(date.month == 12 or date.strftime("%m-%d") in {"11-28", "11-29"}),
                        "promotion_flag": int(date.dayofweek in {4, 5}),
                        "lag_7": lag_7,
                        "rolling_7": rolling_7,
                    }
                ]
            )
            prediction = max(0.0, float(self.pipeline.predict(feature_row[self.FEATURES])[0]))
            rolling_values.append(prediction)
            rows.append({"date": date.date().isoformat(), "predicted_demand": round(prediction, 2)})
        return ForecastResult(product_id=product_id, horizon_days=horizon_days, forecast=rows)

    def forecast_category(self, category: str, horizon_days: int = 30) -> dict:
        if self.history is None:
            raise RuntimeError("Demand forecaster has not been fitted")
        product_ids = self.history.loc[self.history["category"] == category, "product_id"].drop_duplicates().head(25)
        combined: dict[str, float] = {}
        for product_id in product_ids:
            result = self.forecast(str(product_id), horizon_days=horizon_days)
            for row in result.forecast:
                combined[row["date"]] = combined.get(row["date"], 0.0) + float(row["predicted_demand"])
        return {
            "category": category,
            "horizon_days": horizon_days,
            "forecast": [
                {"date": date, "predicted_demand": round(value, 2)}
                for date, value in sorted(combined.items())
            ],
        }

    def seasonal_naive_baseline(self, demand: pd.DataFrame, holdout_days: int = 21) -> dict[str, float]:
        frame = self._make_supervised(demand)
        cutoff = pd.to_datetime(frame["date"]).max() - pd.Timedelta(days=holdout_days)
        test = frame.loc[pd.to_datetime(frame["date"]) > cutoff].copy()
        if test.empty:
            return {"baseline_mae": 0.0, "baseline_rmse": 0.0, "baseline_wape": 0.0}
        predictions = test["lag_7"].to_numpy(dtype=float)
        y_true = test["daily_sales"].to_numpy(dtype=float)
        return {
            "baseline_mae": round(float(mean_absolute_error(y_true, predictions)), 3),
            "baseline_rmse": round(float(np.sqrt(mean_squared_error(y_true, predictions))), 3),
            "baseline_wape": round(float(wape(y_true, predictions)), 4),
        }

    def evaluate(self, demand: pd.DataFrame, holdout_days: int = 21) -> dict[str, float]:
        frame = self._make_supervised(demand)
        cutoff = pd.to_datetime(frame["date"]).max() - pd.Timedelta(days=holdout_days)
        train = frame.loc[pd.to_datetime(frame["date"]) <= cutoff]
        test = frame.loc[pd.to_datetime(frame["date"]) > cutoff]
        if train.empty or test.empty:
            return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}
        self.fit(train)
        predictions = self.pipeline.predict(test[self.FEATURES])  # type: ignore[union-attr]
        y_true = test["daily_sales"].to_numpy(dtype=float)
        return {
            "mae": round(float(mean_absolute_error(y_true, predictions)), 3),
            "rmse": round(float(np.sqrt(mean_squared_error(y_true, predictions))), 3),
            "mape": round(float(mape(y_true, predictions)), 4),
            "wape": round(float(wape(y_true, predictions)), 4),
        }

    def _make_supervised(self, demand: pd.DataFrame) -> pd.DataFrame:
        frame = demand.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values(["product_id", "date"])
        start = frame["date"].min()
        frame["day_index"] = (frame["date"] - start).dt.days
        frame["day_of_week"] = frame["date"].dt.dayofweek
        frame["month"] = frame["date"].dt.month
        frame["lag_7"] = frame.groupby("product_id")["daily_sales"].shift(7)
        frame["rolling_7"] = frame.groupby("product_id")["daily_sales"].transform(
            lambda series: series.shift(1).rolling(7, min_periods=1).mean()
        )
        frame["lag_7"] = frame["lag_7"].fillna(frame.groupby("product_id")["daily_sales"].transform("mean"))
        frame["rolling_7"] = frame["rolling_7"].fillna(frame["lag_7"])
        return frame.dropna(subset=self.FEATURES + ["daily_sales"])

    @staticmethod
    def _seasonality_factor(timestamp: pd.Timestamp) -> float:
        week = timestamp.isocalendar().week
        yearly = 1.0 + 0.18 * np.sin(2 * np.pi * week / 52)
        weekend = 1.08 if timestamp.dayofweek >= 5 else 0.97
        q4 = 1.18 if timestamp.month in {11, 12} else 1.0
        return float(yearly * weekend * q4)

    def _select_model(self):
        if self.backend in {"auto", "lightgbm"}:
            try:
                from lightgbm import LGBMRegressor

                return (
                    LGBMRegressor(
                        n_estimators=260,
                        learning_rate=0.045,
                        num_leaves=31,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=self.random_state,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                    "lightgbm_regressor",
                )
            except ImportError:
                if self.backend == "lightgbm":
                    raise

        return (
            RandomForestRegressor(
                n_estimators=180,
                min_samples_leaf=3,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "sklearn_random_forest",
        )
