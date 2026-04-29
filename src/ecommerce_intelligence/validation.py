from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ValidationReport:
    table_name: str
    row_count: int
    passed: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class DataValidator:
    """Lightweight production-style validation without tying the project to one data tool."""

    EVENT_COLUMNS = {
        "user_id",
        "session_id",
        "timestamp",
        "event_type",
        "product_id",
        "product_category",
        "product_price",
        "discount_percentage",
        "competitor_price",
        "inventory_level",
        "rating",
        "review_count",
        "user_location",
        "device_type",
        "search_query",
        "purchase_label",
    }
    CATALOG_COLUMNS = {"product_id", "category", "brand", "price", "margin", "inventory", "text_description"}
    DEMAND_COLUMNS = {"date", "product_id", "daily_sales", "seasonality_factor", "holiday_flag", "promotion_flag"}

    def validate_events(self, events: pd.DataFrame) -> ValidationReport:
        report = self._base_report("events", events, self.EVENT_COLUMNS)
        self._require_values(report, events, "event_type", {"view", "click", "add_to_cart", "purchase", "search"})
        self._require_range(report, events, "product_price", lower=0)
        self._require_range(report, events, "discount_percentage", lower=0, upper=0.9)
        self._require_range(report, events, "rating", lower=1, upper=5)
        self._require_range(report, events, "purchase_label", lower=0, upper=1)
        return self._finalize(report)

    def validate_catalog(self, catalog: pd.DataFrame) -> ValidationReport:
        report = self._base_report("product_catalog", catalog, self.CATALOG_COLUMNS)
        self._require_unique(report, catalog, "product_id")
        self._require_range(report, catalog, "price", lower=0)
        self._require_range(report, catalog, "margin", lower=0.05, upper=0.8)
        self._require_range(report, catalog, "inventory", lower=0)
        return self._finalize(report)

    def validate_demand(self, demand: pd.DataFrame) -> ValidationReport:
        report = self._base_report("demand", demand, self.DEMAND_COLUMNS)
        self._require_range(report, demand, "daily_sales", lower=0)
        self._require_range(report, demand, "seasonality_factor", lower=0.1, upper=3)
        return self._finalize(report)

    def _base_report(self, table_name: str, frame: pd.DataFrame, required_columns: set[str]) -> ValidationReport:
        report = ValidationReport(table_name=table_name, row_count=len(frame), passed=True)
        missing = sorted(required_columns - set(frame.columns))
        if missing:
            report.errors.append(f"Missing required columns: {missing}")
        if frame.empty:
            report.errors.append("Table is empty")
        null_rates = frame.isna().mean(numeric_only=False)
        high_null = null_rates[null_rates > 0.05]
        if not high_null.empty:
            report.warnings.append(f"Columns over 5% null: {high_null.round(3).to_dict()}")
        return report

    @staticmethod
    def _require_unique(report: ValidationReport, frame: pd.DataFrame, column: str) -> None:
        if column in frame and frame[column].duplicated().any():
            report.errors.append(f"{column} must be unique")

    @staticmethod
    def _require_values(report: ValidationReport, frame: pd.DataFrame, column: str, allowed: set[str]) -> None:
        if column in frame:
            unexpected = set(frame[column].dropna().unique()) - allowed
            if unexpected:
                report.errors.append(f"{column} has unexpected values: {sorted(unexpected)}")

    @staticmethod
    def _require_range(
        report: ValidationReport,
        frame: pd.DataFrame,
        column: str,
        lower: float | None = None,
        upper: float | None = None,
    ) -> None:
        if column not in frame:
            return
        series = pd.to_numeric(frame[column], errors="coerce")
        if lower is not None and (series < lower).any():
            report.errors.append(f"{column} contains values below {lower}")
        if upper is not None and (series > upper).any():
            report.errors.append(f"{column} contains values above {upper}")

    @staticmethod
    def _finalize(report: ValidationReport) -> ValidationReport:
        report.passed = len(report.errors) == 0
        return report
