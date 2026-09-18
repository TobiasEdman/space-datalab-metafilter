"""Meteorological analog-date ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


DEFAULT_FEATURES = (
    "gdd_prev30d_c",
    "precip_prev30d_mm",
    "precip_prev7d_mm",
    "swvl1_prev30d_mean",
    "ssrd_prev30d_mj_m2",
)


@dataclass(frozen=True)
class AnalogMatch:
    """One ranked candidate day and its auditable feature deltas."""

    year: int
    date: str
    distance: float
    feature_values: dict[str, float]
    normalized_deltas: dict[str, float]


class AnalogModel:
    """Rank dates by standardized meteorological distance."""

    def __init__(
        self,
        *,
        features: Iterable[str] = DEFAULT_FEATURES,
        metric: str = "mahalanobis",
        weights: Mapping[str, float] | None = None,
        regularization: float = 1e-6,
    ) -> None:
        self.features = tuple(features)
        if not self.features:
            raise ValueError("features must not be empty")
        if len(set(self.features)) != len(self.features):
            raise ValueError("features must not contain duplicates")
        if metric not in {"euclidean", "mahalanobis"}:
            raise ValueError("metric must be 'euclidean' or 'mahalanobis'")
        regularization = float(regularization)
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization must be finite and positive")
        self.metric = metric
        self.weights = {
            feature: float((weights or {}).get(feature, 1.0))
            for feature in self.features
        }
        if any(
            not np.isfinite(weight) or weight < 0
            for weight in self.weights.values()
        ):
            raise ValueError("feature weights must be finite and non-negative")
        self.regularization = regularization
        self.rejected_rows: dict[int, int] = {}
        self._fitted = False

    def fit(self, frames: Mapping[int, pd.DataFrame]) -> "AnalogModel":
        """Fit pooled normalization and covariance from per-year frames.

        Every year is validated and the new state is built locally before any
        of it is published, so a rejected refit leaves the previous fit intact.
        A row needs a parseable date as well as complete features; rows
        failing either are counted in ``rejected_rows``.
        """
        required = {"date", *self.features}
        by_year: dict[int, pd.DataFrame] = {}
        rejected: dict[int, int] = {}
        prepared = []
        for year, frame in frames.items():
            missing = sorted(required - set(frame.columns))
            if missing:
                raise ValueError(f"year {year} missing columns: {', '.join(missing)}")
            selected = frame.loc[:, ["date", *self.features]].copy()
            dates = pd.to_datetime(selected["date"], errors="coerce")
            feature_values = selected.loc[:, self.features].to_numpy(dtype=float)
            valid = (
                pd.Series(np.isfinite(feature_values).all(axis=1), index=selected.index)
                & dates.notna()
            )
            rejected[int(year)] = int((~valid).sum())
            kept_dates = dates.loc[valid]
            selected = selected.loc[valid].reset_index(drop=True)
            if selected.empty:
                raise ValueError(f"year {year} has no rows with complete features and dates")
            selected["date"] = kept_dates.dt.strftime("%Y-%m-%d").to_numpy()
            selected.insert(0, "year", int(year))
            by_year[int(year)] = selected
            prepared.append(selected)

        if not prepared:
            raise ValueError("frames must not be empty")

        pooled = pd.concat(prepared, ignore_index=True)
        values = pooled.loc[:, self.features].to_numpy(dtype=float)
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale == 0] = 1.0
        standardized = (values - mean) / scale
        weight_vector = np.sqrt(
            np.array([self.weights[name] for name in self.features], dtype=float)
        )
        if self.metric == "mahalanobis":
            covariance = np.atleast_2d(np.cov(standardized, rowvar=False, ddof=0))
            covariance += np.eye(len(self.features)) * self.regularization
            inverse_covariance = np.linalg.pinv(covariance)
        else:
            inverse_covariance = None

        # Publish only now: everything above can raise without side effects.
        self._by_year = by_year
        self.rejected_rows = rejected
        self.mean_ = mean
        self.scale_ = scale
        self.weight_vector_ = weight_vector
        self.inverse_covariance_ = inverse_covariance
        self._fitted = True
        return self

    def query(
        self,
        *,
        reference_year: int,
        reference_date: str,
        candidate_years: Iterable[int] | None = None,
        candidate_dates: Iterable[str] | None = None,
        limit: int | None = None,
    ) -> list[AnalogMatch]:
        """Rank complete candidate rows against one reference day."""
        if not self._fitted:
            raise RuntimeError("fit() must be called before query()")
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        reference = self._row(reference_year, reference_date)
        ref_values = reference.loc[list(self.features)].to_numpy(dtype=float)

        years = (
            [int(year) for year in candidate_years]
            if candidate_years is not None
            else [year for year in self._by_year if year != int(reference_year)]
        )
        allowed_dates = (
            {pd.Timestamp(date).strftime("%Y-%m-%d") for date in candidate_dates}
            if candidate_dates is not None
            else None
        )

        matches = []
        for year in years:
            if year == int(reference_year):
                continue
            if year not in self._by_year:
                raise ValueError(f"candidate year {year} was not fitted")
            candidates = self._by_year[year]
            if allowed_dates is not None:
                candidates = candidates[candidates["date"].isin(allowed_dates)]
            for _, row in candidates.iterrows():
                values = row.loc[list(self.features)].to_numpy(dtype=float)
                delta = (values - ref_values) / self.scale_
                weighted_delta = delta * self.weight_vector_
                if self.metric == "mahalanobis":
                    squared = float(
                        weighted_delta @ self.inverse_covariance_ @ weighted_delta
                    )
                    distance = float(np.sqrt(max(squared, 0.0)))
                else:
                    distance = float(np.linalg.norm(weighted_delta))
                matches.append(
                    AnalogMatch(
                        year=year,
                        date=str(row["date"]),
                        distance=distance,
                        feature_values={
                            name: float(value)
                            for name, value in zip(self.features, values)
                        },
                        normalized_deltas={
                            name: float(value)
                            for name, value in zip(self.features, delta)
                        },
                    )
                )

        matches.sort(key=lambda match: (match.distance, match.date, match.year))
        return matches[:limit] if limit is not None else matches

    def _row(self, year: int, date: str) -> pd.Series:
        if int(year) not in self._by_year:
            raise ValueError(f"reference year {year} was not fitted")
        normalized_date = pd.Timestamp(date).strftime("%Y-%m-%d")
        rows = self._by_year[int(year)]
        row = rows[rows["date"] == normalized_date]
        if row.empty:
            raise ValueError(
                f"reference date {normalized_date} has no complete feature row"
            )
        return row.iloc[0]
