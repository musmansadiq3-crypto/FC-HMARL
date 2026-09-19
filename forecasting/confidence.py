from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np


@dataclass
class ForecastConfidenceConfig:
    delta: float = 1e-6
    minimum_confidence: float = 0.0
    maximum_confidence: float = 1.0
    feature_weights: Sequence[float] = (
        0.25,
        0.25,
        0.25,
        0.25,
    )
    normalization_epsilon: float = 1e-8

    def validate(self) -> None:
        if self.delta <= 0:
            raise ValueError("delta must be positive.")

        if self.normalization_epsilon <= 0:
            raise ValueError(
                "normalization_epsilon must be positive."
            )

        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError(
                "minimum_confidence must lie between 0 and 1."
            )

        if not 0.0 <= self.maximum_confidence <= 1.0:
            raise ValueError(
                "maximum_confidence must lie between 0 and 1."
            )

        if self.minimum_confidence > self.maximum_confidence:
            raise ValueError(
                "minimum_confidence cannot exceed maximum_confidence."
            )

        weights = np.asarray(
            self.feature_weights,
            dtype=np.float64,
        )

        if weights.shape != (4,):
            raise ValueError(
                "feature_weights must contain exactly four values "
                "for PV, Load, EV, and Price."
            )

        if not np.isfinite(weights).all():
            raise ValueError(
                "feature_weights contains NaN or Inf."
            )

        if np.any(weights < 0):
            raise ValueError(
                "feature_weights cannot contain negative values."
            )

        if float(np.sum(weights)) <= 0:
            raise ValueError(
                "feature_weights must have a positive sum."
            )


def validate_array(
    array,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        array,
        dtype=np.float64,
    )

    if array.size == 0:
        raise ValueError(
            f"{name} cannot be empty."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} contains NaN or Inf."
        )

    return array


def validate_same_shapes(
    *arrays: np.ndarray,
) -> None:
    if len(arrays) < 2:
        return

    reference_shape = arrays[0].shape

    for array in arrays[1:]:
        if array.shape != reference_shape:
            raise ValueError(
                "All supplied arrays must have identical shapes."
            )


def calculate_relative_interval_uncertainty(
    forecast,
    lower,
    upper,
    delta: float = 1e-6,
) -> np.ndarray:
    forecast = validate_array(
        forecast,
        "forecast",
    )

    lower = validate_array(
        lower,
        "lower",
    )

    upper = validate_array(
        upper,
        "upper",
    )

    validate_same_shapes(
        forecast,
        lower,
        upper,
    )

    if delta <= 0:
        raise ValueError(
            "delta must be positive."
        )

    if np.any(upper < lower):
        raise ValueError(
            "upper prediction interval cannot be below lower interval."
        )

    interval_width = upper - lower

    denominator = (
        2.0
        * (
            np.abs(forecast)
            + float(delta)
        )
    )

    uncertainty = (
        interval_width
        / denominator
    )

    return uncertainty


def calculate_all_feature_uncertainties(
    forecast,
    lower,
    upper,
    delta: float = 1e-6,
) -> np.ndarray:
    forecast = validate_array(
        forecast,
        "forecast",
    )

    lower = validate_array(
        lower,
        "lower",
    )

    upper = validate_array(
        upper,
        "upper",
    )

    if forecast.ndim != 3:
        raise ValueError(
            "forecast must have shape "
            "(samples, horizon, 4)."
        )

    if lower.ndim != 3:
        raise ValueError(
            "lower must have shape "
            "(samples, horizon, 4)."
        )

    if upper.ndim != 3:
        raise ValueError(
            "upper must have shape "
            "(samples, horizon, 4)."
        )

    validate_same_shapes(
        forecast,
        lower,
        upper,
    )

    if forecast.shape[2] != 4:
        raise ValueError(
            "The confidence formulation requires exactly four "
            "ordered variables: PV, Load, EV, and Price."
        )

    uncertainties = np.empty_like(
        forecast,
        dtype=np.float64,
    )

    for feature_index in range(4):
        uncertainties[:, :, feature_index] = (
            calculate_relative_interval_uncertainty(
                forecast=forecast[:, :, feature_index],
                lower=lower[:, :, feature_index],
                upper=upper[:, :, feature_index],
                delta=delta,
            )
        )

    return uncertainties


@dataclass
class ConfidenceCalibration:
    location: np.ndarray
    scale: np.ndarray
    feature_weights: np.ndarray

    def validate(self) -> None:
        self.location = np.asarray(
            self.location,
            dtype=np.float64,
        )

        self.scale = np.asarray(
            self.scale,
            dtype=np.float64,
        )

        self.feature_weights = np.asarray(
            self.feature_weights,
            dtype=np.float64,
        )

        if self.location.ndim != 2:
            raise ValueError(
                "location must have shape "
                "(horizon, 4)."
            )

        if self.scale.ndim != 2:
            raise ValueError(
                "scale must have shape "
                "(horizon, 4)."
            )

        if self.location.shape != self.scale.shape:
            raise ValueError(
                "location and scale must have identical shapes."
            )

        if self.location.shape[1] != 4:
            raise ValueError(
                "Calibration requires exactly four forecast variables."
            )

        if self.feature_weights.shape != (4,):
            raise ValueError(
                "feature_weights must contain four values."
            )

        if not np.isfinite(self.location).all():
            raise ValueError(
                "location contains NaN or Inf."
            )

        if not np.isfinite(self.scale).all():
            raise ValueError(
                "scale contains NaN or Inf."
            )

        if np.any(self.scale <= 0):
            raise ValueError(
                "scale must be strictly positive."
            )

        if not np.isfinite(
            self.feature_weights
        ).all():
            raise ValueError(
                "feature_weights contains NaN or Inf."
            )

        if np.any(self.feature_weights < 0):
            raise ValueError(
                "feature_weights cannot contain negative values."
            )

        if float(np.sum(self.feature_weights)) <= 0:
            raise ValueError(
                "feature_weights must have positive total weight."
            )


def fit_confidence_calibration(
    validation_forecast,
    validation_lower,
    validation_upper,
    config: Optional[
        ForecastConfidenceConfig
    ] = None,
) -> ConfidenceCalibration:
    if config is None:
        config = ForecastConfidenceConfig()

    config.validate()

    validation_uncertainty = (
        calculate_all_feature_uncertainties(
            forecast=validation_forecast,
            lower=validation_lower,
            upper=validation_upper,
            delta=config.delta,
        )
    )

    location = np.median(
        validation_uncertainty,
        axis=0,
    )

    absolute_deviation = np.abs(
        validation_uncertainty
        - location[None, :, :]
    )

    scale = np.median(
        absolute_deviation,
        axis=0,
    )

    scale = np.maximum(
        scale,
        config.normalization_epsilon,
    )

    weights = np.asarray(
        config.feature_weights,
        dtype=np.float64,
    )

    weights = weights / np.sum(weights)

    calibration = ConfidenceCalibration(
        location=location,
        scale=scale,
        feature_weights=weights,
    )

    calibration.validate()

    return calibration


def normalize_uncertainties(
    uncertainties,
    calibration: ConfidenceCalibration,
) -> np.ndarray:
    uncertainties = validate_array(
        uncertainties,
        "uncertainties",
    )

    calibration.validate()

    if uncertainties.ndim != 3:
        raise ValueError(
            "uncertainties must have shape "
            "(samples, horizon, 4)."
        )

    if (
        uncertainties.shape[1:]
        != calibration.location.shape
    ):
        raise ValueError(
            "uncertainties horizon/features do not match calibration."
        )

    normalized = (
        uncertainties
        / (
            calibration.location[None, :, :]
            + calibration.scale[None, :, :]
        )
    )

    normalized = np.maximum(
        normalized,
        0.0,
    )

    return normalized


def calculate_aggregate_uncertainty(
    normalized_uncertainties,
    feature_weights,
) -> np.ndarray:
    normalized_uncertainties = validate_array(
        normalized_uncertainties,
        "normalized_uncertainties",
    )

    if normalized_uncertainties.ndim != 3:
        raise ValueError(
            "normalized_uncertainties must have shape "
            "(samples, horizon, 4)."
        )

    if normalized_uncertainties.shape[2] != 4:
        raise ValueError(
            "normalized_uncertainties must contain four features."
        )

    weights = np.asarray(
        feature_weights,
        dtype=np.float64,
    )

    if weights.shape != (4,):
        raise ValueError(
            "feature_weights must contain exactly four values."
        )

    if np.any(weights < 0):
        raise ValueError(
            "feature_weights cannot contain negative values."
        )

    if not np.isfinite(weights).all():
        raise ValueError(
            "feature_weights contains NaN or Inf."
        )

    weight_sum = float(
        np.sum(weights)
    )

    if weight_sum <= 0:
        raise ValueError(
            "feature_weights must have positive total weight."
        )

    weights = weights / weight_sum

    aggregate = np.sqrt(
        np.sum(
            weights[None, None, :]
            * np.square(
                normalized_uncertainties
            ),
            axis=2,
        )
    )

    return aggregate


def calculate_adaptive_confidence(
    aggregate_uncertainty,
    minimum_confidence: float = 0.0,
    maximum_confidence: float = 1.0,
) -> np.ndarray:
    aggregate_uncertainty = validate_array(
        aggregate_uncertainty,
        "aggregate_uncertainty",
    )

    if np.any(aggregate_uncertainty < 0):
        raise ValueError(
            "aggregate_uncertainty cannot be negative."
        )

    confidence = np.exp(
        -aggregate_uncertainty
    )

    confidence = np.clip(
        confidence,
        minimum_confidence,
        maximum_confidence,
    )

    return confidence


@dataclass
class ForecastConfidenceResult:
    feature_uncertainties: np.ndarray
    normalized_uncertainties: np.ndarray
    aggregate_uncertainty: np.ndarray
    global_confidence: np.ndarray
    target_names: Optional[
        list[str]
    ] = None

    def validate(self) -> None:
        self.feature_uncertainties = np.asarray(
            self.feature_uncertainties,
            dtype=np.float64,
        )

        self.normalized_uncertainties = np.asarray(
            self.normalized_uncertainties,
            dtype=np.float64,
        )

        self.aggregate_uncertainty = np.asarray(
            self.aggregate_uncertainty,
            dtype=np.float64,
        )

        self.global_confidence = np.asarray(
            self.global_confidence,
            dtype=np.float64,
        )

        if (
            self.feature_uncertainties.shape
            != self.normalized_uncertainties.shape
        ):
            raise ValueError(
                "Feature uncertainty arrays must have identical shapes."
            )

        if self.feature_uncertainties.ndim != 3:
            raise ValueError(
                "feature_uncertainties must have shape "
                "(samples, horizon, 4)."
            )

        expected_shape = (
            self.feature_uncertainties.shape[0],
            self.feature_uncertainties.shape[1],
        )

        if self.aggregate_uncertainty.shape != expected_shape:
            raise ValueError(
                "aggregate_uncertainty must have shape "
                "(samples, horizon)."
            )

        if self.global_confidence.shape != expected_shape:
            raise ValueError(
                "global_confidence must have shape "
                "(samples, horizon)."
            )

        arrays = [
            self.feature_uncertainties,
            self.normalized_uncertainties,
            self.aggregate_uncertainty,
            self.global_confidence,
        ]

        for array in arrays:
            if not np.isfinite(array).all():
                raise ValueError(
                    "Confidence result contains NaN or Inf."
                )

        if np.any(
            self.feature_uncertainties < 0
        ):
            raise ValueError(
                "feature_uncertainties cannot be negative."
            )

        if np.any(
            self.normalized_uncertainties < 0
        ):
            raise ValueError(
                "normalized_uncertainties cannot be negative."
            )

        if np.any(
            self.aggregate_uncertainty < 0
        ):
            raise ValueError(
                "aggregate_uncertainty cannot be negative."
            )

        if np.any(
            self.global_confidence < 0
        ):
            raise ValueError(
                "global_confidence cannot be negative."
            )

        if np.any(
            self.global_confidence > 1
        ):
            raise ValueError(
                "global_confidence cannot exceed 1."
            )

    @property
    def shape(self):
        return self.global_confidence.shape

    @property
    def mean_confidence(self) -> float:
        return float(
            np.mean(
                self.global_confidence
            )
        )

    @property
    def minimum_confidence(self) -> float:
        return float(
            np.min(
                self.global_confidence
            )
        )

    @property
    def maximum_confidence(self) -> float:
        return float(
            np.max(
                self.global_confidence
            )
        )

    def summary(self) -> Dict[str, object]:
        return {
            "shape":
                self.shape,
            "mean_confidence":
                self.mean_confidence,
            "minimum_confidence":
                self.minimum_confidence,
            "maximum_confidence":
                self.maximum_confidence,
            "target_names":
                self.target_names,
        }


class ForecastConfidenceEstimator:
    def __init__(
        self,
        calibration: ConfidenceCalibration,
        config: Optional[
            ForecastConfidenceConfig
        ] = None,
    ) -> None:
        if config is None:
            config = ForecastConfidenceConfig()

        config.validate()
        calibration.validate()

        self.config = config
        self.calibration = calibration

    def calculate(
        self,
        forecast,
        lower,
        upper,
    ) -> ForecastConfidenceResult:
        feature_uncertainties = (
            calculate_all_feature_uncertainties(
                forecast=forecast,
                lower=lower,
                upper=upper,
                delta=self.config.delta,
            )
        )

        normalized_uncertainties = (
            normalize_uncertainties(
                uncertainties=
                    feature_uncertainties,
                calibration=
                    self.calibration,
            )
        )

        aggregate_uncertainty = (
            calculate_aggregate_uncertainty(
                normalized_uncertainties=
                    normalized_uncertainties,
                feature_weights=
                    self.calibration.feature_weights,
            )
        )

        global_confidence = (
            calculate_adaptive_confidence(
                aggregate_uncertainty=
                    aggregate_uncertainty,
                minimum_confidence=
                    self.config.minimum_confidence,
                maximum_confidence=
                    self.config.maximum_confidence,
            )
        )

        result = ForecastConfidenceResult(
            feature_uncertainties=
                feature_uncertainties,
            normalized_uncertainties=
                normalized_uncertainties,
            aggregate_uncertainty=
                aggregate_uncertainty,
            global_confidence=
                global_confidence,
            target_names=[
                "PV",
                "Load",
                "EV",
                "Price",
            ],
        )

        result.validate()

        return result


def calculate_confidence_from_intervals(
    forecast,
    lower,
    upper,
    calibration: ConfidenceCalibration,
    config: Optional[
        ForecastConfidenceConfig
    ] = None,
) -> ForecastConfidenceResult:
    estimator = ForecastConfidenceEstimator(
        calibration=calibration,
        config=config,
    )

    return estimator.calculate(
        forecast=forecast,
        lower=lower,
        upper=upper,
    )


def build_confidence_aware_predictive_state(
    forecast,
    confidence,
) -> np.ndarray:
    forecast = validate_array(
        forecast,
        "forecast",
    )

    confidence = validate_array(
        confidence,
        "confidence",
    )

    if forecast.ndim != 3:
        raise ValueError(
            "forecast must have shape "
            "(samples, horizon, features)."
        )

    if confidence.ndim == 2:
        confidence = confidence[
            :,
            :,
            None
        ]

    elif confidence.ndim != 3:
        raise ValueError(
            "confidence must have shape "
            "(samples, horizon) or "
            "(samples, horizon, 1)."
        )

    if confidence.shape[2] != 1:
        raise ValueError(
            "confidence third dimension must equal 1."
        )

    if (
        forecast.shape[0] != confidence.shape[0]
        or forecast.shape[1] != confidence.shape[1]
    ):
        raise ValueError(
            "forecast and confidence sample/horizon dimensions "
            "must match."
        )

    if np.any(
        confidence < 0
    ) or np.any(
        confidence > 1
    ):
        raise ValueError(
            "confidence must lie in [0, 1]."
        )

    predictive_state = np.concatenate(
        [
            forecast,
            confidence,
        ],
        axis=2,
    )

    return predictive_state