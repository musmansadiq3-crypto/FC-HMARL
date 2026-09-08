from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Sequence
import numpy as np
# ============================================================
# CONFIGURATION
# ============================================================
@dataclass
class ForecastUncertaintyConfig:
    """
    Configuration for forecast uncertainty estimation.
    """
    confidence_level: float = 0.95

    minimum_residual_samples: int = 10

    use_horizon_specific_quantiles: bool = True

    use_target_specific_quantiles: bool = True

    def validate(self) -> None:
        """Validate uncertainty configuration."""

        if not isinstance(
            self.confidence_level,
            (int, float),
        ):
            raise TypeError(
                "confidence_level must be numeric."
            )

        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError(
                "confidence_level must satisfy "
                "0 < confidence_level < 1."
            )

        if not isinstance(
            self.minimum_residual_samples,
            int,
        ):
            raise TypeError(
                "minimum_residual_samples must be an integer."
            )

        if self.minimum_residual_samples <= 0:
            raise ValueError(
                "minimum_residual_samples must be positive."
            )
# ============================================================
# ARRAY VALIDATION
# ============================================================
def validate_forecast_arrays(
    actual,
    forecast,
) -> tuple[np.ndarray, np.ndarray]:
 
    actual = np.asarray(
        actual,
        dtype=np.float64,
    )

    forecast = np.asarray(
        forecast,
        dtype=np.float64,
    )

    if actual.ndim != 3:
        raise ValueError(
            "actual must have shape "
            "(samples, horizon, targets)."
        )

    if forecast.ndim != 3:
        raise ValueError(
            "forecast must have shape "
            "(samples, horizon, targets)."
        )

    if actual.shape != forecast.shape:
        raise ValueError(
            "actual and forecast arrays must "
            "have identical shapes."
        )

    if actual.shape[0] == 0:
        raise ValueError(
            "Forecast arrays cannot be empty."
        )

    if not np.isfinite(
        actual
    ).all():
        raise ValueError(
            "actual contains NaN or Inf."
        )

    if not np.isfinite(
        forecast
    ).all():
        raise ValueError(
            "forecast contains NaN or Inf."
        )

    return actual, forecast
# ============================================================
# RESIDUALS
# ============================================================

def calculate_forecast_residuals(
    actual,
    forecast,
) -> np.ndarray:
    """
    Calculate forecast residuals.

    residual = actual - forecast
    """

    actual, forecast = (
        validate_forecast_arrays(
            actual,
            forecast,
        )
    )

    return actual - forecast
# ============================================================
# ERROR STATISTICS
# ============================================================

def calculate_residual_statistics(
    residuals: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Calculate descriptive residual statistics.

    Statistics are calculated over the sample dimension,
    preserving:

        forecast horizon
        target dimension
    """

    residuals = np.asarray(
        residuals,
        dtype=np.float64,
    )

    if residuals.ndim != 3:
        raise ValueError(
            "residuals must have shape "
            "(samples, horizon, targets)."
        )

    if residuals.shape[0] == 0:
        raise ValueError(
            "residuals cannot be empty."
        )

    if not np.isfinite(
        residuals
    ).all():
        raise ValueError(
            "residuals contain NaN or Inf."
        )

    return {
        "mean":
            np.mean(
                residuals,
                axis=0,
            ),

        "std":
            np.std(
                residuals,
                axis=0,
                ddof=0,
            ),

        "mae":
            np.mean(
                np.abs(
                    residuals
                ),
                axis=0,
            ),

        "rmse":
            np.sqrt(
                np.mean(
                    residuals ** 2,
                    axis=0,
                )
            ),
    }
# ============================================================
# QUANTILE LEVELS
# ============================================================

def confidence_to_quantiles(
    confidence_level: float,
) -> tuple[float, float]:
    """
    Convert confidence level to two-sided quantiles.

    Example
    -------
    confidence_level = 0.95

    returns:
        0.025, 0.975
    """

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must satisfy "
            "0 < confidence_level < 1."
        )

    alpha = 1.0 - confidence_level

    return (
        alpha / 2.0,
        1.0 - alpha / 2.0,
    )
# ============================================================
# RESIDUAL QUANTILE MODEL
# ============================================================

@dataclass
class ResidualQuantileModel:
    """
    Stores learned empirical residual quantiles.
    """
    lower_quantile: np.ndarray
    upper_quantile: np.ndarray
    residual_mean: np.ndarray
    residual_std: np.ndarray
    confidence_level: float
    forecast_horizon: int
    number_of_targets: int
    target_names: Optional[
        list[str]
    ] = None
    def validate(self) -> None:
        """Validate fitted uncertainty model."""

        expected_shape = (
            self.forecast_horizon,
            self.number_of_targets,
        )

        arrays = {
            "lower_quantile":
                self.lower_quantile,

            "upper_quantile":
                self.upper_quantile,

            "residual_mean":
                self.residual_mean,

            "residual_std":
                self.residual_std,
        }

        for name, array in arrays.items():

            array = np.asarray(
                array
            )

            if array.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape "
                    f"{expected_shape}."
                )

            if not np.isfinite(
                array
            ).all():
                raise ValueError(
                    f"{name} contains NaN or Inf."
                )

        if np.any(
            self.lower_quantile
            > self.upper_quantile
        ):
            raise ValueError(
                "lower_quantile cannot exceed "
                "upper_quantile."
            )

        if self.target_names is not None:

            if len(
                self.target_names
            ) != self.number_of_targets:
                raise ValueError(
                    "target_names length does not match "
                    "number_of_targets."
                )


# ============================================================
# UNCERTAINTY RESULT
# ============================================================

@dataclass
class ForecastUncertaintyResult:
    """
    Forecast uncertainty output.
    """

    point_forecast: np.ndarray

    lower_bound: np.ndarray
    upper_bound: np.ndarray

    interval_width: np.ndarray

    uncertainty_std: np.ndarray

    confidence_level: float

    target_names: Optional[
        list[str]
    ] = None

    def validate(self) -> None:

        arrays = {
            "point_forecast":
                self.point_forecast,

            "lower_bound":
                self.lower_bound,

            "upper_bound":
                self.upper_bound,

            "interval_width":
                self.interval_width,

            "uncertainty_std":
                self.uncertainty_std,
        }

        reference_shape = (
            self.point_forecast.shape
        )

        if len(
            reference_shape
        ) != 3:
            raise ValueError(
                "point_forecast must be 3-D."
            )

        for name, array in arrays.items():

            if not isinstance(
                array,
                np.ndarray,
            ):
                raise TypeError(
                    f"{name} must be a NumPy array."
                )

            if array.shape != reference_shape:
                raise ValueError(
                    f"{name} shape does not match "
                    "point_forecast."
                )

            if not np.isfinite(
                array
            ).all():
                raise ValueError(
                    f"{name} contains NaN or Inf."
                )

        if np.any(
            self.lower_bound
            > self.upper_bound
        ):
            raise ValueError(
                "lower_bound exceeds upper_bound."
            )

        if np.any(
            self.interval_width < 0
        ):
            raise ValueError(
                "interval_width cannot be negative."
            )

        if np.any(
            self.uncertainty_std < 0
        ):
            raise ValueError(
                "uncertainty_std cannot be negative."
            )

    @property
    def number_of_samples(
        self,
    ) -> int:

        return int(
            self.point_forecast.shape[
                0
            ]
        )

    @property
    def forecast_horizon(
        self,
    ) -> int:

        return int(
            self.point_forecast.shape[
                1
            ]
        )

    @property
    def number_of_targets(
        self,
    ) -> int:

        return int(
            self.point_forecast.shape[
                2
            ]
        )

    def summary(
        self,
    ) -> Dict[str, object]:

        return {
            "number_of_samples":
                self.number_of_samples,

            "forecast_horizon":
                self.forecast_horizon,

            "number_of_targets":
                self.number_of_targets,

            "confidence_level":
                self.confidence_level,

            "average_interval_width":
                float(
                    np.mean(
                        self.interval_width
                    )
                ),

            "average_uncertainty_std":
                float(
                    np.mean(
                        self.uncertainty_std
                    )
                ),

            "target_names":
                self.target_names,
        }
# ============================================================
# MAIN ESTIMATOR
# ============================================================

class ForecastUncertaintyEstimator:
    """
    Empirical residual-based uncertainty estimator.
    """

    def __init__(
        self,
        config: Optional[
            ForecastUncertaintyConfig
        ] = None,
    ) -> None:

        if config is None:
            config = (
                ForecastUncertaintyConfig()
            )

        config.validate()

        self.config = config

        self.model: Optional[
            ResidualQuantileModel
        ] = None

    # ========================================================
    # FIT
    # ========================================================

    def fit(
        self,
        actual,
        forecast,
        target_names: Optional[
            Sequence[str]
        ] = None,
    ) -> ResidualQuantileModel:
        """
        Fit residual uncertainty distribution.
        """

        residuals = (
            calculate_forecast_residuals(
                actual,
                forecast,
            )
        )

        number_of_samples = (
            residuals.shape[
                0
            ]
        )

        if (
            number_of_samples
            < self.config
            .minimum_residual_samples
        ):
            raise ValueError(
                "Insufficient residual samples. "
                f"Required at least "
                f"{self.config.minimum_residual_samples}, "
                f"received {number_of_samples}."
            )

        horizon = residuals.shape[
            1
        ]

        targets = residuals.shape[
            2
        ]

        lower_probability, upper_probability = (
            confidence_to_quantiles(
                self.config.confidence_level
            )
        )

        # ----------------------------------------------------
        # FULL HORIZON + TARGET SPECIFIC MODEL
        # ----------------------------------------------------

        if (
            self.config
            .use_horizon_specific_quantiles
            and self.config
            .use_target_specific_quantiles
        ):

            lower = np.quantile(
                residuals,
                lower_probability,
                axis=0,
            )

            upper = np.quantile(
                residuals,
                upper_probability,
                axis=0,
            )

            residual_mean = np.mean(
                residuals,
                axis=0,
            )

            residual_std = np.std(
                residuals,
                axis=0,
                ddof=0,
            )

        # ----------------------------------------------------
        # TARGET SPECIFIC ONLY
        # ----------------------------------------------------

        elif (
            not self.config
            .use_horizon_specific_quantiles
            and self.config
            .use_target_specific_quantiles
        ):

            flattened = residuals.reshape(
                -1,
                targets,
            )

            lower_target = np.quantile(
                flattened,
                lower_probability,
                axis=0,
            )

            upper_target = np.quantile(
                flattened,
                upper_probability,
                axis=0,
            )

            mean_target = np.mean(
                flattened,
                axis=0,
            )

            std_target = np.std(
                flattened,
                axis=0,
                ddof=0,
            )

            lower = np.tile(
                lower_target,
                (
                    horizon,
                    1,
                ),
            )

            upper = np.tile(
                upper_target,
                (
                    horizon,
                    1,
                ),
            )

            residual_mean = np.tile(
                mean_target,
                (
                    horizon,
                    1,
                ),
            )

            residual_std = np.tile(
                std_target,
                (
                    horizon,
                    1,
                ),
            )

        # ----------------------------------------------------
        # HORIZON SPECIFIC ONLY
        # ----------------------------------------------------

        elif (
            self.config
            .use_horizon_specific_quantiles
            and not self.config
            .use_target_specific_quantiles
        ):

            flattened = residuals.transpose(
                0,
                2,
                1,
            ).reshape(
                -1,
                horizon,
            )

            lower_horizon = np.quantile(
                flattened,
                lower_probability,
                axis=0,
            )

            upper_horizon = np.quantile(
                flattened,
                upper_probability,
                axis=0,
            )

            mean_horizon = np.mean(
                flattened,
                axis=0,
            )

            std_horizon = np.std(
                flattened,
                axis=0,
                ddof=0,
            )

            lower = np.repeat(
                lower_horizon[
                    :,
                    None
                ],
                targets,
                axis=1,
            )

            upper = np.repeat(
                upper_horizon[
                    :,
                    None
                ],
                targets,
                axis=1,
            )

            residual_mean = np.repeat(
                mean_horizon[
                    :,
                    None
                ],
                targets,
                axis=1,
            )

            residual_std = np.repeat(
                std_horizon[
                    :,
                    None
                ],
                targets,
                axis=1,
            )

        # ----------------------------------------------------
        # GLOBAL RESIDUAL MODEL
        # ----------------------------------------------------

        else:

            flattened = residuals.reshape(
                -1
            )

            lower_value = float(
                np.quantile(
                    flattened,
                    lower_probability,
                )
            )

            upper_value = float(
                np.quantile(
                    flattened,
                    upper_probability,
                )
            )

            mean_value = float(
                np.mean(
                    flattened
                )
            )

            std_value = float(
                np.std(
                    flattened,
                    ddof=0,
                )
            )

            shape = (
                horizon,
                targets,
            )

            lower = np.full(
                shape,
                lower_value,
            )

            upper = np.full(
                shape,
                upper_value,
            )

            residual_mean = np.full(
                shape,
                mean_value,
            )

            residual_std = np.full(
                shape,
                std_value,
            )

        names = None

        if target_names is not None:

            names = list(
                target_names
            )

            if len(
                names
            ) != targets:
                raise ValueError(
                    "target_names length does not "
                    "match target dimension."
                )

        self.model = ResidualQuantileModel(
            lower_quantile=np.asarray(
                lower,
                dtype=np.float64,
            ),

            upper_quantile=np.asarray(
                upper,
                dtype=np.float64,
            ),

            residual_mean=np.asarray(
                residual_mean,
                dtype=np.float64,
            ),

            residual_std=np.asarray(
                residual_std,
                dtype=np.float64,
            ),

            confidence_level=(
                self.config.confidence_level
            ),

            forecast_horizon=horizon,

            number_of_targets=targets,

            target_names=names,
        )

        self.model.validate()

        return self.model

    # ========================================================
    # PREDICT UNCERTAINTY
    # ========================================================

    def transform(
        self,
        point_forecast,
    ) -> ForecastUncertaintyResult:
        """
        Construct prediction intervals for new forecasts.
        """

        if self.model is None:

            raise RuntimeError(
                "Estimator must be fitted before "
                "calling transform()."
            )

        point_forecast = np.asarray(
            point_forecast,
            dtype=np.float64,
        )

        if point_forecast.ndim != 3:

            raise ValueError(
                "point_forecast must have shape "
                "(samples, horizon, targets)."
            )

        if not np.isfinite(
            point_forecast
        ).all():

            raise ValueError(
                "point_forecast contains NaN or Inf."
            )

        if (
            point_forecast.shape[
                1
            ]
            != self.model.forecast_horizon
        ):

            raise ValueError(
                "Forecast horizon does not match "
                "fitted uncertainty model."
            )

        if (
            point_forecast.shape[
                2
            ]
            != self.model.number_of_targets
        ):

            raise ValueError(
                "Target dimension does not match "
                "fitted uncertainty model."
            )

        lower_bound = (
            point_forecast
            + self.model.lower_quantile[
                None,
                :,
                :
            ]
        )

        upper_bound = (
            point_forecast
            + self.model.upper_quantile[
                None,
                :,
                :
            ]
        )

        interval_width = (
            upper_bound
            - lower_bound
        )

        uncertainty_std = np.broadcast_to(
            self.model.residual_std[
                None,
                :,
                :
            ],
            point_forecast.shape,
        ).copy()

        result = ForecastUncertaintyResult(
            point_forecast=(
                point_forecast
            ),

            lower_bound=lower_bound,

            upper_bound=upper_bound,

            interval_width=(
                interval_width
            ),

            uncertainty_std=(
                uncertainty_std
            ),

            confidence_level=(
                self.model.confidence_level
            ),

            target_names=(
                self.model.target_names
            ),
        )

        result.validate()

        return result

    # ========================================================
    # FIT + TRANSFORM
    # ========================================================

    def fit_transform(
        self,
        actual,
        forecast,
        target_names: Optional[
            Sequence[str]
        ] = None,
    ) -> ForecastUncertaintyResult:
        """
        Fit on historical residuals and construct intervals
        around the supplied forecasts.
        """

        self.fit(
            actual=actual,
            forecast=forecast,
            target_names=target_names,
        )

        return self.transform(
            forecast
        )


# ============================================================
# INTERVAL COVERAGE
# ============================================================

def calculate_interval_coverage(
    actual,
    lower_bound,
    upper_bound,
) -> float:
    """
    Calculate empirical prediction-interval coverage.

    Coverage = fraction of actual values lying inside:

        lower_bound <= actual <= upper_bound
    """

    actual = np.asarray(
        actual,
        dtype=np.float64,
    )

    lower_bound = np.asarray(
        lower_bound,
        dtype=np.float64,
    )

    upper_bound = np.asarray(
        upper_bound,
        dtype=np.float64,
    )

    if (
        actual.shape
        != lower_bound.shape
        or actual.shape
        != upper_bound.shape
    ):
        raise ValueError(
            "actual, lower_bound, and upper_bound "
            "must have identical shapes."
        )

    if actual.size == 0:
        raise ValueError(
            "Arrays cannot be empty."
        )

    if not (
        np.isfinite(actual).all()
        and np.isfinite(lower_bound).all()
        and np.isfinite(upper_bound).all()
    ):
        raise ValueError(
            "Coverage arrays contain NaN or Inf."
        )

    if np.any(
        lower_bound
        > upper_bound
    ):
        raise ValueError(
            "lower_bound cannot exceed upper_bound."
        )

    inside = (
        (actual >= lower_bound)
        & (actual <= upper_bound)
    )

    return float(
        np.mean(
            inside
        )
    )


# ============================================================
# NORMALIZED UNCERTAINTY
# ============================================================

def normalize_uncertainty(
    uncertainty,
    epsilon: float = 1e-12,
) -> np.ndarray:
  
    uncertainty = np.asarray(
        uncertainty,
        dtype=np.float64,
    )

    if uncertainty.size == 0:
        raise ValueError(
            "uncertainty cannot be empty."
        )

    if not np.isfinite(
        uncertainty
    ).all():
        raise ValueError(
            "uncertainty contains NaN or Inf."
        )

    if np.any(
        uncertainty < 0
    ):
        raise ValueError(
            "uncertainty cannot be negative."
        )

    minimum = float(
        np.min(
            uncertainty
        )
    )

    maximum = float(
        np.max(
            uncertainty
        )
    )

    scale = maximum - minimum

    if scale <= epsilon:

        return np.zeros_like(
            uncertainty,
            dtype=np.float64,
        )

    normalized = (
        uncertainty
        - minimum
    ) / scale

    return np.clip(
        normalized,
        0.0,
        1.0,
    )
