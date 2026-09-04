"""
Forecast-confidence construction for the FC-HMARL framework.

This module implements the confidence formulation described in
the manuscript.

The forecasting errors are grouped into:

    Renewable/load uncertainty:
        PV forecast error
        Load forecast error

    EV/market uncertainty:
        EV charging forecast error
        Electricity-price forecast error

The manuscript formulation is reconstructed as:

    epsilon_RL =
        sqrt(e_PV^2 + e_Load^2)

    omega_RL =
        exp(-epsilon_RL)

    epsilon_EM =
        sqrt(e_EV^2 + e_Price^2)

    Phi =
        omega_RL * exp(-epsilon_EM)

which is equivalent to:

    Phi =
        exp(-(epsilon_RL + epsilon_EM))

The confidence coefficient Phi therefore satisfies:

    0 < Phi <= 1

for finite forecasting errors.

Higher confidence:
    smaller forecasting error

Lower confidence:
    larger forecasting error

The predictive state is then confidence weighted:

    S_pred = Phi * Z_hat

where Z_hat represents future predictive operating information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class ForecastConfidenceConfig:
    """
    Configuration for forecast-confidence calculation.

    error_scale
    -----------
    Default = 1.0.

    The manuscript exponential formulation is recovered when
    error_scale = 1.0.

    A different value may be used only as an explicit
    implementation/calibration choice.
    """

    error_scale: float = 1.0

    minimum_confidence: float = 0.0

    maximum_confidence: float = 1.0

    def validate(self) -> None:
        """Validate confidence configuration."""

        if not isinstance(
            self.error_scale,
            (int, float),
        ):
            raise TypeError(
                "error_scale must be numeric."
            )

        if self.error_scale <= 0:
            raise ValueError(
                "error_scale must be positive."
            )

        if not isinstance(
            self.minimum_confidence,
            (int, float),
        ):
            raise TypeError(
                "minimum_confidence must be numeric."
            )

        if not isinstance(
            self.maximum_confidence,
            (int, float),
        ):
            raise TypeError(
                "maximum_confidence must be numeric."
            )

        if not (
            0.0
            <= self.minimum_confidence
            <= 1.0
        ):
            raise ValueError(
                "minimum_confidence must lie "
                "between 0 and 1."
            )

        if not (
            0.0
            <= self.maximum_confidence
            <= 1.0
        ):
            raise ValueError(
                "maximum_confidence must lie "
                "between 0 and 1."
            )

        if (
            self.minimum_confidence
            > self.maximum_confidence
        ):
            raise ValueError(
                "minimum_confidence cannot exceed "
                "maximum_confidence."
            )


# ============================================================
# BASIC VALIDATION
# ============================================================

def validate_error_array(
    error,
    name: str = "error",
) -> np.ndarray:
    """
    Validate one forecasting-error array.
    """

    error = np.asarray(
        error,
        dtype=np.float64,
    )

    if error.size == 0:
        raise ValueError(
            f"{name} cannot be empty."
        )

    if not np.isfinite(
        error
    ).all():
        raise ValueError(
            f"{name} contains NaN or Inf."
        )

    return error


def validate_same_shapes(
    *arrays: np.ndarray,
) -> None:
    """
    Verify that all supplied arrays have identical shapes.
    """

    if len(arrays) < 2:
        return

    reference_shape = arrays[
        0
    ].shape

    for array in arrays[
        1:
    ]:

        if array.shape != reference_shape:
            raise ValueError(
                "All forecasting-error arrays "
                "must have identical shapes."
            )


# ============================================================
# RENEWABLE + LOAD UNCERTAINTY
# ============================================================

def calculate_renewable_load_uncertainty(
    pv_error,
    load_error,
) -> np.ndarray:
    """
    Calculate renewable/load forecasting uncertainty.

    Manuscript formulation:

        epsilon_RL =
            sqrt(
                e_PV^2
                +
                e_Load^2
            )
    """

    pv_error = validate_error_array(
        pv_error,
        "pv_error",
    )

    load_error = validate_error_array(
        load_error,
        "load_error",
    )

    validate_same_shapes(
        pv_error,
        load_error,
    )

    uncertainty = np.sqrt(
        np.square(
            pv_error
        )
        +
        np.square(
            load_error
        )
    )

    return uncertainty


# ============================================================
# RENEWABLE + LOAD CONFIDENCE WEIGHT
# ============================================================

def calculate_renewable_load_confidence(
    renewable_load_uncertainty,
    error_scale: float = 1.0,
) -> np.ndarray:
    """
    Convert renewable/load uncertainty into confidence.

    Manuscript formulation:

        omega_RL =
            exp(-epsilon_RL)

    With error_scale = 1.0 this is the manuscript form.
    """

    uncertainty = validate_error_array(
        renewable_load_uncertainty,
        "renewable_load_uncertainty",
    )

    if np.any(
        uncertainty < 0
    ):
        raise ValueError(
            "renewable_load_uncertainty "
            "cannot be negative."
        )

    if error_scale <= 0:
        raise ValueError(
            "error_scale must be positive."
        )

    confidence = np.exp(
        -uncertainty
        / float(
            error_scale
        )
    )

    return confidence


# ============================================================
# EV + MARKET UNCERTAINTY
# ============================================================

def calculate_ev_market_uncertainty(
    ev_error,
    price_error,
) -> np.ndarray:
    """
    Calculate EV-demand and market-price uncertainty.

    Manuscript formulation:

        epsilon_EM =
            sqrt(
                e_EV^2
                +
                e_Price^2
            )
    """

    ev_error = validate_error_array(
        ev_error,
        "ev_error",
    )

    price_error = validate_error_array(
        price_error,
        "price_error",
    )

    validate_same_shapes(
        ev_error,
        price_error,
    )

    uncertainty = np.sqrt(
        np.square(
            ev_error
        )
        +
        np.square(
            price_error
        )
    )

    return uncertainty


# ============================================================
# GLOBAL CONFIDENCE
# ============================================================

def calculate_global_confidence(
    renewable_load_confidence,
    ev_market_uncertainty,
    error_scale: float = 1.0,
    minimum_confidence: float = 0.0,
    maximum_confidence: float = 1.0,
) -> np.ndarray:
    """
    Construct global forecast-confidence coefficient.

    Manuscript formulation:

        Phi =
            omega_RL
            *
            exp(-epsilon_EM)

    With error_scale = 1.0 this reproduces the manuscript
    exponential form.
    """

    renewable_load_confidence = (
        validate_error_array(
            renewable_load_confidence,
            "renewable_load_confidence",
        )
    )

    ev_market_uncertainty = (
        validate_error_array(
            ev_market_uncertainty,
            "ev_market_uncertainty",
        )
    )

    validate_same_shapes(
        renewable_load_confidence,
        ev_market_uncertainty,
    )

    if np.any(
        renewable_load_confidence < 0
    ):
        raise ValueError(
            "renewable_load_confidence cannot "
            "be negative."
        )

    if np.any(
        renewable_load_confidence > 1
    ):
        raise ValueError(
            "renewable_load_confidence cannot "
            "exceed 1."
        )

    if np.any(
        ev_market_uncertainty < 0
    ):
        raise ValueError(
            "ev_market_uncertainty cannot "
            "be negative."
        )

    if error_scale <= 0:
        raise ValueError(
            "error_scale must be positive."
        )

    confidence = (
        renewable_load_confidence
        *
        np.exp(
            -ev_market_uncertainty
            / float(
                error_scale
            )
        )
    )

    return np.clip(
        confidence,
        minimum_confidence,
        maximum_confidence,
    )


# ============================================================
# DIRECT GLOBAL FORMULATION
# ============================================================

def calculate_confidence_from_errors(
    pv_error,
    load_error,
    ev_error,
    price_error,
    error_scale: float = 1.0,
) -> np.ndarray:
    """
    Directly calculate Phi from all four forecasting errors.

    Equivalent formulation:

        Phi =
            exp(
                -(
                    epsilon_RL
                    +
                    epsilon_EM
                )
            )

    when error_scale = 1.
    """

    epsilon_rl = (
        calculate_renewable_load_uncertainty(
            pv_error,
            load_error,
        )
    )

    epsilon_em = (
        calculate_ev_market_uncertainty(
            ev_error,
            price_error,
        )
    )

    omega_rl = (
        calculate_renewable_load_confidence(
            epsilon_rl,
            error_scale=error_scale,
        )
    )

    return calculate_global_confidence(
        renewable_load_confidence=omega_rl,
        ev_market_uncertainty=epsilon_em,
        error_scale=error_scale,
    )


# ============================================================
# RESULT CONTAINER
# ============================================================

@dataclass
class ForecastConfidenceResult:
    """
    Stores all confidence-related outputs.
    """

    pv_error: np.ndarray

    load_error: np.ndarray

    ev_error: np.ndarray

    price_error: np.ndarray

    renewable_load_uncertainty: np.ndarray

    renewable_load_confidence: np.ndarray

    ev_market_uncertainty: np.ndarray

    global_confidence: np.ndarray

    target_names: Optional[
        list[str]
    ] = None

    def validate(self) -> None:
        """Validate confidence result."""

        arrays = [
            self.pv_error,
            self.load_error,
            self.ev_error,
            self.price_error,
            self.renewable_load_uncertainty,
            self.renewable_load_confidence,
            self.ev_market_uncertainty,
            self.global_confidence,
        ]

        arrays = [
            np.asarray(
                array,
                dtype=np.float64,
            )
            for array in arrays
        ]

        validate_same_shapes(
            *arrays
        )

        for array in arrays:

            if not np.isfinite(
                array
            ).all():
                raise ValueError(
                    "Confidence result contains "
                    "NaN or Inf."
                )

        if np.any(
            self.renewable_load_uncertainty
            < 0
        ):
            raise ValueError(
                "renewable_load_uncertainty "
                "cannot be negative."
            )

        if np.any(
            self.ev_market_uncertainty
            < 0
        ):
            raise ValueError(
                "ev_market_uncertainty "
                "cannot be negative."
            )

        if np.any(
            self.global_confidence < 0
        ):

            raise ValueError(
                "global_confidence cannot "
                "be negative."
            )

        if np.any(
            self.global_confidence > 1
        ):

            raise ValueError(
                "global_confidence cannot "
                "exceed 1."
            )

    @property
    def shape(
        self,
    ):

        return self.global_confidence.shape

    @property
    def mean_confidence(
        self,
    ) -> float:

        return float(
            np.mean(
                self.global_confidence
            )
        )

    @property
    def minimum_confidence(
        self,
    ) -> float:

        return float(
            np.min(
                self.global_confidence
            )
        )

    @property
    def maximum_confidence(
        self,
    ) -> float:

        return float(
            np.max(
                self.global_confidence
            )
        )

    def summary(
        self,
    ) -> Dict[str, object]:

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


# ============================================================
# MAIN CONFIDENCE ESTIMATOR
# ============================================================

class ForecastConfidenceEstimator:
    """
    Main interface for calculating manuscript-based
    forecast confidence.
    """

    def __init__(
        self,
        config: Optional[
            ForecastConfidenceConfig
        ] = None,
    ) -> None:

        if config is None:

            config = (
                ForecastConfidenceConfig()
            )

        config.validate()

        self.config = config

    def calculate(
        self,
        pv_error,
        load_error,
        ev_error,
        price_error,
    ) -> ForecastConfidenceResult:
        """
        Calculate complete forecast-confidence information.
        """

        pv_error = validate_error_array(
            pv_error,
            "pv_error",
        )

        load_error = validate_error_array(
            load_error,
            "load_error",
        )

        ev_error = validate_error_array(
            ev_error,
            "ev_error",
        )

        price_error = validate_error_array(
            price_error,
            "price_error",
        )

        validate_same_shapes(
            pv_error,
            load_error,
            ev_error,
            price_error,
        )

        epsilon_rl = (
            calculate_renewable_load_uncertainty(
                pv_error,
                load_error,
            )
        )

        omega_rl = (
            calculate_renewable_load_confidence(
                epsilon_rl,
                error_scale=(
                    self.config.error_scale
                ),
            )
        )

        epsilon_em = (
            calculate_ev_market_uncertainty(
                ev_error,
                price_error,
            )
        )

        phi = calculate_global_confidence(
            renewable_load_confidence=omega_rl,
            ev_market_uncertainty=epsilon_em,
            error_scale=(
                self.config.error_scale
            ),
            minimum_confidence=(
                self.config.minimum_confidence
            ),
            maximum_confidence=(
                self.config.maximum_confidence
            ),
        )

        result = ForecastConfidenceResult(
            pv_error=pv_error,

            load_error=load_error,

            ev_error=ev_error,

            price_error=price_error,

            renewable_load_uncertainty=(
                epsilon_rl
            ),

            renewable_load_confidence=(
                omega_rl
            ),

            ev_market_uncertainty=(
                epsilon_em
            ),

            global_confidence=phi,

            target_names=[
                "PV",
                "Load",
                "EV",
                "Price",
            ],
        )

        result.validate()

        return result


# ============================================================
# CONFIDENCE FROM ACTUAL AND FORECAST ARRAYS
# ============================================================

def confidence_from_actual_forecast(
    actual,
    forecast,
    config: Optional[
        ForecastConfidenceConfig
    ] = None,
) -> ForecastConfidenceResult:
    """
    Calculate confidence directly from actual and forecast data.

    Required shape:

        (samples, horizon, 4)

    Feature order:

        0 = PV
        1 = Load
        2 = EV charging demand
        3 = Electricity price

    Forecast residual definition:

        error = actual - forecast
    """

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
            "(samples, horizon, 4)."
        )

    if forecast.ndim != 3:
        raise ValueError(
            "forecast must have shape "
            "(samples, horizon, 4)."
        )

    if actual.shape != forecast.shape:
        raise ValueError(
            "actual and forecast must have "
            "identical shapes."
        )

    if actual.shape[
        2
    ] != 4:
        raise ValueError(
            "The FC-HMARL confidence formulation "
            "requires exactly four ordered variables: "
            "PV, Load, EV, and Price."
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

    residual = (
        actual
        - forecast
    )

    estimator = (
        ForecastConfidenceEstimator(
            config
        )
    )

    return estimator.calculate(
        pv_error=residual[
            :,
            :,
            0
        ],

        load_error=residual[
            :,
            :,
            1
        ],

        ev_error=residual[
            :,
            :,
            2
        ],

        price_error=residual[
            :,
            :,
            3
        ],
    )


# ============================================================
# CONFIDENCE-AWARE PREDICTIVE STATE
# ============================================================

def build_confidence_aware_predictive_state(
    forecast,
    confidence,
) -> np.ndarray:
    """
    Build manuscript confidence-aware predictive state.

    Manuscript concept:

        S_pred(t+h)
            =
        Phi(t+h) * Z_hat(t+h)

    Parameters
    ----------
    forecast:
        Forecast information.

        Typical shape:

            (samples, horizon, features)

    confidence:
        Global confidence coefficient.

        Accepted shapes:

            (samples, horizon)

        or

            (samples, horizon, 1)

    Returns
    -------
    np.ndarray
        Confidence-weighted predictive information.
    """

    forecast = np.asarray(
        forecast,
        dtype=np.float64,
    )

    confidence = np.asarray(
        confidence,
        dtype=np.float64,
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

    if confidence.shape[
        2
    ] != 1:

        raise ValueError(
            "confidence third dimension "
            "must equal 1."
        )

    if (
        forecast.shape[
            0
        ]
        != confidence.shape[
            0
        ]
        or forecast.shape[
            1
        ]
        != confidence.shape[
            1
        ]
    ):

        raise ValueError(
            "forecast and confidence sample/horizon "
            "dimensions must match."
        )

    if not np.isfinite(
        forecast
    ).all():

        raise ValueError(
            "forecast contains NaN or Inf."
        )

    if not np.isfinite(
        confidence
    ).all():

        raise ValueError(
            "confidence contains NaN or Inf."
        )

    if np.any(
        confidence < 0
    ) or np.any(
        confidence > 1
    ):

        raise ValueError(
            "confidence must lie in [0, 1]."
        )

    return (
        forecast
        * confidence
    )