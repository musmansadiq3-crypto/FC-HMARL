from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np


def _finite_nonnegative(value, name: str) -> float:
    value = float(value)

    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    if value < 0.0:
        raise ValueError(f"{name} must be non-negative.")

    return value


def _confidence_array(
    confidence: Iterable[float],
) -> np.ndarray:
    phi = np.asarray(
        confidence,
        dtype=np.float64,
    ).reshape(-1)

    if phi.size == 0:
        raise ValueError(
            "confidence must contain at least one value."
        )

    if not np.isfinite(phi).all():
        raise ValueError(
            "confidence contains NaN or Inf."
        )

    if np.any(phi < 0.0) or np.any(phi > 1.0):
        raise ValueError(
            "confidence values must lie in [0, 1]."
        )

    return phi


def mean_forecast_confidence(
    confidence: Iterable[float] | float,
) -> float:
    phi = np.asarray(
        confidence,
        dtype=np.float64,
    )

    if phi.ndim == 0:
        value = float(phi)

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "confidence must lie in [0, 1]."
            )

        return value

    phi = _confidence_array(phi)

    return float(
        np.mean(phi)
    )


@dataclass
class OperationalExposureConfig:
    eta_market: float = 0.25
    eta_bess: float = 0.25
    eta_sharing: float = 0.25
    eta_constraint: float = 0.25

    def validate(self) -> None:
        values = np.asarray(
            [
                self.eta_market,
                self.eta_bess,
                self.eta_sharing,
                self.eta_constraint,
            ],
            dtype=np.float64,
        )

        if not np.isfinite(values).all():
            raise ValueError(
                "Exposure weights contain NaN or Inf."
            )

        if np.any(values < 0.0):
            raise ValueError(
                "Exposure weights cannot be negative."
            )

        if float(np.sum(values)) <= 0.0:
            raise ValueError(
                "Exposure weights must have positive total weight."
            )

    @property
    def normalized_weights(self) -> np.ndarray:
        self.validate()

        weights = np.asarray(
            [
                self.eta_market,
                self.eta_bess,
                self.eta_sharing,
                self.eta_constraint,
            ],
            dtype=np.float64,
        )

        return weights / np.sum(weights)


@dataclass
class OperationalExposureResult:
    market_exposure: float
    bess_exposure: float
    sharing_exposure: float
    constraint_exposure: float
    total_exposure: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def calculate_constraint_exposure(
    soc_values: Optional[
        Sequence[float]
    ] = None,
    minimum_soc: Optional[
        Sequence[float] | float
    ] = None,
    maximum_soc: Optional[
        Sequence[float] | float
    ] = None,
    grid_exchange: Optional[
        Sequence[float]
    ] = None,
    maximum_grid_exchange: Optional[
        Sequence[float] | float
    ] = None,
) -> float:
    components = []

    if soc_values is not None:
        soc = np.asarray(
            soc_values,
            dtype=np.float64,
        ).reshape(-1)

        if not np.isfinite(soc).all():
            raise ValueError(
                "soc_values contains NaN or Inf."
            )

        if minimum_soc is not None:
            soc_min = np.broadcast_to(
                np.asarray(
                    minimum_soc,
                    dtype=np.float64,
                ),
                soc.shape,
            )

            lower_violation = np.maximum(
                soc_min - soc,
                0.0,
            )

            denominator = np.maximum(
                np.abs(soc_min),
                1e-8,
            )

            components.extend(
                (
                    lower_violation
                    / denominator
                ).tolist()
            )

        if maximum_soc is not None:
            soc_max = np.broadcast_to(
                np.asarray(
                    maximum_soc,
                    dtype=np.float64,
                ),
                soc.shape,
            )

            upper_violation = np.maximum(
                soc - soc_max,
                0.0,
            )

            denominator = np.maximum(
                np.abs(soc_max),
                1e-8,
            )

            components.extend(
                (
                    upper_violation
                    / denominator
                ).tolist()
            )

    if grid_exchange is not None:
        grid = np.asarray(
            grid_exchange,
            dtype=np.float64,
        ).reshape(-1)

        if not np.isfinite(grid).all():
            raise ValueError(
                "grid_exchange contains NaN or Inf."
            )

        if maximum_grid_exchange is None:
            raise ValueError(
                "maximum_grid_exchange is required "
                "when grid_exchange is supplied."
            )

        grid_max = np.broadcast_to(
            np.asarray(
                maximum_grid_exchange,
                dtype=np.float64,
            ),
            grid.shape,
        )

        denominator = np.maximum(
            np.abs(grid_max),
            1e-8,
        )

        violation = np.maximum(
            np.abs(grid) - denominator,
            0.0,
        )

        components.extend(
            (
                violation
                / denominator
            ).tolist()
        )

    if len(components) == 0:
        return 0.0

    return float(
        np.mean(
            components
        )
    )


def calculate_operational_exposure(
    market_power: float,
    maximum_market_power: float,
    bess_power: Sequence[float] | float,
    maximum_bess_power: Sequence[float] | float,
    executed_sharing_power:
        Sequence[float] | np.ndarray | float,
    maximum_sharing_power: float,
    constraint_exposure: float = 0.0,
    config: Optional[
        OperationalExposureConfig
    ] = None,
) -> OperationalExposureResult:
    if config is None:
        config = OperationalExposureConfig()

    config.validate()

    market_power = float(
        market_power
    )

    maximum_market_power = (
        _finite_nonnegative(
            maximum_market_power,
            "maximum_market_power",
        )
    )

    if maximum_market_power <= 0.0:
        raise ValueError(
            "maximum_market_power must be positive."
        )

    market_exposure = (
        abs(market_power)
        / maximum_market_power
    )

    bess = np.asarray(
        bess_power,
        dtype=np.float64,
    ).reshape(-1)

    bess_max = np.broadcast_to(
        np.asarray(
            maximum_bess_power,
            dtype=np.float64,
        ),
        bess.shape,
    )

    if not np.isfinite(bess).all():
        raise ValueError(
            "bess_power contains NaN or Inf."
        )

    if (
        not np.isfinite(bess_max).all()
        or np.any(bess_max <= 0.0)
    ):
        raise ValueError(
            "maximum_bess_power must be positive."
        )

    bess_exposure = float(
        np.sum(
            np.abs(bess)
        )
        / np.sum(
            bess_max
        )
    )

    sharing = np.asarray(
        executed_sharing_power,
        dtype=np.float64,
    )

    if not np.isfinite(sharing).all():
        raise ValueError(
            "executed_sharing_power contains NaN or Inf."
        )

    maximum_sharing_power = (
        _finite_nonnegative(
            maximum_sharing_power,
            "maximum_sharing_power",
        )
    )

    if maximum_sharing_power <= 0.0:
        raise ValueError(
            "maximum_sharing_power must be positive."
        )

    if sharing.ndim == 2:
        sharing = sharing.copy()

        if sharing.shape[0] == sharing.shape[1]:
            np.fill_diagonal(
                sharing,
                0.0,
            )

    sharing_exposure = float(
        np.sum(
            np.abs(sharing)
        )
        / maximum_sharing_power
    )

    constraint_exposure = (
        _finite_nonnegative(
            constraint_exposure,
            "constraint_exposure",
        )
    )

    weights = config.normalized_weights

    total_exposure = float(
        weights[0]
        * market_exposure
        + weights[1]
        * bess_exposure
        + weights[2]
        * sharing_exposure
        + weights[3]
        * constraint_exposure
    )

    return OperationalExposureResult(
        market_exposure=float(
            market_exposure
        ),
        bess_exposure=float(
            bess_exposure
        ),
        sharing_exposure=float(
            sharing_exposure
        ),
        constraint_exposure=float(
            constraint_exposure
        ),
        total_exposure=float(
            total_exposure
        ),
    )


def confidence_risk_penalty(
    confidence: float | Iterable[float],
    risk_aversion: float = 1.0,
    exposure: float = 1.0,
) -> float:
    """
    Adaptive confidence-aware training penalty:

        C_risk(t)
        = rho * [1 - Phi_bar(t)] * Xi(t)
    """

    rho = _finite_nonnegative(
        risk_aversion,
        "risk_aversion",
    )

    exposure = _finite_nonnegative(
        exposure,
        "exposure",
    )

    phi_bar = mean_forecast_confidence(
        confidence
    )

    return float(
        rho
        * (
            1.0
            - phi_bar
        )
        * exposure
    )


def confidence_risk_penalty_series(
    confidence: Iterable[float],
    risk_aversion: float = 1.0,
    exposure: Optional[
        Iterable[float]
    ] = None,
) -> np.ndarray:
    rho = _finite_nonnegative(
        risk_aversion,
        "risk_aversion",
    )

    phi = _confidence_array(
        confidence
    )

    if exposure is None:
        xi = np.ones_like(
            phi
        )

    else:
        xi = np.asarray(
            exposure,
            dtype=np.float64,
        ).reshape(-1)

        if xi.shape != phi.shape:
            raise ValueError(
                "exposure and confidence must have "
                "identical shapes."
            )

        if not np.isfinite(xi).all():
            raise ValueError(
                "exposure contains NaN or Inf."
            )

        if np.any(xi < 0.0):
            raise ValueError(
                "exposure cannot be negative."
            )

    return (
        rho
        * (
            1.0
            - phi
        )
        * xi
    )


def total_confidence_risk_cost(
    confidence: Iterable[float],
    risk_aversion: float = 1.0,
    exposure: Optional[
        Iterable[float]
    ] = None,
) -> float:
    penalties = (
        confidence_risk_penalty_series(
            confidence=confidence,
            risk_aversion=risk_aversion,
            exposure=exposure,
        )
    )

    return float(
        np.sum(
            penalties
        )
    )


def mean_confidence_risk_penalty(
    confidence: Iterable[float],
    risk_aversion: float = 1.0,
    exposure: Optional[
        Iterable[float]
    ] = None,
) -> float:
    penalties = (
        confidence_risk_penalty_series(
            confidence=confidence,
            risk_aversion=risk_aversion,
            exposure=exposure,
        )
    )

    return float(
        np.mean(
            penalties
        )
    )


def uncertainty_from_confidence(
    confidence: Iterable[float],
) -> np.ndarray:
    phi = _confidence_array(
        confidence
    )

    return (
        1.0
        - phi
    )


@dataclass(frozen=True)
class RiskPenaltySummary:
    risk_aversion: float
    sample_count: int
    mean_confidence: float
    minimum_confidence: float
    maximum_confidence: float
    mean_uncertainty: float
    maximum_uncertainty: float
    mean_exposure: float
    mean_risk_penalty: float
    total_risk_penalty: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def summarize_confidence_risk(
    confidence: Iterable[float],
    risk_aversion: float = 1.0,
    exposure: Optional[
        Iterable[float]
    ] = None,
) -> RiskPenaltySummary:
    rho = _finite_nonnegative(
        risk_aversion,
        "risk_aversion",
    )

    phi = _confidence_array(
        confidence
    )

    if exposure is None:
        xi = np.ones_like(
            phi
        )

    else:
        xi = np.asarray(
            exposure,
            dtype=np.float64,
        ).reshape(-1)

        if xi.shape != phi.shape:
            raise ValueError(
                "exposure and confidence must have "
                "identical shapes."
            )

        if not np.isfinite(xi).all():
            raise ValueError(
                "exposure contains NaN or Inf."
            )

        if np.any(xi < 0.0):
            raise ValueError(
                "exposure cannot be negative."
            )

    uncertainty = (
        1.0
        - phi
    )

    penalties = (
        rho
        * uncertainty
        * xi
    )

    return RiskPenaltySummary(
        risk_aversion=rho,
        sample_count=int(
            phi.size
        ),
        mean_confidence=float(
            np.mean(phi)
        ),
        minimum_confidence=float(
            np.min(phi)
        ),
        maximum_confidence=float(
            np.max(phi)
        ),
        mean_uncertainty=float(
            np.mean(uncertainty)
        ),
        maximum_uncertainty=float(
            np.max(uncertainty)
        ),
        mean_exposure=float(
            np.mean(xi)
        ),
        mean_risk_penalty=float(
            np.mean(penalties)
        ),
        total_risk_penalty=float(
            np.sum(penalties)
        ),
    )


def risk_penalty_reduction_percent(
    baseline_confidence: Iterable[float],
    proposed_confidence: Iterable[float],
    risk_aversion: float = 1.0,
    baseline_exposure: Optional[
        Iterable[float]
    ] = None,
    proposed_exposure: Optional[
        Iterable[float]
    ] = None,
) -> float:
    baseline = (
        total_confidence_risk_cost(
            confidence=
                baseline_confidence,
            risk_aversion=
                risk_aversion,
            exposure=
                baseline_exposure,
        )
    )

    proposed = (
        total_confidence_risk_cost(
            confidence=
                proposed_confidence,
            risk_aversion=
                risk_aversion,
            exposure=
                proposed_exposure,
        )
    )

    if abs(
        baseline
    ) < 1e-12:
        return 0.0

    return float(
        100.0
        * (
            baseline
            - proposed
        )
        / abs(
            baseline
        )
    )