"""
Risk-penalty utilities for FC-HMARL.

The core confidence-aware penalty follows the manuscript relationship

    C_risk = rho * (1 - Phi)

where rho >= 0 is the risk-aversion coefficient and Phi is the forecast
confidence in [0, 1].

Software aggregation helpers in this module are reconstructed evaluation
utilities; they do not alter the trained policy or checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable

import numpy as np


def _confidence_array(confidence: Iterable[float]) -> np.ndarray:
    phi = np.asarray(confidence, dtype=float).reshape(-1)

    if phi.size == 0:
        raise ValueError(
            "confidence must contain at least one value."
        )

    if not np.all(np.isfinite(phi)):
        raise ValueError(
            "confidence contains NaN or infinite values."
        )

    if np.any(phi < 0.0) or np.any(phi > 1.0):
        raise ValueError(
            "confidence values must lie in [0, 1]."
        )

    return phi


def confidence_risk_penalty(
    confidence: float,
    risk_aversion: float = 1.0,
) -> float:
    """
    Manuscript confidence-aware risk penalty for one decision:

        C_risk = rho * (1 - Phi)
    """
    rho = float(risk_aversion)
    phi = float(confidence)

    if rho < 0.0:
        raise ValueError(
            "risk_aversion must be >= 0."
        )

    if not 0.0 <= phi <= 1.0:
        raise ValueError(
            "confidence must lie in [0, 1]."
        )

    return float(rho * (1.0 - phi))


def confidence_risk_penalty_series(
    confidence: Iterable[float],
    risk_aversion: float = 1.0,
) -> np.ndarray:
    """Vector form of C_risk = rho(1-Phi)."""
    rho = float(risk_aversion)

    if rho < 0.0:
        raise ValueError(
            "risk_aversion must be >= 0."
        )

    phi = _confidence_array(confidence)

    return rho * (1.0 - phi)


def total_confidence_risk_cost(
    confidence: Iterable[float],
    risk_aversion: float = 1.0,
) -> float:
    """Sum confidence-aware risk penalties over a trajectory."""
    penalties = confidence_risk_penalty_series(
        confidence,
        risk_aversion,
    )
    return float(np.sum(penalties))


def mean_confidence_risk_penalty(
    confidence: Iterable[float],
    risk_aversion: float = 1.0,
) -> float:
    """Mean confidence-aware risk penalty over a trajectory."""
    penalties = confidence_risk_penalty_series(
        confidence,
        risk_aversion,
    )
    return float(np.mean(penalties))


def uncertainty_from_confidence(
    confidence: Iterable[float],
) -> np.ndarray:
    """Convert confidence Phi to uncertainty score 1-Phi."""
    phi = _confidence_array(confidence)
    return 1.0 - phi


@dataclass(frozen=True)
class RiskPenaltySummary:
    risk_aversion: float
    sample_count: int
    mean_confidence: float
    minimum_confidence: float
    maximum_confidence: float
    mean_uncertainty: float
    maximum_uncertainty: float
    mean_risk_penalty: float
    total_risk_penalty: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def summarize_confidence_risk(
    confidence: Iterable[float],
    risk_aversion: float = 1.0,
) -> RiskPenaltySummary:
    """
    Summarize confidence and manuscript risk penalties.
    """
    rho = float(risk_aversion)

    if rho < 0.0:
        raise ValueError(
            "risk_aversion must be >= 0."
        )

    phi = _confidence_array(confidence)
    uncertainty = 1.0 - phi
    penalties = rho * uncertainty

    return RiskPenaltySummary(
        risk_aversion=rho,
        sample_count=int(phi.size),
        mean_confidence=float(np.mean(phi)),
        minimum_confidence=float(np.min(phi)),
        maximum_confidence=float(np.max(phi)),
        mean_uncertainty=float(np.mean(uncertainty)),
        maximum_uncertainty=float(np.max(uncertainty)),
        mean_risk_penalty=float(np.mean(penalties)),
        total_risk_penalty=float(np.sum(penalties)),
    )


def risk_penalty_reduction_percent(
    baseline_confidence: Iterable[float],
    proposed_confidence: Iterable[float],
    risk_aversion: float = 1.0,
) -> float:
    """
    Percentage reduction in total confidence-risk penalty.

    This helper is appropriate only when comparing confidence sequences
    under the same risk-aversion coefficient and evaluation horizon.
    """
    baseline = total_confidence_risk_cost(
        baseline_confidence,
        risk_aversion,
    )
    proposed = total_confidence_risk_cost(
        proposed_confidence,
        risk_aversion,
    )

    if abs(baseline) < 1e-12:
        return 0.0

    return float(
        100.0 * (baseline - proposed) / abs(baseline)
    )
