"""
Conditional Value-at-Risk (CVaR) utilities for FC-HMARL evaluation.

These routines are evaluation utilities reconstructed for the project.
They operate on empirical episode losses/costs. Larger values are assumed
to represent worse outcomes.

For a confidence level alpha, VaR_alpha is the empirical alpha-quantile and
CVaR_alpha is the mean of the upper-tail outcomes at/above that threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable

import numpy as np


def _loss_array(losses: Iterable[float]) -> np.ndarray:
    arr = np.asarray(losses, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("losses must contain at least one value.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("losses contains NaN or infinite values.")
    return arr


def validate_confidence_level(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must satisfy 0 < alpha < 1.")
    return alpha


def value_at_risk(
    losses: Iterable[float],
    alpha: float = 0.95,
) -> float:
    """Empirical VaR at confidence level alpha."""
    arr = _loss_array(losses)
    alpha = validate_confidence_level(alpha)
    return float(np.quantile(arr, alpha))


def conditional_value_at_risk(
    losses: Iterable[float],
    alpha: float = 0.95,
) -> float:
    """
    Empirical upper-tail CVaR.

    Larger loss = worse outcome. The implementation averages the worst
    ceil((1-alpha)*N) observations, which is stable for small evaluation
    samples and avoids an empty empirical tail.
    """
    arr = _loss_array(losses)
    alpha = validate_confidence_level(alpha)

    tail_count = max(
        1,
        int(np.ceil((1.0 - alpha) * arr.size)),
    )

    ordered = np.sort(arr)
    tail = ordered[-tail_count:]

    return float(np.mean(tail))


def lower_tail_cvar_for_returns(
    returns: Iterable[float],
    alpha: float = 0.95,
) -> float:
    """
    CVaR-style lower-tail statistic for rewards/returns.

    Lower return = worse outcome. This returns the mean of the worst
    ceil((1-alpha)*N) return observations.
    """
    arr = _loss_array(returns)
    alpha = validate_confidence_level(alpha)

    tail_count = max(
        1,
        int(np.ceil((1.0 - alpha) * arr.size)),
    )

    ordered = np.sort(arr)
    tail = ordered[:tail_count]

    return float(np.mean(tail))


@dataclass(frozen=True)
class CVaRSummary:
    alpha: float
    sample_count: int
    mean_loss: float
    std_loss: float
    var: float
    cvar: float
    worst_loss: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def summarize_losses(
    losses: Iterable[float],
    alpha: float = 0.95,
) -> CVaRSummary:
    """Return empirical mean/std/VaR/CVaR/worst-loss summary."""
    arr = _loss_array(losses)
    alpha = validate_confidence_level(alpha)

    return CVaRSummary(
        alpha=alpha,
        sample_count=int(arr.size),
        mean_loss=float(np.mean(arr)),
        std_loss=float(np.std(arr, ddof=0)),
        var=value_at_risk(arr, alpha),
        cvar=conditional_value_at_risk(arr, alpha),
        worst_loss=float(np.max(arr)),
    )


def cvar_improvement_percent(
    baseline_losses: Iterable[float],
    proposed_losses: Iterable[float],
    alpha: float = 0.95,
) -> float:
    """
    Percentage reduction in CVaR from baseline to proposed method.

    Positive = proposed method has lower/wetter tail loss.
    """
    baseline = conditional_value_at_risk(
        baseline_losses,
        alpha,
    )
    proposed = conditional_value_at_risk(
        proposed_losses,
        alpha,
    )

    if abs(baseline) < 1e-12:
        return 0.0

    return float(
        100.0 * (baseline - proposed) / abs(baseline)
    )
