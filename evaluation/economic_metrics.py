from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Mapping, Sequence
import numpy as np
def _as_1d(values: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr
def _same_length(**arrays: np.ndarray) -> None:
    lengths = {name: arr.size for name, arr in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            "All economic trajectories must have equal length. "
            f"Received lengths: {lengths}"
        )
def purchase_cost_usd(
    grid_power_kw: Iterable[float],
    buy_price_usd_per_kwh: Iterable[float],
    timestep_hours: float = 1.0,
) -> float:
    """Grid import purchase cost [USD]."""
    if timestep_hours <= 0:
        raise ValueError("timestep_hours must be > 0.")

    grid = _as_1d(grid_power_kw, "grid_power_kw")
    buy = _as_1d(
        buy_price_usd_per_kwh,
        "buy_price_usd_per_kwh",
    )

    _same_length(grid=grid, buy=buy)

    imported_kw = np.maximum(grid, 0.0)

    return float(
        np.sum(imported_kw * buy) * timestep_hours
    )
def sale_revenue_usd(
    grid_power_kw: Iterable[float],
    sell_price_usd_per_kwh: Iterable[float],
    timestep_hours: float = 1.0,
) -> float:
    """Grid export revenue [USD]."""
    if timestep_hours <= 0:
        raise ValueError("timestep_hours must be > 0.")

    grid = _as_1d(grid_power_kw, "grid_power_kw")
    sell = _as_1d(
        sell_price_usd_per_kwh,
        "sell_price_usd_per_kwh",
    )

    _same_length(grid=grid, sell=sell)

    exported_kw = np.maximum(-grid, 0.0)

    return float(
        np.sum(exported_kw * sell) * timestep_hours
    )


def reserve_revenue_usd(
    reserve_power_kw: Iterable[float],
    reserve_price_usd_per_kwh: Iterable[float],
    timestep_hours: float = 1.0,
) -> float:
    """
    Reserve-service revenue [USD].

    Reserve power is treated as nonnegative awarded/provided capacity.
    """
    if timestep_hours <= 0:
        raise ValueError("timestep_hours must be > 0.")

    reserve = np.maximum(
        _as_1d(reserve_power_kw, "reserve_power_kw"),
        0.0,
    )
    price = _as_1d(
        reserve_price_usd_per_kwh,
        "reserve_price_usd_per_kwh",
    )

    _same_length(reserve=reserve, price=price)

    return float(
        np.sum(reserve * price) * timestep_hours
    )


def bess_degradation_cost_usd(
    bess_power_kw: Iterable[float],
    degradation_cost_usd_per_kwh: float = 0.02,
    timestep_hours: float = 1.0,
) -> float:
    """
    BESS throughput degradation cost [USD].

    Default 0.02 USD/kWh follows the manuscript parameter table used
    elsewhere in this reconstruction.
    """
    if degradation_cost_usd_per_kwh < 0:
        raise ValueError(
            "degradation_cost_usd_per_kwh must be >= 0."
        )
    if timestep_hours <= 0:
        raise ValueError("timestep_hours must be > 0.")

    bess = np.abs(
        _as_1d(bess_power_kw, "bess_power_kw")
    )

    throughput_kwh = float(
        np.sum(bess) * timestep_hours
    )

    return float(
        throughput_kwh
        * degradation_cost_usd_per_kwh
    )


def imbalance_cost_usd(
    imbalance_kw: Iterable[float],
    penalty_usd_per_kw: float,
) -> float:
    """
    Imbalance penalty [USD].

    This generic implementation applies a linear penalty to absolute
    imbalance magnitude. The coefficient is an explicit experiment
    parameter because the manuscript does not uniquely specify its
    software value in the reconstructed implementation.
    """
    if penalty_usd_per_kw < 0:
        raise ValueError(
            "penalty_usd_per_kw must be >= 0."
        )

    imbalance = np.abs(
        _as_1d(imbalance_kw, "imbalance_kw")
    )

    return float(
        np.sum(imbalance) * penalty_usd_per_kw
    )


def risk_cost_usd(
    confidence: Iterable[float],
    risk_aversion: float,
) -> float:
    """
    Confidence-aware risk cost.

    Implements the manuscript relationship:

        C_risk = rho * (1 - Phi)

    summed over the supplied trajectory.
    """
    if risk_aversion < 0:
        raise ValueError("risk_aversion must be >= 0.")

    phi = _as_1d(confidence, "confidence")

    if np.any(phi < 0.0) or np.any(phi > 1.0):
        raise ValueError(
            "confidence values must lie in [0, 1]."
        )

    return float(
        np.sum(risk_aversion * (1.0 - phi))
    )


def net_market_profit_usd(
    *,
    purchase_cost: float,
    sale_revenue: float,
    reserve_revenue: float = 0.0,
) -> float:
    """
    Market profit before degradation, imbalance, and risk penalties.
    """
    return float(
        sale_revenue
        + reserve_revenue
        - purchase_cost
    )


def total_operating_cost_usd(
    *,
    purchase_cost: float,
    degradation_cost: float,
    imbalance_cost: float = 0.0,
    risk_cost: float = 0.0,
    sale_revenue: float = 0.0,
    reserve_revenue: float = 0.0,
) -> float:
    """
    Net operating cost [USD].

    Positive values represent net cost. Revenues reduce the cost.
    """
    return float(
        purchase_cost
        + degradation_cost
        + imbalance_cost
        + risk_cost
        - sale_revenue
        - reserve_revenue
    )


@dataclass(frozen=True)
class EconomicMetrics:
    """Economic summary for one evaluated trajectory."""

    purchase_cost_usd: float
    sale_revenue_usd: float
    reserve_revenue_usd: float
    degradation_cost_usd: float
    imbalance_cost_usd: float
    risk_cost_usd: float

    gross_market_revenue_usd: float
    market_profit_usd: float
    total_operating_cost_usd: float
    net_profit_usd: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_economic_metrics(
    *,
    grid_power_kw: Iterable[float],
    buy_price_usd_per_kwh: Iterable[float],
    sell_price_usd_per_kwh: Iterable[float],
    bess_power_kw: Iterable[float],
    reserve_power_kw: Iterable[float],
    reserve_price_usd_per_kwh: Iterable[float],
    imbalance_kw: Iterable[float],
    confidence: Iterable[float],
    degradation_cost_usd_per_kwh: float = 0.02,
    imbalance_penalty_usd_per_kw: float = 1.0,
    risk_aversion: float = 1.0,
    timestep_hours: float = 1.0,
) -> EconomicMetrics:
    """
    Compute economic metrics for one VPP evaluation trajectory.
    """

    purchase = purchase_cost_usd(
        grid_power_kw,
        buy_price_usd_per_kwh,
        timestep_hours,
    )

    sale = sale_revenue_usd(
        grid_power_kw,
        sell_price_usd_per_kwh,
        timestep_hours,
    )

    reserve = reserve_revenue_usd(
        reserve_power_kw,
        reserve_price_usd_per_kwh,
        timestep_hours,
    )

    degradation = bess_degradation_cost_usd(
        bess_power_kw,
        degradation_cost_usd_per_kwh,
        timestep_hours,
    )

    imbalance = imbalance_cost_usd(
        imbalance_kw,
        imbalance_penalty_usd_per_kw,
    )

    risk = risk_cost_usd(
        confidence,
        risk_aversion,
    )

    gross_revenue = float(
        sale + reserve
    )

    market_profit = net_market_profit_usd(
        purchase_cost=purchase,
        sale_revenue=sale,
        reserve_revenue=reserve,
    )

    total_cost = total_operating_cost_usd(
        purchase_cost=purchase,
        degradation_cost=degradation,
        imbalance_cost=imbalance,
        risk_cost=risk,
        sale_revenue=sale,
        reserve_revenue=reserve,
    )

    net_profit = float(-total_cost)

    return EconomicMetrics(
        purchase_cost_usd=purchase,
        sale_revenue_usd=sale,
        reserve_revenue_usd=reserve,
        degradation_cost_usd=degradation,
        imbalance_cost_usd=imbalance,
        risk_cost_usd=risk,
        gross_market_revenue_usd=gross_revenue,
        market_profit_usd=market_profit,
        total_operating_cost_usd=total_cost,
        net_profit_usd=net_profit,
    )

def percentage_improvement(
    baseline_value: float,
    proposed_value: float,
    *,
    lower_is_better: bool = True,
) -> float:
    baseline = float(baseline_value)
    proposed = float(proposed_value)

    if abs(baseline) < 1e-12:
        return 0.0

    if lower_is_better:
        return float(
            100.0 * (baseline - proposed) / abs(baseline)
        )

    return float(
        100.0 * (proposed - baseline) / abs(baseline)
    )


def aggregate_metric_dicts(
    metrics: Sequence[Mapping[str, float]],
) -> Dict[str, float]:
    """
    Aggregate numeric economic metric dictionaries across episodes.

    Produces <metric>_mean and <metric>_std.
    """
    if not metrics:
        raise ValueError(
            "metrics must contain at least one episode."
        )

    numeric_keys = []

    for key in metrics[0].keys():
        values = [m.get(key) for m in metrics]

        if all(
            isinstance(
                value,
                (
                    int,
                    float,
                    np.integer,
                    np.floating,
                ),
            )
            for value in values
        ):
            numeric_keys.append(key)

    result: Dict[str, float] = {}

    for key in numeric_keys:

        values = np.asarray(
            [m[key] for m in metrics],
            dtype=float,
        )

        result[f"{key}_mean"] = float(
            np.mean(values)
        )

        result[f"{key}_std"] = float(
            np.std(values, ddof=0)
        )

    return result
