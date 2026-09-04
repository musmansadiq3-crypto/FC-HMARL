"""
Operational metrics for FC-HMARL evaluation.

This module contains generic, reproducible post-processing metrics for
24-hour VPP evaluation. It does not change the trained policy or the
environment. The metrics are computed only from recorded trajectories.

Reconstructed implementation note:
The manuscript does not uniquely specify software-level metric APIs.
These functions provide transparent evaluation utilities consistent
with the physical quantities already used in the FC-HMARL project.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


_EPS = 1e-12


def _as_1d(values: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def _nonnegative(arr: np.ndarray) -> np.ndarray:
    return np.maximum(arr, 0.0)


def peak_value(values: Iterable[float]) -> float:
    """Maximum value in a trajectory."""
    return float(np.max(_as_1d(values, "values")))


def minimum_value(values: Iterable[float]) -> float:
    """Minimum value in a trajectory."""
    return float(np.min(_as_1d(values, "values")))


def mean_value(values: Iterable[float]) -> float:
    """Arithmetic mean."""
    return float(np.mean(_as_1d(values, "values")))


def rms_value(values: Iterable[float]) -> float:
    """Root-mean-square value."""
    arr = _as_1d(values, "values")
    return float(np.sqrt(np.mean(arr ** 2)))


def standard_deviation(values: Iterable[float]) -> float:
    """Population standard deviation."""
    return float(np.std(_as_1d(values, "values"), ddof=0))


def peak_to_average_ratio(values: Iterable[float]) -> float:
    """
    Peak-to-average ratio using absolute magnitudes.

    Returns 0 when the mean absolute magnitude is numerically zero.
    """
    arr = np.abs(_as_1d(values, "values"))
    mean_abs = float(np.mean(arr))
    if mean_abs <= _EPS:
        return 0.0
    return float(np.max(arr) / mean_abs)


def energy_from_power(
    power_kw: Iterable[float],
    timestep_hours: float = 1.0,
) -> float:
    """Integrate a sampled power trajectory into energy [kWh]."""
    if timestep_hours <= 0:
        raise ValueError("timestep_hours must be > 0.")
    arr = _as_1d(power_kw, "power_kw")
    return float(np.sum(arr) * timestep_hours)


def imported_energy_kwh(
    grid_power_kw: Iterable[float],
    timestep_hours: float = 1.0,
) -> float:
    """
    Grid-import energy [kWh].

    Convention used by the reconstructed VPP environment:
    positive grid power = import, negative grid power = export.
    """
    arr = _as_1d(grid_power_kw, "grid_power_kw")
    return energy_from_power(_nonnegative(arr), timestep_hours)


def exported_energy_kwh(
    grid_power_kw: Iterable[float],
    timestep_hours: float = 1.0,
) -> float:
    """Grid-export energy [kWh], returned as a positive quantity."""
    arr = _as_1d(grid_power_kw, "grid_power_kw")
    return energy_from_power(_nonnegative(-arr), timestep_hours)


def net_grid_energy_kwh(
    grid_power_kw: Iterable[float],
    timestep_hours: float = 1.0,
) -> float:
    """
    Net grid energy [kWh].

    Positive = net import, negative = net export.
    """
    return energy_from_power(grid_power_kw, timestep_hours)


def renewable_energy_kwh(
    pv_power_kw: Iterable[float],
    timestep_hours: float = 1.0,
) -> float:
    """PV energy [kWh]."""
    return energy_from_power(_nonnegative(_as_1d(pv_power_kw, "pv_power_kw")),
                             timestep_hours)


def load_energy_kwh(
    load_power_kw: Iterable[float],
    timestep_hours: float = 1.0,
) -> float:
    """Load energy [kWh]."""
    return energy_from_power(_nonnegative(_as_1d(load_power_kw, "load_power_kw")),
                             timestep_hours)


def ev_energy_kwh(
    ev_power_kw: Iterable[float],
    timestep_hours: float = 1.0,
) -> float:
    """
    Aggregate EV charging energy [kWh].

    Positive EV power is interpreted as charging demand.
    """
    return energy_from_power(_nonnegative(_as_1d(ev_power_kw, "ev_power_kw")),
                             timestep_hours)


def bess_throughput_kwh(
    bess_power_kw: Iterable[float],
    timestep_hours: float = 1.0,
) -> float:
    """
    BESS energy throughput [kWh].

    Both charging and discharging contribute to throughput.
    """
    arr = _as_1d(bess_power_kw, "bess_power_kw")
    return energy_from_power(np.abs(arr), timestep_hours)


def sharing_energy_kwh(
    sharing_power_kw: Iterable[float],
    timestep_hours: float = 1.0,
) -> float:
    """
    Total scheduled sharing energy [kWh].

    Input should be a trajectory of already-aggregated nonnegative
    sharing power values. Pairwise double counting must be resolved
    before calling this function if the source matrix stores both
    directions separately.
    """
    arr = _nonnegative(_as_1d(sharing_power_kw, "sharing_power_kw"))
    return energy_from_power(arr, timestep_hours)


def renewable_fraction_of_demand(
    pv_power_kw: Iterable[float],
    load_power_kw: Iterable[float],
    ev_power_kw: Optional[Iterable[float]] = None,
    timestep_hours: float = 1.0,
) -> float:
    """
    PV-energy to total-demand-energy ratio.

    Total demand = load + positive EV charging demand.
    This is an accounting ratio, not a causal self-consumption metric.
    """
    pv = renewable_energy_kwh(pv_power_kw, timestep_hours)
    load = load_energy_kwh(load_power_kw, timestep_hours)

    ev = 0.0
    if ev_power_kw is not None:
        ev = ev_energy_kwh(ev_power_kw, timestep_hours)

    demand = load + ev
    if demand <= _EPS:
        return 0.0
    return float(pv / demand)


def count_violation_steps(
    violation_values: Iterable[float],
    tolerance: float = 1e-9,
) -> int:
    """
    Count time steps whose violation magnitude exceeds tolerance.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0.")
    arr = np.abs(_as_1d(violation_values, "violation_values"))
    return int(np.sum(arr > tolerance))


def total_violation_magnitude(
    violation_values: Iterable[float],
) -> float:
    """Sum of absolute violation magnitudes."""
    arr = np.abs(_as_1d(violation_values, "violation_values"))
    return float(np.sum(arr))


@dataclass(frozen=True)
class OperationalMetrics:
    """Compact operational summary for one evaluated trajectory."""

    horizon_steps: int
    timestep_hours: float

    peak_grid_import_kw: float
    peak_grid_export_kw: float
    mean_absolute_grid_exchange_kw: float
    grid_exchange_std_kw: float
    peak_to_average_grid_ratio: float

    imported_energy_kwh: float
    exported_energy_kwh: float
    net_grid_energy_kwh: float

    pv_energy_kwh: float
    load_energy_kwh: float
    ev_energy_kwh: float
    bess_throughput_kwh: float
    sharing_energy_kwh: float

    renewable_fraction_of_demand: float

    power_balance_violation_steps: int
    transformer_violation_steps: int
    total_power_balance_violation: float
    total_transformer_violation: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_operational_metrics(
    *,
    grid_power_kw: Iterable[float],
    pv_power_kw: Iterable[float],
    load_power_kw: Iterable[float],
    ev_power_kw: Iterable[float],
    bess_power_kw: Iterable[float],
    sharing_power_kw: Iterable[float],
    power_balance_violation: Iterable[float],
    transformer_violation: Iterable[float],
    timestep_hours: float = 1.0,
    violation_tolerance: float = 1e-9,
) -> OperationalMetrics:
    """
    Compute a complete operational summary for one VPP trajectory.

    All trajectory inputs must have equal length.
    """
    if timestep_hours <= 0:
        raise ValueError("timestep_hours must be > 0.")

    arrays = {
        "grid_power_kw": _as_1d(grid_power_kw, "grid_power_kw"),
        "pv_power_kw": _as_1d(pv_power_kw, "pv_power_kw"),
        "load_power_kw": _as_1d(load_power_kw, "load_power_kw"),
        "ev_power_kw": _as_1d(ev_power_kw, "ev_power_kw"),
        "bess_power_kw": _as_1d(bess_power_kw, "bess_power_kw"),
        "sharing_power_kw": _as_1d(sharing_power_kw, "sharing_power_kw"),
        "power_balance_violation":
            _as_1d(power_balance_violation, "power_balance_violation"),
        "transformer_violation":
            _as_1d(transformer_violation, "transformer_violation"),
    }

    lengths = {name: arr.size for name, arr in arrays.items()}
    unique_lengths = set(lengths.values())

    if len(unique_lengths) != 1:
        raise ValueError(
            "All operational trajectories must have equal length. "
            f"Received lengths: {lengths}"
        )

    grid = arrays["grid_power_kw"]
    pv = arrays["pv_power_kw"]
    load = arrays["load_power_kw"]
    ev = arrays["ev_power_kw"]
    bess = arrays["bess_power_kw"]
    sharing = arrays["sharing_power_kw"]
    pbv = arrays["power_balance_violation"]
    txv = arrays["transformer_violation"]

    import_power = np.maximum(grid, 0.0)
    export_power = np.maximum(-grid, 0.0)

    return OperationalMetrics(
        horizon_steps=int(grid.size),
        timestep_hours=float(timestep_hours),

        peak_grid_import_kw=float(np.max(import_power)),
        peak_grid_export_kw=float(np.max(export_power)),
        mean_absolute_grid_exchange_kw=float(np.mean(np.abs(grid))),
        grid_exchange_std_kw=float(np.std(grid, ddof=0)),
        peak_to_average_grid_ratio=peak_to_average_ratio(grid),

        imported_energy_kwh=energy_from_power(import_power, timestep_hours),
        exported_energy_kwh=energy_from_power(export_power, timestep_hours),
        net_grid_energy_kwh=energy_from_power(grid, timestep_hours),

        pv_energy_kwh=renewable_energy_kwh(pv, timestep_hours),
        load_energy_kwh=load_energy_kwh(load, timestep_hours),
        ev_energy_kwh=ev_energy_kwh(ev, timestep_hours),
        bess_throughput_kwh=bess_throughput_kwh(bess, timestep_hours),
        sharing_energy_kwh=sharing_energy_kwh(sharing, timestep_hours),

        renewable_fraction_of_demand=renewable_fraction_of_demand(
            pv,
            load,
            ev,
            timestep_hours,
        ),

        power_balance_violation_steps=count_violation_steps(
            pbv,
            violation_tolerance,
        ),
        transformer_violation_steps=count_violation_steps(
            txv,
            violation_tolerance,
        ),
        total_power_balance_violation=total_violation_magnitude(pbv),
        total_transformer_violation=total_violation_magnitude(txv),
    )


def aggregate_metric_dicts(
    metrics: Sequence[Mapping[str, float]],
) -> Dict[str, float]:
    """
    Aggregate numeric metric dictionaries across evaluation episodes.

    Returns <metric>_mean and <metric>_std for every numeric key found.
    """
    if not metrics:
        raise ValueError("metrics must contain at least one episode.")

    numeric_keys = []

    for key in metrics[0].keys():
        values = [m.get(key) for m in metrics]
        if all(isinstance(v, (int, float, np.integer, np.floating))
               for v in values):
            numeric_keys.append(key)

    result: Dict[str, float] = {}

    for key in numeric_keys:
        values = np.asarray([m[key] for m in metrics], dtype=float)
        result[f"{key}_mean"] = float(np.mean(values))
        result[f"{key}_std"] = float(np.std(values, ddof=0))

    return result
