"""
Constraint utilities for the FC-HMARL VPP environment.

This module implements the physical feasibility checks required by the
manuscript, including:

1. Microgrid power-balance validation
2. PCC / transformer exchange limits
3. Generic clipping utilities
4. Constraint-violation magnitudes for RL penalty functions

Manuscript basis
----------------
The microgrid power balance requires that generation, load, BESS,
EV charging, internal sharing, and grid exchange balance at every
scheduling instant.

The manuscript also imposes a transformer / network capacity limit
on the combined exchange through the PCC.

Important reconstruction note
-----------------------------
Transformer ratings are reported in kVA in Table 2, while the power
balance equations use active power in kW.

The recovered manuscript does not provide a power factor conversion.
Therefore, this implementation makes power factor an explicit input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class ConstraintParameters:
    """
    Parameters used by the physical constraint layer.

    Parameters
    ----------
    transformer_rating_kva:
        Transformer apparent-power rating [kVA].

    power_factor:
        Reconstruction parameter used to convert transformer rating
        to an active-power limit:

            P_max = S_rated * power_factor

        Must satisfy:
            0 < power_factor <= 1

    balance_tolerance_kw:
        Numerical tolerance used when checking power balance.
    """

    transformer_rating_kva: float
    power_factor: float = 1.0
    balance_tolerance_kw: float = 1e-6

    def validate(self) -> None:
        """
        Validate constraint parameters.
        """

        if self.transformer_rating_kva <= 0:
            raise ValueError(
                "transformer_rating_kva must be greater than zero."
            )

        if not 0 < self.power_factor <= 1:
            raise ValueError(
                "power_factor must satisfy 0 < power_factor <= 1."
            )

        if self.balance_tolerance_kw < 0:
            raise ValueError(
                "balance_tolerance_kw must be non-negative."
            )

    @property
    def transformer_active_power_limit_kw(self) -> float:
        """
        Convert transformer apparent-power rating to an active-power
        exchange limit.
        """

        return float(
            self.transformer_rating_kva
            * self.power_factor
        )


# ============================================================
# BASIC CLIPPING UTILITIES
# ============================================================

def clip_value(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """
    Clip a scalar into the interval [minimum, maximum].
    """

    value = float(value)
    minimum = float(minimum)
    maximum = float(maximum)

    if minimum > maximum:
        raise ValueError(
            "minimum cannot be greater than maximum."
        )

    return float(
        np.clip(
            value,
            minimum,
            maximum,
        )
    )


def positive_part(
    value: float,
) -> float:
    """
    Return max(value, 0).

    Useful for constraint-violation penalties.
    """

    return float(
        max(
            float(value),
            0.0,
        )
    )


# ============================================================
# POWER BALANCE
# ============================================================

def microgrid_power_balance_residual(
    net_local_power_kw: float,
    incoming_sharing_kw: float,
    outgoing_sharing_kw: float,
    grid_power_kw: float,
) -> float:
    """
    Evaluate microgrid power-balance residual.

    Manuscript structure:

        P_net
        + incoming sharing
        - outgoing sharing
        + P_grid
        = 0

    Therefore:

        residual =
            P_net
            + incoming_sharing
            - outgoing_sharing
            + grid_power

    A physically balanced microgrid has residual approximately zero.

    Parameters
    ----------
    net_local_power_kw:
        Local net power of the microgrid before sharing and grid
        interaction.

    incoming_sharing_kw:
        Total power received from neighboring microgrids.

    outgoing_sharing_kw:
        Total power exported to neighboring microgrids.

    grid_power_kw:
        Utility-grid interaction power.

    Returns
    -------
    float
        Power-balance residual [kW].
    """

    return float(
        float(net_local_power_kw)
        + float(incoming_sharing_kw)
        - float(outgoing_sharing_kw)
        + float(grid_power_kw)
    )


def is_power_balanced(
    net_local_power_kw: float,
    incoming_sharing_kw: float,
    outgoing_sharing_kw: float,
    grid_power_kw: float,
    tolerance_kw: float = 1e-6,
) -> bool:
    """
    Check whether a microgrid satisfies power balance.
    """

    if tolerance_kw < 0:
        raise ValueError(
            "tolerance_kw must be non-negative."
        )

    residual = microgrid_power_balance_residual(
        net_local_power_kw=net_local_power_kw,
        incoming_sharing_kw=incoming_sharing_kw,
        outgoing_sharing_kw=outgoing_sharing_kw,
        grid_power_kw=grid_power_kw,
    )

    return bool(
        abs(residual) <= tolerance_kw
    )


def power_balance_violation_kw(
    net_local_power_kw: float,
    incoming_sharing_kw: float,
    outgoing_sharing_kw: float,
    grid_power_kw: float,
) -> float:
    """
    Return absolute power-balance mismatch.

    This value can later be used in the RL operational-violation
    penalty.
    """

    residual = microgrid_power_balance_residual(
        net_local_power_kw=net_local_power_kw,
        incoming_sharing_kw=incoming_sharing_kw,
        outgoing_sharing_kw=outgoing_sharing_kw,
        grid_power_kw=grid_power_kw,
    )

    return float(
        abs(residual)
    )


# ============================================================
# GRID POWER REQUIRED FOR EXACT BALANCE
# ============================================================

def required_grid_power_for_balance(
    net_local_power_kw: float,
    incoming_sharing_kw: float,
    outgoing_sharing_kw: float,
) -> float:
    """
    Compute utility-grid power required to close the balance.

    Starting from:

        P_net
        + P_in
        - P_out
        + P_grid
        = 0

    Therefore:

        P_grid =
            -P_net
            -P_in
            +P_out
    """

    return float(
        -float(net_local_power_kw)
        - float(incoming_sharing_kw)
        + float(outgoing_sharing_kw)
    )


# ============================================================
# PCC / TRANSFORMER LIMIT
# ============================================================

def pcc_exchange_kw(
    grid_power_kw: float,
    outgoing_sharing_kw: float,
) -> float:
    """
    Calculate combined PCC-side exchange magnitude.

    Reconstruction interpretation of the manuscript network constraint:

        | P_grid + outgoing sharing |

    This is kept separate as a function so that the later microgrid
    environment can explicitly apply the same definition everywhere.
    """

    return float(
        float(grid_power_kw)
        + float(outgoing_sharing_kw)
    )


def is_transformer_limit_satisfied(
    grid_power_kw: float,
    outgoing_sharing_kw: float,
    active_power_limit_kw: float,
) -> bool:
    """
    Check whether combined PCC exchange stays within transformer limit.
    """

    if active_power_limit_kw < 0:
        raise ValueError(
            "active_power_limit_kw must be non-negative."
        )

    exchange = pcc_exchange_kw(
        grid_power_kw=grid_power_kw,
        outgoing_sharing_kw=outgoing_sharing_kw,
    )

    return bool(
        abs(exchange)
        <= float(active_power_limit_kw)
    )


def transformer_violation_kw(
    grid_power_kw: float,
    outgoing_sharing_kw: float,
    active_power_limit_kw: float,
) -> float:
    """
    Return amount by which PCC exchange exceeds transformer limit.
    """

    if active_power_limit_kw < 0:
        raise ValueError(
            "active_power_limit_kw must be non-negative."
        )

    exchange = abs(
        pcc_exchange_kw(
            grid_power_kw=grid_power_kw,
            outgoing_sharing_kw=outgoing_sharing_kw,
        )
    )

    violation = (
        exchange
        - float(active_power_limit_kw)
    )

    return positive_part(
        violation
    )


def clip_pcc_exchange(
    grid_power_kw: float,
    outgoing_sharing_kw: float,
    active_power_limit_kw: float,
) -> float:
    """
    Return a grid-power value that keeps the combined PCC exchange
    inside the allowed active-power limit.

    Constraint:

        -P_max
        <= P_grid + P_out
        <= P_max

    Solving for P_grid:

        -P_max - P_out
        <= P_grid
        <= P_max - P_out
    """

    if active_power_limit_kw < 0:
        raise ValueError(
            "active_power_limit_kw must be non-negative."
        )

    lower_bound = (
        -float(active_power_limit_kw)
        - float(outgoing_sharing_kw)
    )

    upper_bound = (
        float(active_power_limit_kw)
        - float(outgoing_sharing_kw)
    )

    return clip_value(
        value=grid_power_kw,
        minimum=lower_bound,
        maximum=upper_bound,
    )


# ============================================================
# FULL CONSTRAINT EVALUATION
# ============================================================

def evaluate_microgrid_constraints(
    net_local_power_kw: float,
    incoming_sharing_kw: float,
    outgoing_sharing_kw: float,
    grid_power_kw: float,
    parameters: ConstraintParameters,
) -> Dict[str, float | bool]:
    """
    Evaluate all currently reconstructed microgrid constraints.

    Returns
    -------
    dict
        Includes:
        - power-balance residual
        - power-balance feasibility
        - transformer active-power limit
        - PCC exchange
        - transformer feasibility
        - violation magnitudes
    """

    parameters.validate()

    balance_residual = (
        microgrid_power_balance_residual(
            net_local_power_kw=net_local_power_kw,
            incoming_sharing_kw=incoming_sharing_kw,
            outgoing_sharing_kw=outgoing_sharing_kw,
            grid_power_kw=grid_power_kw,
        )
    )

    balance_violation = abs(
        balance_residual
    )

    active_limit = (
        parameters
        .transformer_active_power_limit_kw
    )

    exchange = pcc_exchange_kw(
        grid_power_kw=grid_power_kw,
        outgoing_sharing_kw=outgoing_sharing_kw,
    )

    transformer_violation = (
        transformer_violation_kw(
            grid_power_kw=grid_power_kw,
            outgoing_sharing_kw=outgoing_sharing_kw,
            active_power_limit_kw=active_limit,
        )
    )

    return {
        "power_balance_residual_kw":
            balance_residual,

        "power_balance_violation_kw":
            balance_violation,

        "power_balance_satisfied":
            bool(
                balance_violation
                <= parameters.balance_tolerance_kw
            ),

        "transformer_active_power_limit_kw":
            active_limit,

        "pcc_exchange_kw":
            exchange,

        "transformer_violation_kw":
            transformer_violation,

        "transformer_limit_satisfied":
            bool(
                transformer_violation
                <= 0.0
            ),
    }