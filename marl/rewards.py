"""
Reward construction for the FC-HMARL framework.

This module implements the local-agent, coordinator-agent,
confidence-risk, and hierarchical total reward formulations.

Manuscript-supported formulations
---------------------------------

Local microgrid reward:

    R_i(t) =
        -(
            C_i^grid(t)
            +
            C_i^deg(t)
            +
            C_i^viol(t)
        )

Operational violation cost:

    C_i^viol(t) =
        beta_soc * max(0, SOC_i(t) - SOC_i^max)^2
        +
        beta_grid * max(
            0,
            P_i^grid(t) - P_i^grid,max
        )^2

Coordinator reward:

    R_VPP(t) =
        Profit(t)
        -
        C^imb(t)
        -
        C^risk(t)

Forecast-confidence risk cost:

    C^risk(t) =
        rho * [1 - Phi(t)]

Hierarchical reward:

    R(t) =
        sum_i R_i(t)
        +
        R_VPP(t)

Notes
-----
The manuscript specifies the mathematical structures above,
but does not provide all numerical penalty coefficients in the
reward equations.

Therefore, coefficient defaults in this implementation are
explicit reconstruction choices and can be overridden through
configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class RewardConfig:
    """
    Configuration for FC-HMARL reward construction.

    beta_soc
        Penalty coefficient for upper SOC violation.

    beta_grid
        Penalty coefficient for grid-exchange violation.

    risk_aversion
        rho in:

            C_risk = rho * (1 - Phi)

    Numeric defaults are reconstruction choices unless explicitly
    replaced by manuscript-/experiment-specific parameters.
    """

    beta_soc: float = 1.0

    beta_grid: float = 1.0

    risk_aversion: float = 1.0

    def validate(self) -> None:
        """Validate reward configuration."""

        numeric_fields = {
            "beta_soc":
                self.beta_soc,

            "beta_grid":
                self.beta_grid,

            "risk_aversion":
                self.risk_aversion,
        }

        for name, value in (
            numeric_fields.items()
        ):

            if not isinstance(
                value,
                (int, float),
            ):

                raise TypeError(
                    f"{name} must be numeric."
                )

            if not np.isfinite(
                float(value)
            ):

                raise ValueError(
                    f"{name} must be finite."
                )

            if float(value) < 0:

                raise ValueError(
                    f"{name} cannot be negative."
                )


# ============================================================
# BASIC SCALAR VALIDATION
# ============================================================

def validate_finite_scalar(
    value,
    name: str,
) -> float:
    """
    Convert and validate one scalar reward quantity.
    """

    array = np.asarray(
        value,
        dtype=np.float64,
    )

    if array.ndim != 0:

        raise ValueError(
            f"{name} must be a scalar."
        )

    value = float(
        array
    )

    if not np.isfinite(
        value
    ):

        raise ValueError(
            f"{name} must be finite."
        )

    return value


def validate_nonnegative_scalar(
    value,
    name: str,
) -> float:
    """
    Validate a non-negative scalar cost.
    """

    value = validate_finite_scalar(
        value,
        name,
    )

    if value < 0:

        raise ValueError(
            f"{name} cannot be negative."
        )

    return value


# ============================================================
# SOC VIOLATION
# ============================================================

def calculate_soc_violation(
    soc,
    maximum_soc,
) -> float:
    """
    Calculate upper SOC constraint violation.

    Manuscript penalty component:

        max(
            0,
            SOC_i - SOC_i^max
        )
    """

    soc = validate_finite_scalar(
        soc,
        "soc",
    )

    maximum_soc = (
        validate_finite_scalar(
            maximum_soc,
            "maximum_soc",
        )
    )

    return max(
        0.0,
        soc - maximum_soc,
    )


# ============================================================
# GRID EXCHANGE VIOLATION
# ============================================================

def calculate_grid_violation(
    grid_exchange,
    maximum_grid_exchange,
) -> float:
    """
    Calculate upper grid-exchange violation.

    Manuscript penalty component:

        max(
            0,
            P_i^grid
            -
            P_i^grid,max
        )
    """

    grid_exchange = (
        validate_finite_scalar(
            grid_exchange,
            "grid_exchange",
        )
    )

    maximum_grid_exchange = (
        validate_finite_scalar(
            maximum_grid_exchange,
            "maximum_grid_exchange",
        )
    )

    return max(
        0.0,
        grid_exchange
        - maximum_grid_exchange,
    )


# ============================================================
# OPERATIONAL VIOLATION COST
# ============================================================

def calculate_violation_cost(
    soc,
    maximum_soc,
    grid_exchange,
    maximum_grid_exchange,
    beta_soc: float = 1.0,
    beta_grid: float = 1.0,
) -> float:
    """
    Calculate manuscript operational-violation penalty.

    C_i^viol =
        beta_soc
        * max(
            0,
            SOC_i - SOC_i^max
        )^2

        +

        beta_grid
        * max(
            0,
            P_i^grid - P_i^grid,max
        )^2
    """

    beta_soc = (
        validate_nonnegative_scalar(
            beta_soc,
            "beta_soc",
        )
    )

    beta_grid = (
        validate_nonnegative_scalar(
            beta_grid,
            "beta_grid",
        )
    )

    soc_violation = (
        calculate_soc_violation(
            soc,
            maximum_soc,
        )
    )

    grid_violation = (
        calculate_grid_violation(
            grid_exchange,
            maximum_grid_exchange,
        )
    )

    cost = (
        beta_soc
        * soc_violation ** 2

        +

        beta_grid
        * grid_violation ** 2
    )

    return float(
        cost
    )


# ============================================================
# LOCAL MICROGRID REWARD
# ============================================================

def calculate_local_reward(
    grid_cost,
    battery_degradation_cost,
    violation_cost,
) -> float:
    """
    Calculate one local-agent reward.

    Manuscript formulation:

        R_i =
            -(
                C_i^grid
                +
                C_i^deg
                +
                C_i^viol
            )
    """

    grid_cost = (
        validate_nonnegative_scalar(
            grid_cost,
            "grid_cost",
        )
    )

    battery_degradation_cost = (
        validate_nonnegative_scalar(
            battery_degradation_cost,
            "battery_degradation_cost",
        )
    )

    violation_cost = (
        validate_nonnegative_scalar(
            violation_cost,
            "violation_cost",
        )
    )

    reward = -(
        grid_cost
        + battery_degradation_cost
        + violation_cost
    )

    return float(
        reward
    )


# ============================================================
# COMPLETE LOCAL REWARD
# ============================================================

@dataclass
class LocalRewardResult:
    """
    Detailed result for one local microgrid reward.
    """

    grid_cost: float

    battery_degradation_cost: float

    violation_cost: float

    soc_violation: float

    grid_violation: float

    reward: float

    def summary(
        self,
    ) -> Dict[str, float]:

        return {
            "grid_cost":
                self.grid_cost,

            "battery_degradation_cost":
                self.battery_degradation_cost,

            "violation_cost":
                self.violation_cost,

            "soc_violation":
                self.soc_violation,

            "grid_violation":
                self.grid_violation,

            "reward":
                self.reward,
        }


def calculate_complete_local_reward(
    grid_cost,
    battery_degradation_cost,
    soc,
    maximum_soc,
    grid_exchange,
    maximum_grid_exchange,
    config: Optional[
        RewardConfig
    ] = None,
) -> LocalRewardResult:
    """
    Calculate the complete local reward from costs and
    physical operating limits.
    """

    if config is None:

        config = (
            RewardConfig()
        )

    config.validate()

    grid_cost = (
        validate_nonnegative_scalar(
            grid_cost,
            "grid_cost",
        )
    )

    battery_degradation_cost = (
        validate_nonnegative_scalar(
            battery_degradation_cost,
            "battery_degradation_cost",
        )
    )

    soc_violation = (
        calculate_soc_violation(
            soc,
            maximum_soc,
        )
    )

    grid_violation = (
        calculate_grid_violation(
            grid_exchange,
            maximum_grid_exchange,
        )
    )

    violation_cost = (
        calculate_violation_cost(
            soc=soc,
            maximum_soc=maximum_soc,
            grid_exchange=grid_exchange,
            maximum_grid_exchange=(
                maximum_grid_exchange
            ),
            beta_soc=config.beta_soc,
            beta_grid=config.beta_grid,
        )
    )

    reward = calculate_local_reward(
        grid_cost=grid_cost,
        battery_degradation_cost=(
            battery_degradation_cost
        ),
        violation_cost=(
            violation_cost
        ),
    )

    return LocalRewardResult(
        grid_cost=grid_cost,

        battery_degradation_cost=(
            battery_degradation_cost
        ),

        violation_cost=(
            violation_cost
        ),

        soc_violation=(
            soc_violation
        ),

        grid_violation=(
            grid_violation
        ),

        reward=reward,
    )


# ============================================================
# FORECAST-CONFIDENCE RISK COST
# ============================================================

def calculate_confidence_risk_cost(
    confidence,
    risk_aversion: float = 1.0,
) -> float:
    """
    Calculate manuscript confidence-based risk penalty.

        C_risk =
            rho * (1 - Phi)

    where:

        Phi = forecast-confidence coefficient
        rho = risk-aversion coefficient
    """

    confidence = (
        validate_finite_scalar(
            confidence,
            "confidence",
        )
    )

    risk_aversion = (
        validate_nonnegative_scalar(
            risk_aversion,
            "risk_aversion",
        )
    )

    if not 0.0 <= confidence <= 1.0:

        raise ValueError(
            "confidence must lie in [0, 1]."
        )

    risk_cost = (
        risk_aversion
        * (
            1.0
            - confidence
        )
    )

    return float(
        risk_cost
    )


# ============================================================
# COORDINATOR REWARD
# ============================================================

def calculate_coordinator_reward(
    profit,
    imbalance_cost,
    risk_cost,
) -> float:
    """
    Calculate upper-level VPP coordinator reward.

    Manuscript formulation:

        R_VPP =
            Profit
            -
            C_imb
            -
            C_risk
    """

    profit = validate_finite_scalar(
        profit,
        "profit",
    )

    imbalance_cost = (
        validate_nonnegative_scalar(
            imbalance_cost,
            "imbalance_cost",
        )
    )

    risk_cost = (
        validate_nonnegative_scalar(
            risk_cost,
            "risk_cost",
        )
    )

    reward = (
        profit
        - imbalance_cost
        - risk_cost
    )

    return float(
        reward
    )


# ============================================================
# COMPLETE COORDINATOR REWARD
# ============================================================

@dataclass
class CoordinatorRewardResult:
    """
    Detailed coordinator reward result.
    """

    profit: float

    imbalance_cost: float

    risk_cost: float

    confidence: float

    reward: float

    def summary(
        self,
    ) -> Dict[str, float]:

        return {
            "profit":
                self.profit,

            "imbalance_cost":
                self.imbalance_cost,

            "risk_cost":
                self.risk_cost,

            "confidence":
                self.confidence,

            "reward":
                self.reward,
        }


def calculate_complete_coordinator_reward(
    profit,
    imbalance_cost,
    confidence,
    config: Optional[
        RewardConfig
    ] = None,
) -> CoordinatorRewardResult:
    """
    Calculate complete upper-level coordinator reward.
    """

    if config is None:

        config = (
            RewardConfig()
        )

    config.validate()

    profit = validate_finite_scalar(
        profit,
        "profit",
    )

    imbalance_cost = (
        validate_nonnegative_scalar(
            imbalance_cost,
            "imbalance_cost",
        )
    )

    confidence = (
        validate_finite_scalar(
            confidence,
            "confidence",
        )
    )

    risk_cost = (
        calculate_confidence_risk_cost(
            confidence=confidence,
            risk_aversion=(
                config.risk_aversion
            ),
        )
    )

    reward = (
        calculate_coordinator_reward(
            profit=profit,
            imbalance_cost=(
                imbalance_cost
            ),
            risk_cost=risk_cost,
        )
    )

    return CoordinatorRewardResult(
        profit=profit,

        imbalance_cost=(
            imbalance_cost
        ),

        risk_cost=risk_cost,

        confidence=confidence,

        reward=reward,
    )


# ============================================================
# TOTAL HIERARCHICAL REWARD
# ============================================================

def calculate_hierarchical_reward(
    local_rewards: Sequence[
        float
    ],
    coordinator_reward,
) -> float:
    """
    Calculate the cooperative FC-HMARL reward.

    Manuscript formulation:

        R(t) =
            sum_i R_i(t)
            +
            R_VPP(t)
    """

    local_rewards = np.asarray(
        local_rewards,
        dtype=np.float64,
    )

    if local_rewards.ndim != 1:

        raise ValueError(
            "local_rewards must be one-dimensional."
        )

    if local_rewards.size == 0:

        raise ValueError(
            "local_rewards cannot be empty."
        )

    if not np.isfinite(
        local_rewards
    ).all():

        raise ValueError(
            "local_rewards contain NaN or Inf."
        )

    coordinator_reward = (
        validate_finite_scalar(
            coordinator_reward,
            "coordinator_reward",
        )
    )

    total_reward = (
        np.sum(
            local_rewards
        )
        + coordinator_reward
    )

    return float(
        total_reward
    )


# ============================================================
# HIERARCHICAL RESULT
# ============================================================

@dataclass
class HierarchicalRewardResult:
    """
    Complete reward information for one FC-HMARL step.
    """

    local_rewards: np.ndarray

    coordinator_reward: float

    total_reward: float

    def validate(
        self,
    ) -> None:

        if not isinstance(
            self.local_rewards,
            np.ndarray,
        ):

            raise TypeError(
                "local_rewards must be a NumPy array."
            )

        if self.local_rewards.ndim != 1:

            raise ValueError(
                "local_rewards must be one-dimensional."
            )

        if self.local_rewards.size == 0:

            raise ValueError(
                "local_rewards cannot be empty."
            )

        if not np.isfinite(
            self.local_rewards
        ).all():

            raise ValueError(
                "local_rewards contain NaN or Inf."
            )

        validate_finite_scalar(
            self.coordinator_reward,
            "coordinator_reward",
        )

        validate_finite_scalar(
            self.total_reward,
            "total_reward",
        )

    @property
    def number_of_local_agents(
        self,
    ) -> int:

        return int(
            self.local_rewards.size
        )

    @property
    def mean_local_reward(
        self,
    ) -> float:

        return float(
            np.mean(
                self.local_rewards
            )
        )

    def summary(
        self,
    ) -> Dict[str, float | int]:

        return {
            "number_of_local_agents":
                self.number_of_local_agents,

            "mean_local_reward":
                self.mean_local_reward,

            "coordinator_reward":
                self.coordinator_reward,

            "total_reward":
                self.total_reward,
        }


# ============================================================
# MAIN REWARD BUILDER
# ============================================================

class HierarchicalRewardBuilder:
    """
    Main interface for FC-HMARL reward construction.
    """

    def __init__(
        self,
        config: Optional[
            RewardConfig
        ] = None,
    ) -> None:

        if config is None:

            config = (
                RewardConfig()
            )

        config.validate()

        self.config = config

    def build(
        self,
        local_reward_inputs:
        Sequence[
            Dict[str, float]
        ],
        profit,
        imbalance_cost,
        confidence,
    ) -> HierarchicalRewardResult:
        """
        Build all local rewards, coordinator reward,
        and total hierarchical reward.

        Each local input dictionary must contain:

            grid_cost
            battery_degradation_cost
            soc
            maximum_soc
            grid_exchange
            maximum_grid_exchange
        """

        if len(
            local_reward_inputs
        ) == 0:

            raise ValueError(
                "local_reward_inputs cannot be empty."
            )

        required_keys = {
            "grid_cost",
            "battery_degradation_cost",
            "soc",
            "maximum_soc",
            "grid_exchange",
            "maximum_grid_exchange",
        }

        rewards = []

        for index, values in enumerate(
            local_reward_inputs
        ):

            missing = (
                required_keys
                - set(
                    values.keys()
                )
            )

            if missing:

                raise KeyError(
                    f"Local reward input {index} "
                    f"is missing keys: "
                    f"{sorted(missing)}"
                )

            result = (
                calculate_complete_local_reward(
                    grid_cost=values[
                        "grid_cost"
                    ],

                    battery_degradation_cost=values[
                        "battery_degradation_cost"
                    ],

                    soc=values[
                        "soc"
                    ],

                    maximum_soc=values[
                        "maximum_soc"
                    ],

                    grid_exchange=values[
                        "grid_exchange"
                    ],

                    maximum_grid_exchange=values[
                        "maximum_grid_exchange"
                    ],

                    config=self.config,
                )
            )

            rewards.append(
                result.reward
            )

        local_rewards = np.asarray(
            rewards,
            dtype=np.float64,
        )

        coordinator_result = (
            calculate_complete_coordinator_reward(
                profit=profit,
                imbalance_cost=imbalance_cost,
                confidence=confidence,
                config=self.config,
            )
        )

        total_reward = (
            calculate_hierarchical_reward(
                local_rewards=local_rewards,
                coordinator_reward=(
                    coordinator_result.reward
                ),
            )
        )

        result = HierarchicalRewardResult(
            local_rewards=local_rewards,

            coordinator_reward=(
                coordinator_result.reward
            ),

            total_reward=total_reward,
        )

        result.validate()

        return result