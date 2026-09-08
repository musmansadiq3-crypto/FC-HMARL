from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence
import numpy as np
from environment.vpp_env import VPPEnvironment
from marl.environment_adapter import (
    HierarchicalActionBundle,
)

from marl.state_builder import (
    HierarchicalStateBuilder,
    HierarchicalStateResult,
)

from marl.rewards import (
    HierarchicalRewardBuilder,
    HierarchicalRewardResult,
)

from marl.training_loop import (
    TrainingObservation,
    EnvironmentStepResult,
)


# ============================================================
# EXOGENOUS VPP INPUT
# ============================================================

@dataclass
class VPPExogenousInput:
    """
    Exogenous physical quantities for one hourly VPP interval.
    """

    time: float

    loads_kw: Sequence[float]

    irradiances_w_m2: Sequence[float]

    buy_prices_usd_per_kwh: Sequence[float]

    sell_prices_usd_per_kwh: Sequence[float]

    predictive_state: np.ndarray

    confidence: float

    market_price: Optional[float] = None

    ev_requested_charging_powers_kw: Optional[
        Sequence
    ] = None

    def validate(
        self,
        number_of_microgrids: int,
    ) -> None:

        if number_of_microgrids <= 0:
            raise ValueError(
                "number_of_microgrids must be positive."
            )

        self.time = float(
            self.time
        )

        if not np.isfinite(
            self.time
        ):
            raise ValueError(
                "time must be finite."
            )

        if self.time < 0:
            raise ValueError(
                "time cannot be negative."
            )

        for name, values in (
            (
                "loads_kw",
                self.loads_kw,
            ),
            (
                "irradiances_w_m2",
                self.irradiances_w_m2,
            ),
            (
                "buy_prices_usd_per_kwh",
                self.buy_prices_usd_per_kwh,
            ),
            (
                "sell_prices_usd_per_kwh",
                self.sell_prices_usd_per_kwh,
            ),
        ):

            array = np.asarray(
                values,
                dtype=np.float64,
            )

            if array.shape != (
                number_of_microgrids,
            ):
                raise ValueError(
                    f"{name} must contain exactly "
                    f"{number_of_microgrids} values."
                )

            if not np.isfinite(
                array
            ).all():
                raise ValueError(
                    f"{name} contains NaN or Inf."
                )

        predictive = np.asarray(
            self.predictive_state,
            dtype=np.float64,
        )

        if not np.isfinite(
            predictive
        ).all():
            raise ValueError(
                "predictive_state contains NaN or Inf."
            )

        self.confidence = float(
            self.confidence
        )

        if not np.isfinite(
            self.confidence
        ):
            raise ValueError(
                "confidence must be finite."
            )

        if not (
            0.0
            <= self.confidence
            <= 1.0
        ):
            raise ValueError(
                "confidence must lie in [0, 1]."
            )

        if self.market_price is None:

            buy_prices = np.asarray(
                self.buy_prices_usd_per_kwh,
                dtype=np.float64,
            )

            self.market_price = float(
                np.mean(
                    buy_prices
                )
            )

        else:

            self.market_price = float(
                self.market_price
            )

            if not np.isfinite(
                self.market_price
            ):
                raise ValueError(
                    "market_price must be finite."
                )


# ============================================================
# PHYSICAL ACTION COMMAND
# ============================================================

@dataclass
class VPPPhysicalAction:
    """
    Physical action passed directly to VPPEnvironment.step().
    """

    bess_actions_kw: Sequence[float]

    sharing_matrix_kw: np.ndarray

    reserve_actions_kw: Optional[
        Sequence[float]
    ] = None

    grid_power_actions_kw: Optional[
        Sequence[float]
    ] = None

    ev_requested_charging_powers_kw: Optional[
        Sequence
    ] = None

    def validate(
        self,
        number_of_microgrids: int,
    ) -> None:

        bess = np.asarray(
            self.bess_actions_kw,
            dtype=np.float64,
        )

        if bess.shape != (
            number_of_microgrids,
        ):
            raise ValueError(
                "bess_actions_kw must contain "
                "one action per microgrid."
            )

        if not np.isfinite(
            bess
        ).all():
            raise ValueError(
                "bess_actions_kw contains NaN or Inf."
            )

        sharing = np.asarray(
            self.sharing_matrix_kw,
            dtype=np.float64,
        )

        expected_shape = (
            number_of_microgrids,
            number_of_microgrids,
        )

        if sharing.shape != expected_shape:
            raise ValueError(
                "sharing_matrix_kw must have shape "
                f"{expected_shape}."
            )

        if not np.isfinite(
            sharing
        ).all():
            raise ValueError(
                "sharing_matrix_kw contains NaN or Inf."
            )

        if np.any(
            sharing < 0.0
        ):
            raise ValueError(
                "sharing_matrix_kw cannot contain "
                "negative directional transfers."
            )

        if self.reserve_actions_kw is not None:

            reserve = np.asarray(
                self.reserve_actions_kw,
                dtype=np.float64,
            )

            if reserve.shape != (
                number_of_microgrids,
            ):
                raise ValueError(
                    "reserve_actions_kw must contain "
                    "one value per microgrid."
                )

            if not np.isfinite(
                reserve
            ).all():
                raise ValueError(
                    "reserve_actions_kw contains NaN or Inf."
                )

            if np.any(
                reserve < 0.0
            ):
                raise ValueError(
                    "reserve actions cannot be negative."
                )

        if self.grid_power_actions_kw is not None:

            grid = np.asarray(
                self.grid_power_actions_kw,
                dtype=np.float64,
            )

            if grid.shape != (
                number_of_microgrids,
            ):
                raise ValueError(
                    "grid_power_actions_kw must contain "
                    "one value per microgrid."
                )

            if not np.isfinite(
                grid
            ).all():
                raise ValueError(
                    "grid_power_actions_kw contains NaN or Inf."
                )


# ============================================================
# BRIDGE CONFIGURATION
# ============================================================

@dataclass
class VPPTrainingBridgeConfig:
    """
    Configuration needed only for quantities not directly returned
    by VPPEnvironment.

    maximum_socs
        Maximum SOC used by the local reward-violation calculation.

    maximum_grid_exchanges_kw
        Per-MG grid-exchange limits used by the reward calculation.

    battery_degradation_cost_per_kwh
        Reconstruction parameter converting BESS throughput to
        degradation cost.

    imbalance_penalty_per_kw
        Reconstruction parameter converting physical power-balance
        violation into coordinator imbalance cost.

    timestep_hours
        Physical simulation interval. Manuscript operation is hourly.
    """

    maximum_socs: Sequence[float]

    maximum_grid_exchanges_kw: Sequence[float]

    battery_degradation_cost_per_kwh: float = 0.0

    imbalance_penalty_per_kw: float = 1.0

    timestep_hours: float = 1.0

    number_of_microgrids: int = 5

    def validate(
        self,
    ) -> None:

        if self.number_of_microgrids <= 0:
            raise ValueError(
                "number_of_microgrids must be positive."
            )

        maximum_socs = np.asarray(
            self.maximum_socs,
            dtype=np.float64,
        )

        if maximum_socs.shape != (
            self.number_of_microgrids,
        ):
            raise ValueError(
                "maximum_socs must contain one "
                "value per microgrid."
            )

        if not np.isfinite(
            maximum_socs
        ).all():
            raise ValueError(
                "maximum_socs contains NaN or Inf."
            )

        maximum_grid = np.asarray(
            self.maximum_grid_exchanges_kw,
            dtype=np.float64,
        )

        if maximum_grid.shape != (
            self.number_of_microgrids,
        ):
            raise ValueError(
                "maximum_grid_exchanges_kw must contain "
                "one value per microgrid."
            )

        if np.any(
            maximum_grid <= 0
        ):
            raise ValueError(
                "maximum_grid_exchanges_kw values "
                "must be positive."
            )

        if not np.isfinite(
            maximum_grid
        ).all():
            raise ValueError(
                "maximum_grid_exchanges_kw contains "
                "NaN or Inf."
            )

        self.battery_degradation_cost_per_kwh = float(
            self.battery_degradation_cost_per_kwh
        )

        self.imbalance_penalty_per_kw = float(
            self.imbalance_penalty_per_kw
        )

        self.timestep_hours = float(
            self.timestep_hours
        )

        if (
            self.battery_degradation_cost_per_kwh
            < 0.0
        ):
            raise ValueError(
                "battery_degradation_cost_per_kwh "
                "cannot be negative."
            )

        if self.imbalance_penalty_per_kw < 0.0:
            raise ValueError(
                "imbalance_penalty_per_kw "
                "cannot be negative."
            )

        if self.timestep_hours <= 0.0:
            raise ValueError(
                "timestep_hours must be positive."
            )


# ============================================================
# CALLBACK TYPES
# ============================================================

ExogenousProvider = Callable[
    [int, int],
    VPPExogenousInput,
]

ActionMapper = Callable[
    [
        HierarchicalActionBundle,
        VPPExogenousInput,
        VPPEnvironment,
    ],
    VPPPhysicalAction,
]


# ============================================================
# MAIN PRODUCTION BRIDGE
# ============================================================

class FCHMARLVPPTrainingBridge:
    """
    Connect the actual VPPEnvironment to the FC-HMARL training loop.
    """

    def __init__(
        self,
        environment: VPPEnvironment,
        state_builder: HierarchicalStateBuilder,
        reward_builder: HierarchicalRewardBuilder,
        exogenous_provider: ExogenousProvider,
        action_mapper: ActionMapper,
        config: VPPTrainingBridgeConfig,
    ) -> None:

        if not isinstance(
            environment,
            VPPEnvironment,
        ):
            raise TypeError(
                "environment must be a VPPEnvironment."
            )

        if not isinstance(
            state_builder,
            HierarchicalStateBuilder,
        ):
            raise TypeError(
                "state_builder must be a "
                "HierarchicalStateBuilder."
            )

        if not isinstance(
            reward_builder,
            HierarchicalRewardBuilder,
        ):
            raise TypeError(
                "reward_builder must be a "
                "HierarchicalRewardBuilder."
            )

        if not callable(
            exogenous_provider
        ):
            raise TypeError(
                "exogenous_provider must be callable."
            )

        if not callable(
            action_mapper
        ):
            raise TypeError(
                "action_mapper must be callable."
            )

        config.validate()

        if (
            environment.num_microgrids
            != config.number_of_microgrids
        ):
            raise ValueError(
                "Environment microgrid count does not "
                "match bridge configuration."
            )

        if (
            state_builder.config.number_of_microgrids
            != config.number_of_microgrids
        ):
            raise ValueError(
                "State-builder microgrid count does not "
                "match bridge configuration."
            )

        self.environment = environment

        self.state_builder = state_builder

        self.reward_builder = reward_builder

        self.exogenous_provider = (
            exogenous_provider
        )

        self.action_mapper = (
            action_mapper
        )

        self.config = config

        self.current_episode = 0

        self.last_physical_result: Dict[
            str,
            object
        ] = {}

        self.last_state_result: Optional[
            HierarchicalStateResult
        ] = None

        self.last_reward_result: Optional[
            HierarchicalRewardResult
        ] = None

    # ========================================================
    # LOCAL OBSERVATION CONVERSION
    # ========================================================

    def _microgrid_observations(
        self,
    ):

        states = (
            self.environment
            .get_local_states()
        )

        observations = []

        for state in states:

            observations.append(
                {
                    "soc":
                        float(
                            state[
                                "bess_soc"
                            ]
                        ),

                    "pv_power":
                        float(
                            state[
                                "pv_power_kw"
                            ]
                        ),

                    "load_power":
                        float(
                            state[
                                "load_kw"
                            ]
                        ),

                    "ev_power":
                        float(
                            state[
                                "ev_power_kw"
                            ]
                        ),

                    "grid_exchange":
                        float(
                            state[
                                "grid_power_kw"
                            ]
                        ),
                }
            )

        return observations

    # ========================================================
    # HIERARCHICAL STATE
    # ========================================================

    def _build_state(
        self,
        exogenous: VPPExogenousInput,
        sharing_matrix,
    ) -> HierarchicalStateResult:

        result = self.state_builder.build(
            microgrid_observations=(
                self._microgrid_observations()
            ),

            predictive_state=(
                exogenous.predictive_state
            ),

            market_price=(
                exogenous.market_price
            ),

            sharing_matrix=(
                sharing_matrix
            ),
        )

        self.last_state_result = result

        return result

    @staticmethod
    def _training_observation(
        state_result:
        HierarchicalStateResult,
    ) -> TrainingObservation:

        return TrainingObservation(
            local_states=[
                state.copy()
                for state
                in state_result.local_states
            ],

            coordinator_state=(
                state_result
                .coordinator_state
                .copy()
            ),
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        episode: int,
    ) -> TrainingObservation:
        """
        Training-loop reset callback.

        The physical VPP is reset first.

        Since no sharing has yet occurred at t=0, the initial
        coordinator sharing state is represented by a zero matrix.
        """

        self.current_episode = int(
            episode
        )

        self.environment.reset()

        self.last_physical_result = {}

        self.last_reward_result = None

        exogenous = self.exogenous_provider(
            self.current_episode,
            0,
        )

        if not isinstance(
            exogenous,
            VPPExogenousInput,
        ):
            raise TypeError(
                "exogenous_provider must return "
                "VPPExogenousInput."
            )

        exogenous.validate(
            self.config.number_of_microgrids
        )

        initial_sharing = np.zeros(
            (
                self.config.number_of_microgrids,
                self.config.number_of_microgrids,
            ),
            dtype=np.float64,
        )

        state_result = (
            self._build_state(
                exogenous=exogenous,
                sharing_matrix=(
                    initial_sharing
                ),
            )
        )

        return self._training_observation(
            state_result
        )

    # ========================================================
    # REWARD CONSTRUCTION
    # ========================================================

    def _build_reward(
        self,
        physical_result:
        Dict[str, object],
        confidence: float,
    ) -> HierarchicalRewardResult:

        local_results = (
            physical_result[
                "local_results"
            ]
        )

        local_inputs = []

        for index, result in enumerate(
            local_results
        ):

            market = result[
                "market"
            ]

            grid_purchase_cost = float(
                market[
                    "grid_purchase_cost_usd"
                ]
            )

            bess_power_kw = abs(
                float(
                    result[
                        "bess_power_kw"
                    ]
                )
            )

            degradation_cost = (
                bess_power_kw
                * self.config.timestep_hours
                * self.config
                .battery_degradation_cost_per_kwh
            )

            local_inputs.append(
                {
                    "grid_cost":
                        grid_purchase_cost,

                    "battery_degradation_cost":
                        degradation_cost,

                    "soc":
                        float(
                            self.environment
                            .microgrids[
                                index
                            ]
                            .bess
                            .soc
                        ),

                    "maximum_soc":
                        float(
                            self.config
                            .maximum_socs[
                                index
                            ]
                        ),
                   
                    "grid_exchange":
                        float(
                            abs(
                                float(
                                    result[
                                        "requested_grid_power_kw"
                                    ]
                                )
                                + float(
                                    result[
                                        "outgoing_sharing_kw"
                                    ]
                                )
                            )
                        ),

                    # Use the same active-power limit as the physical
                    # transformer/PCC safety layer. This is important when
                    # the reconstruction power factor is not 1.0.
                    "maximum_grid_exchange":
                        float(
                            result[
                                "transformer_active_power_limit_kw"
                            ]
                        ),
                }
            )

        totals = physical_result[
            "totals"
        ]

        constraints = physical_result[
            "constraints"
        ]

        profit = float(
            totals[
                "market_profit_usd"
            ]
        )

        imbalance_cost = (
            float(
                constraints[
                    "total_power_balance_violation_kw"
                ]
            )
            * self.config
            .imbalance_penalty_per_kw
        )

        reward_result = (
            self.reward_builder.build(
                local_reward_inputs=(
                    local_inputs
                ),

                profit=profit,

                imbalance_cost=(
                    imbalance_cost
                ),

                confidence=(
                    confidence
                ),
            )
        )

        self.last_reward_result = (
            reward_result
        )

        return reward_result

    # ========================================================
    # ONE TRAINING STEP
    # ========================================================

    def step(
        self,
        actions:
        HierarchicalActionBundle,
    ) -> EnvironmentStepResult:
        """
        Training-loop step callback.
        """

        if not isinstance(
            actions,
            HierarchicalActionBundle,
        ):
            raise TypeError(
                "actions must be a "
                "HierarchicalActionBundle."
            )

        physical_step_index = int(
            self.environment.current_step
        )

        exogenous = self.exogenous_provider(
            self.current_episode,
            physical_step_index,
        )

        if not isinstance(
            exogenous,
            VPPExogenousInput,
        ):
            raise TypeError(
                "exogenous_provider must return "
                "VPPExogenousInput."
            )

        exogenous.validate(
            self.config.number_of_microgrids
        )

        physical_action = (
            self.action_mapper(
                actions,
                exogenous,
                self.environment,
            )
        )

        if not isinstance(
            physical_action,
            VPPPhysicalAction,
        ):
            raise TypeError(
                "action_mapper must return "
                "VPPPhysicalAction."
            )

        physical_action.validate(
            self.config.number_of_microgrids
        )

        ev_requests = (
            physical_action
            .ev_requested_charging_powers_kw
        )

        if ev_requests is None:

            ev_requests = (
                exogenous
                .ev_requested_charging_powers_kw
            )

        physical_result = (
            self.environment.step(
                time=exogenous.time,

                loads_kw=(
                    exogenous.loads_kw
                ),

                irradiances_w_m2=(
                    exogenous
                    .irradiances_w_m2
                ),

                bess_actions_kw=(
                    physical_action
                    .bess_actions_kw
                ),

                sharing_matrix_kw=(
                    physical_action
                    .sharing_matrix_kw
                ),

                buy_prices_usd_per_kwh=(
                    exogenous
                    .buy_prices_usd_per_kwh
                ),

                sell_prices_usd_per_kwh=(
                    exogenous
                    .sell_prices_usd_per_kwh
                ),

                reserve_actions_kw=(
                    physical_action
                    .reserve_actions_kw
                ),

                ev_requested_charging_powers_kw=(
                    ev_requests
                ),

                grid_power_actions_kw=(
                    physical_action
                    .grid_power_actions_kw
                ),
            )
        )

        self.last_physical_result = (
            physical_result
        )

        reward_result = (
            self._build_reward(
                physical_result=(
                    physical_result
                ),
                confidence=(
                    exogenous.confidence
                ),
            )
        )

        done = bool(
            physical_result[
                "done"
            ]
        )

        # ----------------------------------------------------
        # NEXT OBSERVATION
        # ----------------------------------------------------

        if done:

            next_exogenous = (
                exogenous
            )

        else:

            next_step_index = int(
                self.environment
                .current_step
            )

            next_exogenous = (
                self.exogenous_provider(
                    self.current_episode,
                    next_step_index,
                )
            )

            if not isinstance(
                next_exogenous,
                VPPExogenousInput,
            ):
                raise TypeError(
                    "exogenous_provider must return "
                    "VPPExogenousInput."
                )

            next_exogenous.validate(
                self.config.number_of_microgrids
            )

        feasible_sharing_matrix = (
            physical_result[
                "sharing"
            ][
                "feasible_matrix_kw"
            ]
        )

        next_state_result = (
            self._build_state(
                exogenous=(
                    next_exogenous
                ),
                sharing_matrix=(
                    feasible_sharing_matrix
                ),
            )
        )

        next_observation = (
            self._training_observation(
                next_state_result
            )
        )

        info = {
            "physical_result":
                physical_result,

            "hierarchical_state":
                next_state_result,

            "hierarchical_reward":
                reward_result,

            "confidence":
                float(
                    exogenous.confidence
                ),

            "physical_step":
                int(
                    physical_result[
                        "step"
                    ]
                ),
        }

        return EnvironmentStepResult(
            next_observation=(
                next_observation
            ),

            local_rewards=(
                reward_result
                .local_rewards
                .copy()
            ),

            coordinator_reward=(
                float(
                    reward_result
                    .coordinator_reward
                )
            ),

            done=done,

            info=info,
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(
        self,
    ) -> Dict[str, object]:

        return {
            "environment":
                self.environment.name,

            "number_of_microgrids":
                self.config.number_of_microgrids,

            "episode_length":
                self.environment
                .episode_length_hours,

            "current_episode":
                self.current_episode,

            "current_step":
                self.environment
                .current_step,

            "has_physical_result":
                bool(
                    self.last_physical_result
                ),

            "has_state_result":
                (
                    self.last_state_result
                    is not None
                ),

            "has_reward_result":
                (
                    self.last_reward_result
                    is not None
                ),
        }
