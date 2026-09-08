from __future__ import annotations
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
)

import numpy as np

from environment.vpp_env import (
    VPPEnvironment,
)

from marl.state_builder import (
    HierarchicalStateBuilder,
)

from marl.rewards import (
    HierarchicalRewardBuilder,
)

from marl.action_mapper import (
    FCHMARLActionMapper,
)

from marl.training_loop import (
    EnvironmentStepResult,
    HierarchicalActionBundle,
    TrainingObservation,
)


# ============================================================
# EXOGENOUS INPUT
# ============================================================

@dataclass
class VPPExogenousInput:
    time: float

    loads_kw: Sequence[float]

    irradiances_w_m2: Sequence[float]

    buy_prices_usd_per_kwh: Sequence[float]

    sell_prices_usd_per_kwh: Sequence[float]

    predictive_state: np.ndarray

    confidence: float

    market_price: Optional[float] = None

    ev_requested_charging_powers_kw: Optional[
        Sequence[float]
    ] = None

    def validate(
        self,
        number_of_microgrids: int,
    ) -> None:
        """
        Validate one exogenous input record.
        """

        if not isinstance(
            number_of_microgrids,
            int,
        ):
            raise TypeError(
                "number_of_microgrids must be an integer."
            )

        if number_of_microgrids <= 0:
            raise ValueError(
                "number_of_microgrids must be positive."
            )

        scalar_time = float(
            self.time
        )

        if not np.isfinite(
            scalar_time
        ):
            raise ValueError(
                "time must be finite."
            )

        self._validate_vector(
            self.loads_kw,
            number_of_microgrids,
            "loads_kw",
        )

        self._validate_vector(
            self.irradiances_w_m2,
            number_of_microgrids,
            "irradiances_w_m2",
        )

        self._validate_vector(
            self.buy_prices_usd_per_kwh,
            number_of_microgrids,
            "buy_prices_usd_per_kwh",
        )

        self._validate_vector(
            self.sell_prices_usd_per_kwh,
            number_of_microgrids,
            "sell_prices_usd_per_kwh",
        )

        predictive_state = np.asarray(
            self.predictive_state,
            dtype=np.float64,
        )

        if predictive_state.ndim != 2:
            raise ValueError(
                "predictive_state must be two-dimensional."
            )

        if not np.isfinite(
            predictive_state
        ).all():
            raise ValueError(
                "predictive_state contains NaN or Inf."
            )

        confidence = float(
            self.confidence
        )

        if not np.isfinite(
            confidence
        ):
            raise ValueError(
                "confidence must be finite."
            )

        if not (
            0.0
            <= confidence
            <= 1.0
        ):
            raise ValueError(
                "confidence must lie in [0, 1]."
            )

        if self.market_price is not None:

            market_price = float(
                self.market_price
            )

            if not np.isfinite(
                market_price
            ):
                raise ValueError(
                    "market_price must be finite."
                )

        if (
            self.ev_requested_charging_powers_kw
            is not None
        ):
            self._validate_vector(
                self.ev_requested_charging_powers_kw,
                number_of_microgrids,
                (
                    "ev_requested_"
                    "charging_powers_kw"
                ),
            )

    @staticmethod
    def _validate_vector(
        values,
        expected_size: int,
        name: str,
    ) -> np.ndarray:
        """
        Validate a one-dimensional finite vector.
        """

        array = np.asarray(
            values,
            dtype=np.float64,
        )

        if array.ndim != 1:
            raise ValueError(
                f"{name} must be one-dimensional."
            )

        if array.size != expected_size:
            raise ValueError(
                f"{name} must contain "
                f"{expected_size} values."
            )

        if not np.isfinite(
            array
        ).all():
            raise ValueError(
                f"{name} contains NaN or Inf."
            )

        return array

    @property
    def resolved_market_price(
        self,
    ) -> float:
        """
        Scalar price inserted into coordinator state.

        Reconstruction convenience:
        use mean purchase price when a separate market price
        was not supplied.
        """

        if self.market_price is not None:
            return float(
                self.market_price
            )

        prices = np.asarray(
            self.buy_prices_usd_per_kwh,
            dtype=np.float64,
        )

        return float(
            np.mean(
                prices
            )
        )


# ============================================================
# PHYSICAL ACTION CONTAINER
# ============================================================

@dataclass
class VPPPhysicalAction:
    """
    Physical control commands sent to VPPEnvironment.
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
        Sequence[float]
    ] = None

    def validate(
        self,
        number_of_microgrids: int,
    ) -> None:
        """
        Validate mapped physical actions.
        """

        bess_actions = np.asarray(
            self.bess_actions_kw,
            dtype=np.float64,
        )

        if bess_actions.shape != (
            number_of_microgrids,
        ):
            raise ValueError(
                "bess_actions_kw has incorrect shape."
            )

        if not np.isfinite(
            bess_actions
        ).all():
            raise ValueError(
                "bess_actions_kw contains NaN or Inf."
            )

        sharing = np.asarray(
            self.sharing_matrix_kw,
            dtype=np.float64,
        )

        if sharing.shape != (
            number_of_microgrids,
            number_of_microgrids,
        ):
            raise ValueError(
                "sharing_matrix_kw has incorrect shape."
            )

        if not np.isfinite(
            sharing
        ).all():
            raise ValueError(
                "sharing_matrix_kw contains NaN or Inf."
            )

        optional_vectors = {
            "reserve_actions_kw":
                self.reserve_actions_kw,

            "grid_power_actions_kw":
                self.grid_power_actions_kw,

            "ev_requested_charging_powers_kw":
                self.ev_requested_charging_powers_kw,
        }

        for (
            name,
            values,
        ) in optional_vectors.items():

            if values is None:
                continue

            array = np.asarray(
                values,
                dtype=np.float64,
            )

            if array.shape != (
                number_of_microgrids,
            ):
                raise ValueError(
                    f"{name} has incorrect shape."
                )

            if not np.isfinite(
                array
            ).all():
                raise ValueError(
                    f"{name} contains NaN or Inf."
                )


# ============================================================
# BRIDGE CONFIGURATION
# ============================================================

@dataclass
class VPPTrainingBridgeConfig:
    """
    Configuration for the real FC-HMARL/VPP bridge.

    maximum_socs
        Maximum allowable SOC for each microgrid.

    maximum_grid_exchanges_kw
        Grid/transformer exchange limit used by the local
        violation-cost calculation.

    battery_degradation_cost_per_kwh
        Economic BESS cycling coefficient.

    imbalance_penalty_per_kw
        Reconstruction coefficient used to translate physical
        power imbalance into coordinator imbalance cost.

    timestep_hours
        Simulation interval.

    number_of_microgrids
        Number of local microgrids.
    """

    maximum_socs: Sequence[float]

    maximum_grid_exchanges_kw: Sequence[float]

    battery_degradation_cost_per_kwh: float = 0.02

    imbalance_penalty_per_kw: float = 1.0

    timestep_hours: float = 1.0

    number_of_microgrids: int = 5

    def validate(
        self,
    ) -> None:
        """
        Validate bridge configuration.
        """

        if not isinstance(
            self.number_of_microgrids,
            int,
        ):
            raise TypeError(
                "number_of_microgrids must be an integer."
            )

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
                "maximum_socs has incorrect shape."
            )

        if not np.isfinite(
            maximum_socs
        ).all():
            raise ValueError(
                "maximum_socs contains NaN or Inf."
            )

        if np.any(
            maximum_socs <= 0.0
        ):
            raise ValueError(
                "maximum_socs must be positive."
            )

        maximum_grid = np.asarray(
            self.maximum_grid_exchanges_kw,
            dtype=np.float64,
        )

        if maximum_grid.shape != (
            self.number_of_microgrids,
        ):
            raise ValueError(
                "maximum_grid_exchanges_kw "
                "has incorrect shape."
            )

        if not np.isfinite(
            maximum_grid
        ).all():
            raise ValueError(
                "maximum_grid_exchanges_kw "
                "contains NaN or Inf."
            )

        if np.any(
            maximum_grid <= 0.0
        ):
            raise ValueError(
                "maximum_grid_exchanges_kw "
                "must contain positive values."
            )

        if (
            not np.isfinite(
                self.battery_degradation_cost_per_kwh
            )
            or
            self.battery_degradation_cost_per_kwh
            < 0.0
        ):
            raise ValueError(
                "battery_degradation_cost_per_kwh "
                "must be finite and non-negative."
            )

        if (
            not np.isfinite(
                self.imbalance_penalty_per_kw
            )
            or
            self.imbalance_penalty_per_kw
            < 0.0
        ):
            raise ValueError(
                "imbalance_penalty_per_kw must "
                "be finite and non-negative."
            )

        if (
            not np.isfinite(
                self.timestep_hours
            )
            or
            self.timestep_hours <= 0.0
        ):
            raise ValueError(
                "timestep_hours must be positive and finite."
            )


# ============================================================
# EXOGENOUS PROVIDER TYPE
# ============================================================

ExogenousProvider = Callable[
    [
        int,
        int,
    ],
    VPPExogenousInput,
]


# ============================================================
# MAIN FC-HMARL / PHYSICAL-VPP BRIDGE
# ============================================================

class FCHMARLVPPTrainingBridge:
    """
    Bridge between FCHMARLTrainingLoop and VPPEnvironment.

    The bridge:

    1. receives hierarchical SAC actions,
    2. maps normalized decisions to physical commands,
    3. executes the physical VPP,
    4. computes rewards,
    5. creates the next hierarchical state,
    6. returns EnvironmentStepResult.
    """

    def __init__(
        self,
        environment: VPPEnvironment,
        state_builder: HierarchicalStateBuilder,
        reward_builder: HierarchicalRewardBuilder,
        exogenous_provider: ExogenousProvider,
        action_mapper: FCHMARLActionMapper,
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

        if not isinstance(
            action_mapper,
            FCHMARLActionMapper,
        ):
            raise TypeError(
                "action_mapper must be an "
                "FCHMARLActionMapper."
            )

        if not isinstance(
            config,
            VPPTrainingBridgeConfig,
        ):
            raise TypeError(
                "config must be a "
                "VPPTrainingBridgeConfig."
            )

        config.validate()

        if (
            len(
                environment.microgrids
            )
            != config.number_of_microgrids
        ):
            raise ValueError(
                "Environment microgrid count does not "
                "match bridge configuration."
            )

        self.environment = environment

        self.state_builder = state_builder

        self.reward_builder = reward_builder

        self.exogenous_provider = (
            exogenous_provider
        )

        self.action_mapper = action_mapper

        self.config = config

        self.current_episode = 0

        self.current_step = 0

        self.last_exogenous_input: Optional[
            VPPExogenousInput
        ] = None

        self.last_physical_action: Optional[
            object
        ] = None

        self.last_environment_result = None

        self.last_sharing_matrix = np.zeros(
            (
                self.config.number_of_microgrids,
                self.config.number_of_microgrids,
            ),
            dtype=np.float64,
        )

    # ========================================================
    # EXOGENOUS DATA
    # ========================================================

    def get_exogenous_input(
        self,
        episode: int,
        step: int,
    ) -> VPPExogenousInput:
        """
        Retrieve and validate one exogenous record.
        """

        exogenous = (
            self.exogenous_provider(
                episode,
                step,
            )
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

        return exogenous

    # ========================================================
    # ENVIRONMENT STATE -> STATE BUILDER FORMAT
    # ========================================================

    def get_microgrid_observations(
        self,
    ) -> List[
        Dict[str, float]
    ]:
        """
        Convert VPPEnvironment microgrid states to the dictionary
        format expected by HierarchicalStateBuilder.

        Microgrid.get_state() currently supplies:

            bess_soc
            pv_power_kw
            load_kw
            ev_power_kw
            grid_power_kw
        """

        raw_states = (
            self.environment
            .get_local_states()
        )

        if (
            len(
                raw_states
            )
            != self.config.number_of_microgrids
        ):
            raise RuntimeError(
                "Environment returned an incorrect "
                "number of local states."
            )

        observations: List[
            Dict[str, float]
        ] = []

        for (
            index,
            raw_state,
        ) in enumerate(
            raw_states
        ):

            required_keys = (
                "bess_soc",
                "pv_power_kw",
                "load_kw",
                "ev_power_kw",
                "grid_power_kw",
            )

            missing = [
                key
                for key
                in required_keys
                if key not in raw_state
            ]

            if missing:
                raise KeyError(
                    f"Microgrid state {index} is missing "
                    f"required fields: {missing}"
                )

            observation = {
                "soc":
                    float(
                        raw_state[
                            "bess_soc"
                        ]
                    ),

                "pv_power":
                    float(
                        raw_state[
                            "pv_power_kw"
                        ]
                    ),

                "load_power":
                    float(
                        raw_state[
                            "load_kw"
                        ]
                    ),

                "ev_power":
                    float(
                        raw_state[
                            "ev_power_kw"
                        ]
                    ),

                "grid_exchange":
                    float(
                        raw_state[
                            "grid_power_kw"
                        ]
                    ),
            }

            observations.append(
                observation
            )

        return observations

    # ========================================================
    # TRAINING OBSERVATION
    # ========================================================

    def build_training_observation(
        self,
        exogenous: VPPExogenousInput,
        sharing_matrix,
    ) -> TrainingObservation:
        """
        Build the observation consumed by the training loop.
        """

        state_result = (
            self.state_builder.build(
                microgrid_observations=(
                    self.get_microgrid_observations()
                ),

                predictive_state=(
                    exogenous.predictive_state
                ),

                market_price=(
                    exogenous.resolved_market_price
                ),

                sharing_matrix=(
                    sharing_matrix
                ),
            )
        )

        # Use an actual Python list for local states.
        local_states = [
            np.asarray(
                state,
                dtype=np.float32,
            )
            for state
            in state_result.local_states
        ]

        coordinator_state = np.asarray(
            state_result.coordinator_state,
            dtype=np.float32,
        )

        observation = TrainingObservation(
            local_states=local_states,

            coordinator_state=(
                coordinator_state
            ),
        )

        observation.validate(
            self.config.number_of_microgrids
        )

        return observation

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        episode: int = 0,
    ) -> TrainingObservation:
        """
        Reset physical VPP and return initial FC-HMARL state.
        """

        self.environment.reset()

        self.current_episode = int(
            episode
        )

        self.current_step = 0

        self.last_physical_action = None

        self.last_environment_result = None

        self.last_sharing_matrix = np.zeros(
            (
                self.config.number_of_microgrids,
                self.config.number_of_microgrids,
            ),
            dtype=np.float64,
        )

        exogenous = (
            self.get_exogenous_input(
                self.current_episode,
                self.current_step,
            )
        )

        self.last_exogenous_input = (
            exogenous
        )

        return (
            self.build_training_observation(
                exogenous=exogenous,

                sharing_matrix=(
                    self.last_sharing_matrix
                ),
            )
        )

    # ========================================================
    # ACTION MAPPING
    # ========================================================

    def map_actions(
        self,
        action_bundle: HierarchicalActionBundle,
        exogenous: VPPExogenousInput,
    ):
        """
        Convert hierarchical normalized actions into physical
        VPP control variables.

        The mapper itself owns the exact reconstructed software
        encoding.
        """

        # The existing action mapper accepts the hierarchical
        # bundle together with the physical environment.
        #
        # EV charging requests are supplied through exogenous
        # data rather than invented by the mapper.

        physical_action = (
            self.action_mapper.map(
                local_actions=(
                    action_bundle.local_actions
                ),

                coordinator_action=(
                    action_bundle.coordinator_action
                ),

                environment=(
                    self.environment
                ),

                ev_requested_charging_powers_kw=(
                    exogenous
                    .ev_requested_charging_powers_kw
                ),
            )
        )

        return physical_action

    # ========================================================
    # LOCAL REWARD INPUTS
    # ========================================================

    def calculate_local_rewards(
        self,
        environment_result,
    ) -> List[float]:
        """
        Calculate the five local rewards.

        Manuscript form:

            r_i = -(
                C_grid_i
                + C_deg_i
                + C_viol_i
            )

        The reward builder contains the reconstructed implementation
        of this expression.
        """

        rewards: List[
            float
        ] = []

        maximum_socs = np.asarray(
            self.config.maximum_socs,
            dtype=np.float64,
        )

        maximum_grid = np.asarray(
            self.config.maximum_grid_exchanges_kw,
            dtype=np.float64,
        )

        if (
            len(
                environment_result.local_results
            )
            != self.config.number_of_microgrids
        ):
            raise RuntimeError(
                "Environment returned an incorrect "
                "number of local results."
            )

        for (
            index,
            local_result,
        ) in enumerate(
            environment_result.local_results
        ):

            market_result = (
                local_result.market
            )

            grid_cost = float(
                market_result[
                    "grid_purchase_cost_usd"
                ]
            )

            bess_power_kw = float(
                local_result.bess_power_kw
            )

            degradation_cost = (
                abs(
                    bess_power_kw
                )
                * self.config.timestep_hours
                * self.config
                .battery_degradation_cost_per_kwh
            )

            soc = float(
                local_result.bess_soc
            )

            grid_exchange = float(
                local_result.grid_power_kw
            )

            local_reward = (
                self.reward_builder.local_reward(
                    grid_cost=(
                        grid_cost
                    ),

                    degradation_cost=(
                        degradation_cost
                    ),

                    soc=soc,

                    maximum_soc=float(
                        maximum_socs[
                            index
                        ]
                    ),

                    grid_exchange=(
                        grid_exchange
                    ),

                    maximum_grid_exchange=float(
                        maximum_grid[
                            index
                        ]
                    ),
                )
            )

            rewards.append(
                float(
                    local_reward
                )
            )

        return rewards

    # ========================================================
    # COORDINATOR REWARD
    # ========================================================

    def calculate_coordinator_reward(
        self,
        environment_result,
        confidence: float,
    ) -> float:
        """
        Calculate upper-level FC-HMARL reward.

        Manuscript form:

            r_VPP =
                profit
                - imbalance cost
                - risk cost

        with:

            risk cost =
                rho * (1 - confidence)
        """

        totals = (
            environment_result.totals
        )

        coordinator_profit = float(
            totals[
                "market_profit_usd"
            ]
        )

        constraints = (
            environment_result.constraints
        )

        # The physical environment reports total balance
        # violation.  The monetary multiplier is a bridge-level
        # reconstruction parameter.

        balance_violation = float(
            constraints.get(
                "total_power_balance_violation_kw",
                constraints.get(
                    "total_balance_violation_kw",
                    0.0,
                ),
            )
        )

        imbalance_cost = (
            abs(
                balance_violation
            )
            * self.config
            .imbalance_penalty_per_kw
        )

        reward = (
            self.reward_builder
            .coordinator_reward(
                profit=(
                    coordinator_profit
                ),

                imbalance_cost=(
                    imbalance_cost
                ),

                confidence=float(
                    confidence
                ),
            )
        )

        return float(
            reward
        )

    # ========================================================
    # PHYSICAL ENVIRONMENT STEP
    # ========================================================

    def execute_environment_step(
        self,
        exogenous: VPPExogenousInput,
        physical_action,
    ):
        """
        Execute one real VPP interval.
        """

        result = (
            self.environment.step(
                time=float(
                    exogenous.time
                ),

                loads_kw=np.asarray(
                    exogenous.loads_kw,
                    dtype=np.float64,
                ),

                irradiances_w_m2=np.asarray(
                    exogenous.irradiances_w_m2,
                    dtype=np.float64,
                ),

                bess_actions_kw=np.asarray(
                    physical_action.bess_actions_kw,
                    dtype=np.float64,
                ),

                sharing_matrix_kw=np.asarray(
                    physical_action.sharing_matrix_kw,
                    dtype=np.float64,
                ),

                buy_prices_usd_per_kwh=np.asarray(
                    exogenous
                    .buy_prices_usd_per_kwh,
                    dtype=np.float64,
                ),

                sell_prices_usd_per_kwh=np.asarray(
                    exogenous
                    .sell_prices_usd_per_kwh,
                    dtype=np.float64,
                ),

                reserve_actions_kw=(
                    physical_action
                    .reserve_actions_kw
                ),

                ev_requested_charging_powers_kw=(
                    physical_action
                    .ev_requested_charging_powers_kw
                ),

                grid_power_actions_kw=(
                    physical_action
                    .grid_power_actions_kw
                ),
            )
        )

        return result

    # ========================================================
    # STEP
    # ========================================================

    def step(
        self,
        action_bundle: HierarchicalActionBundle,
    ) -> EnvironmentStepResult:
        """
        Execute one complete FC-HMARL/VPP interaction.

        IMPORTANT FIX
        -------------
        local_rewards is explicitly converted into a Python
        list[float] before EnvironmentStepResult is returned.

        This satisfies training_loop.py's Sequence[float]
        interface and fixes:

            TypeError:
            local_rewards must be a sequence.
        """

        if not isinstance(
            action_bundle,
            HierarchicalActionBundle,
        ):
            raise TypeError(
                "action_bundle must be a "
                "HierarchicalActionBundle."
            )

        # ----------------------------------------------------
        # 1. CURRENT EXOGENOUS INFORMATION
        # ----------------------------------------------------

        exogenous = (
            self.get_exogenous_input(
                self.current_episode,
                self.current_step,
            )
        )

        self.last_exogenous_input = (
            exogenous
        )

        # ----------------------------------------------------
        # 2. MAP NORMALIZED SAC ACTIONS -> PHYSICAL ACTIONS
        # ----------------------------------------------------

        physical_action = (
            self.map_actions(
                action_bundle=(
                    action_bundle
                ),

                exogenous=(
                    exogenous
                ),
            )
        )

        self.last_physical_action = (
            physical_action
        )

        # ----------------------------------------------------
        # 3. OPERATE REAL PHYSICAL VPP
        # ----------------------------------------------------

        environment_result = (
            self.execute_environment_step(
                exogenous=exogenous,

                physical_action=(
                    physical_action
                ),
            )
        )

        self.last_environment_result = (
            environment_result
        )

        # Use the actual feasible sharing schedule applied by
        # the environment / mapper for the next coordinator state.

        self.last_sharing_matrix = np.asarray(
            physical_action.sharing_matrix_kw,
            dtype=np.float64,
        ).copy()

        # ----------------------------------------------------
        # 4. COMPUTE LOCAL REWARDS
        # ----------------------------------------------------

        calculated_local_rewards = (
            self.calculate_local_rewards(
                environment_result
            )
        )

        # ====================================================
        # CRITICAL RUNTIME FIX
        # ====================================================
        #
        # Do NOT return np.ndarray here.
        #
        # EnvironmentStepResult.validate() checks:
        #
        #     isinstance(local_rewards, Sequence)
        #
        # A NumPy ndarray does not satisfy the required
        # Sequence check in the current training_loop.py.
        #
        # Therefore force a genuine Python list containing
        # genuine Python float values.
        # ====================================================

        local_rewards: List[
            float
        ] = [
            float(
                reward
            )
            for reward
            in calculated_local_rewards
        ]

        if (
            len(
                local_rewards
            )
            != self.config.number_of_microgrids
        ):
            raise RuntimeError(
                "Incorrect number of local rewards. "
                f"Expected "
                f"{self.config.number_of_microgrids}, "
                f"received {len(local_rewards)}."
            )

        if not np.isfinite(
            np.asarray(
                local_rewards,
                dtype=np.float64,
            )
        ).all():
            raise RuntimeError(
                "Local rewards contain NaN or Inf."
            )

        # ----------------------------------------------------
        # 5. COORDINATOR REWARD
        # ----------------------------------------------------

        coordinator_reward = float(
            self.calculate_coordinator_reward(
                environment_result=(
                    environment_result
                ),

                confidence=float(
                    exogenous.confidence
                ),
            )
        )

        if not np.isfinite(
            coordinator_reward
        ):
            raise RuntimeError(
                "Coordinator reward is NaN or Inf."
            )

        # ----------------------------------------------------
        # 6. TERMINATION FLAG
        # ----------------------------------------------------

        done = bool(
            environment_result.done
        )

        # ----------------------------------------------------
        # 7. MOVE INTERNAL CLOCK
        # ----------------------------------------------------

        self.current_step += 1

        # ----------------------------------------------------
        # 8. BUILD NEXT STATE
        # ----------------------------------------------------

        if not done:

            next_exogenous = (
                self.get_exogenous_input(
                    self.current_episode,
                    self.current_step,
                )
            )

        else:

            # The replay transition still requires a valid
            # next observation on the terminal step.
            #
            # Reuse the terminal exogenous predictive state
            # instead of requesting hour 24 from a provider
            # that may only expose indices 0...23.

            next_exogenous = (
                exogenous
            )

        next_observation = (
            self.build_training_observation(
                exogenous=(
                    next_exogenous
                ),

                sharing_matrix=(
                    self.last_sharing_matrix
                ),
            )
        )

        # ----------------------------------------------------
        # 9. DIAGNOSTIC INFORMATION
        # ----------------------------------------------------

        info = {
            "episode":
                int(
                    self.current_episode
                ),

            "step":
                int(
                    self.current_step - 1
                ),

            "time":
                float(
                    exogenous.time
                ),

            "confidence":
                float(
                    exogenous.confidence
                ),

            "local_rewards":
                local_rewards.copy(),

            "coordinator_reward":
                float(
                    coordinator_reward
                ),

            "done":
                bool(
                    done
                ),

            "environment_result":
                environment_result,

            "physical_action":
                physical_action,
        }

        # ----------------------------------------------------
        # 10. RETURN TRAINING-LOOP CONTRACT
        # ----------------------------------------------------

        result = EnvironmentStepResult(
            next_observation=(
                next_observation
            ),

            # =================================================
            # FIXED:
            # Python list, NOT numpy.ndarray
            # =================================================
            local_rewards=[
                float(
                    reward
                )
                for reward
                in local_rewards
            ],

            # Python scalar, not np.float32/np.float64.
            coordinator_reward=float(
                coordinator_reward
            ),

            # Python bool, not np.bool_.
            done=bool(
                done
            ),

            info=info,
        )

        # Validate here as well, so bridge errors are detected
        # before propagating into the outer training loop.
        result.validate(
            self.config.number_of_microgrids
        )

        return result

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(
        self,
    ) -> Dict[
        str,
        object,
    ]:
        """
        Return bridge status information.
        """

        return {
            "number_of_microgrids":
                self.config.number_of_microgrids,

            "current_episode":
                self.current_episode,

            "current_step":
                self.current_step,

            "battery_degradation_cost_per_kwh":
                (
                    self.config
                    .battery_degradation_cost_per_kwh
                ),

            "imbalance_penalty_per_kw":
                (
                    self.config
                    .imbalance_penalty_per_kw
                ),

            "timestep_hours":
                self.config.timestep_hours,

            "has_last_exogenous_input":
                (
                    self.last_exogenous_input
                    is not None
                ),

            "has_last_physical_action":
                (
                    self.last_physical_action
                    is not None
                ),

            "has_last_environment_result":
                (
                    self.last_environment_result
                    is not None
                ),
        }


# ============================================================
# PUBLIC ALIASES
# ============================================================

__all__ = [
    "VPPExogenousInput",
    "VPPPhysicalAction",
    "VPPTrainingBridgeConfig",
    "FCHMARLVPPTrainingBridge",
]
