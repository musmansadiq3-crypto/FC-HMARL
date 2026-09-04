"""
FC-HMARL offline training entry point.

This file connects the complete reconstructed implementation:

    Five-microgrid VPP
        ->
    Exogenous operating data
        ->
    Predictive state / confidence
        ->
    Hierarchical state builder
        ->
    5 local SAC agents
        ->
    1 coordinator SAC agent
        ->
    FC-HMARL action mapper
        ->
    Physical VPP environment
        ->
    Local + coordinator rewards
        ->
    Replay buffers
        ->
    SAC training loop

Two execution modes are supported.

1. Smoke test
   -----------
   python train.py --smoke-test

   Runs:
       1 episode
       24 hourly steps

   A deliberately smaller replay/batch configuration is used so
   gradient updates can actually occur during the 24-step smoke run.

   Extended smoke test:
   --------------------
   python train.py --smoke-test --episodes 10

   Runs:
       10 episodes
       24 hourly steps

   The reduced replay buffer and batch size are retained so SAC
   updates begin quickly during validation.

2. Full reconstructed manuscript-scale training
   ----------------------------------------------
   python train.py

   Runs:
       5000 episodes
       24 steps per episode

   Manuscript-supported SAC settings:
       gamma                  = 0.99
       learning rate          = 1e-4
       replay capacity        = 1,000,000
       batch size             = 512
       tau                    = 0.005
       initial noise          = 0.20
       noise decay            = 0.999

Important:
----------
The deterministic exogenous profiles in this file are currently an
integration/reconstruction dataset. They are NOT claimed to be the
lost original manuscript training data.

After this executable training chain is validated, the synthetic
provider can be replaced by the final forecasting/data pipeline.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import numpy as np
import torch


# ============================================================
# PHYSICAL ENVIRONMENT
# ============================================================

from environment.bess import (
    BESSParameters,
    BatteryEnergyStorageSystem,
)

from environment.pv import (
    PVParameters,
    PhotovoltaicSystem,
)

from environment.ev_fleet import (
    EVFleet,
    EVFleetParameters,
    EVRecord,
)

from environment.market import (
    ElectricityMarket,
    MarketParameters,
)

from environment.microgrid import (
    Microgrid,
    MicrogridParameters,
)

from environment.energy_sharing import (
    EnergySharingNetwork,
    EnergySharingParameters,
)

from environment.vpp_env import (
    VPPEnvironment,
)


# ============================================================
# AGENTS
# ============================================================

from agents.local_agent import (
    LocalAgent,
    LocalAgentConfig,
)

from agents.coordinator_agent import (
    CoordinatorAgent,
    CoordinatorAgentConfig,
)


# ============================================================
# MARL
# ============================================================

from marl.state_builder import (
    StateBuilderConfig,
    HierarchicalStateBuilder,
)

from marl.rewards import (
    RewardConfig,
    HierarchicalRewardBuilder,
)

from marl.action_mapper import (
    ActionMapperConfig,
    FCHMARLActionMapper,
)

from marl.vpp_training_bridge import (
    VPPExogenousInput,
    VPPTrainingBridgeConfig,
    FCHMARLVPPTrainingBridge,
)

from marl.training_loop import (
    TrainingLoopConfig,
    FCHMARLTrainingLoop,
)


# ============================================================
# EVALUATION
# ============================================================

from evaluation.training_history_logger import (
    export_training_results,
)


# ============================================================
# GLOBAL DIMENSIONS
# ============================================================

NUMBER_OF_MICROGRIDS = 5

FORECAST_HORIZON = 24
FORECAST_FEATURES = 4

# Eq. (49)-style reconstruction:
#
# 5 current physical variables
# +
# 24 x 4 predictive variables
#
# = 101
LOCAL_STATE_DIMENSION = (
    5
    + FORECAST_HORIZON
    * FORECAST_FEATURES
)

# Reconstructed software action encoding:
#
# [BESS,
#  share-to-peer-1,
#  share-to-peer-2,
#  share-to-peer-3,
#  share-to-peer-4]
LOCAL_ACTION_DIMENSION = 5

# Eq. (55)-style reconstruction:
#
# 3 coordinator variables
# +
# 24 x 4 predictive variables
#
# = 99
COORDINATOR_STATE_DIMENSION = (
    3
    + FORECAST_HORIZON
    * FORECAST_FEATURES
)

# Reconstructed software encoding:
#
# [market/grid command,
#  reserve command,
#  sharing multiplier]
#
# The manuscript gives conceptual coordinator actions,
# but not this exact Python-vector implementation.
COORDINATOR_ACTION_DIMENSION = 3


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_global_seed(
    seed: int,
) -> None:
    """
    Set deterministic random seeds where practical.
    """

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# MICROGRID FACTORY
# ============================================================

def build_microgrid(
    name: str,
    pv_capacity_kw: float,
    bess_capacity_kwh: float,
    bess_power_kw: float,
    peak_load_kw: float,
    transformer_kva: float,
) -> Microgrid:
    """
    Construct one physical microgrid.

    These parameters match the physical fixture already validated
    by tests/test_vpp_env.py and
    tests/test_real_vpp_training_bridge.py.
    """

    pv_system = PhotovoltaicSystem(
        PVParameters(
            rated_capacity_kw=(
                pv_capacity_kw
            ),

            efficiency=1.0,

            critical_irradiance_w_m2=(
                200.0
            ),

            stc_irradiance_w_m2=(
                1000.0
            ),
        ),

        name=f"{name}_PV",
    )

    bess = BatteryEnergyStorageSystem(
        BESSParameters(
            capacity_kwh=(
                bess_capacity_kwh
            ),

            rated_power_kw=(
                bess_power_kw
            ),

            charging_efficiency=0.95,

            discharging_efficiency=0.95,

            self_discharge_rate=0.001,

            minimum_soc=0.20,

            maximum_soc=0.95,

            time_step_hours=1.0,
        ),

        initial_soc=0.60,

        name=f"{name}_BESS",
    )

    # --------------------------------------------------------
    # Integration-test EV fleet.
    #
    # These individual EV records are reconstruction values.
    # They should later be replaced by final EV-data profiles.
    # --------------------------------------------------------

    ev_fleet = EVFleet(
        EVFleetParameters(
            number_of_evs=2,

            maximum_aggregate_charging_power_kw=(
                20.0
            ),

            time_step_hours=1.0,
        ),

        vehicles=[
            EVRecord(
                ev_id=f"{name}_EV1",
                arrival_time=8.0,
                departure_time=12.0,
                charging_power_kw=7.0,
            ),

            EVRecord(
                ev_id=f"{name}_EV2",
                arrival_time=9.0,
                departure_time=15.0,
                charging_power_kw=11.0,
            ),
        ],

        name=f"{name}_EV_Fleet",
    )

    market = ElectricityMarket(
        MarketParameters(
            minimum_buy_price_usd_per_kwh=(
                0.12
            ),

            maximum_buy_price_usd_per_kwh=(
                0.32
            ),

            minimum_sell_price_usd_per_kwh=(
                0.08
            ),

            maximum_sell_price_usd_per_kwh=(
                0.24
            ),

            reserve_price_usd_per_kwh=(
                0.05
            ),

            time_step_hours=1.0,
        ),

        name=f"{name}_Market",
    )

    microgrid = Microgrid(
        parameters=MicrogridParameters(
            name=name,

            peak_load_kw=(
                peak_load_kw
            ),

            transformer_rating_kva=(
                transformer_kva
            ),

            power_factor=1.0,
        ),

        pv_system=pv_system,

        bess=bess,

        ev_fleet=ev_fleet,

        market=market,
    )

    return microgrid


# ============================================================
# BUILD FIVE-MICROGRID VPP
# ============================================================

def build_vpp_environment() -> VPPEnvironment:
    """
    Build the complete 5-MG physical VPP.
    """

    microgrids = [

        build_microgrid(
            name="MG1",
            pv_capacity_kw=500.0,
            bess_capacity_kwh=1000.0,
            bess_power_kw=250.0,
            peak_load_kw=750.0,
            transformer_kva=1000.0,
        ),

        build_microgrid(
            name="MG2",
            pv_capacity_kw=600.0,
            bess_capacity_kwh=1200.0,
            bess_power_kw=300.0,
            peak_load_kw=850.0,
            transformer_kva=1250.0,
        ),

        build_microgrid(
            name="MG3",
            pv_capacity_kw=450.0,
            bess_capacity_kwh=900.0,
            bess_power_kw=250.0,
            peak_load_kw=700.0,
            transformer_kva=1000.0,
        ),

        build_microgrid(
            name="MG4",
            pv_capacity_kw=550.0,
            bess_capacity_kwh=1100.0,
            bess_power_kw=300.0,
            peak_load_kw=800.0,
            transformer_kva=1250.0,
        ),

        build_microgrid(
            name="MG5",
            pv_capacity_kw=700.0,
            bess_capacity_kwh=1400.0,
            bess_power_kw=350.0,
            peak_load_kw=950.0,
            transformer_kva=1500.0,
        ),
    ]

    # --------------------------------------------------------
    # Reconstruction:
    # fully connected directed sharing network.
    #
    # The manuscript did not give the exact pairwise software
    # topology/capacity matrix.
    # --------------------------------------------------------

    connectivity_matrix = np.ones(
        (
            NUMBER_OF_MICROGRIDS,
            NUMBER_OF_MICROGRIDS,
        ),
        dtype=int,
    )

    np.fill_diagonal(
        connectivity_matrix,
        0,
    )

    maximum_power_matrix_kw = np.full(
        (
            NUMBER_OF_MICROGRIDS,
            NUMBER_OF_MICROGRIDS,
        ),
        250.0,
        dtype=float,
    )

    np.fill_diagonal(
        maximum_power_matrix_kw,
        0.0,
    )

    sharing_network = (
        EnergySharingNetwork(
            parameters=(
                EnergySharingParameters(
                    number_of_microgrids=(
                        NUMBER_OF_MICROGRIDS
                    ),

                    efficiency=0.98,
                )
            ),

            connectivity_matrix=(
                connectivity_matrix
            ),

            maximum_power_matrix_kw=(
                maximum_power_matrix_kw
            ),

            name="FC_HMARL_Sharing",
        )
    )

    environment = VPPEnvironment(
        microgrids=microgrids,

        energy_sharing_network=(
            sharing_network
        ),

        episode_length_hours=24,

        name="FC_HMARL_VPP",
    )

    return environment


# ============================================================
# DAILY LOAD PROFILE
# ============================================================

def daily_load_multiplier(
    hour: int,
) -> float:
    """
    Reconstructed normalized daily demand pattern.

    This is used only until the final historical-data loader is
    connected.
    """

    profile = np.asarray(
        [
            0.58,
            0.55,
            0.53,
            0.52,
            0.54,
            0.60,
            0.69,
            0.78,
            0.84,
            0.89,
            0.94,
            0.98,
            1.00,
            0.97,
            0.92,
            0.88,
            0.90,
            0.96,
            1.00,
            0.98,
            0.91,
            0.82,
            0.72,
            0.64,
        ],
        dtype=np.float64,
    )

    return float(
        profile[
            int(hour) % 24
        ]
    )


# ============================================================
# SOLAR IRRADIANCE PROFILE
# ============================================================

def daily_irradiance(
    hour: int,
) -> float:
    """
    Smooth deterministic daytime irradiance profile.
    """

    hour = int(
        hour
    ) % 24

    if (
        hour < 6
        or hour > 18
    ):
        return 0.0

    solar_angle = (
        np.pi
        * (hour - 6)
        / 12.0
    )

    irradiance = (
        900.0
        * np.sin(
            solar_angle
        )
    )

    return float(
        max(
            0.0,
            irradiance,
        )
    )


# ============================================================
# MARKET PRICE PROFILE
# ============================================================

def daily_buy_price(
    hour: int,
) -> float:
    """
    Reconstructed day-ahead price profile.
    """

    hour = int(
        hour
    ) % 24

    # Valley period.
    if 0 <= hour < 6:
        return 0.14

    # Morning / daytime.
    if 6 <= hour < 9:
        return 0.18

    # Morning peak.
    if 9 <= hour < 13:
        return 0.27

    # Afternoon.
    if 13 <= hour < 19:
        return 0.21

    # Evening peak.
    if 19 <= hour < 22:
        return 0.30

    return 0.17


def daily_sell_price(
    hour: int,
) -> float:
    """
    Reconstructed export price based on buy price.
    """

    buy_price = (
        daily_buy_price(
            hour
        )
    )

    return float(
        max(
            0.08,
            0.75 * buy_price,
        )
    )


# ============================================================
# PREDICTIVE STATE
# ============================================================

def build_predictive_state(
    current_hour: int,
) -> np.ndarray:
    """
    Construct 24 x 4 predictive state.

    Column order:

        0 -> aggregate PV prediction
        1 -> aggregate load prediction
        2 -> aggregate EV prediction
        3 -> electricity-price prediction

    The structure follows the reconstructed forecasting interface.
    The values here are deterministic integration data, not the
    lost manuscript forecast dataset.
    """

    predictive_state = np.zeros(
        (
            FORECAST_HORIZON,
            FORECAST_FEATURES,
        ),
        dtype=np.float64,
    )

    total_peak_load = 4050.0

    total_pv_capacity = 2800.0

    for forecast_step in range(
        FORECAST_HORIZON
    ):

        hour = (
            current_hour
            + forecast_step
        ) % 24

        irradiance = (
            daily_irradiance(
                hour
            )
        )

        predicted_pv = (
            total_pv_capacity
            * irradiance
            / 1000.0
        )

        predicted_load = (
            total_peak_load
            * daily_load_multiplier(
                hour
            )
        )

        if 8 <= hour < 15:
            predicted_ev = 50.0
        else:
            predicted_ev = 0.0

        predicted_price = (
            daily_buy_price(
                hour
            )
        )

        predictive_state[
            forecast_step,
            0,
        ] = predicted_pv

        predictive_state[
            forecast_step,
            1,
        ] = predicted_load

        predictive_state[
            forecast_step,
            2,
        ] = predicted_ev

        predictive_state[
            forecast_step,
            3,
        ] = predicted_price

    return predictive_state


# ============================================================
# EXOGENOUS INPUT PROVIDER
# ============================================================

def training_exogenous_provider(
    episode: int,
    step: int,
) -> VPPExogenousInput:
    """
    Supply physical and predictive inputs to the VPP bridge.

    The current implementation intentionally uses deterministic
    reconstructed profiles so that the complete FC-HMARL software
    pipeline can be trained and tested reproducibly.

    Later this function will be replaced/extended by the actual
    forecasting pipeline.
    """

    _ = episode

    hour = int(
        step % 24
    )

    peak_loads = np.asarray(
        [
            750.0,
            850.0,
            700.0,
            800.0,
            950.0,
        ],
        dtype=np.float64,
    )

    load_multiplier = (
        daily_load_multiplier(
            hour
        )
    )

    loads_kw = (
        peak_loads
        * load_multiplier
    )

    irradiance = (
        daily_irradiance(
            hour
        )
    )

    irradiances_w_m2 = np.full(
        NUMBER_OF_MICROGRIDS,
        irradiance,
        dtype=np.float64,
    )

    buy_price = (
        daily_buy_price(
            hour
        )
    )

    sell_price = (
        daily_sell_price(
            hour
        )
    )

    buy_prices = np.full(
        NUMBER_OF_MICROGRIDS,
        buy_price,
        dtype=np.float64,
    )

    sell_prices = np.full(
        NUMBER_OF_MICROGRIDS,
        sell_price,
        dtype=np.float64,
    )

    predictive_state = (
        build_predictive_state(
            hour
        )
    )

    # --------------------------------------------------------
    # Confidence placeholder / reconstruction.
    #
    # The final pipeline should calculate Phi using the
    # forecasting errors:
    #
    # epsilon_RL = sqrt(ePV^2 + eLoad^2)
    # omega_RL   = exp(-epsilon_RL)
    #
    # epsilon_EM = sqrt(eEV^2 + ePrice^2)
    #
    # Phi = omega_RL * exp(-epsilon_EM)
    #
    # For the integration trainer we use a stable finite value.
    # --------------------------------------------------------

    confidence = 0.95

    return VPPExogenousInput(
        time=float(
            hour
        ),

        loads_kw=loads_kw,

        irradiances_w_m2=(
            irradiances_w_m2
        ),

        buy_prices_usd_per_kwh=(
            buy_prices
        ),

        sell_prices_usd_per_kwh=(
            sell_prices
        ),

        predictive_state=(
            predictive_state
        ),

        confidence=confidence,

        market_price=buy_price,
    )


# ============================================================
# STATE BUILDER
# ============================================================

def build_state_builder(
) -> HierarchicalStateBuilder:
    """
    Build hierarchical FC-HMARL states.
    """

    config = StateBuilderConfig(
        number_of_microgrids=(
            NUMBER_OF_MICROGRIDS
        ),

        forecast_horizon=(
            FORECAST_HORIZON
        ),

        forecast_features=(
            FORECAST_FEATURES
        ),
    )

    return HierarchicalStateBuilder(
        config
    )


# ============================================================
# REWARD BUILDER
# ============================================================

def build_reward_builder(
) -> HierarchicalRewardBuilder:
    """
    Build local and coordinator reward functions.

    beta_soc = 1 and beta_grid = 1 remain configurable
    reconstruction choices.

    risk_aversion = 1 corresponds to rho = 1 here.
    """

    config = RewardConfig(
        beta_soc=1.0,
        beta_grid=1.0,
        risk_aversion=1.0,
    )

    return HierarchicalRewardBuilder(
        config
    )


# ============================================================
# ACTION MAPPER
# ============================================================

def build_action_mapper(
) -> FCHMARLActionMapper:
    """
    Map normalized SAC outputs into physical VPP actions.
    """

    config = ActionMapperConfig(
        use_dynamic_bess_feasibility=True,

        # Physical environment automatically closes
        # local grid power balance.
        coordinator_controls_grid=False,

        coordinator_controls_reserve=True,

        coordinator_controls_sharing=True,

        reserve_fraction_of_bess_rating=1.0,
    )

    return FCHMARLActionMapper(
        config
    )


# ============================================================
# TRAINING BRIDGE
# ============================================================

def build_training_bridge(
    environment: VPPEnvironment,
) -> FCHMARLVPPTrainingBridge:
    """
    Connect hierarchical actions to the real physical VPP.
    """

    maximum_socs = [
        microgrid.bess.maximum_soc
        for microgrid
        in environment.microgrids
    ]

    maximum_grid_exchange_kw = [
        microgrid.transformer_rating_kva
        for microgrid
        in environment.microgrids
    ]

    bridge_config = (
        VPPTrainingBridgeConfig(
            maximum_socs=(
                maximum_socs
            ),

            maximum_grid_exchanges_kw=(
                maximum_grid_exchange_kw
            ),

            # No unsupported manuscript degradation coefficient
            # is inserted here.
            battery_degradation_cost_per_kwh=(
                0.0
            ),

            # Reconstruction coefficient used for executable
            # integration. Replace when final calibrated economic
            # parameter is available.
            imbalance_penalty_per_kw=1.0,

            timestep_hours=1.0,

            number_of_microgrids=(
                NUMBER_OF_MICROGRIDS
            ),
        )
    )

    bridge = (
        FCHMARLVPPTrainingBridge(
            environment=environment,

            state_builder=(
                build_state_builder()
            ),

            reward_builder=(
                build_reward_builder()
            ),

            exogenous_provider=(
                training_exogenous_provider
            ),

            action_mapper=(
                build_action_mapper()
            ),

            config=bridge_config,
        )
    )

    return bridge


# ============================================================
# LOCAL AGENTS
# ============================================================

def build_local_agents(
    seed: int,
    smoke_test: bool,
    device: str,
) -> List[LocalAgent]:
    """
    Create the five local SAC agents.
    """

    local_agents = []

    # --------------------------------------------------------
    # Smoke-test buffer:
    # deliberately small enough for updates in one episode.
    #
    # Full training:
    # manuscript-scale replay/batch settings.
    # --------------------------------------------------------

    if smoke_test:

        replay_capacity = 2048

        batch_size = 16

    else:

        replay_capacity = 1_000_000

        batch_size = 512

    for microgrid_index in range(
        NUMBER_OF_MICROGRIDS
    ):

        config = LocalAgentConfig(
            microgrid_id=(
                microgrid_index
                + 1
            ),

            state_dimension=(
                LOCAL_STATE_DIMENSION
            ),

            action_dimension=(
                LOCAL_ACTION_DIMENSION
            ),

            # Network widths are reconstruction choices.
            hidden_dimensions=(
                256,
                256,
            ),

            activation="relu",

            action_low=-1.0,

            action_high=1.0,

            discount_factor=0.99,

            learning_rate=1e-4,

            replay_buffer_capacity=(
                replay_capacity
            ),

            batch_size=(
                batch_size
            ),

            soft_update_coefficient=(
                0.005
            ),

            # Reconstruction because manuscript does not
            # provide numerical SAC alpha.
            entropy_coefficient=0.20,

            initial_exploration_noise=(
                0.20
            ),

            exploration_noise_decay=(
                0.999
            ),

            minimum_exploration_noise=(
                0.0
            ),

            gradient_clip_norm=None,

            seed=(
                seed
                + microgrid_index
            ),

            device=device,

            dtype="float32",
        )

        local_agents.append(
            LocalAgent(
                config
            )
        )

    return local_agents


# ============================================================
# COORDINATOR AGENT
# ============================================================

def build_coordinator_agent(
    seed: int,
    smoke_test: bool,
    device: str,
) -> CoordinatorAgent:
    """
    Create the upper-level VPP SAC coordinator.
    """

    if smoke_test:

        replay_capacity = 2048

        batch_size = 16

    else:

        replay_capacity = 1_000_000

        batch_size = 512

    config = CoordinatorAgentConfig(
        state_dimension=(
            COORDINATOR_STATE_DIMENSION
        ),

        action_dimension=(
            COORDINATOR_ACTION_DIMENSION
        ),

        coordinator_id=1,

        hidden_dimensions=(
            256,
            256,
        ),

        activation="relu",

        action_low=-1.0,

        action_high=1.0,

        discount_factor=0.99,

        learning_rate=1e-4,

        replay_buffer_capacity=(
            replay_capacity
        ),

        batch_size=(
            batch_size
        ),

        soft_update_coefficient=0.005,

        entropy_coefficient=0.20,

        initial_exploration_noise=0.20,

        exploration_noise_decay=0.999,

        minimum_exploration_noise=0.0,

        gradient_clip_norm=None,

        seed=seed + 100,

        device=device,

        dtype="float32",
    )

    return CoordinatorAgent(
        config
    )


# ============================================================
# DEVICE SELECTION
# ============================================================

def resolve_device(
    requested_device: str,
) -> str:
    """
    Resolve CPU/CUDA selection.
    """

    requested_device = (
        requested_device
        .strip()
        .lower()
    )

    if requested_device == "auto":

        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    if requested_device == "cuda":

        if not torch.cuda.is_available():

            print(
                "WARNING: CUDA was requested but is "
                "not available. Falling back to CPU."
            )

            return "cpu"

    return requested_device


# ============================================================
# CALLBACK CONNECTION
# ============================================================

def make_reset_function(
    bridge: FCHMARLVPPTrainingBridge,
):
    """
    Adapt bridge.reset() to TrainingLoop reset_function API.
    """

    def reset_function(
        episode: int,
    ):

        return bridge.reset(
            episode=episode
        )

    return reset_function


def make_step_function(
    bridge: FCHMARLVPPTrainingBridge,
):
    """
    Adapt bridge.step() to TrainingLoop step_function API.

    TrainingLoop passes:
        episode
        time_step
        action_bundle

    The physical bridge already tracks the active episode/time
    internally through reset() and successive calls to step().
    """

    def step_function(
        episode: int,
        time_step: int,
        action_bundle,
    ):

        _ = episode
        _ = time_step

        return bridge.step(
            action_bundle
        )

    return step_function


# ============================================================
# BUILD COMPLETE TRAINER
# ============================================================

def build_training_loop(
    smoke_test: bool,
    episodes: int | None,
    seed: int,
    device: str,
) -> FCHMARLTrainingLoop:
    """
    Assemble complete FC-HMARL training system.
    """

    environment = (
        build_vpp_environment()
    )

    bridge = build_training_bridge(
        environment
    )

    local_agents = (
        build_local_agents(
            seed=seed,
            smoke_test=smoke_test,
            device=device,
        )
    )

    coordinator_agent = (
        build_coordinator_agent(
            seed=seed,
            smoke_test=smoke_test,
            device=device,
        )
    )

    if smoke_test:

        # ----------------------------------------------------
        # Multi-episode validation mode
        # ----------------------------------------------------
        #
        # Default:
        #     --smoke-test
        #         -> 1 episode
        #
        # Extended:
        #     --smoke-test --episodes 10
        #         -> 10 episodes
        #
        # The reduced replay buffer and batch size are retained
        # so SAC updates begin quickly during validation.
        # ----------------------------------------------------

        if episodes is None:
            maximum_episodes = 1
        else:
            maximum_episodes = episodes

        checkpoint_interval = 1

        enable_checkpointing = False

    else:

        if episodes is None:
            maximum_episodes = 5000
        else:
            maximum_episodes = episodes

        checkpoint_interval = 100

        enable_checkpointing = True

    training_config = TrainingLoopConfig(
        maximum_episodes=(
            maximum_episodes
        ),

        steps_per_episode=24,

        updates_per_step=1,

        checkpoint_interval=(
            checkpoint_interval
        ),

        enable_checkpointing=(
            enable_checkpointing
        ),

        checkpoint_directory=(
            "outputs/checkpoints"
        ),

        deterministic_actions=False,

        verbose=True,
    )

    trainer = FCHMARLTrainingLoop(
        local_agents=local_agents,

        coordinator_agent=(
            coordinator_agent
        ),

        reset_function=(
            make_reset_function(
                bridge
            )
        ),

        step_function=(
            make_step_function(
                bridge
            )
        ),

        config=training_config,

        stop_function=None,
    )

    # Keep references for diagnostics.
    trainer.physical_environment = (
        environment
    )

    trainer.vpp_training_bridge = (
        bridge
    )

    return trainer


# ============================================================
# SAVE TRAINING SUMMARY
# ============================================================

def make_json_safe(
    value,
):
    """
    Convert NumPy/path objects to JSON-compatible values.
    """

    if isinstance(
        value,
        np.ndarray,
    ):

        return value.tolist()

    if isinstance(
        value,
        (
            np.integer,
        ),
    ):

        return int(
            value
        )

    if isinstance(
        value,
        (
            np.floating,
        ),
    ):

        return float(
            value
        )

    if isinstance(
        value,
        Path,
    ):

        return str(
            value
        )

    if isinstance(
        value,
        dict,
    ):

        return {
            str(key):
                make_json_safe(
                    item
                )

            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):

        return [
            make_json_safe(
                item
            )
            for item in value
        ]

    return value


def save_training_summary(
    trainer: FCHMARLTrainingLoop,
    smoke_test: bool,
) -> Path:
    """
    Save top-level training summary.
    """

    output_directory = Path(
        "outputs/results"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if smoke_test:

        output_path = (
            output_directory
            / "smoke_training_summary.json"
        )

    else:

        output_path = (
            output_directory
            / "training_summary.json"
        )

    summary = (
        trainer.summary()
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            make_json_safe(
                summary
            ),
            file,
            indent=4,
        )

    return output_path


# ============================================================
# PRINT SYSTEM INFORMATION
# ============================================================

def print_training_configuration(
    smoke_test: bool,
    episodes: int,
    device: str,
    seed: int,
) -> None:
    """
    Print configuration before training begins.
    """

    print()
    print("=" * 72)
    print(
        "FC-HMARL TRAINING"
    )
    print("=" * 72)

    if smoke_test:

        print(
            "Mode                   : "
            "smoke / validation training"
        )

    else:

        print(
            "Mode                   : "
            "full training"
        )

    print(
        "Microgrids             : "
        f"{NUMBER_OF_MICROGRIDS}"
    )

    print(
        "Local agents           : "
        f"{NUMBER_OF_MICROGRIDS}"
    )

    print(
        "Coordinator agents     : 1"
    )

    print(
        "Local state dimension  : "
        f"{LOCAL_STATE_DIMENSION}"
    )

    print(
        "Local action dimension : "
        f"{LOCAL_ACTION_DIMENSION}"
    )

    print(
        "Coord. state dimension : "
        f"{COORDINATOR_STATE_DIMENSION}"
    )

    print(
        "Coord. action dimension: "
        f"{COORDINATOR_ACTION_DIMENSION}"
    )

    print(
        "Episodes               : "
        f"{episodes}"
    )

    print(
        "Steps / episode        : 24"
    )

    print(
        "Device                 : "
        f"{device}"
    )

    print(
        "Random seed            : "
        f"{seed}"
    )

    if smoke_test:

        print(
            "Replay capacity        : 2,048"
        )

        print(
            "Batch size             : 16"
        )

    else:

        print(
            "Replay capacity        : 1,000,000"
        )

        print(
            "Batch size             : 512"
        )

    print(
        "Discount factor        : 0.99"
    )

    print(
        "Learning rate          : 1e-4"
    )

    print(
        "Soft update tau        : 0.005"
    )

    print(
        "Entropy coefficient    : 0.20 "
        "(reconstruction)"
    )

    print(
        "Initial exploration    : 0.20"
    )

    print(
        "Exploration decay      : 0.999"
    )

    print("=" * 72)
    print()


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_arguments():
    """
    Parse command-line options.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Train the reconstructed "
            "FC-HMARL VPP controller."
        )
    )

    parser.add_argument(
        "--smoke-test",

        action="store_true",

        help=(
            "Run smoke/validation training using "
            "a reduced replay buffer/batch for "
            "integration validation. Default: 1 episode. "
            "Use --episodes to override."
        ),
    )

    parser.add_argument(
        "--episodes",

        type=int,

        default=None,

        help=(
            "Override number of episodes "
            "for non-smoke training, or extend "
            "smoke-test episode count."
        ),
    )

    parser.add_argument(
        "--seed",

        type=int,

        default=42,

        help=(
            "Global random seed."
        ),
    )

    parser.add_argument(
        "--device",

        type=str,

        default="cpu",

        choices=[
            "cpu",
            "cuda",
            "auto",
        ],

        help=(
            "PyTorch training device."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    """
    FC-HMARL training program.
    """

    args = parse_arguments()

    if (
        args.episodes is not None        and args.episodes <= 0
    ):

        raise ValueError(
            "--episodes must be positive."
        )

    device = resolve_device(
        args.device
    )

    set_global_seed(
        args.seed
    )

    # --------------------------------------------------------
    # Determine number of episodes
    # --------------------------------------------------------
    #
    # --smoke-test without --episodes -> 1 episode
    # --smoke-test --episodes 10      -> 10 episodes
    # no flags                         -> 5000 episodes
    # --episodes 100                   -> 100 episodes
    # --------------------------------------------------------

    if args.smoke_test:

        if args.episodes is None:
            number_of_episodes = 1
        else:
            number_of_episodes = args.episodes

    elif args.episodes is None:

        number_of_episodes = 5000

    else:

        number_of_episodes = args.episodes

    print_training_configuration(
        smoke_test=(
            args.smoke_test
        ),

        episodes=(
            number_of_episodes
        ),

        device=device,

        seed=args.seed,
    )

    print(
        "Building physical "
        "five-microgrid VPP..."
    )

    trainer = build_training_loop(
        smoke_test=(
            args.smoke_test
        ),

        episodes=(
            args.episodes
        ),

        seed=args.seed,

        device=device,
    )

    print(
        "VPP environment built."
    )

    print(
        "Five local SAC agents built."
    )

    print(
        "Coordinator SAC agent built."
    )

    print(
        "Training bridge connected."
    )

    print(
        "Starting FC-HMARL training..."
    )

    print()

    history = trainer.train()

    print()
    print("=" * 72)
    print(
        "TRAINING COMPLETE"
    )
    print("=" * 72)

    history_summary = (
        history.summary()
    )

    print(
        "Episodes completed : "
        f"{history_summary.get('number_of_episodes')}"
    )

    print(
        "Total environment "
        "steps : "
        f"{history_summary.get('total_steps')}"
    )

    print(
        "Final return       : "
        f"{history_summary.get('final_return')}"
    )

    print(
        "Mean return        : "
        f"{history_summary.get('mean_return')}"
    )

    # ============================================================
    # SAVE JSON SUMMARY
    # ============================================================

    summary_path = (
        save_training_summary(
            trainer=trainer,
            smoke_test=(
                args.smoke_test
            ),
        )
    )

    print(
        "Summary saved      : "
        f"{summary_path}"
    )

    # ============================================================
    # EXPORT TRAINING HISTORY (CSV + FIGURES)
    # ============================================================

    if args.smoke_test:

        moving_average_window = min(
            5,
            max(
                1,
                len(
                    history.episodes
                ),
            ),
        )

    else:

        moving_average_window = 20

    training_outputs = (
        export_training_results(
            history=history,
            output_directory=(
                "outputs/results"
            ),
            moving_average_window=(
                moving_average_window
            ),
        )
    )

    print()
    print(
        "Training history CSV : "
        f"{training_outputs['training_history_csv']}"
    )

    print(
        "Convergence figure   : "
        f"{training_outputs['training_convergence_figure']}"
    )

    print(
        "Return components    : "
        f"{training_outputs['return_components_figure']}"
    )

    print(
        "SAC update figure    : "
        f"{training_outputs['update_count_figure']}"
    )

    print("=" * 72)

    if args.smoke_test:

        print()

        print(
            "Smoke test finished."
        )

        print(
            "Inspect the episode line above. "
            "Local/coordinator update counts "
            "should be greater than zero once "
            "the replay buffers reach batch "
            "size 16."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()