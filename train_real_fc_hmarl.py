# ============================================================
# FC-HMARL
# STEP 7L-B
# REAL-DATA HIERARCHICAL SAC TRAINING
# ============================================================
#
# TRAINING DATA:
#   outputs/rl_data/real_rl_training_archive.npz
#
# DATA SEPARATION:
#   TRAIN      -> FC-HMARL learning
#   VALIDATION -> forecast selection + confidence calibration
#   TEST       -> final evaluation only
#
# IMPORTANT:
# This script does NOT load the forecasting TEST archive.
#
# ============================================================

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch


# ============================================================
# EXISTING PROJECT MODULES
# ============================================================

from environment.bess import (
    BESSParameters,
    BatteryEnergyStorageSystem,
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

from environment.pv import (
    PVParameters,
    PhotovoltaicSystem,
)

from environment.energy_sharing import (
    EnergySharingNetwork,
    EnergySharingParameters,
)

from environment.vpp_env import (
    VPPEnvironment,
)


from marl.action_mapper import (
    ActionMapperConfig,
    FCHMARLActionMapper,
)

from marl.state_builder import (
    StateBuilderConfig,
    HierarchicalStateBuilder,
)

from marl.rewards import (
    RewardConfig,
    HierarchicalRewardBuilder,
)

from marl.vpp_training_bridge import (
    FCHMARLVPPTrainingBridge,
    VPPTrainingBridgeConfig,
    VPPExogenousInput,
)

from marl.training_loop import (
    FCHMARLTrainingLoop,
    TrainingLoopConfig,
)


# ------------------------------------------------------------
# Reuse already validated agent builders from train.py.
# ------------------------------------------------------------

from train import (
    build_local_agents,
    build_coordinator_agent,
    make_reset_function,
    make_step_function,
    export_training_results,
    resolve_device,
    set_global_seed,
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"D:\Molvi paper review\FC_HMARL"
)

RL_ARCHIVE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "rl_data"
    / "real_rl_training_archive.npz"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "results"
    / "real_fc_hmarl"
)

CHECKPOINT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "real_fc_hmarl"
)

SUMMARY_FILE = (
    OUTPUT_DIRECTORY
    / "real_fc_hmarl_training_summary.json"
)


# ============================================================
# 2. SYSTEM CONSTANTS
# ============================================================

NUMBER_OF_MICROGRIDS = 5

EPISODE_LENGTH = 24

FORECAST_HORIZON = 24

NUMBER_OF_FORECAST_FEATURES = 4


# ============================================================
# 3. MANUSCRIPT MICROGRID PARAMETERS
# ============================================================

PV_CAPACITY_KW = np.asarray(
    [
        500.0,
        600.0,
        450.0,
        550.0,
        700.0,
    ],
    dtype=float,
)

BESS_CAPACITY_KWH = np.asarray(
    [
        1000.0,
        1200.0,
        900.0,
        1100.0,
        1400.0,
    ],
    dtype=float,
)

BESS_POWER_KW = np.asarray(
    [
        250.0,
        300.0,
        250.0,
        300.0,
        350.0,
    ],
    dtype=float,
)

EV_COUNTS = np.asarray(
    [
        200,
        250,
        180,
        220,
        300,
    ],
    dtype=int,
)

PEAK_LOAD_KW = np.asarray(
    [
        750.0,
        850.0,
        700.0,
        800.0,
        950.0,
    ],
    dtype=float,
)

TRANSFORMER_KVA = np.asarray(
    [
        1000.0,
        1250.0,
        1000.0,
        1250.0,
        1500.0,
    ],
    dtype=float,
)


# ============================================================
# 4. PHYSICAL / MARKET PARAMETERS
# ============================================================

BESS_CHARGING_EFFICIENCY = 0.95
BESS_DISCHARGING_EFFICIENCY = 0.95
BESS_SELF_DISCHARGE_RATE = 0.001

BESS_MINIMUM_SOC = 0.20
BESS_MAXIMUM_SOC = 0.95

# Reconstruction choice because manuscript does not provide
# one universal initial SOC.
INITIAL_SOC = 0.60


# ------------------------------------------------------------
# EV reconstruction parameters
# ------------------------------------------------------------

EV_CHARGER_POWER_KW = 7.2

# Reconstruction:
# maximum simultaneous fraction used for converting the ACN
# representative aggregate shape to manuscript-scale fleets.
EV_SIMULTANEOUS_FRACTION = 0.25


# ------------------------------------------------------------
# Manuscript market limits
# ------------------------------------------------------------

MINIMUM_BUY_PRICE = 0.12
MAXIMUM_BUY_PRICE = 0.32

MINIMUM_SELL_PRICE = 0.08
MAXIMUM_SELL_PRICE = 0.24

RESERVE_PRICE = 0.05


# ------------------------------------------------------------
# Physical sharing reconstruction
# ------------------------------------------------------------

SHARING_EFFICIENCY = 0.98

MAXIMUM_PAIRWISE_SHARING_KW = 250.0


# ------------------------------------------------------------
# Manuscript degradation coefficient
# ------------------------------------------------------------

BATTERY_DEGRADATION_COST = 0.02


# ------------------------------------------------------------
# Reconstruction because exact manuscript imbalance penalty
# coefficient is not recoverable.
# ------------------------------------------------------------

IMBALANCE_PENALTY_PER_KW = 1.0


# ============================================================
# 5. FIXED TRAINING-DATA SCALING
# ============================================================
#
# These values come from the TRAINING scaler.
#
# We deliberately do NOT normalize using the maximum value
# inside each 24-hour episode.
#
# Therefore episode-future information is not required to
# scale the physical state.
# ============================================================

TRAIN_LOAD_MIN = 0.164380
TRAIN_LOAD_MAX = 0.925029

TRAIN_EV_MIN = 0.0
TRAIN_EV_MAX = 42.707362

TRAIN_PRICE_MIN = 0.008954
TRAIN_PRICE_MAX = 0.056380


# ============================================================
# 6. UTILITY FUNCTIONS
# ============================================================

def section(title: str) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def safe_fraction(
    value: float,
    minimum: float,
    maximum: float,
) -> float:

    denominator = maximum - minimum

    if denominator <= 0.0:

        raise ValueError(
            "Scaling maximum must exceed minimum."
        )

    result = (
        float(value) - minimum
    ) / denominator

    return float(
        np.clip(
            result,
            0.0,
            1.0,
        )
    )


# ============================================================
# 7. PER-EV CHARGING REQUEST BUILDER
# ============================================================

def build_per_ev_charging_requests(
    aggregate_power_by_mg,
):

    aggregate_power_by_mg = np.asarray(
        aggregate_power_by_mg,
        dtype=float,
    )

    requests_by_microgrid = []

    for mg_index in range(
        NUMBER_OF_MICROGRIDS
    ):

        number_of_evs = int(
            EV_COUNTS[mg_index]
        )

        aggregate_request = max(
            float(
                aggregate_power_by_mg[
                    mg_index
                ]
            ),
            0.0,
        )

        maximum_power = (
            number_of_evs
            * EV_CHARGER_POWER_KW
        )

        aggregate_request = min(
            aggregate_request,
            maximum_power,
        )

        per_ev_request = (
            aggregate_request
            / number_of_evs
        )

        per_ev_request = min(
            per_ev_request,
            EV_CHARGER_POWER_KW,
        )

        requests = np.full(
            number_of_evs,
            per_ev_request,
            dtype=float,
        )

        requests_by_microgrid.append(
            requests
        )

    return requests_by_microgrid


# ============================================================
# 8. LOAD REAL RL TRAINING ARCHIVE
# ============================================================

class RealRLTrainingData:

    def __init__(
        self,
        archive_path: Path,
    ):

        if not archive_path.exists():

            raise FileNotFoundError(
                f"RL training archive not found:\n"
                f"{archive_path}"
            )

        archive = np.load(
            archive_path,
            allow_pickle=True,
        )

        required = [
            "current_actual_original",
            "current_actual_normalized",
            "forecast_original",
            "forecast_normalized",
            "predictive_state_matrix",
            "predictive_state_flat",
            "causal_confidence_24h",
            "training_indices",
        ]

        for key in required:

            if key not in archive.files:

                raise RuntimeError(
                    f"Missing RL archive key: {key}"
                )


        self.current_actual_original = (
            np.asarray(
                archive[
                    "current_actual_original"
                ],
                dtype=np.float32,
            )
        )

        self.current_actual_normalized = (
            np.asarray(
                archive[
                    "current_actual_normalized"
                ],
                dtype=np.float32,
            )
        )

        self.forecast_original = (
            np.asarray(
                archive[
                    "forecast_original"
                ],
                dtype=np.float32,
            )
        )

        self.forecast_normalized = (
            np.asarray(
                archive[
                    "forecast_normalized"
                ],
                dtype=np.float32,
            )
        )

        self.predictive_state_matrix = (
            np.asarray(
                archive[
                    "predictive_state_matrix"
                ],
                dtype=np.float32,
            )
        )

        self.predictive_state_flat = (
            np.asarray(
                archive[
                    "predictive_state_flat"
                ],
                dtype=np.float32,
            )
        )

        self.causal_confidence_24h = (
            np.asarray(
                archive[
                    "causal_confidence_24h"
                ],
                dtype=np.float32,
            )
        )

        self.training_indices = (
            np.asarray(
                archive[
                    "training_indices"
                ]
            )
        )


        self.number_of_samples = len(
            self.current_actual_original
        )


        self.validate()


    def validate(self):

        n = self.number_of_samples

        if self.current_actual_original.shape != (
            n,
            4,
        ):

            raise RuntimeError(
                "Unexpected current physical-data shape."
            )


        if self.predictive_state_matrix.shape != (
            n,
            24,
            4,
        ):

            raise RuntimeError(
                "Unexpected predictive-state shape."
            )


        if self.predictive_state_flat.shape != (
            n,
            96,
        ):

            raise RuntimeError(
                "Unexpected flattened predictive-state shape."
            )


        if self.causal_confidence_24h.shape != (
            24,
        ):

            raise RuntimeError(
                "Expected 24 causal-confidence values."
            )


        if n < EPISODE_LENGTH:

            raise RuntimeError(
                "Not enough data for one 24-hour episode."
            )


# ============================================================
# 9. REAL TRAINING EXOGENOUS PROVIDER
# ============================================================

class RealTrainingExogenousProvider:
    """
    Converts the leakage-free RL TRAIN archive into the
    physical inputs expected by the tested VPP bridge.

    Episode starting points are selected reproducibly from
    valid locations in the TRAIN split.

    No TEST data are loaded.
    """

    def __init__(
        self,
        data: RealRLTrainingData,
        maximum_episodes: int,
        seed: int,
    ):

        self.data = data

        self.maximum_episodes = int(
            maximum_episodes
        )

        self.seed = int(
            seed
        )


        # ----------------------------------------------------
        # Valid start positions.
        #
        # Each episode needs:
        #
        #   start ... start + 23
        #
        # ----------------------------------------------------

        maximum_start = (
            data.number_of_samples
            - EPISODE_LENGTH
        )

        self.valid_starts = np.arange(
            maximum_start + 1,
            dtype=int,
        )


        # ----------------------------------------------------
        # Deterministic random episode schedule.
        #
        # Repeated cycles are shuffled independently when more
        # episodes are requested than unique start positions.
        # ----------------------------------------------------

        rng = np.random.default_rng(
            self.seed
        )

        episode_starts = []

        while len(
            episode_starts
        ) < self.maximum_episodes:

            shuffled = rng.permutation(
                self.valid_starts
            )

            episode_starts.extend(
                shuffled.tolist()
            )


        self.episode_starts = np.asarray(
            episode_starts[
                :self.maximum_episodes
            ],
            dtype=int,
        )


    def get_sample_index(
        self,
        episode: int,
        step: int,
    ) -> int:

        # Training loop episodes are 1-based.

        episode_index = max(
            int(episode) - 1,
            0,
        )

        if episode_index >= len(
            self.episode_starts
        ):

            raise IndexError(
                "Episode exceeds prepared start schedule."
            )


        step = int(step)

        if not 0 <= step < EPISODE_LENGTH:

            raise IndexError(
                f"Invalid episode step: {step}"
            )


        start = int(
            self.episode_starts[
                episode_index
            ]
        )

        return start + step


    def __call__(
        self,
        episode: int,
        step: int,
    ) -> VPPExogenousInput:

        index = self.get_sample_index(
            episode,
            step,
        )


        actual = (
            self.data
            .current_actual_original[
                index
            ]
        )


        # Feature order:
        #
        # 0 = PV
        # 1 = Load
        # 2 = EV
        # 3 = Price

        pv_reference = float(
            actual[0]
        )

        load_reference = float(
            actual[1]
        )

        ev_reference = float(
            actual[2]
        )

        price_reference = float(
            actual[3]
        )


        # ====================================================
        # PV
        # ====================================================
        #
        # Reference PV signal was constructed as:
        #
        #   PV_1kW = GHI / 1000
        #
        # therefore:
        #
        #   GHI = PV_1kW * 1000
        #
        # Fixed physical conversion; no episode look-ahead.
        # ====================================================

        irradiance = float(
            np.clip(
                pv_reference * 1000.0,
                0.0,
                1000.0,
            )
        )


        irradiances = np.full(
            NUMBER_OF_MICROGRIDS,
            irradiance,
            dtype=float,
        )


        # ====================================================
        # LOAD
        # ====================================================

        load_shape = safe_fraction(
            load_reference,
            0.0,
            TRAIN_LOAD_MAX,
        )


        loads = (
            PEAK_LOAD_KW
            * load_shape
        )


        # ====================================================
        # EV
        # ====================================================

        ev_shape = safe_fraction(
            ev_reference,
            TRAIN_EV_MIN,
            TRAIN_EV_MAX,
        )


        maximum_ev_power = (
            EV_COUNTS.astype(float)
            * EV_CHARGER_POWER_KW
            * EV_SIMULTANEOUS_FRACTION
        )


        aggregate_ev_requests = (
            maximum_ev_power
            * ev_shape
        )


        per_ev_requests = (
            build_per_ev_charging_requests(
                aggregate_ev_requests
            )
        )


        # ====================================================
        # MARKET PRICE
        # ====================================================

        price_shape = safe_fraction(
            price_reference,
            TRAIN_PRICE_MIN,
            TRAIN_PRICE_MAX,
        )


        buy_price = (
            MINIMUM_BUY_PRICE
            +
            price_shape
            * (
                MAXIMUM_BUY_PRICE
                - MINIMUM_BUY_PRICE
            )
        )


        sell_price = (
            MINIMUM_SELL_PRICE
            +
            price_shape
            * (
                MAXIMUM_SELL_PRICE
                - MINIMUM_SELL_PRICE
            )
        )


        buy_prices = np.full(
            NUMBER_OF_MICROGRIDS,
            buy_price,
            dtype=float,
        )

        sell_prices = np.full(
            NUMBER_OF_MICROGRIDS,
            sell_price,
            dtype=float,
        )


        # ====================================================
        # PREDICTIVE STATE
        # ====================================================

        predictive_state = (
            self.data
            .predictive_state_matrix[
                index
            ]
            .copy()
        )


        # ====================================================
        # CAUSAL SCALAR CONFIDENCE
        # ====================================================
        #
        # Reconstruction choice:
        #
        # Use horizon-1 Phi for the immediate coordinator
        # action because every rolling decision acts on the
        # immediate next control interval.
        #
        # The complete 24-h confidence information remains
        # embedded in S_pred.
        # ====================================================

        scalar_confidence = float(
            self.data
            .causal_confidence_24h[0]
        )


        return VPPExogenousInput(

            time=
                float(step),

            loads_kw=
                loads,

            irradiances_w_m2=
                irradiances,

            buy_prices_usd_per_kwh=
                buy_prices,

            sell_prices_usd_per_kwh=
                sell_prices,

            predictive_state=
                predictive_state,

            confidence=
                scalar_confidence,

            # Coordinator lambda state:
            #
            # Keep the actual benchmark market-price signal,
            # rather than replacing it with the mapped retail
            # settlement tariff.
            market_price=
                price_reference,

            ev_requested_charging_powers_kw=
                per_ev_requests,
        )


# ============================================================
# 10. BUILD PHYSICAL MICROGRID
# ============================================================

def build_real_microgrid(
    index: int,
) -> Microgrid:

    mg_number = index + 1

    name = f"MG{mg_number}"


    # --------------------------------------------------------
    # PV
    # --------------------------------------------------------

    pv_parameters = PVParameters(

        rated_capacity_kw=
            float(
                PV_CAPACITY_KW[index]
            ),

        efficiency=
            1.0,

        critical_irradiance_w_m2=
            200.0,

        stc_irradiance_w_m2=
            1000.0,
    )


    pv_system = PhotovoltaicSystem(

        parameters=
            pv_parameters,

        name=
            f"{name}_PV",
    )


    # --------------------------------------------------------
    # BESS
    # --------------------------------------------------------

    bess_parameters = BESSParameters(

        capacity_kwh=
            float(
                BESS_CAPACITY_KWH[
                    index
                ]
            ),

        rated_power_kw=
            float(
                BESS_POWER_KW[
                    index
                ]
            ),

        charging_efficiency=
            BESS_CHARGING_EFFICIENCY,

        discharging_efficiency=
            BESS_DISCHARGING_EFFICIENCY,

        self_discharge_rate=
            BESS_SELF_DISCHARGE_RATE,

        minimum_soc=
            BESS_MINIMUM_SOC,

        maximum_soc=
            BESS_MAXIMUM_SOC,

        time_step_hours=
            1.0,
    )


    bess = BatteryEnergyStorageSystem(

        parameters=
            bess_parameters,

        initial_soc=
            INITIAL_SOC,

        name=
            f"{name}_BESS",
    )


    # --------------------------------------------------------
    # EV FLEET
    # --------------------------------------------------------

    number_of_evs = int(
        EV_COUNTS[index]
    )


    ev_parameters = EVFleetParameters(

        number_of_evs=
            number_of_evs,

        maximum_aggregate_charging_power_kw=
            (
                number_of_evs
                * EV_CHARGER_POWER_KW
            ),

        time_step_hours=
            1.0,
    )


    # Reconstruction for aggregate EV interface.
    #
    # ACN timing is represented in the aggregate time-varying
    # demand profile. The individual placeholder records keep
    # the tested EVFleet implementation operational.
    #
    # Do NOT describe these 0-24 availability records as
    # observed ACN individual sessions.

    ev_records = [

        EVRecord(

            arrival_time=
                0.0,

            departure_time=
                24.0,

            charging_power_kw=
                EV_CHARGER_POWER_KW,

            ev_id=
                f"{name}_EV_{j + 1:04d}",
        )

        for j in range(
            number_of_evs
        )
    ]


    ev_fleet = EVFleet(

        parameters=
            ev_parameters,

        vehicles=
            ev_records,

        name=
            f"{name}_EV_Fleet",
    )


    # --------------------------------------------------------
    # MARKET
    # --------------------------------------------------------

    market_parameters = MarketParameters(

        minimum_buy_price_usd_per_kwh=
            MINIMUM_BUY_PRICE,

        maximum_buy_price_usd_per_kwh=
            MAXIMUM_BUY_PRICE,

        minimum_sell_price_usd_per_kwh=
            MINIMUM_SELL_PRICE,

        maximum_sell_price_usd_per_kwh=
            MAXIMUM_SELL_PRICE,

        reserve_price_usd_per_kwh=
            RESERVE_PRICE,

        time_step_hours=
            1.0,
    )


    market = ElectricityMarket(

        parameters=
            market_parameters,

        name=
            f"{name}_Market",
    )


    # --------------------------------------------------------
    # MICROGRID PARAMETERS
    # --------------------------------------------------------

    parameters = MicrogridParameters(

        name=
            name,

        peak_load_kw=
            float(
                PEAK_LOAD_KW[index]
            ),

        transformer_rating_kva=
            float(
                TRANSFORMER_KVA[
                    index
                ]
            ),

        power_factor=
            1.0,

        balance_tolerance_kw=
            1e-6,
    )


    return Microgrid(

        parameters=
            parameters,

        pv_system=
            pv_system,

        bess=
            bess,

        ev_fleet=
            ev_fleet,

        market=
            market,
    )


# ============================================================
# 11. BUILD FIVE-MG VPP
# ============================================================

def build_real_vpp_environment():

    microgrids = [

        build_real_microgrid(i)

        for i in range(
            NUMBER_OF_MICROGRIDS
        )
    ]


    connectivity = np.ones(
        (
            NUMBER_OF_MICROGRIDS,
            NUMBER_OF_MICROGRIDS,
        ),
        dtype=int,
    )

    np.fill_diagonal(
        connectivity,
        0,
    )


    maximum_power = np.full(
        (
            NUMBER_OF_MICROGRIDS,
            NUMBER_OF_MICROGRIDS,
        ),
        MAXIMUM_PAIRWISE_SHARING_KW,
        dtype=float,
    )

    np.fill_diagonal(
        maximum_power,
        0.0,
    )


    sharing_parameters = EnergySharingParameters(

        number_of_microgrids=
            NUMBER_OF_MICROGRIDS,

        efficiency=
            SHARING_EFFICIENCY,

        tolerance_kw=
            1e-6,
    )


    sharing_network = EnergySharingNetwork(

        parameters=
            sharing_parameters,

        connectivity_matrix=
            connectivity,

        maximum_power_matrix_kw=
            maximum_power,

        name=
            "FC_HMARL_Real_Energy_Sharing",
    )


    environment = VPPEnvironment(

        microgrids=
            microgrids,

        energy_sharing_network=
            sharing_network,

        episode_length_hours=
            EPISODE_LENGTH,

        name=
            "FC_HMARL_Real_Data_VPP",
    )


    return environment


# ============================================================
# 12. BUILD FC-HMARL BRIDGE
# ============================================================

def build_real_training_bridge(
    data,
    maximum_episodes,
    seed,
):

    environment = (
        build_real_vpp_environment()
    )


    state_builder = HierarchicalStateBuilder(

        StateBuilderConfig(

            number_of_microgrids=
                NUMBER_OF_MICROGRIDS,

            forecast_horizon=
                FORECAST_HORIZON,

            forecast_features=
                NUMBER_OF_FORECAST_FEATURES,

            flatten_predictive_state=
                True,

            dtype=
                "float32",
        )
    )


    reward_builder = HierarchicalRewardBuilder(

        RewardConfig(

            beta_soc=
                1.0,

            beta_grid=
                1.0,

            risk_aversion=
                1.0,
        )
    )


    exogenous_provider = (
        RealTrainingExogenousProvider(

            data=
                data,

            maximum_episodes=
                maximum_episodes,

            seed=
                seed,
        )
    )


    action_mapper = FCHMARLActionMapper(

        ActionMapperConfig(

            use_dynamic_bess_feasibility=
                True,

            coordinator_controls_grid=
                False,

            coordinator_controls_reserve=
                True,

            coordinator_controls_sharing=
                True,

            reserve_fraction_of_bess_rating=
                1.0,

            action_tolerance=
                1e-8,
        )
    )


    bridge_config = VPPTrainingBridgeConfig(

        maximum_socs=
            np.full(
                NUMBER_OF_MICROGRIDS,
                BESS_MAXIMUM_SOC,
                dtype=float,
            ),

        maximum_grid_exchanges_kw=
            TRANSFORMER_KVA.copy(),

        battery_degradation_cost_per_kwh=
            BATTERY_DEGRADATION_COST,

        imbalance_penalty_per_kw=
            IMBALANCE_PENALTY_PER_KW,

        timestep_hours=
            1.0,

        number_of_microgrids=
            NUMBER_OF_MICROGRIDS,
    )


    bridge = FCHMARLVPPTrainingBridge(

        environment=
            environment,

        state_builder=
            state_builder,

        reward_builder=
            reward_builder,

        exogenous_provider=
            exogenous_provider,

        action_mapper=
            action_mapper,

        config=
            bridge_config,
    )


    return bridge


# ============================================================
# 12.5. LOAD COMPLETE TRAINING CHECKPOINT
# ============================================================

def load_complete_training_checkpoint(
    local_agents,
    coordinator_agent,
    checkpoint_directory,
    resume_episode,
    device,
):
    """
    Restore a complete FC-HMARL training checkpoint.

    Required files:
        5 local-agent .pt files
        5 local-agent replay .npz files
        1 coordinator .pt file
        1 coordinator replay .npz file
        1 global training-state .pt file

    Legacy checkpoints without replay sidecars are intentionally
    rejected for exact-resume training.
    """

    checkpoint_directory = Path(
        checkpoint_directory
    )

    resume_episode = int(
        resume_episode
    )

    if resume_episode <= 0:
        raise ValueError(
            "resume_episode must be positive."
        )

    required_files = []

    # ========================================================
    # LOCAL AGENTS
    # ========================================================

    local_checkpoint_paths = []

    for agent in local_agents:

        checkpoint_path = (
            checkpoint_directory
            / (
                f"local_agent_"
                f"{agent.microgrid_id}_"
                f"episode_{resume_episode}.pt"
            )
        )

        replay_path = checkpoint_path.with_suffix(
            ".replay.npz"
        )

        required_files.extend(
            [
                checkpoint_path,
                replay_path,
            ]
        )

        local_checkpoint_paths.append(
            checkpoint_path
        )

    # ========================================================
    # COORDINATOR
    # ========================================================

    coordinator_checkpoint_path = (
        checkpoint_directory
        / (
            "coordinator_"
            f"episode_{resume_episode}.pt"
        )
    )

    coordinator_replay_path = (
        coordinator_checkpoint_path.with_suffix(
            ".replay.npz"
        )
    )

    required_files.extend(
        [
            coordinator_checkpoint_path,
            coordinator_replay_path,
        ]
    )

    # ========================================================
    # GLOBAL TRAINING STATE
    # ========================================================

    training_state_path = (
        checkpoint_directory
        / (
            "training_state_"
            f"episode_{resume_episode}.pt"
        )
    )

    required_files.append(
        training_state_path
    )

    # ========================================================
    # VALIDATE ALL REQUIRED FILES
    # ========================================================

    missing_files = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing_files:

        message = [
            "",
            "Exact resume is not possible.",
            "",
            "The following checkpoint files are missing:",
        ]

        message.extend(
            [
                f"  {path}"
                for path in missing_files
            ]
        )

        message.extend(
            [
                "",
                "This usually means the checkpoint was created ",
                "before replay-buffer persistence was added.",
                "",
                "Legacy checkpoints may still be evaluated, but ",
                "they must not be presented as exact SAC resume ",
                "checkpoints.",
            ]
        )

        raise FileNotFoundError(
            "\n".join(message)
        )

    # ========================================================
    # LOAD LOCAL AGENTS
    # ========================================================

    for (
        agent,
        checkpoint_path,
    ) in zip(
        local_agents,
        local_checkpoint_paths,
    ):

        agent.load(
            checkpoint_path,
            load_optimizers=True,
        )

    # ========================================================
    # LOAD COORDINATOR
    # ========================================================

    coordinator_agent.load(
        coordinator_checkpoint_path,
        load_optimizers=True,
    )

    # ========================================================
    # LOAD GLOBAL TRAINING STATE
    # ========================================================

    training_state = torch.load(
        training_state_path,
        map_location=device,
        weights_only=False,
    )

    saved_episode = int(
        training_state.get(
            "episode",
            -1,
        )
    )

    if saved_episode != resume_episode:

        raise ValueError(
            "Training-state episode mismatch: "
            f"requested {resume_episode}, "
            f"file contains {saved_episode}."
        )

    saved_number_of_local_agents = int(
        training_state.get(
            "number_of_local_agents",
            len(local_agents),
        )
    )

    if saved_number_of_local_agents != len(
        local_agents
    ):

        raise ValueError(
            "Number of local agents in checkpoint "
            "does not match current configuration."
        )

    # ========================================================
    # RESTORE GLOBAL RNG STATES
    # ========================================================

    python_random_state = (
        training_state.get(
            "python_random_state",
            None,
        )
    )

    if python_random_state is not None:
        random.setstate(
            python_random_state
        )

    numpy_random_state = (
        training_state.get(
            "numpy_random_state",
            None,
        )
    )

    if numpy_random_state is not None:
        np.random.set_state(
            numpy_random_state
        )

    torch_cpu_rng_state = (
        training_state.get(
            "torch_cpu_rng_state",
            None,
        )
    )

    if torch_cpu_rng_state is not None:
        torch.set_rng_state(
            torch_cpu_rng_state
        )

    torch_cuda_rng_state_all = (
        training_state.get(
            "torch_cuda_rng_state_all",
            None,
        )
    )

    if (
        torch_cuda_rng_state_all is not None
        and torch.cuda.is_available()
    ):

        torch.cuda.set_rng_state_all(
            torch_cuda_rng_state_all
        )

    return training_state


# ============================================================
# 13. COMMAND-LINE ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(

        description=(
            "Train FC-HMARL using the leakage-free "
            "real-data training archive."
        )
    )


    parser.add_argument(

        "--smoke-test",

        action="store_true",

        help=(
            "Run a short validation training session."
        ),
    )


    parser.add_argument(

        "--episodes",

        type=int,

        default=None,

        help=(
            "Override number of training episodes."
        ),
    )


    parser.add_argument(

        "--seed",

        type=int,

        default=42,
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
    )


    parser.add_argument(
        "--resume-episode",
        type=int,
        default=None,
        help=(
            "Resume training from a complete checkpoint episode. "
            "Training continues from resume_episode + 1."
        ),
    )


    return parser.parse_args()


# ============================================================
# 14. MAIN
# ============================================================

def main():

    args = parse_arguments()


    section(
        "FC-HMARL STEP 7L-B - REAL-DATA TRAINING"
    )


    # --------------------------------------------------------
    # Number of episodes
    # --------------------------------------------------------

    if args.episodes is not None:

        maximum_episodes = int(
            args.episodes
        )

    elif args.smoke_test:

        # Two episodes are enough to exercise replay updates
        # with the smoke-test batch size.

        maximum_episodes = 2

    else:

        # Manuscript setting.
        maximum_episodes = 5000


    if maximum_episodes <= 0:

        raise ValueError(
            "Episodes must be positive."
        )


    device = resolve_device(
        args.device
    )


    set_global_seed(
        args.seed
    )


    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )


    print(
        f"Mode                : "
        f"{'SMOKE TEST' if args.smoke_test else 'FULL'}"
    )

    print(
        f"Episodes            : "
        f"{maximum_episodes}"
    )

    print(
        f"Steps per episode   : "
        f"{EPISODE_LENGTH}"
    )

    print(
        f"Seed                : "
        f"{args.seed}"
    )

    print(
        f"Device              : "
        f"{device}"
    )


    # ========================================================
    # LOAD TRAINING ARCHIVE
    # ========================================================

    section(
        "LOADING LEAKAGE-FREE RL TRAINING ARCHIVE"
    )


    data = RealRLTrainingData(
        RL_ARCHIVE_FILE
    )


    print(
        f"Training samples           : "
        f"{data.number_of_samples}"
    )

    print(
        f"Predictive-state matrix    : "
        f"{data.predictive_state_matrix.shape}"
    )

    print(
        f"Predictive-state flat      : "
        f"{data.predictive_state_flat.shape}"
    )

    print(
        f"Causal confidence          : "
        f"{data.causal_confidence_24h.shape}"
    )

    print(
        f"Lead-1 causal Phi          : "
        f"{data.causal_confidence_24h[0]:.8f}"
    )

    print(
        f"Mean 24-h causal Phi       : "
        f"{data.causal_confidence_24h.mean():.8f}"
    )


    # ========================================================
    # BUILD REAL PHYSICAL BRIDGE
    # ========================================================

    section(
        "BUILDING REAL-DATA PHYSICAL VPP BRIDGE"
    )


    bridge = build_real_training_bridge(

        data=
            data,

        maximum_episodes=
            maximum_episodes,

        seed=
            args.seed,
    )


    # Validate observation once before training.

    initial_observation = bridge.reset(
        episode=1
    )


    print(
        f"Local agents/states        : "
        f"{len(initial_observation.local_states)}"
    )


    for i, local_state in enumerate(
        initial_observation.local_states,
        start=1,
    ):

        print(
            f"MG{i} state dimension      : "
            f"{np.asarray(local_state).shape}"
        )


    print(
        f"Coordinator dimension      : "
        f"{np.asarray(initial_observation.coordinator_state).shape}"
    )


    # Reset again when trainer begins episode 1.
    bridge.environment.reset()


    # ========================================================
    # BUILD AGENTS
    # ========================================================

    section(
        "BUILDING FC-HMARL SAC AGENTS"
    )


    local_agents = build_local_agents(

        seed=
            args.seed,

        smoke_test=
            args.smoke_test,

        device=
            device,
    )


    coordinator_agent = (
        build_coordinator_agent(

            seed=
                args.seed,

            smoke_test=
                args.smoke_test,

            device=
                device,
        )
    )


    print(
        f"Local SAC agents           : "
        f"{len(local_agents)}"
    )

    print(
        "Coordinator SAC agents     : 1"
    )


    # ========================================================
    # RESUME CHECKPOINT IF REQUESTED
    # ========================================================

    resume_episode = (
        args.resume_episode
    )

    start_episode = 1


    if resume_episode is not None:

        resume_episode = int(
            resume_episode
        )

        if args.smoke_test:

            raise ValueError(
                "--resume-episode cannot be used "
                "together with --smoke-test."
            )

        if resume_episode >= maximum_episodes:

            raise ValueError(
                "resume_episode must be smaller than "
                "the target --episodes value."
            )


        section(
            "RESTORING COMPLETE FC-HMARL CHECKPOINT"
        )


        training_state = (
            load_complete_training_checkpoint(

                local_agents=
                    local_agents,

                coordinator_agent=
                    coordinator_agent,

                checkpoint_directory=
                    CHECKPOINT_DIRECTORY,

                resume_episode=
                    resume_episode,

                device=
                    device,
            )
        )


        start_episode = (
            resume_episode
            + 1
        )


        print(
            f"Resume checkpoint      : "
            f"episode {resume_episode}"
        )

        print(
            f"Next episode           : "
            f"{start_episode}"
        )

        print(
            f"Target episode         : "
            f"{maximum_episodes}"
        )

        print(
            "[OK] Actor states restored."
        )

        print(
            "[OK] Critic states restored."
        )

        print(
            "[OK] Target critics restored."
        )

        print(
            "[OK] Optimizer states restored."
        )

        print(
            "[OK] Replay buffers restored."
        )

        print(
            "[OK] Exploration/counters restored."
        )

        print(
            "[OK] Global RNG state restored."
        )


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    checkpoint_interval = (
        100
    )


    training_config = TrainingLoopConfig(

        maximum_episodes=
            maximum_episodes,

        steps_per_episode=
            EPISODE_LENGTH,

        updates_per_step=
            1,

        checkpoint_interval=
            checkpoint_interval,

        enable_checkpointing=
            not args.smoke_test,

        checkpoint_directory=
            str(
                CHECKPOINT_DIRECTORY
            ),

        deterministic_actions=
            False,

        verbose=
            True,
    )


    trainer = FCHMARLTrainingLoop(

        local_agents=
            local_agents,

        coordinator_agent=
            coordinator_agent,

        reset_function=
            make_reset_function(
                bridge
            ),

        step_function=
            make_step_function(
                bridge
            ),

        config=
            training_config,

        # Manuscript states training proceeds until
        # convergence but gives no numerical threshold.
        #
        # Therefore no invented early stopping rule.
        stop_function=
            None,
    )


    # ========================================================
    # TRAIN
    # ========================================================

    section(
        "STARTING REAL-DATA FC-HMARL TRAINING"
    )


    history = trainer.train(
        start_episode=start_episode
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    section(
        "SAVING REAL-DATA TRAINING RESULTS"
    )


    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


    result_paths = export_training_results(

        history,

        output_directory=
            OUTPUT_DIRECTORY,

        moving_average_window=
            20,
    )


    print(
        "Exported training outputs:"
    )


    for name, path in (
        result_paths.items()
    ):

        print(
            f"  {name}: {path}"
        )


    # ========================================================
    # TRAINER SUMMARY
    # ========================================================

    trainer_summary = trainer.summary()


    summary = {

        "mode":
            (
                "smoke_test"
                if args.smoke_test
                else "full_training"
            ),

        "requested_episodes":
            maximum_episodes,

        "completed_episodes":
            len(
                history.episodes
            ),

        "steps_per_episode":
            EPISODE_LENGTH,

        "training_samples_available":
            data.number_of_samples,

        "seed":
            args.seed,

        "device":
            device,

        "local_state_dimension":
            101,

        "local_action_dimension":
            5,

        "coordinator_state_dimension":
            99,

        "coordinator_action_dimension":
            3,

        "predictive_state_dimension":
            96,

        "lead_1_causal_confidence":
            float(
                data.causal_confidence_24h[
                    0
                ]
            ),

        "mean_24h_causal_confidence":
            float(
                data.causal_confidence_24h.mean()
            ),

        "training_split":
            "train",

        "forecast_selection_split":
            "validation",

        "confidence_calibration_split":
            "validation",

        "test_data_used_during_training":
            False,

        "battery_degradation_cost_usd_per_kwh":
            BATTERY_DEGRADATION_COST,

        "initial_soc":
            INITIAL_SOC,

        "ev_charger_power_kw":
            EV_CHARGER_POWER_KW,

        "ev_simultaneous_fraction":
            EV_SIMULTANEOUS_FRACTION,

        "pairwise_sharing_capacity_kw":
            MAXIMUM_PAIRWISE_SHARING_KW,

        "resume_episode":
            (
                int(resume_episode)
                if resume_episode is not None
                else None
            ),

        "start_episode":
            int(start_episode),

        "exact_resume":
            bool(
                resume_episode is not None
            ),

        "notes": [

            (
                "Initial SOC is a reconstruction choice."
            ),

            (
                "EV per-vehicle availability records are "
                "interface reconstruction records; the "
                "time-varying aggregate demand originates "
                "from the ACN-derived training profile."
            ),

            (
                "EV simultaneous fraction is a "
                "reconstruction choice."
            ),

            (
                "Pairwise sharing capacity is a "
                "reconstruction choice."
            ),

            (
                "Lead-1 Phi is used as the scalar "
                "coordinator risk confidence at each "
                "rolling decision."
            ),

            (
                "Full confidence-aware 24-hour forecast "
                "remains embedded in the 96-dimensional "
                "predictive state."
            ),
        ],

        "trainer_summary":
            trainer_summary,
    }


    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            default=str,
        )


    # ========================================================
    # FINAL CONSOLE SUMMARY
    # ========================================================

    section(
        "STEP 7L-B TRAINING COMPLETE"
    )


    print(
        f"Completed episodes : "
        f"{len(history.episodes)}"
    )


    if len(history.episodes) > 0:

        returns = np.asarray(
            [
                episode.total_return

                for episode
                in history.episodes
            ],
            dtype=float,
        )


        print(
            f"First return       : "
            f"{returns[0]:.6f}"
        )

        print(
            f"Final return       : "
            f"{returns[-1]:.6f}"
        )

        print(
            f"Mean return        : "
            f"{returns.mean():.6f}"
        )


        if len(returns) >= 2:

            print(
                f"Return std         : "
                f"{returns.std():.6f}"
            )


    print()

    print(
        "[OK] Training used the RL TRAIN archive."
    )

    print(
        "[OK] Forecast methods were selected "
        "using VALIDATION."
    )

    print(
        "[OK] Causal Phi was calibrated "
        "using VALIDATION."
    )

    print(
        "[OK] Final TEST split was not loaded "
        "during FC-HMARL training."
    )

    print(
        "[OK] Real forecast-derived S_pred "
        "entered every hierarchical state."
    )

    print(
        "[OK] Five manuscript-scale microgrids "
        "were used."
    )


    print()

    print(
        f"Results directory:\n"
        f"{OUTPUT_DIRECTORY}"
    )

    print()

    print(
        f"Training summary:\n"
        f"{SUMMARY_FILE}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()