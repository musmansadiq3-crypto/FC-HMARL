from pathlib import Path
import numpy as np
import pandas as pd
# ------------------------------------------------------------
# PHYSICAL ENVIRONMENT
# ------------------------------------------------------------

from environment.vpp_env import VPPEnvironment

from environment.microgrid import (
    Microgrid,
    MicrogridParameters,
)

from environment.bess import (
    BESSParameters,
    BatteryEnergyStorageSystem,
)

from environment.ev_fleet import (
    EVFleet,
    EVFleetParameters,
    EVRecord,
)

from environment.pv import (
    PVParameters,
    PhotovoltaicSystem,
)

from environment.market import (
    MarketParameters,
    ElectricityMarket,
)

from environment.energy_sharing import (
    EnergySharingParameters,
    EnergySharingNetwork,
)

# ------------------------------------------------------------
# FC-HMARL
# ------------------------------------------------------------

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

from marl.training_loop import (
    HierarchicalActionBundle,
)

from marl.vpp_training_bridge import (
    VPPExogenousInput,
    VPPTrainingBridgeConfig,
    FCHMARLVPPTrainingBridge,
)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"D:\Molvi paper review\FC_HMARL"
)

PREDICTIVE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "forecasting"
    / "predictive_state"
    / "confidence_aware_predictive_state.npz"
)

FORECAST_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "forecasting"
    / "causal_confidence"
    / "final_leakage_free_test_forecast.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "integration"
    / "real_vpp_bridge"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

HOURLY_OUTPUT = (
    OUTPUT_DIR
    / "real_vpp_bridge_24h.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "real_vpp_bridge_summary.csv"
)


# ============================================================
# 2. BASIC CONSTANTS
# ============================================================

NUMBER_OF_MICROGRIDS = 5
EPISODE_LENGTH = 24

FORECAST_HORIZON = 24
FORECAST_FEATURES = 4

EXPECTED_LOCAL_DIM = 101
EXPECTED_COORDINATOR_DIM = 99


# ============================================================
# 3. MANUSCRIPT MICROGRID PARAMETERS
# ============================================================

PV_CAPACITY_KW = np.array(
    [500, 600, 450, 550, 700],
    dtype=float,
)

BESS_CAPACITY_KWH = np.array(
    [1000, 1200, 900, 1100, 1400],
    dtype=float,
)

BESS_POWER_KW = np.array(
    [250, 300, 250, 300, 350],
    dtype=float,
)

EV_COUNTS = np.array(
    [200, 250, 180, 220, 300],
    dtype=int,
)

PEAK_LOAD_KW = np.array(
    [750, 850, 700, 800, 950],
    dtype=float,
)

TRANSFORMER_KVA = np.array(
    [1000, 1250, 1000, 1250, 1500],
    dtype=float,
)


# ============================================================
# 4. MANUSCRIPT PHYSICAL PARAMETERS
# ============================================================

BESS_CHARGING_EFFICIENCY = 0.95
BESS_DISCHARGING_EFFICIENCY = 0.95

BESS_SELF_DISCHARGE = 0.001

MINIMUM_SOC = 0.20
MAXIMUM_SOC = 0.95

SHARING_EFFICIENCY = 0.98

BATTERY_DEGRADATION_COST = 0.02

BUY_PRICE_MIN = 0.12
BUY_PRICE_MAX = 0.32

SELL_PRICE_MIN = 0.08
SELL_PRICE_MAX = 0.24

RESERVE_PRICE = 0.05


# ============================================================
# 5. EXPLICIT RECONSTRUCTION CHOICES
# ============================================================

# Manuscript does not recover universal initial SOC.
INITIAL_BESS_SOC = 0.60

# Typical Level-2 charging-power assumption.
# This is NOT claimed as manuscript-specified.
EV_CHARGER_POWER_KW = 7.2

# Reconstruction:
# assumed maximum simultaneous charging fraction used to scale
# the representative ACN temporal profile.
EV_SIMULTANEOUS_FRACTION = 0.25

# Reconstruction:
# fully connected MG sharing network.
SHARING_LIMIT_KW = 250.0

# Reconstruction:
# imbalance penalty used by bridge reward.
IMBALANCE_PENALTY_PER_KW = 1.0


# ============================================================
# 6. HELPERS
# ============================================================

def section(title):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def safe_normalize(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    minimum = float(
        np.min(values)
    )

    maximum = float(
        np.max(values)
    )

    if maximum - minimum < 1e-12:

        return np.zeros_like(
            values
        )

    return (
        values - minimum
    ) / (
        maximum - minimum
    )


def safe_positive_peak_normalize(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = np.clip(
        values,
        0.0,
        None,
    )

    maximum = float(
        np.max(values)
    )

    if maximum <= 1e-12:

        return np.zeros_like(
            values
        )

    return values / maximum


def build_per_ev_charging_requests(
    aggregate_power_by_mg,
    ev_counts,
    charger_power_kw,
):
    """
    Convert one aggregate EV charging request per microgrid
    into one charging-power request per EV.

    The aggregate requested power is preserved as closely as
    possible while respecting the per-EV charger-power limit.
    """

    aggregate_power_by_mg = np.asarray(
        aggregate_power_by_mg,
        dtype=float,
    )

    all_requests = []

    for mg_index, number_of_evs in enumerate(ev_counts):

        number_of_evs = int(number_of_evs)

        aggregate_request = float(
            aggregate_power_by_mg[mg_index]
        )

        aggregate_request = max(
            aggregate_request,
            0.0,
        )

        maximum_fleet_power = (
            number_of_evs
            * charger_power_kw
        )

        aggregate_request = min(
            aggregate_request,
            maximum_fleet_power,
        )

        # Equal allocation across all EVs.
        per_ev_power = (
            aggregate_request
            / number_of_evs
        )

        per_ev_power = min(
            per_ev_power,
            charger_power_kw,
        )

        requests = np.full(
            number_of_evs,
            per_ev_power,
            dtype=float,
        )

        all_requests.append(
            requests
        )

    return all_requests


# ============================================================
# 7. LOAD FORECAST / PREDICTIVE DATA
# ============================================================

section(
    "FC-HMARL STEP 7K - LOADING REAL DATA"
)

if not PREDICTIVE_FILE.exists():

    raise FileNotFoundError(
        f"Predictive-state file not found:\n"
        f"{PREDICTIVE_FILE}"
    )

if not FORECAST_FILE.exists():

    raise FileNotFoundError(
        f"Leakage-free forecast file not found:\n"
        f"{FORECAST_FILE}"
    )


pred_archive = np.load(
    PREDICTIVE_FILE,
    allow_pickle=True,
)

forecast_archive = np.load(
    FORECAST_FILE,
    allow_pickle=True,
)


print(
    "Predictive-state keys:"
)

for key in pred_archive.files:

    print(
        f"  {key}"
    )


print()

print(
    "Forecast archive keys:"
)

for key in forecast_archive.files:

    print(
        f"  {key}"
    )


predictive_states = np.asarray(
    pred_archive[
        "predictive_state_matrix"
    ],
    dtype=np.float32,
)

causal_confidence_24h = np.asarray(
    pred_archive[
        "causal_confidence_24h"
    ],
    dtype=float,
)


# ------------------------------------------------------------
# Actual physical values.
# ------------------------------------------------------------

if "actual_original" not in forecast_archive.files:

    raise KeyError(
        "final_leakage_free_test_forecast.npz "
        "does not contain 'actual_original'."
    )


actual_original = np.asarray(
    forecast_archive[
        "actual_original"
    ],
    dtype=float,
)


print()

print(
    f"Predictive states : "
    f"{predictive_states.shape}"
)

print(
    f"Actual test data  : "
    f"{actual_original.shape}"
)

print(
    f"Confidence        : "
    f"{causal_confidence_24h.shape}"
)


if predictive_states.shape[1:] != (
    24,
    4,
):

    raise RuntimeError(
        "Predictive state must have shape "
        "(samples, 24, 4)."
    )


if actual_original.shape[1:] != (
    24,
    4,
):

    raise RuntimeError(
        "Actual archive must have shape "
        "(samples, 24, 4)."
    )


if len(predictive_states) < 24:

    raise RuntimeError(
        "At least 24 rolling forecast samples "
        "are required."
    )

section(
    "BUILDING ROLLING 24-HOUR REAL-DATA EPISODE"
)


episode_actual = np.stack(
    [
        actual_original[
            t,
            0,
            :
        ]
        for t in range(
            EPISODE_LENGTH
        )
    ],
    axis=0,
)


episode_predictive = predictive_states[
    :EPISODE_LENGTH
]


print(
    f"Episode actual shape     : "
    f"{episode_actual.shape}"
)

print(
    f"Episode predictive shape : "
    f"{episode_predictive.shape}"
)


# Feature ordering established in forecasting pipeline:
#
# 0 PV
# 1 Load
# 2 EV
# 3 Price

pv_reference = episode_actual[
    :,
    0
]

load_reference = episode_actual[
    :,
    1
]

ev_reference = episode_actual[
    :,
    2
]

price_reference = episode_actual[
    :,
    3
]


# ============================================================
# 9. PHYSICAL PROFILE SCALING
# ============================================================

section(
    "SCALING BENCHMARK PROFILES TO FIVE MICROGRIDS"
)


# ------------------------------------------------------------
# PV
# ------------------------------------------------------------
#
# Forecast preprocessing defined:
#
#   PV_ref_kW = clip(GHI / 1000, 0, 1)
#
# Therefore:
#
#   reconstructed irradiance = PV_ref * 1000 W/m2
#
# The PV system itself then scales generation using the
# manuscript-installed PV capacities.
# ------------------------------------------------------------

irradiance_profile = np.clip(
    pv_reference * 1000.0,
    0.0,
    1000.0,
)


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------
#
# Pecan profile is household-scale.
#
# Reconstruction:
# preserve its temporal shape and scale its maximum to each
# manuscript MG peak-load rating.
# ------------------------------------------------------------

load_shape = safe_positive_peak_normalize(
    load_reference
)


loads_by_hour = (
    load_shape[:, None]
    *
    PEAK_LOAD_KW[None, :]
)


# ------------------------------------------------------------
# EV
# ------------------------------------------------------------
#
# Preserve ACN temporal shape.
#
# Reconstruction:
#
# peak EV demand_i
#   =
# EV count_i
# x charger power
# x simultaneous fraction
# ------------------------------------------------------------

ev_shape = safe_positive_peak_normalize(
    ev_reference
)


EV_PROFILE_PEAK_KW = (
    EV_COUNTS.astype(float)
    *
    EV_CHARGER_POWER_KW
    *
    EV_SIMULTANEOUS_FRACTION
)


ev_by_hour = (
    ev_shape[:, None]
    *
    EV_PROFILE_PEAK_KW[None, :]
)


# ------------------------------------------------------------
# MARKET PRICE
# ------------------------------------------------------------
#
# PJM data provide real temporal price variation but the
# manuscript specifies a different market-price range.
#
# Reconstruction:
# preserve PJM relative temporal shape and map it into the
# manuscript buy/sell ranges.
# ------------------------------------------------------------

price_shape = safe_normalize(
    price_reference
)


buy_price_profile = (
    BUY_PRICE_MIN
    +
    price_shape
    *
    (
        BUY_PRICE_MAX
        -
        BUY_PRICE_MIN
    )
)


sell_price_profile = (
    SELL_PRICE_MIN
    +
    price_shape
    *
    (
        SELL_PRICE_MAX
        -
        SELL_PRICE_MIN
    )
)


# Same market signal is used across the five interconnected MGs.

buy_prices_by_hour = np.repeat(
    buy_price_profile[:, None],
    NUMBER_OF_MICROGRIDS,
    axis=1,
)


sell_prices_by_hour = np.repeat(
    sell_price_profile[:, None],
    NUMBER_OF_MICROGRIDS,
    axis=1,
)


# ------------------------------------------------------------
# Scalar confidence for coordinator risk.
# ------------------------------------------------------------
#
# Manuscript risk equation uses scalar Phi:
#
#     C_risk = rho(1 - Phi)
#
# Our calibrated Phi is horizon-specific (24 values).
#
# Reconstruction for current bridge:
# use mean causal 24-hour confidence as the scalar confidence
# while retaining the full horizon-wise Phi inside S_pred.
# ------------------------------------------------------------

SCALAR_CAUSAL_CONFIDENCE = float(
    np.mean(
        causal_confidence_24h
    )
)


print(
    f"Load shape range         : "
    f"{load_shape.min():.6f} -> "
    f"{load_shape.max():.6f}"
)

print(
    f"EV shape range           : "
    f"{ev_shape.min():.6f} -> "
    f"{ev_shape.max():.6f}"
)

print(
    f"Irradiance range         : "
    f"{irradiance_profile.min():.3f} -> "
    f"{irradiance_profile.max():.3f} W/m2"
)

print(
    f"Buy-price range          : "
    f"{buy_price_profile.min():.6f} -> "
    f"{buy_price_profile.max():.6f} USD/kWh"
)

print(
    f"Sell-price range         : "
    f"{sell_price_profile.min():.6f} -> "
    f"{sell_price_profile.max():.6f} USD/kWh"
)

print(
    f"Scalar causal confidence : "
    f"{SCALAR_CAUSAL_CONFIDENCE:.8f}"
)


# ============================================================
# 10. BUILD FIVE PHYSICAL MICROGRIDS
# ============================================================

section(
    "BUILDING FIVE-MICROGRID PHYSICAL VPP"
)


microgrids = []


for i in range(
    NUMBER_OF_MICROGRIDS
):

    # --------------------------------------------------------
    # PV
    # --------------------------------------------------------

    pv_parameters = PVParameters(

        rated_capacity_kw=
            float(
                PV_CAPACITY_KW[i]
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
            f"MG{i + 1}_PV",
    )


    # --------------------------------------------------------
    # BESS
    # --------------------------------------------------------

    bess_parameters = BESSParameters(

        capacity_kwh=
            float(
                BESS_CAPACITY_KWH[i]
            ),

        rated_power_kw=
            float(
                BESS_POWER_KW[i]
            ),

        charging_efficiency=
            BESS_CHARGING_EFFICIENCY,

        discharging_efficiency=
            BESS_DISCHARGING_EFFICIENCY,

        self_discharge_rate=
            BESS_SELF_DISCHARGE,

        minimum_soc=
            MINIMUM_SOC,

        maximum_soc=
            MAXIMUM_SOC,

        time_step_hours=
            1.0,
    )


    bess = BatteryEnergyStorageSystem(

        parameters=
            bess_parameters,

        initial_soc=
            INITIAL_BESS_SOC,

        name=
            f"MG{i + 1}_BESS",
    )


    # --------------------------------------------------------
    # EV fleet
    # --------------------------------------------------------

    ev_maximum_power = float(

        EV_COUNTS[i]
        *
        EV_CHARGER_POWER_KW
    )


    ev_parameters = EVFleetParameters(

        number_of_evs=
            int(
                EV_COUNTS[i]
            ),

        maximum_aggregate_charging_power_kw=
            ev_maximum_power,

        time_step_hours=
            1.0,
    )


    # --------------------------------------------------------
    # EV RECORD INITIALIZATION
    # --------------------------------------------------------
    #
    # EVFleet.step() requires explicit EVRecord objects.
    #
    # For this bridge-validation stage, we create a complete
    # fleet matching the manuscript EV counts.
    #
    # The hourly aggregate charging request is still supplied
    # externally through:
    #
    # ev_requested_charging_powers_kw
    #
    # Therefore these EV records provide the fleet structure
    # required by the physical environment.
    #
    # Arrival/departure timing here is a reconstruction choice
    # for interface validation and is NOT claimed as recovered
    # manuscript data.
    # --------------------------------------------------------

    ev_records = [

        EVRecord(

            arrival_time=0.0,

            departure_time=24.0,

            charging_power_kw=
                EV_CHARGER_POWER_KW,

            ev_id=
                f"MG{i + 1}_EV_{j + 1:04d}",
        )

        for j in range(
            int(
                EV_COUNTS[i]
            )
        )
    ]


    ev_fleet = EVFleet(

        parameters=
            ev_parameters,

        vehicles=
            ev_records,

        name=
            f"MG{i + 1}_EV",
    )


    # --------------------------------------------------------
    # Market
    # --------------------------------------------------------

    market_parameters = MarketParameters(

        minimum_buy_price_usd_per_kwh=
            BUY_PRICE_MIN,

        maximum_buy_price_usd_per_kwh=
            BUY_PRICE_MAX,

        minimum_sell_price_usd_per_kwh=
            SELL_PRICE_MIN,

        maximum_sell_price_usd_per_kwh=
            SELL_PRICE_MAX,

        reserve_price_usd_per_kwh=
            RESERVE_PRICE,

        time_step_hours=
            1.0,
    )


    market = ElectricityMarket(

        parameters=
            market_parameters,

        name=
            f"MG{i + 1}_Market",
    )


    # --------------------------------------------------------
    # MG
    # --------------------------------------------------------

    microgrid_parameters = MicrogridParameters(

        name=
            f"MG{i + 1}",

        peak_load_kw=
            float(
                PEAK_LOAD_KW[i]
            ),

        transformer_rating_kva=
            float(
                TRANSFORMER_KVA[i]
            ),

        power_factor=
            1.0,

        balance_tolerance_kw=
            1e-6,
    )


    microgrid = Microgrid(

        parameters=
            microgrid_parameters,

        pv_system=
            pv_system,

        bess=
            bess,

        ev_fleet=
            ev_fleet,

        market=
            market,
    )


    microgrids.append(
        microgrid
    )


    print(
        f"MG{i + 1}: "
        f"PV={PV_CAPACITY_KW[i]:.0f} kW | "
        f"BESS={BESS_CAPACITY_KWH[i]:.0f} kWh/"
        f"{BESS_POWER_KW[i]:.0f} kW | "
        f"EVs={EV_COUNTS[i]} | "
        f"Peak Load={PEAK_LOAD_KW[i]:.0f} kW | "
        f"Transformer={TRANSFORMER_KVA[i]:.0f} kVA"
    )


# ============================================================
# 11. BUILD ENERGY SHARING NETWORK
# ============================================================

connectivity_matrix = np.ones(
    (
        NUMBER_OF_MICROGRIDS,
        NUMBER_OF_MICROGRIDS,
    ),
    dtype=float,
)


np.fill_diagonal(
    connectivity_matrix,
    0.0,
)


maximum_power_matrix_kw = np.full(
    (
        NUMBER_OF_MICROGRIDS,
        NUMBER_OF_MICROGRIDS,
    ),
    SHARING_LIMIT_KW,
    dtype=float,
)


np.fill_diagonal(
    maximum_power_matrix_kw,
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
        connectivity_matrix,

    maximum_power_matrix_kw=
        maximum_power_matrix_kw,

    name=
        "FC_HMARL_Real_Data_Sharing",
)


# ============================================================
# 12. BUILD VPP ENVIRONMENT
# ============================================================

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


print()

print(
    "[OK] Five-MG physical VPP constructed."
)


# ============================================================
# 13. REAL EXOGENOUS PROVIDER
# ============================================================

class RealDataExogenousProvider:

    def __init__(self):

        self.current_step = 0
        self.episode = 0


    def reset(self, episode=0):

        self.current_step = 0
        self.episode = int(
            episode
        )


    def __call__(self, *args, **kwargs):

        # ----------------------------------------------------
        # Bridge implementations may supply no argument,
        # a time index, or episode/time values.
        #
        # We retain our own current_step so the provider is
        # robust to those interface variants.
        # ----------------------------------------------------

        if self.current_step >= EPISODE_LENGTH:

            index = (
                EPISODE_LENGTH - 1
            )

        else:

            index = self.current_step


        exogenous = VPPExogenousInput(

            time=
                float(
                    index
                ),

            loads_kw=
                loads_by_hour[
                    index
                ].copy(),

            irradiances_w_m2=
                np.repeat(
                    irradiance_profile[
                        index
                    ],
                    NUMBER_OF_MICROGRIDS,
                ),

            buy_prices_usd_per_kwh=
                buy_prices_by_hour[
                    index
                ].copy(),

            sell_prices_usd_per_kwh=
                sell_prices_by_hour[
                    index
                ].copy(),

            predictive_state=
                episode_predictive[
                    index
                ].copy(),

            confidence=
                SCALAR_CAUSAL_CONFIDENCE,

            market_price=
                float(
                    buy_price_profile[
                        index
                    ]
                ),

            ev_requested_charging_powers_kw=
                build_per_ev_charging_requests(
                    aggregate_power_by_mg=
                        ev_by_hour[index],

                    ev_counts=
                        EV_COUNTS,

                    charger_power_kw=
                        EV_CHARGER_POWER_KW,
                ),
        )


        self.current_step += 1

        return exogenous


exogenous_provider = (
    RealDataExogenousProvider()
)


# ============================================================
# 14. STATE BUILDER
# ============================================================

state_builder = HierarchicalStateBuilder(

    StateBuilderConfig(

        number_of_microgrids=
            NUMBER_OF_MICROGRIDS,

        forecast_horizon=
            FORECAST_HORIZON,

        forecast_features=
            FORECAST_FEATURES,

        flatten_predictive_state=
            True,

        dtype=
            "float32",
    )
)


# ============================================================
# 15. REWARD BUILDER
# ============================================================
#
# beta values remain implementation-level reconstruction
# defaults because exact coefficients were not recovered.
# ============================================================

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


# ============================================================
# 16. ACTION MAPPER
# ============================================================

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


# ============================================================
# 17. TRAINING BRIDGE
# ============================================================

bridge_config = VPPTrainingBridgeConfig(

    maximum_socs=
        [MAXIMUM_SOC]
        * NUMBER_OF_MICROGRIDS,

    maximum_grid_exchanges_kw=
        TRANSFORMER_KVA.tolist(),

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


# ============================================================
# 18. RESET BRIDGE
# ============================================================

section(
    "RESETTING REAL-DATA VPP TRAINING BRIDGE"
)


# Explicit reset so provider and bridge start together.

exogenous_provider.reset(
    episode=0
)


observation = bridge.reset(
    episode=0
)


print(
    f"Number of local states : "
    f"{len(observation.local_states)}"
)


for i, state in enumerate(
    observation.local_states
):

    state = np.asarray(
        state
    )

    print(
        f"MG{i + 1} state shape     : "
        f"{state.shape}"
    )

    if state.shape != (
        EXPECTED_LOCAL_DIM,
    ):

        raise RuntimeError(
            f"MG{i + 1} state dimension mismatch."
        )


coordinator_state = np.asarray(
    observation.coordinator_state
)


print(
    f"Coordinator shape       : "
    f"{coordinator_state.shape}"
)


if coordinator_state.shape != (
    EXPECTED_COORDINATOR_DIM,
):

    raise RuntimeError(
        "Coordinator state dimension mismatch."
    )


print()

print(
    "[OK] Initial real-data FC-HMARL "
    "observation constructed."
)


# ============================================================
# 19. 24-HOUR ZERO-CONTROL VALIDATION
# ============================================================
#
# This is NOT the learned policy.
#
# It is deliberately a neutral control test.
#
# Local action software structure:
#
#   [BESS, sharing-to-other-4-MGs]
#
# => dimension 5
#
# Coordinator software structure:
#
#   [market/grid-related, reserve, sharing multiplier]
#
# => dimension 3
#
# Local BESS action = 0
# Local sharing actions = 0
#
# Coordinator values = -1 so optional positive controls such
# as reserve/sharing multiplier are driven toward their
# minimum rather than accidentally requesting participation.
# ============================================================

section(
    "RUNNING 24-HOUR PHYSICAL BRIDGE EPISODE"
)


local_zero_actions = [

    np.zeros(
        5,
        dtype=np.float32,
    )

    for _ in range(
        NUMBER_OF_MICROGRIDS
    )
]


coordinator_neutral_action = np.array(

    [
        -1.0,
        -1.0,
        -1.0,
    ],

    dtype=np.float32,
)


hourly_records = []


for hour in range(
    EPISODE_LENGTH
):

    action_bundle = (
        HierarchicalActionBundle(

            local_actions=[
                action.copy()
                for action
                in local_zero_actions
            ],

            coordinator_action=
                coordinator_neutral_action.copy(),
        )
    )


    result = bridge.step(
        action_bundle
    )


    local_rewards = np.asarray(
        result.local_rewards,
        dtype=float,
    )


    coordinator_reward = float(
        result.coordinator_reward
    )


    next_local_dims = [

        int(
            np.asarray(
                state
            ).size
        )

        for state
        in result.next_observation.local_states
    ]


    next_coordinator_dim = int(

        np.asarray(
            result.next_observation.coordinator_state
        ).size
    )


    if next_local_dims != (
        [EXPECTED_LOCAL_DIM]
        * NUMBER_OF_MICROGRIDS
    ):

        raise RuntimeError(

            f"Hour {hour}: "
            f"local-state dimension failure."
        )


    if next_coordinator_dim != (
        EXPECTED_COORDINATOR_DIM
    ):

        raise RuntimeError(

            f"Hour {hour}: "
            f"coordinator dimension failure."
        )


    hourly_records.append(
        {
            "hour":
                hour + 1,

            "irradiance_w_m2":
                float(
                    irradiance_profile[
                        hour
                    ]
                ),

            "load_MG1_kw":
                float(
                    loads_by_hour[
                        hour,
                        0
                    ]
                ),

            "load_MG2_kw":
                float(
                    loads_by_hour[
                        hour,
                        1
                    ]
                ),

            "load_MG3_kw":
                float(
                    loads_by_hour[
                        hour,
                        2
                    ]
                ),

            "load_MG4_kw":
                float(
                    loads_by_hour[
                        hour,
                        3
                    ]
                ),

            "load_MG5_kw":
                float(
                    loads_by_hour[
                        hour,
                        4
                    ]
                ),

            "ev_MG1_kw":
                float(
                    ev_by_hour[
                        hour,
                        0
                    ]
                ),

            "ev_MG2_kw":
                float(
                    ev_by_hour[
                        hour,
                        1
                    ]
                ),

            "ev_MG3_kw":
                float(
                    ev_by_hour[
                        hour,
                        2
                    ]
                ),

            "ev_MG4_kw":
                float(
                    ev_by_hour[
                        hour,
                        3
                    ]
                ),

            "ev_MG5_kw":
                float(
                    ev_by_hour[
                        hour,
                        4
                    ]
                ),

            "buy_price":
                float(
                    buy_price_profile[
                        hour
                    ]
                ),

            "sell_price":
                float(
                    sell_price_profile[
                        hour
                    ]
                ),

            "confidence":
                SCALAR_CAUSAL_CONFIDENCE,

            "local_reward_MG1":
                float(
                    local_rewards[0]
                ),

            "local_reward_MG2":
                float(
                    local_rewards[1]
                ),

            "local_reward_MG3":
                float(
                    local_rewards[2]
                ),

            "local_reward_MG4":
                float(
                    local_rewards[3]
                ),

            "local_reward_MG5":
                float(
                    local_rewards[4]
                ),

            "coordinator_reward":
                coordinator_reward,

            "done":
                bool(
                    result.done
                ),

            "local_state_dimension":
                next_local_dims[0],

            "coordinator_state_dimension":
                next_coordinator_dim,
        }
    )


    print(
        f"Hour {hour + 1:02d} | "
        f"Load total="
        f"{loads_by_hour[hour].sum():8.2f} kW | "
        f"EV total="
        f"{ev_by_hour[hour].sum():8.2f} kW | "
        f"Irr="
        f"{irradiance_profile[hour]:7.2f} | "
        f"Price="
        f"{buy_price_profile[hour]:.4f} | "
        f"LocalRewardSum="
        f"{local_rewards.sum():10.3f} | "
        f"CoordReward="
        f"{coordinator_reward:10.3f} | "
        f"Done={result.done}"
    )


    observation = (
        result.next_observation
    )


# ============================================================
# 20. DONE CHECK
# ============================================================

section(
    "VERIFYING EPISODE TERMINATION"
)


if not result.done:

    raise RuntimeError(
        "Bridge did not terminate after "
        "the expected 24-hour episode."
    )


print(
    "[OK] Episode terminated after 24 hours."
)


# ============================================================
# 21. BRIDGE SUMMARY
# ============================================================

section(
    "BRIDGE SUMMARY"
)


try:

    bridge_summary = (
        bridge.summary()
    )

    for key, value in (
        bridge_summary.items()
    ):

        print(
            f"{key}: {value}"
        )

except Exception as exc:

    bridge_summary = {
        "summary_error":
            str(exc)
    }

    print(
        "Bridge summary unavailable:"
    )

    print(
        exc
    )


# ============================================================
# 22. OPTIONAL PHYSICAL STATE INSPECTION
# ============================================================

section(
    "FINAL PHYSICAL ENVIRONMENT STATE"
)


try:

    local_physical_states = (
        environment.get_local_states()
    )

    print(
        f"Number of physical MG states: "
        f"{len(local_physical_states)}"
    )

    for i, state in enumerate(
        local_physical_states
    ):

        print(
            f"MG{i + 1}: {state}"
        )

except Exception as exc:

    print(
        "get_local_states() inspection "
        "was not available:"
    )

    print(
        exc
    )


# ============================================================
# 23. SAVE RESULTS
# ============================================================

hourly_df = pd.DataFrame(
    hourly_records
)


hourly_df.to_csv(
    HOURLY_OUTPUT,
    index=False,
)


summary_rows = [
    {
        "metric":
            "episode_hours",
        "value":
            EPISODE_LENGTH,
    },
    {
        "metric":
            "number_of_microgrids",
        "value":
            NUMBER_OF_MICROGRIDS,
    },
    {
        "metric":
            "local_state_dimension",
        "value":
            EXPECTED_LOCAL_DIM,
    },
    {
        "metric":
            "coordinator_state_dimension",
        "value":
            EXPECTED_COORDINATOR_DIM,
    },
    {
        "metric":
            "predictive_state_dimension",
        "value":
            96,
    },
    {
        "metric":
            "scalar_causal_confidence",
        "value":
            SCALAR_CAUSAL_CONFIDENCE,
    },
    {
        "metric":
            "total_load_energy_proxy_kwh",
        "value":
            float(
                loads_by_hour.sum()
            ),
    },
    {
        "metric":
            "total_ev_energy_proxy_kwh",
        "value":
            float(
                ev_by_hour.sum()
            ),
    },
    {
        "metric":
            "sum_local_rewards",
        "value":
            float(
                hourly_df[
                    [
                        "local_reward_MG1",
                        "local_reward_MG2",
                        "local_reward_MG3",
                        "local_reward_MG4",
                        "local_reward_MG5",
                    ]
                ].to_numpy().sum()
            ),
    },
    {
        "metric":
            "sum_coordinator_reward",
        "value":
            float(
                hourly_df[
                    "coordinator_reward"
                ].sum()
            ),
    },
]


pd.DataFrame(
    summary_rows
).to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


# ============================================================
# 24. FINAL VALIDATION REPORT
# ============================================================

section(
    "STEP 7K FINAL VALIDATION"
)


print(
    f"24-hour steps completed       : "
    f"{len(hourly_df)}"
)

print(
    f"Local state dimension         : "
    f"{EXPECTED_LOCAL_DIM}"
)

print(
    f"Coordinator state dimension   : "
    f"{EXPECTED_COORDINATOR_DIM}"
)

print(
    f"Predictive-state dimension    : "
    f"96"
)

print(
    f"Mean causal confidence        : "
    f"{SCALAR_CAUSAL_CONFIDENCE:.8f}"
)

print(
    f"Final done                    : "
    f"{bool(result.done)}"
)


print()

print(
    "[OK] Real forecast-derived S_pred "
    "entered FC-HMARL."
)

print(
    "[OK] Real benchmark temporal profiles "
    "entered the physical VPP."
)

print(
    "[OK] Five local observations remained "
    "101-dimensional."
)

print(
    "[OK] Coordinator observation remained "
    "99-dimensional."
)

print(
    "[OK] 24-hour VPP bridge episode completed."
)

print(
    "[OK] No future actual trajectory was "
    "inserted into S_pred."
)


section(
    "STEP 7K COMPLETE"
)


print(
    f"Hourly results:\n"
    f"{HOURLY_OUTPUT}"
)

print()

print(
    f"Summary:\n"
    f"{SUMMARY_OUTPUT}"
)

print()

print(
    "STEP 7K COMPLETE."
)
