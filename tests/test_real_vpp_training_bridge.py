"""
Real integration tests for the FC-HMARL VPP training bridge.

This test connects:

    HierarchicalActionBundle
        ->
    FCHMARLActionMapper
        ->
    VPPEnvironment
        ->
    HierarchicalStateBuilder
        ->
    HierarchicalRewardBuilder
        ->
    EnvironmentStepResult

The five-microgrid capacities follow the same physical fixture already
validated in tests/test_vpp_env.py.

The small EV fleet, pairwise sharing capacities, predictive-state data,
and test actions are integration-test/reconstruction values.
"""

import numpy as np
import pytest


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
# FC-HMARL
# ============================================================

from marl.state_builder import (
    StateBuilderConfig,
    HierarchicalStateBuilder,
)

from marl.rewards import (
    RewardConfig,
    HierarchicalRewardBuilder,
)

from marl.environment_adapter import (
    HierarchicalActionBundle,
)

from marl.training_loop import (
    TrainingObservation,
    EnvironmentStepResult,
)

from marl.vpp_training_bridge import (
    VPPExogenousInput,
    VPPTrainingBridgeConfig,
    FCHMARLVPPTrainingBridge,
)

from marl.action_mapper import (
    ActionMapperConfig,
    FCHMARLActionMapper,
)


# ============================================================
# MICROGRID FACTORY
# ============================================================

def build_microgrid(
    name,
    pv_capacity_kw,
    bess_capacity_kwh,
    bess_power_kw,
    peak_load_kw,
    transformer_kva,
):
    """
    Build one microgrid using the same physical parameters
    used in the already validated VPP environment tests.
    """

    pv = PhotovoltaicSystem(
        PVParameters(
            rated_capacity_kw=pv_capacity_kw,
            efficiency=1.0,
            critical_irradiance_w_m2=200.0,
            stc_irradiance_w_m2=1000.0,
        ),
        name=f"{name}_PV",
    )

    bess = BatteryEnergyStorageSystem(
        BESSParameters(
            capacity_kwh=bess_capacity_kwh,
            rated_power_kw=bess_power_kw,
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

    ev_fleet = EVFleet(
        EVFleetParameters(
            number_of_evs=2,
            maximum_aggregate_charging_power_kw=20.0,
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
            minimum_buy_price_usd_per_kwh=0.12,
            maximum_buy_price_usd_per_kwh=0.32,
            minimum_sell_price_usd_per_kwh=0.08,
            maximum_sell_price_usd_per_kwh=0.24,
            reserve_price_usd_per_kwh=0.05,
            time_step_hours=1.0,
        ),
        name=f"{name}_Market",
    )

    return Microgrid(
        parameters=MicrogridParameters(
            name=name,
            peak_load_kw=peak_load_kw,
            transformer_rating_kva=transformer_kva,
            power_factor=1.0,
        ),
        pv_system=pv,
        bess=bess,
        ev_fleet=ev_fleet,
        market=market,
    )


# ============================================================
# REAL FIVE-MG ENVIRONMENT
# ============================================================

@pytest.fixture
def real_vpp():

    microgrids = [
        build_microgrid(
            "MG1",
            pv_capacity_kw=500.0,
            bess_capacity_kwh=1000.0,
            bess_power_kw=250.0,
            peak_load_kw=750.0,
            transformer_kva=1000.0,
        ),

        build_microgrid(
            "MG2",
            pv_capacity_kw=600.0,
            bess_capacity_kwh=1200.0,
            bess_power_kw=300.0,
            peak_load_kw=850.0,
            transformer_kva=1250.0,
        ),

        build_microgrid(
            "MG3",
            pv_capacity_kw=450.0,
            bess_capacity_kwh=900.0,
            bess_power_kw=250.0,
            peak_load_kw=700.0,
            transformer_kva=1000.0,
        ),

        build_microgrid(
            "MG4",
            pv_capacity_kw=550.0,
            bess_capacity_kwh=1100.0,
            bess_power_kw=300.0,
            peak_load_kw=800.0,
            transformer_kva=1250.0,
        ),

        build_microgrid(
            "MG5",
            pv_capacity_kw=700.0,
            bess_capacity_kwh=1400.0,
            bess_power_kw=350.0,
            peak_load_kw=950.0,
            transformer_kva=1500.0,
        ),
    ]

    connectivity = np.ones(
        (5, 5),
        dtype=int,
    )

    np.fill_diagonal(
        connectivity,
        0,
    )

    maximum_power = np.full(
        (5, 5),
        250.0,
        dtype=float,
    )

    np.fill_diagonal(
        maximum_power,
        0.0,
    )

    sharing_network = EnergySharingNetwork(
        parameters=EnergySharingParameters(
            number_of_microgrids=5,
            efficiency=0.98,
        ),
        connectivity_matrix=connectivity,
        maximum_power_matrix_kw=maximum_power,
        name="Five_MG_Sharing",
    )

    return VPPEnvironment(
        microgrids=microgrids,
        energy_sharing_network=sharing_network,
        episode_length_hours=24,
        name="Real_FC_HMARL_VPP",
    )


# ============================================================
# EXOGENOUS PROVIDER
# ============================================================

def exogenous_provider(
    episode,
    step,
):
    """
    Deterministic 24-hour integration-test input.

    IMPORTANT:
    This is a helper function, not a pytest test.
    Therefore its name does NOT begin with "test_".

    Predictive state:
        shape = 24 x 4

    Feature order:
        PV
        Load
        EV
        Price
    """

    _ = episode

    hour = int(
        step % 24
    )

    if 6 <= hour <= 18:
        irradiance = 800.0
    else:
        irradiance = 0.0

    loads = np.array(
        [
            500.0,
            550.0,
            450.0,
            525.0,
            625.0,
        ],
        dtype=np.float64,
    )

    irradiances = np.full(
        5,
        irradiance,
        dtype=np.float64,
    )

    buy_prices = np.full(
        5,
        0.20,
        dtype=np.float64,
    )

    sell_prices = np.full(
        5,
        0.12,
        dtype=np.float64,
    )

    predictive_state = np.zeros(
        (24, 4),
        dtype=np.float64,
    )

    # Integration-test values only.
    predictive_state[:, 0] = 500.0
    predictive_state[:, 1] = 530.0
    predictive_state[:, 2] = 10.0
    predictive_state[:, 3] = 0.20

    return VPPExogenousInput(
        time=float(hour),
        loads_kw=loads,
        irradiances_w_m2=irradiances,
        buy_prices_usd_per_kwh=buy_prices,
        sell_prices_usd_per_kwh=sell_prices,
        predictive_state=predictive_state,
        confidence=0.95,
        market_price=0.20,
    )


# ============================================================
# COMPLETE BRIDGE FIXTURE
# ============================================================

@pytest.fixture
def real_bridge(
    real_vpp,
):

    state_builder = HierarchicalStateBuilder(
        StateBuilderConfig(
            number_of_microgrids=5,
            forecast_horizon=24,
            forecast_features=4,
        )
    )

    reward_builder = HierarchicalRewardBuilder(
        RewardConfig(
            beta_soc=1.0,
            beta_grid=1.0,
            risk_aversion=1.0,
        )
    )

    action_mapper = FCHMARLActionMapper(
        ActionMapperConfig(
            use_dynamic_bess_feasibility=True,

            # Let the physical environment automatically
            # close the local grid power balance.
            coordinator_controls_grid=False,

            coordinator_controls_reserve=True,

            coordinator_controls_sharing=True,

            reserve_fraction_of_bess_rating=1.0,
        )
    )

    maximum_socs = [
        mg.bess.maximum_soc
        for mg in real_vpp.microgrids
    ]

    maximum_grid_exchange = [
        mg.transformer_rating_kva
        for mg in real_vpp.microgrids
    ]

    bridge_config = VPPTrainingBridgeConfig(
        maximum_socs=maximum_socs,

        maximum_grid_exchanges_kw=(
            maximum_grid_exchange
        ),

        # No manuscript-specific degradation coefficient
        # is claimed in this integration test.
        battery_degradation_cost_per_kwh=0.0,

        # Explicit test/reconstruction value.
        imbalance_penalty_per_kw=1.0,

        timestep_hours=1.0,

        number_of_microgrids=5,
    )

    return FCHMARLVPPTrainingBridge(
        environment=real_vpp,
        state_builder=state_builder,
        reward_builder=reward_builder,

        # Corrected helper name.
        exogenous_provider=exogenous_provider,

        action_mapper=action_mapper,
        config=bridge_config,
    )


# ============================================================
# ACTION HELPER
# ============================================================

def zero_physical_stress_action():
    """
    Build a valid hierarchical action.

    Local action dimension = 5:

        [BESS,
         share-to-peer-1,
         share-to-peer-2,
         share-to-peer-3,
         share-to-peer-4]

    Coordinator action:

        [market/grid command,
         reserve command,
         sharing multiplier]

    reserve command = -1
        -> zero reserve participation

    sharing multiplier = +1
        -> preserve local sharing request
    """

    local_actions = [
        np.zeros(
            5,
            dtype=np.float32,
        )
        for _ in range(5)
    ]

    coordinator_action = np.array(
        [
            0.0,
            -1.0,
            1.0,
        ],
        dtype=np.float32,
    )

    return HierarchicalActionBundle(
        local_actions=local_actions,
        coordinator_action=coordinator_action,
    )


# ============================================================
# RESET TESTS
# ============================================================

def test_real_bridge_reset_returns_training_observation(
    real_bridge,
):

    observation = real_bridge.reset(
        episode=0
    )

    assert isinstance(
        observation,
        TrainingObservation,
    )


def test_real_bridge_reset_returns_five_local_states(
    real_bridge,
):

    observation = real_bridge.reset(
        episode=0
    )

    assert len(
        observation.local_states
    ) == 5


def test_real_local_state_dimensions(
    real_bridge,
):

    observation = real_bridge.reset(
        episode=0
    )

    # 5 physical variables
    # + 24 x 4 predictive variables
    # = 101
    for state in observation.local_states:

        assert state.shape == (
            101,
        )


def test_real_coordinator_state_dimension(
    real_bridge,
):

    observation = real_bridge.reset(
        episode=0
    )

    # 3 coordinator scalars
    # + 24 x 4 predictive values
    # = 99
    assert (
        observation.coordinator_state.shape
        == (
            99,
        )
    )


# ============================================================
# REAL PHYSICAL STEP
# ============================================================

def test_real_bridge_one_complete_step(
    real_bridge,
):

    real_bridge.reset(
        episode=0
    )

    action = (
        zero_physical_stress_action()
    )

    result = real_bridge.step(
        action
    )

    assert isinstance(
        result,
        EnvironmentStepResult,
    )


def test_real_bridge_step_advances_environment(
    real_bridge,
):

    real_bridge.reset(
        episode=0
    )

    real_bridge.step(
        zero_physical_stress_action()
    )

    assert (
        real_bridge.environment.current_step
        == 1
    )


def test_real_bridge_returns_five_local_rewards(
    real_bridge,
):

    real_bridge.reset(
        episode=0
    )

    result = real_bridge.step(
        zero_physical_stress_action()
    )

    assert result.local_rewards.shape == (
        5,
    )

    assert np.isfinite(
        result.local_rewards
    ).all()


def test_real_bridge_coordinator_reward_finite(
    real_bridge,
):

    real_bridge.reset(
        episode=0
    )

    result = real_bridge.step(
        zero_physical_stress_action()
    )

    assert np.isfinite(
        result.coordinator_reward
    )


def test_real_bridge_not_done_after_first_step(
    real_bridge,
):

    real_bridge.reset(
        episode=0
    )

    result = real_bridge.step(
        zero_physical_stress_action()
    )

    assert result.done is False


# ============================================================
# PHYSICAL RESULT TESTS
# ============================================================

def test_real_physical_power_balance(
    real_bridge,
):

    real_bridge.reset(
        episode=0
    )

    result = real_bridge.step(
        zero_physical_stress_action()
    )

    physical = result.info[
        "physical_result"
    ]

    assert physical[
        "constraints"
    ][
        "all_power_balanced"
    ]


def test_real_physical_transformers_feasible(
    real_bridge,
):

    real_bridge.reset(
        episode=0
    )

    result = real_bridge.step(
        zero_physical_stress_action()
    )

    physical = result.info[
        "physical_result"
    ]

    assert physical[
        "constraints"
    ][
        "all_transformers_feasible"
    ]


def test_zero_local_actions_produce_zero_sharing(
    real_bridge,
):

    real_bridge.reset(
        episode=0
    )

    result = real_bridge.step(
        zero_physical_stress_action()
    )

    physical = result.info[
        "physical_result"
    ]

    assert physical[
        "totals"
    ][
        "sharing_scheduled_kw"
    ] == pytest.approx(
        0.0
    )


def test_zero_local_bess_commands(
    real_bridge,
):

    real_bridge.reset(
        episode=0
    )

    result = real_bridge.step(
        zero_physical_stress_action()
    )

    physical = result.info[
        "physical_result"
    ]

    for local_result in physical[
        "local_results"
    ]:

        assert local_result[
            "bess_power_kw"
        ] == pytest.approx(
            0.0
        )


# ============================================================
# NONZERO BESS ACTION
# ============================================================

def test_positive_bess_action_causes_discharge(
    real_bridge,
):

    real_bridge.reset(
        episode=0
    )

    local_actions = [
        np.zeros(
            5,
            dtype=np.float32,
        )
        for _ in range(5)
    ]

    # Positive BESS command -> discharge.
    local_actions[0][0] = 0.5

    action = HierarchicalActionBundle(
        local_actions=local_actions,
        coordinator_action=np.array(
            [
                0.0,
                -1.0,
                1.0,
            ],
            dtype=np.float32,
        ),
    )

    result = real_bridge.step(
        action
    )

    physical = result.info[
        "physical_result"
    ]

    assert physical[
        "local_results"
    ][0][
        "bess_power_kw"
    ] > 0.0


def test_negative_bess_action_causes_charge(
    real_bridge,
):

    real_bridge.reset(
        episode=0
    )

    local_actions = [
        np.zeros(
            5,
            dtype=np.float32,
        )
        for _ in range(5)
    ]

    # Negative BESS command -> charge.
    local_actions[0][0] = -0.5

    action = HierarchicalActionBundle(
        local_actions=local_actions,
        coordinator_action=np.array(
            [
                0.0,
                -1.0,
                1.0,
            ],
            dtype=np.float32,
        ),
    )

    result = real_bridge.step(
        action
    )

    physical = result.info[
        "physical_result"
    ]

    assert physical[
        "local_results"
    ][0][
        "bess_power_kw"
    ] < 0.0


# ============================================================
# SHARING ACTION
# ============================================================

def test_nonzero_sharing_action_reaches_environment(
    real_bridge,
):

    real_bridge.reset(
        episode=0
    )

    local_actions = [
        np.zeros(
            5,
            dtype=np.float32,
        )
        for _ in range(5)
    ]

    # MG1 -> MG2
    #
    # For source MG index 0,
    # local_action[1] maps to destination MG index 1.
    local_actions[0][1] = 0.5

    action = HierarchicalActionBundle(
        local_actions=local_actions,
        coordinator_action=np.array(
            [
                0.0,
                -1.0,
                1.0,
            ],
            dtype=np.float32,
        ),
    )

    result = real_bridge.step(
        action
    )

    physical = result.info[
        "physical_result"
    ]

    sharing_matrix = physical[
        "sharing"
    ][
        "feasible_matrix_kw"
    ]

    # 0.5 normalized request
    # x 250 kW pairwise capacity
    # = 125 kW
    assert sharing_matrix[
        0,
        1,
    ] == pytest.approx(
        125.0
    )


# ============================================================
# COMPLETE 24-HOUR EPISODE
# ============================================================

def test_real_bridge_complete_24_hour_episode(
    real_bridge,
):

    real_bridge.reset(
        episode=0
    )

    final_result = None

    for step in range(
        24
    ):

        final_result = real_bridge.step(
            zero_physical_stress_action()
        )

        if step < 23:

            assert final_result.done is False

    assert final_result is not None

    assert final_result.done is True

    assert (
        real_bridge.environment.current_step
        == 24
    )


# ============================================================
# INTERNAL BRIDGE STORAGE
# ============================================================

def test_bridge_stores_state_reward_and_physical_results(
    real_bridge,
):

    real_bridge.reset(
        episode=0
    )

    real_bridge.step(
        zero_physical_stress_action()
    )

    summary = real_bridge.summary()

    assert summary[
        "has_physical_result"
    ]

    assert summary[
        "has_state_result"
    ]

    assert summary[
        "has_reward_result"
    ]

# ============================================================
# STEP 7R-J: LOCAL PCC CREDIT-ASSIGNMENT REGRESSION
# ============================================================

def pcc_stress_action():
    """Create a deterministic action that overloads MG1 before projection."""
    local_actions = [
        np.zeros(5, dtype=np.float32)
        for _ in range(5)
    ]

    # Full BESS charging in MG1.
    local_actions[0][0] = -1.0

    # Full MG1 -> MG2 sharing request.
    local_actions[0][1] = 1.0

    coordinator_action = np.array(
        [0.0, -1.0, 1.0],
        dtype=np.float32,
    )

    return HierarchicalActionBundle(
        local_actions=local_actions,
        coordinator_action=coordinator_action,
    )


def test_local_reward_uses_requested_pcc_loading_before_projection(
    real_bridge,
):
    """Projection must not erase the local PCC-violation reward signal."""
    real_bridge.reset(episode=0)
    result = real_bridge.step(pcc_stress_action())

    physical = result.info["physical_result"]
    mg1 = physical["local_results"][0]

    requested_pcc = abs(
        float(mg1["requested_grid_power_kw"])
        + float(mg1["outgoing_sharing_kw"])
    )
    active_limit = float(
        mg1["transformer_active_power_limit_kw"]
    )

    assert requested_pcc > active_limit
    assert mg1["constraints"]["transformer_limit_satisfied"]

    expected_violation = requested_pcc - active_limit
    expected_grid_cost = float(
        mg1["market"]["grid_purchase_cost_usd"]
    )

    # Integration fixture: beta_grid=1, degradation coefficient=0,
    # and upper-SOC violation is zero in this step.
    expected_reward = -(
        expected_grid_cost
        + expected_violation ** 2
    )

    assert result.local_rewards[0] == pytest.approx(
        expected_reward
    )


def test_local_pcc_penalty_is_symmetric_for_requested_export(
    real_bridge,
):
    """Excess requested export must be penalized by PCC magnitude."""
    real_bridge.reset(episode=0)

    local_results = []
    for mg in real_bridge.environment.microgrids:
        local_results.append(
            {
                "market": {
                    "grid_purchase_cost_usd": 0.0,
                },
                "bess_power_kw": 0.0,
                "requested_grid_power_kw": 0.0,
                "grid_power_kw": 0.0,
                "outgoing_sharing_kw": 0.0,
                "transformer_active_power_limit_kw": (
                    mg.transformer_rating_kva
                    * mg.parameters.power_factor
                ),
            }
        )

    # MG1 requests 1250 kW export through a 1000 kW active PCC limit.
    local_results[0]["requested_grid_power_kw"] = -1250.0

    physical_result = {
        "local_results": local_results,
        "totals": {
            "market_profit_usd": 0.0,
        },
        "constraints": {
            "total_power_balance_violation_kw": 0.0,
        },
    }

    reward = real_bridge._build_reward(
        physical_result=physical_result,
        confidence=1.0,
    )

    assert reward.local_rewards[0] == pytest.approx(
        -(250.0 ** 2)
    )


def test_coordinator_imbalance_penalty_remains_active(
    real_bridge,
):
    """Step 7R-J must preserve the coordinator physical-imbalance penalty."""
    real_bridge.reset(episode=0)
    result = real_bridge.step(pcc_stress_action())

    physical = result.info["physical_result"]
    total_imbalance = float(
        physical["constraints"][
            "total_power_balance_violation_kw"
        ]
    )

    assert total_imbalance > 0.0
    assert np.isfinite(result.coordinator_reward)
    assert result.coordinator_reward <= float(
        physical["totals"]["market_profit_usd"]
    )

