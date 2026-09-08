import numpy as np
import pytest
from marl.rewards import (
    CoordinatorRewardResult,
    HierarchicalRewardBuilder,
    HierarchicalRewardResult,
    LocalRewardResult,
    RewardConfig,
    calculate_complete_coordinator_reward,
    calculate_complete_local_reward,
    calculate_confidence_risk_cost,
    calculate_coordinator_reward,
    calculate_grid_violation,
    calculate_hierarchical_reward,
    calculate_local_reward,
    calculate_soc_violation,
    calculate_violation_cost,
)
# ============================================================
# CONFIG
# ============================================================
def test_default_beta_soc():

    config = RewardConfig()

    assert config.beta_soc == pytest.approx(
        1.0
    )

def test_default_beta_grid():

    config = RewardConfig()

    assert config.beta_grid == pytest.approx(
        1.0
    )
def test_default_risk_aversion():
    config = RewardConfig()
    assert config.risk_aversion == pytest.approx(
        1.0
    )
def test_valid_config():
    RewardConfig().validate()

def test_negative_beta_soc():

    with pytest.raises(ValueError):

        RewardConfig(
            beta_soc=-1
        ).validate()

def test_negative_beta_grid():

    with pytest.raises(ValueError):

        RewardConfig(
            beta_grid=-1
        ).validate()

def test_negative_risk_aversion():

    with pytest.raises(ValueError):

        RewardConfig(
            risk_aversion=-1
        ).validate()

# ============================================================
# SOC VIOLATION
# ============================================================

def test_no_soc_violation():

    result = calculate_soc_violation(
        soc=0.8,
        maximum_soc=0.9,
    )

    assert result == pytest.approx(
        0.0
    )

def test_soc_violation():

    result = calculate_soc_violation(
        soc=1.0,
        maximum_soc=0.9,
    )

    assert result == pytest.approx(
        0.1
    )

# ============================================================
# GRID VIOLATION
# ============================================================

def test_no_grid_violation():

    result = calculate_grid_violation(
        grid_exchange=50,
        maximum_grid_exchange=100,
    )

    assert result == pytest.approx(
        0.0
    )

def test_grid_violation():

    result = calculate_grid_violation(
        grid_exchange=125,
        maximum_grid_exchange=100,
    )

    assert result == pytest.approx(
        25.0
    )

# ============================================================
# VIOLATION COST
# ============================================================

def test_zero_violation_cost():

    result = calculate_violation_cost(
        soc=0.8,
        maximum_soc=0.9,
        grid_exchange=50,
        maximum_grid_exchange=100,
    )

    assert result == pytest.approx(
        0.0
    )

def test_soc_only_violation_cost():

    result = calculate_violation_cost(
        soc=1.0,
        maximum_soc=0.9,
        grid_exchange=50,
        maximum_grid_exchange=100,
        beta_soc=2.0,
        beta_grid=1.0,
    )

    assert result == pytest.approx(
        2.0
        * 0.1 ** 2
    )

def test_grid_only_violation_cost():

    result = calculate_violation_cost(
        soc=0.8,
        maximum_soc=0.9,
        grid_exchange=110,
        maximum_grid_exchange=100,
        beta_soc=1.0,
        beta_grid=2.0,
    )

    assert result == pytest.approx(
        2.0
        * 10.0 ** 2
    )

def test_combined_violation_cost():

    result = calculate_violation_cost(
        soc=1.0,
        maximum_soc=0.9,
        grid_exchange=110,
        maximum_grid_exchange=100,
        beta_soc=2.0,
        beta_grid=3.0,
    )

    expected = (
        2.0
        * 0.1 ** 2
        +
        3.0
        * 10.0 ** 2
    )

    assert result == pytest.approx(
        expected
    )

# ============================================================
# LOCAL REWARD
# ============================================================

def test_local_reward():

    reward = calculate_local_reward(
        grid_cost=10.0,
        battery_degradation_cost=2.0,
        violation_cost=3.0,
    )

    assert reward == pytest.approx(
        -15.0
    )

def test_zero_local_cost_reward():

    reward = calculate_local_reward(
        0.0,
        0.0,
        0.0,
    )

    assert reward == pytest.approx(
        0.0
    )

def test_negative_grid_cost_rejected():

    with pytest.raises(ValueError):

        calculate_local_reward(
            -1.0,
            0.0,
            0.0,
        )

# ============================================================
# COMPLETE LOCAL
# ============================================================

def test_complete_local_result():

    result = calculate_complete_local_reward(
        grid_cost=10.0,
        battery_degradation_cost=2.0,
        soc=0.8,
        maximum_soc=0.9,
        grid_exchange=50,
        maximum_grid_exchange=100,
    )

    assert isinstance(
        result,
        LocalRewardResult,
    )


def test_complete_local_reward_value():

    result = calculate_complete_local_reward(
        grid_cost=10.0,
        battery_degradation_cost=2.0,
        soc=0.8,
        maximum_soc=0.9,
        grid_exchange=50,
        maximum_grid_exchange=100,
    )

    assert result.reward == pytest.approx(
        -12.0
    )


# ============================================================
# CONFIDENCE RISK
# ============================================================

def test_full_confidence_zero_risk():

    risk = calculate_confidence_risk_cost(
        confidence=1.0,
        risk_aversion=10.0,
    )

    assert risk == pytest.approx(
        0.0
    )


def test_zero_confidence_maximum_risk():

    risk = calculate_confidence_risk_cost(
        confidence=0.0,
        risk_aversion=10.0,
    )

    assert risk == pytest.approx(
        10.0
    )


def test_half_confidence_risk():

    risk = calculate_confidence_risk_cost(
        confidence=0.5,
        risk_aversion=10.0,
    )

    assert risk == pytest.approx(
        5.0
    )

def test_invalid_confidence():

    with pytest.raises(ValueError):

        calculate_confidence_risk_cost(
            confidence=1.2,
        )

# ============================================================
# COORDINATOR REWARD
# ============================================================

def test_coordinator_reward():

    reward = calculate_coordinator_reward(
        profit=100.0,
        imbalance_cost=10.0,
        risk_cost=5.0,
    )

    assert reward == pytest.approx(
        85.0
    )

def test_negative_profit_allowed():

    reward = calculate_coordinator_reward(
        profit=-20.0,
        imbalance_cost=10.0,
        risk_cost=5.0,
    )

    assert reward == pytest.approx(
        -35.0
    )

def test_negative_imbalance_rejected():

    with pytest.raises(ValueError):

        calculate_coordinator_reward(
            profit=100,
            imbalance_cost=-1,
            risk_cost=0,
        )

# ============================================================
# COMPLETE COORDINATOR
# ============================================================

def test_complete_coordinator_result():

    result = (
        calculate_complete_coordinator_reward(
            profit=100.0,
            imbalance_cost=10.0,
            confidence=0.8,
            config=RewardConfig(
                risk_aversion=10.0
            ),
        )
    )

    assert isinstance(
        result,
        CoordinatorRewardResult,
    )


def test_complete_coordinator_value():

    result = (
        calculate_complete_coordinator_reward(
            profit=100.0,
            imbalance_cost=10.0,
            confidence=0.8,
            config=RewardConfig(
                risk_aversion=10.0
            ),
        )
    )

    # Risk =
    # 10 * (1 - 0.8) = 2
    #
    # reward =
    # 100 - 10 - 2 = 88

    assert result.risk_cost == pytest.approx(
        2.0
    )

    assert result.reward == pytest.approx(
        88.0
    )

# ============================================================
# HIERARCHICAL REWARD
# ============================================================

def test_hierarchical_reward():

    result = calculate_hierarchical_reward(
        local_rewards=[
            -10,
            -20,
            -30,
        ],
        coordinator_reward=100,
    )

    assert result == pytest.approx(
        40.0
    )

def test_empty_local_rewards_rejected():

    with pytest.raises(ValueError):

        calculate_hierarchical_reward(
            [],
            100,
        )

# ============================================================
# FULL BUILDER
# ============================================================

@pytest.fixture
def local_inputs():

    return [
        {
            "grid_cost": 10.0,
            "battery_degradation_cost": 1.0,
            "soc": 0.8,
            "maximum_soc": 0.9,
            "grid_exchange": 50.0,
            "maximum_grid_exchange": 100.0,
        },
        {
            "grid_cost": 20.0,
            "battery_degradation_cost": 2.0,
            "soc": 0.7,
            "maximum_soc": 0.9,
            "grid_exchange": 60.0,
            "maximum_grid_exchange": 100.0,
        },
        {
            "grid_cost": 30.0,
            "battery_degradation_cost": 3.0,
            "soc": 0.6,
            "maximum_soc": 0.9,
            "grid_exchange": 70.0,
            "maximum_grid_exchange": 100.0,
        },
    ]

def test_reward_builder_creation():

    builder = HierarchicalRewardBuilder()

    assert isinstance(
        builder.config,
        RewardConfig,
    )

def test_reward_builder(
    local_inputs,
):

    builder = HierarchicalRewardBuilder(
        RewardConfig(
            risk_aversion=10.0
        )
    )

    result = builder.build(
        local_reward_inputs=local_inputs,
        profit=100.0,
        imbalance_cost=5.0,
        confidence=0.8,
    )

    assert isinstance(
        result,
        HierarchicalRewardResult,
    )


def test_reward_builder_local_count(
    local_inputs,
):

    builder = HierarchicalRewardBuilder()

    result = builder.build(
        local_inputs,
        profit=100,
        imbalance_cost=0,
        confidence=1,
    )

    assert result.number_of_local_agents == 3


def test_reward_builder_local_values(
    local_inputs,
):

    builder = HierarchicalRewardBuilder()

    result = builder.build(
        local_inputs,
        profit=100,
        imbalance_cost=0,
        confidence=1,
    )

    assert np.allclose(
        result.local_rewards,
        [
            -11,
            -22,
            -33,
        ],
    )


def test_reward_builder_total(
    local_inputs,
):

    builder = HierarchicalRewardBuilder()

    result = builder.build(
        local_inputs,
        profit=100,
        imbalance_cost=0,
        confidence=1,
    )

    # local sum = -66
    # coordinator = 100
    # total = 34

    assert result.total_reward == pytest.approx(
        34.0
    )


def test_reward_builder_low_confidence_reduces_reward(
    local_inputs,
):

    config = RewardConfig(
        risk_aversion=10.0
    )

    builder = HierarchicalRewardBuilder(
        config
    )

    high = builder.build(
        local_inputs,
        profit=100,
        imbalance_cost=0,
        confidence=1.0,
    )

    low = builder.build(
        local_inputs,
        profit=100,
        imbalance_cost=0,
        confidence=0.0,
    )

    assert (
        high.total_reward
        >
        low.total_reward
    )


def test_missing_local_key(
    local_inputs,
):

    local_inputs = [
        dict(
            item
        )
        for item in local_inputs
    ]

    del local_inputs[
        0
    ][
        "grid_cost"
    ]

    builder = HierarchicalRewardBuilder()

    with pytest.raises(KeyError):

        builder.build(
            local_inputs,
            profit=100,
            imbalance_cost=0,
            confidence=1,
        )


def test_empty_builder_inputs():

    builder = HierarchicalRewardBuilder()

    with pytest.raises(ValueError):

        builder.build(
            [],
            profit=100,
            imbalance_cost=0,
            confidence=1,
        )


def test_hierarchical_result_summary(
    local_inputs,
):

    builder = HierarchicalRewardBuilder()

    result = builder.build(
        local_inputs,
        profit=100,
        imbalance_cost=0,
        confidence=1,
    )

    summary = result.summary()

    assert summary[
        "number_of_local_agents"
    ] == 3

    assert summary[
        "mean_local_reward"
    ] == pytest.approx(
        -22.0
    )

    assert summary[
        "coordinator_reward"
    ] == pytest.approx(
        100.0
    )

    assert summary[
        "total_reward"
    ] == pytest.approx(
        34.0
    )
