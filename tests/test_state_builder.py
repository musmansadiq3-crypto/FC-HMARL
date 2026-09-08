import numpy as np
import pytest
from marl.state_builder import (
    HierarchicalStateBuilder,
    HierarchicalStateResult,
    StateBuilderConfig,
    build_coordinator_state,
    build_global_state,
    build_local_state,
    calculate_total_sharing_activity,
    calculate_vpp_power,
    flatten_predictive_state,
    validate_predictive_state,
)

# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def small_config():

    return StateBuilderConfig(
        number_of_microgrids=3,
        forecast_horizon=4,
        forecast_features=2,
        dtype="float32",
    )

@pytest.fixture
def predictive_state():

    return np.arange(
        8,
        dtype=np.float32,
    ).reshape(
        4,
        2,
    )

@pytest.fixture
def observations():

    return [
        {
            "soc": 0.50,
            "pv_power": 10.0,
            "load_power": 20.0,
            "ev_power": 3.0,
            "grid_exchange": 13.0,
        },
        {
            "soc": 0.60,
            "pv_power": 12.0,
            "load_power": 18.0,
            "ev_power": 4.0,
            "grid_exchange": 10.0,
        },
        {
            "soc": 0.70,
            "pv_power": 15.0,
            "load_power": 21.0,
            "ev_power": 2.0,
            "grid_exchange": 8.0,
        },
    ]

# ============================================================
# CONFIGURATION
# ============================================================

def test_default_microgrids():

    config = StateBuilderConfig()

    assert config.number_of_microgrids == 5


def test_default_forecast_horizon():

    config = StateBuilderConfig()

    assert config.forecast_horizon == 24

def test_default_forecast_features():

    config = StateBuilderConfig()

    assert config.forecast_features == 4


def test_default_predictive_dimension():

    config = StateBuilderConfig()

    assert (
        config.predictive_state_dimension
        == 96
    )


def test_default_local_dimension():

    config = StateBuilderConfig()

    assert (
        config.local_state_dimension
        == 101
    )

def test_default_coordinator_dimension():

    config = StateBuilderConfig()

    assert (
        config.coordinator_state_dimension
        == 99
    )

def test_default_global_dimension():

    config = StateBuilderConfig()

    assert (
        config.global_state_dimension
        == 604
    )


def test_valid_config():

    StateBuilderConfig().validate()


def test_invalid_microgrid_number():

    config = StateBuilderConfig(
        number_of_microgrids=0
    )

    with pytest.raises(ValueError):
        config.validate()


def test_invalid_horizon():

    config = StateBuilderConfig(
        forecast_horizon=0
    )

    with pytest.raises(ValueError):
        config.validate()


def test_invalid_features():

    config = StateBuilderConfig(
        forecast_features=0
    )

    with pytest.raises(ValueError):
        config.validate()


def test_invalid_dtype():

    config = StateBuilderConfig(
        dtype="int32"
    )

    with pytest.raises(ValueError):
        config.validate()

# ============================================================
# PREDICTIVE STATE
# ============================================================

def test_validate_predictive_matrix(
    predictive_state,
):

    result = validate_predictive_state(
        predictive_state,
        forecast_horizon=4,
        forecast_features=2,
    )

    assert result.shape == (
        4,
        2,
    )

def test_flatten_predictive_state(
    predictive_state,
):

    result = flatten_predictive_state(
        predictive_state,
        forecast_horizon=4,
        forecast_features=2,
    )

    assert result.shape == (
        8,
    )


def test_flatten_preserves_order(
    predictive_state,
):

    result = flatten_predictive_state(
        predictive_state,
        forecast_horizon=4,
        forecast_features=2,
    )

    assert np.allclose(
        result,
        np.arange(
            8
        ),
    )


def test_predictive_size_mismatch():

    state = np.zeros(
        (
            3,
            2,
        )
    )

    with pytest.raises(ValueError):

        validate_predictive_state(
            state,
            forecast_horizon=4,
            forecast_features=2,
        )


def test_predictive_nan_rejected():

    state = np.zeros(
        (
            4,
            2,
        )
    )

    state[
        0,
        0
    ] = np.nan

    with pytest.raises(ValueError):

        validate_predictive_state(
            state,
            4,
            2,
        )


# ============================================================
# LOCAL STATE
# ============================================================

def test_local_state_shape(
    small_config,
    predictive_state,
):

    state = build_local_state(
        soc=0.5,
        pv_power=10.0,
        load_power=20.0,
        ev_power=3.0,
        grid_exchange=13.0,
        predictive_state=(
            predictive_state
        ),
        config=small_config,
    )

    assert state.shape == (
        13,
    )


def test_local_state_first_values(
    small_config,
    predictive_state,
):

    state = build_local_state(
        0.5,
        10.0,
        20.0,
        3.0,
        13.0,
        predictive_state,
        small_config,
    )

    assert np.allclose(
        state[:5],
        [
            0.5,
            10.0,
            20.0,
            3.0,
            13.0,
        ],
    )


def test_local_predictive_part(
    small_config,
    predictive_state,
):

    state = build_local_state(
        0.5,
        10.0,
        20.0,
        3.0,
        13.0,
        predictive_state,
        small_config,
    )

    assert np.allclose(
        state[
            5:
        ],
        np.arange(
            8
        ),
    )


def test_local_state_float32(
    small_config,
    predictive_state,
):

    state = build_local_state(
        0.5,
        10,
        20,
        3,
        13,
        predictive_state,
        small_config,
    )

    assert state.dtype == np.float32


def test_invalid_local_scalar(
    small_config,
    predictive_state,
):

    with pytest.raises(ValueError):

        build_local_state(
            soc=[
                0.5,
                0.6,
            ],
            pv_power=10,
            load_power=20,
            ev_power=3,
            grid_exchange=13,
            predictive_state=(
                predictive_state
            ),
            config=small_config,
        )


# ============================================================
# COORDINATOR STATE
# ============================================================

def test_coordinator_state_shape(
    small_config,
    predictive_state,
):

    state = build_coordinator_state(
        vpp_power=31.0,
        energy_sharing=5.0,
        market_price=0.12,
        predictive_state=(
            predictive_state
        ),
        config=small_config,
    )

    assert state.shape == (
        11,
    )


def test_coordinator_first_values(
    small_config,
    predictive_state,
):

    state = build_coordinator_state(
        31.0,
        5.0,
        0.12,
        predictive_state,
        small_config,
    )

    assert np.allclose(
        state[:3],
        [
            31.0,
            5.0,
            0.12,
        ],
    )

# ============================================================
# VPP AGGREGATION
# ============================================================

def test_vpp_power_sum():

    result = calculate_vpp_power(
        [
            13.0,
            10.0,
            8.0,
        ]
    )

    assert result == pytest.approx(
        31.0
    )


def test_vpp_power_rejects_nan():

    with pytest.raises(ValueError):

        calculate_vpp_power(
            [
                1.0,
                np.nan,
            ]
        )

# ============================================================
# SHARING
# ============================================================

def test_total_sharing_activity():

    matrix = np.array(
        [
            [
                0,
                2,
                0,
            ],
            [
                0,
                0,
                3,
            ],
            [
                1,
                0,
                0,
            ],
        ],
        dtype=float,
    )

    result = (
        calculate_total_sharing_activity(
            matrix
        )
    )

    assert result == pytest.approx(
        6.0
    )


def test_sharing_diagonal_ignored():

    matrix = np.array(
        [
            [
                100,
                2,
            ],
            [
                3,
                100,
            ],
        ],
        dtype=float,
    )

    result = (
        calculate_total_sharing_activity(
            matrix
        )
    )

    assert result == pytest.approx(
        5.0
    )


def test_negative_sharing_rejected():

    matrix = np.array(
        [
            [
                0,
                -1,
            ],
            [
                0,
                0,
            ],
        ],
        dtype=float,
    )

    with pytest.raises(ValueError):

        calculate_total_sharing_activity(
            matrix
        )


def test_non_square_sharing_rejected():

    matrix = np.zeros(
        (
            2,
            3,
        )
    )

    with pytest.raises(ValueError):

        calculate_total_sharing_activity(
            matrix
        )

# ============================================================
# GLOBAL STATE
# ============================================================

def test_global_state_shape(
    small_config,
    predictive_state,
):

    local_states = [
        build_local_state(
            0.5,
            10,
            20,
            3,
            13,
            predictive_state,
            small_config,
        )
        for _ in range(
            3
        )
    ]

    coordinator = (
        build_coordinator_state(
            39,
            5,
            0.1,
            predictive_state,
            small_config,
        )
    )

    state = build_global_state(
        local_states,
        coordinator,
        small_config,
    )

    assert state.shape == (
        50,
    )


def test_wrong_number_local_states(
    small_config,
    predictive_state,
):

    local_states = [
        build_local_state(
            0.5,
            10,
            20,
            3,
            13,
            predictive_state,
            small_config,
        )
        for _ in range(
            2
        )
    ]

    coordinator = (
        build_coordinator_state(
            26,
            5,
            0.1,
            predictive_state,
            small_config,
        )
    )

    with pytest.raises(ValueError):

        build_global_state(
            local_states,
            coordinator,
            small_config,
        )


# ============================================================
# COMPLETE BUILDER
# ============================================================

def test_builder_creation(
    small_config,
):

    builder = (
        HierarchicalStateBuilder(
            small_config
        )
    )

    assert (
        builder.config
        is small_config
    )


def test_complete_state_build(
    small_config,
    predictive_state,
    observations,
):

    builder = (
        HierarchicalStateBuilder(
            small_config
        )
    )

    sharing_matrix = np.array(
        [
            [
                0,
                1,
                0,
            ],
            [
                0,
                0,
                2,
            ],
            [
                1,
                0,
                0,
            ],
        ],
        dtype=float,
    )

    result = builder.build(
        microgrid_observations=(
            observations
        ),
        predictive_state=(
            predictive_state
        ),
        market_price=0.10,
        sharing_matrix=(
            sharing_matrix
        ),
    )

    assert isinstance(
        result,
        HierarchicalStateResult,
    )


def test_complete_local_state_count(
    small_config,
    predictive_state,
    observations,
):

    builder = (
        HierarchicalStateBuilder(
            small_config
        )
    )

    result = builder.build(
        observations,
        predictive_state,
        market_price=0.1,
        sharing_matrix=np.zeros(
            (
                3,
                3,
            )
        ),
    )

    assert len(
        result.local_states
    ) == 3


def test_complete_coordinator_shape(
    small_config,
    predictive_state,
    observations,
):

    builder = (
        HierarchicalStateBuilder(
            small_config
        )
    )

    result = builder.build(
        observations,
        predictive_state,
        market_price=0.1,
        sharing_matrix=np.zeros(
            (
                3,
                3,
            )
        ),
    )

    assert result.coordinator_state.shape == (
        11,
    )


def test_complete_global_shape(
    small_config,
    predictive_state,
    observations,
):

    builder = (
        HierarchicalStateBuilder(
            small_config
        )
    )

    result = builder.build(
        observations,
        predictive_state,
        market_price=0.1,
        sharing_matrix=np.zeros(
            (
                3,
                3,
            )
        ),
    )

    assert result.global_state.shape == (
        50,
    )


def test_builder_vpp_power(
    small_config,
    predictive_state,
    observations,
):

    builder = (
        HierarchicalStateBuilder(
            small_config
        )
    )

    result = builder.build(
        observations,
        predictive_state,
        market_price=0.1,
        sharing_matrix=np.zeros(
            (
                3,
                3,
            )
        ),
    )

    assert result.coordinator_state[
        0
    ] == pytest.approx(
        31.0
    )


def test_builder_sharing_activity(
    small_config,
    predictive_state,
    observations,
):

    matrix = np.array(
        [
            [
                0,
                1,
                0,
            ],
            [
                0,
                0,
                2,
            ],
            [
                3,
                0,
                0,
            ],
        ]
    )

    builder = (
        HierarchicalStateBuilder(
            small_config
        )
    )

    result = builder.build(
        observations,
        predictive_state,
        market_price=0.1,
        sharing_matrix=matrix,
    )

    assert result.coordinator_state[
        1
    ] == pytest.approx(
        6.0
    )


def test_builder_market_price(
    small_config,
    predictive_state,
    observations,
):

    builder = (
        HierarchicalStateBuilder(
            small_config
        )
    )

    result = builder.build(
        observations,
        predictive_state,
        market_price=0.25,
        sharing_matrix=np.zeros(
            (
                3,
                3,
            )
        ),
    )

    assert result.coordinator_state[
        2
    ] == pytest.approx(
        0.25
    )


def test_missing_observation_key(
    small_config,
    predictive_state,
    observations,
):

    observations = [
        dict(
            observation
        )
        for observation in observations
    ]

    del observations[
        0
    ][
        "soc"
    ]

    builder = (
        HierarchicalStateBuilder(
            small_config
        )
    )

    with pytest.raises(KeyError):

        builder.build(
            observations,
            predictive_state,
            market_price=0.1,
            sharing_matrix=np.zeros(
                (
                    3,
                    3,
                )
            ),
        )


def test_wrong_observation_count(
    small_config,
    predictive_state,
    observations,
):

    builder = (
        HierarchicalStateBuilder(
            small_config
        )
    )

    with pytest.raises(ValueError):

        builder.build(
            observations[:2],
            predictive_state,
            market_price=0.1,
            sharing_matrix=np.zeros(
                (
                    3,
                    3,
                )
            ),
        )


def test_result_summary(
    small_config,
    predictive_state,
    observations,
):

    builder = (
        HierarchicalStateBuilder(
            small_config
        )
    )

    result = builder.build(
        observations,
        predictive_state,
        market_price=0.1,
        sharing_matrix=np.zeros(
            (
                3,
                3,
            )
        ),
    )

    summary = result.summary()

    assert summary[
        "number_of_local_agents"
    ] == 3

    assert summary[
        "local_state_dimension"
    ] == 13

    assert summary[
        "coordinator_state_dimension"
    ] == 11

    assert summary[
        "global_state_dimension"
    ] == 50

    assert summary[
        "predictive_state_dimension"
    ] == 8
