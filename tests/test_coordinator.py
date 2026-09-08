import numpy as np
import pytest
from agents.coordinator_agent import (
    CoordinatorActionResult,
    CoordinatorAgent,
    CoordinatorAgentConfig,
    CoordinatorTrainingResult,
    build_coordinator_agent,
)
from agents.sac_agent import (
    SACAgent,
    SACAgentConfig,
    SACUpdateResult,
)
# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def config():

    return CoordinatorAgentConfig(
        coordinator_id=1,

        state_dimension=12,

        action_dimension=2,

        hidden_dimensions=(
            32,
            32,
        ),

        action_low=[
            -2.0,
            -1.0,
        ],

        action_high=[
            2.0,
            1.0,
        ],

        replay_buffer_capacity=100,

        batch_size=8,

        seed=42,
    )


@pytest.fixture
def coordinator(
    config,
):

    return CoordinatorAgent(
        config
    )
def make_state(
    value=0.0,
):
    return np.full(
        12,
        value,
        dtype=np.float32,
    )
def make_transition(
    index,
):
    rng = np.random.default_rng(
        500 + index
    )
    state = rng.normal(
        size=12
    ).astype(
        np.float32
    )

    action = rng.uniform(
        low=[
            -2.0,
            -1.0,
        ],
        high=[
            2.0,
            1.0,
        ],
    ).astype(
        np.float32
    )

    reward = float(
        rng.normal()
    )

    next_state = rng.normal(
        size=12
    ).astype(
        np.float32
    )

    done = bool(
        index % 6 == 0
    )

    return (
        state,
        action,
        reward,
        next_state,
        done,
    )
def fill_buffer(
    coordinator,
    count=16,
):
    for index in range(
        count
    ):
        coordinator.store_transition(
            *make_transition(
                index
            )
        )
# ============================================================
# CONFIGURATION
# ============================================================

def test_valid_config(
    config,
):
    config.validate()
def test_invalid_coordinator_id():

    with pytest.raises(
        ValueError
    ):
        CoordinatorAgentConfig(
            coordinator_id=0,
            state_dimension=12,
            action_dimension=2,
        ).validate()
def test_invalid_state_dimension():

    with pytest.raises(
        ValueError
    ):
        CoordinatorAgentConfig(
            state_dimension=0,
            action_dimension=2,
        ).validate()
def test_invalid_action_dimension():
    with pytest.raises(
        ValueError
    ):
        CoordinatorAgentConfig(
            state_dimension=12,
            action_dimension=0,
        ).validate()
def test_to_sac_config(
    config,
):
    sac_config = (
        config.to_sac_config()
    )
    assert isinstance(
        sac_config,
        SACAgentConfig,
    )
    assert (
        sac_config.state_dimension
        == 12
    )
    assert (
        sac_config.action_dimension
        == 2
    )
def test_sac_parameters_preserved(
    config,
):
    sac = config.to_sac_config()

    assert (
        sac.discount_factor
        == pytest.approx(
            0.99
        )
    )
    assert (
        sac.learning_rate
        == pytest.approx(
            1e-4
        )
    )
    assert (
        sac.soft_update_coefficient
        == pytest.approx(
            0.005
        )
    )
# ============================================================
# CONSTRUCTION
# ============================================================

def test_coordinator_creation(
    coordinator,
):
    assert isinstance(
        coordinator,
        CoordinatorAgent,
    )
def test_contains_sac_agent(
    coordinator,
):
    assert isinstance(
        coordinator.sac_agent,
        SACAgent,
    )
def test_coordinator_identity(
    coordinator,
):
    assert (
        coordinator.coordinator_id
        == 1
    )
def test_state_dimension_property(
    coordinator,
):
    assert (
        coordinator.state_dimension
        == 12
    )
def test_action_dimension_property(
    coordinator,
):
    assert (
        coordinator.action_dimension
        == 2
    )
def test_actor_property(
    coordinator,
):
    assert (
        coordinator.actor
        is coordinator.sac_agent.actor
    )
def test_critic_property(
    coordinator,
):
    assert (
        coordinator.critic
        is coordinator.sac_agent.critic
    )


def test_target_critic_property(
    coordinator,
):

    assert (
        coordinator.target_critic
        is coordinator.sac_agent.target_critic
    )


def test_replay_buffer_property(
    coordinator,
):

    assert (
        coordinator.replay_buffer
        is coordinator.sac_agent.replay_buffer
    )
# ============================================================
# STATE VALIDATION
# ============================================================

def test_valid_state(
    coordinator,
):
    state = (
        coordinator.validate_state(
            make_state()
        )
    )
    assert state.shape == (
        12,
    )
def test_state_wrong_dimension(
    coordinator,
):
    with pytest.raises(
        ValueError
    ):
        coordinator.validate_state(
            np.zeros(
                11
            )
        )
def test_state_wrong_rank(
    coordinator,
):

    with pytest.raises(
        ValueError
    ):
        coordinator.validate_state(
            np.zeros(
                (
                    1,
                    12,
                )
            )
        )
def test_state_nan_rejected(
    coordinator,
):
    state = make_state()

    state[
        5
    ] = np.nan

    with pytest.raises(
        ValueError
    ):

        coordinator.validate_state(
            state
        )
# ============================================================
# ACTION VALIDATION
# ============================================================

def test_valid_action(
    coordinator,
):
    action = (
        coordinator.validate_action(
            [
                0.5,
                0.2,
            ]
        )
    )
    assert action.shape == (
        2,
    )
def test_action_wrong_dimension(
    coordinator,
):
    with pytest.raises(
        ValueError
    ):

        coordinator.validate_action(
            [
                0.0,
                0.0,
                0.0,
            ]
        )
def test_action_nan_rejected(
    coordinator,
):
    with pytest.raises(
        ValueError
    ):
        coordinator.validate_action(
            [
                np.nan,
                0.0,
            ]
        )


def test_action_below_bound(
    coordinator,
):

    with pytest.raises(
        ValueError
    ):

        coordinator.validate_action(
            [
                -3.0,
                0.0,
            ]
        )


def test_action_above_bound(
    coordinator,
):

    with pytest.raises(
        ValueError
    ):

        coordinator.validate_action(
            [
                0.0,
                2.0,
            ]
        )
# ============================================================
# ACTION SELECTION
# ============================================================
def test_select_action_shape(
    coordinator,
):

    action = (
        coordinator.select_action(
            make_state()
        )
    )

    assert action.shape == (
        2,
    )


def test_select_action_finite(
    coordinator,
):

    action = (
        coordinator.select_action(
            make_state()
        )
    )

    assert np.isfinite(
        action
    ).all()


def test_select_action_within_bounds(
    coordinator,
):

    for _ in range(
        20
    ):

        action = (
            coordinator.select_action(
                make_state()
            )
        )

        assert (
            -2.0
            <= action[
                0
            ]
            <= 2.0
        )

        assert (
            -1.0
            <= action[
                1
            ]
            <= 1.0
        )


def test_deterministic_action(
    coordinator,
):

    state = make_state(
        0.5
    )

    first = (
        coordinator.select_action(
            state,
            deterministic=True,
        )
    )

    second = (
        coordinator.select_action(
            state,
            deterministic=True,
        )
    )

    assert np.allclose(
        first,
        second,
    )


def test_action_selection_counter(
    coordinator,
):

    coordinator.select_action(
        make_state()
    )

    coordinator.select_action(
        make_state()
    )

    assert (
        coordinator.total_actions_selected
        == 2
    )


def test_last_state_recorded(
    coordinator,
):

    state = make_state(
        1.0
    )

    coordinator.select_action(
        state
    )

    assert np.allclose(
        coordinator.last_state,
        state,
    )


def test_last_action_recorded(
    coordinator,
):

    action = (
        coordinator.select_action(
            make_state()
        )
    )

    assert np.allclose(
        coordinator.last_action,
        action,
    )


# ============================================================
# ACTION RESULT
# ============================================================

def test_action_result(
    coordinator,
):

    result = (
        coordinator.select_action_result(
            make_state(),
            deterministic=True,
        )
    )

    assert isinstance(
        result,
        CoordinatorActionResult,
    )

    assert (
        result.coordinator_id
        == 1
    )

    assert result.action.shape == (
        2,
    )

    assert (
        result.deterministic
        is True
    )


def test_action_result_summary():

    result = CoordinatorActionResult(
        coordinator_id=1,

        action=np.array(
            [
                0.5,
                -0.2,
            ]
        ),

        deterministic=True,
    )

    result.validate(
        expected_dimension=2
    )

    summary = (
        result.summary()
    )

    assert (
        summary[
            "coordinator_id"
        ]
        == 1
    )

    assert (
        summary[
            "action_dimension"
        ]
        == 2
    )


# ============================================================
# TRANSITION STORAGE
# ============================================================

def test_store_transition(
    coordinator,
):

    index = (
        coordinator.store_transition(
            *make_transition(
                1
            )
        )
    )

    assert index == 0

    assert len(
        coordinator.replay_buffer
    ) == 1


def test_transition_counter(
    coordinator,
):

    coordinator.store_transition(
        *make_transition(
            1
        )
    )

    coordinator.store_transition(
        *make_transition(
            2
        )
    )

    assert (
        coordinator.total_transitions_stored
        == 2
    )


def test_store_invalid_action(
    coordinator,
):

    (
        state,
        _,
        reward,
        next_state,
        done,
    ) = make_transition(
        1
    )

    with pytest.raises(
        ValueError
    ):

        coordinator.store_transition(
            state=state,

            action=[
                5.0,
                0.0,
            ],

            reward=reward,

            next_state=next_state,

            done=done,
        )


# ============================================================
# TRAINING READINESS
# ============================================================

def test_not_ready_initially(
    coordinator,
):

    assert (
        coordinator.ready_to_update()
        is False
    )


def test_ready_after_batch_size(
    coordinator,
):

    fill_buffer(
        coordinator,
        8,
    )

    assert (
        coordinator.ready_to_update()
        is True
    )


# ============================================================
# TRAINING
# ============================================================

def test_update_if_not_ready(
    coordinator,
):

    result = (
        coordinator.update_if_ready()
    )

    assert isinstance(
        result,
        CoordinatorTrainingResult,
    )

    assert (
        result.updated
        is False
    )

    assert (
        result.sac_result
        is None
    )


def test_update_if_ready(
    coordinator,
):

    fill_buffer(
        coordinator,
        16,
    )

    result = (
        coordinator.update_if_ready()
    )

    assert (
        result.updated
        is True
    )

    assert isinstance(
        result.sac_result,
        SACUpdateResult,
    )


def test_update_counter(
    coordinator,
):

    fill_buffer(
        coordinator,
        16,
    )

    coordinator.update()

    assert (
        coordinator.total_training_updates
        == 1
    )


# ============================================================
# TRAINING RESULT
# ============================================================

def test_training_result_not_updated():

    result = CoordinatorTrainingResult(
        coordinator_id=1,

        updated=False,

        sac_result=None,
    )

    result.validate()

    summary = (
        result.summary()
    )

    assert (
        summary[
            "updated"
        ]
        is False
    )


def test_invalid_training_result():

    with pytest.raises(
        ValueError
    ):

        CoordinatorTrainingResult(
            coordinator_id=1,

            updated=True,

            sac_result=None,
        ).validate()


# ============================================================
# TRAIN / EVAL
# ============================================================

def test_initial_training_mode(
    coordinator,
):

    assert (
        coordinator.training
        is True
    )


def test_eval_mode(
    coordinator,
):

    coordinator.eval_mode()

    assert (
        coordinator.training
        is False
    )


def test_train_mode(
    coordinator,
):

    coordinator.eval_mode()

    coordinator.train_mode()

    assert (
        coordinator.training
        is True
    )


def test_hierarchical_training_interface(
    coordinator,
):

    coordinator.set_training_mode(
        False
    )

    assert (
        coordinator.training
        is False
    )

    coordinator.set_training_mode(
        True
    )

    assert (
        coordinator.training
        is True
    )


# ============================================================
# RESET
# ============================================================

def test_reset_last_state_action(
    coordinator,
):

    coordinator.select_action(
        make_state()
    )

    assert (
        coordinator.last_state
        is not None
    )

    assert (
        coordinator.last_action
        is not None
    )

    coordinator.reset()

    assert (
        coordinator.last_state
        is None
    )

    assert (
        coordinator.last_action
        is None
    )


def test_reset_preserves_replay_memory(
    coordinator,
):

    coordinator.store_transition(
        *make_transition(
            1
        )
    )

    coordinator.reset()

    assert len(
        coordinator.replay_buffer
    ) == 1


# ============================================================
# SAVE / LOAD
# ============================================================

def test_save_and_load(
    coordinator,
    tmp_path,
):

    fill_buffer(
        coordinator,
        16,
    )

    coordinator.update()

    path = (
        tmp_path
        / "coordinator.pt"
    )

    coordinator.save(
        path
    )

    second = CoordinatorAgent(
        coordinator.config
    )

    second.load(
        path
    )

    for (
        first_parameter,
        second_parameter,
    ) in zip(
        coordinator.actor.parameters(),
        second.actor.parameters(),
    ):

        assert np.allclose(
            first_parameter
            .detach()
            .cpu()
            .numpy(),

            second_parameter
            .detach()
            .cpu()
            .numpy(),
        )


def test_load_missing_file(
    coordinator,
    tmp_path,
):

    with pytest.raises(
        FileNotFoundError
    ):

        coordinator.load(
            tmp_path
            / "missing.pt"
        )


# ============================================================
# SUMMARY
# ============================================================

def test_summary(
    coordinator,
):

    summary = (
        coordinator.summary()
    )

    assert (
        summary[
            "agent_type"
        ]
        == "vpp_coordinator"
    )

    assert (
        summary[
            "coordinator_id"
        ]
        == 1
    )

    assert (
        summary[
            "state_dimension"
        ]
        == 12
    )

    assert (
        summary[
            "action_dimension"
        ]
        == 2
    )


# ============================================================
# FACTORY
# ============================================================

def test_factory():

    coordinator = (
        build_coordinator_agent(
            state_dimension=12,

            action_dimension=2,

            hidden_dimensions=(
                16,
                16,
            ),

            replay_buffer_capacity=50,

            batch_size=4,

            seed=42,
        )
    )

    assert isinstance(
        coordinator,
        CoordinatorAgent,
    )


def test_factory_id():

    coordinator = (
        build_coordinator_agent(
            state_dimension=12,

            action_dimension=2,

            hidden_dimensions=(
                16,
                16,
            ),

            replay_buffer_capacity=50,

            batch_size=4,
        )
    )

    assert (
        coordinator.coordinator_id
        == 1
    )


# ============================================================
# HIERARCHICAL CONTROLLER COMPATIBILITY
# ============================================================

def test_hierarchical_select_action_interface(
    coordinator,
):

    action = (
        coordinator.select_action(
            make_state(),
            deterministic=True,
        )
    )

    assert isinstance(
        action,
        np.ndarray,
    )

    assert action.ndim == 1


def test_hierarchical_reset_interface(
    coordinator,
):

    coordinator.reset()


def test_hierarchical_set_training_interface(
    coordinator,
):

    coordinator.set_training_mode(
        False
    )

    assert (
        coordinator.training
        is False
    )
