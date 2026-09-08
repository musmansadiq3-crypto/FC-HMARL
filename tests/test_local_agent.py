import numpy as np
import pytest
from agents.local_agent import (
    LocalActionResult,
    LocalAgent,
    LocalAgentConfig,
    LocalTrainingResult,
    build_local_agents,
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

    return LocalAgentConfig(
        microgrid_id=1,

        state_dimension=10,

        action_dimension=2,

        hidden_dimensions=(
            32,
            32,
        ),

        action_low=[
            -1.0,
            -2.0,
        ],

        action_high=[
            1.0,
            2.0,
        ],

        replay_buffer_capacity=100,

        batch_size=8,

        seed=42,
    )


@pytest.fixture
def agent(
    config,
):

    return LocalAgent(
        config
    )


def make_state(
    value=0.0,
):

    return np.full(
        10,
        value,
        dtype=np.float32,
    )


def make_transition(
    index,
):

    rng = np.random.default_rng(
        100 + index
    )

    state = rng.normal(
        size=10
    ).astype(
        np.float32
    )

    action = rng.uniform(
        low=[
            -1.0,
            -2.0,
        ],
        high=[
            1.0,
            2.0,
        ],
    ).astype(
        np.float32
    )

    reward = float(
        rng.normal()
    )

    next_state = rng.normal(
        size=10
    ).astype(
        np.float32
    )

    done = bool(
        index % 7 == 0
    )

    return (
        state,
        action,
        reward,
        next_state,
        done,
    )


def fill_buffer(
    agent,
    count=16,
):

    for index in range(
        count
    ):

        agent.store_transition(
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


def test_invalid_microgrid_id():

    with pytest.raises(
        ValueError
    ):

        LocalAgentConfig(
            microgrid_id=0,
            state_dimension=10,
            action_dimension=2,
        ).validate()


def test_invalid_state_dimension():

    with pytest.raises(
        ValueError
    ):

        LocalAgentConfig(
            microgrid_id=1,
            state_dimension=0,
            action_dimension=2,
        ).validate()


def test_invalid_action_dimension():

    with pytest.raises(
        ValueError
    ):

        LocalAgentConfig(
            microgrid_id=1,
            state_dimension=10,
            action_dimension=0,
        ).validate()


def test_to_sac_config(
    config,
):

    sac = config.to_sac_config()

    assert isinstance(
        sac,
        SACAgentConfig,
    )

    assert (
        sac.state_dimension
        == 10
    )

    assert (
        sac.action_dimension
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

def test_local_agent_creation(
    agent,
):

    assert isinstance(
        agent,
        LocalAgent,
    )


def test_contains_sac_agent(
    agent,
):

    assert isinstance(
        agent.sac_agent,
        SACAgent,
    )


def test_microgrid_identity(
    agent,
):

    assert agent.microgrid_id == 1


def test_state_dimension_property(
    agent,
):

    assert agent.state_dimension == 10


def test_action_dimension_property(
    agent,
):

    assert agent.action_dimension == 2


def test_actor_property(
    agent,
):

    assert (
        agent.actor
        is agent.sac_agent.actor
    )


def test_critic_property(
    agent,
):

    assert (
        agent.critic
        is agent.sac_agent.critic
    )


def test_target_critic_property(
    agent,
):

    assert (
        agent.target_critic
        is agent.sac_agent.target_critic
    )


def test_replay_buffer_property(
    agent,
):

    assert (
        agent.replay_buffer
        is agent.sac_agent.replay_buffer
    )


# ============================================================
# STATE VALIDATION
# ============================================================

def test_valid_state(
    agent,
):

    state = agent.validate_state(
        make_state()
    )

    assert state.shape == (
        10,
    )


def test_state_wrong_dimension(
    agent,
):

    with pytest.raises(
        ValueError
    ):

        agent.validate_state(
            np.zeros(
                9
            )
        )


def test_state_wrong_rank(
    agent,
):

    with pytest.raises(
        ValueError
    ):

        agent.validate_state(
            np.zeros(
                (
                    1,
                    10,
                )
            )
        )


def test_state_nan_rejected(
    agent,
):

    state = make_state()

    state[
        3
    ] = np.nan

    with pytest.raises(
        ValueError
    ):

        agent.validate_state(
            state
        )


# ============================================================
# ACTION VALIDATION
# ============================================================

def test_valid_action(
    agent,
):

    action = agent.validate_action(
        np.array(
            [
                0.5,
                1.0,
            ]
        )
    )

    assert action.shape == (
        2,
    )


def test_action_wrong_dimension(
    agent,
):

    with pytest.raises(
        ValueError
    ):

        agent.validate_action(
            np.zeros(
                3
            )
        )


def test_action_nan_rejected(
    agent,
):

    with pytest.raises(
        ValueError
    ):

        agent.validate_action(
            [
                np.nan,
                0.0,
            ]
        )


def test_action_below_bound(
    agent,
):

    with pytest.raises(
        ValueError
    ):

        agent.validate_action(
            [
                -2.0,
                0.0,
            ]
        )


def test_action_above_bound(
    agent,
):

    with pytest.raises(
        ValueError
    ):

        agent.validate_action(
            [
                0.0,
                3.0,
            ]
        )


# ============================================================
# ACTION SELECTION
# ============================================================

def test_select_action_shape(
    agent,
):

    action = agent.select_action(
        make_state()
    )

    assert action.shape == (
        2,
    )


def test_select_action_finite(
    agent,
):

    action = agent.select_action(
        make_state()
    )

    assert np.isfinite(
        action
    ).all()


def test_select_action_within_bounds(
    agent,
):

    for _ in range(
        20
    ):

        action = agent.select_action(
            make_state()
        )

        assert (
            -1.0
            <= action[
                0
            ]
            <= 1.0
        )

        assert (
            -2.0
            <= action[
                1
            ]
            <= 2.0
        )


def test_deterministic_action(
    agent,
):

    state = make_state(
        0.5
    )

    first = agent.select_action(
        state,
        deterministic=True,
    )

    second = agent.select_action(
        state,
        deterministic=True,
    )

    assert np.allclose(
        first,
        second,
    )


def test_action_selection_counter(
    agent,
):

    agent.select_action(
        make_state()
    )

    agent.select_action(
        make_state()
    )

    assert (
        agent.total_actions_selected
        == 2
    )


def test_last_state_recorded(
    agent,
):

    state = make_state(
        2.0
    )

    agent.select_action(
        state
    )

    assert np.allclose(
        agent.last_state,
        state,
    )


def test_last_action_recorded(
    agent,
):

    action = agent.select_action(
        make_state()
    )

    assert np.allclose(
        agent.last_action,
        action,
    )


# ============================================================
# ACTION RESULT
# ============================================================

def test_select_action_result(
    agent,
):

    result = (
        agent.select_action_result(
            make_state(),
            deterministic=True,
        )
    )

    assert isinstance(
        result,
        LocalActionResult,
    )

    assert result.microgrid_id == 1

    assert result.action.shape == (
        2,
    )

    assert (
        result.deterministic
        is True
    )


def test_action_result_summary():

    result = LocalActionResult(
        microgrid_id=2,

        action=np.array(
            [
                0.1,
                0.2,
            ]
        ),

        deterministic=True,
    )

    result.validate(
        expected_dimension=2
    )

    summary = result.summary()

    assert (
        summary[
            "microgrid_id"
        ]
        == 2
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
    agent,
):

    index = agent.store_transition(
        *make_transition(
            1
        )
    )

    assert index == 0

    assert len(
        agent.replay_buffer
    ) == 1


def test_transition_counter(
    agent,
):

    agent.store_transition(
        *make_transition(
            1
        )
    )

    agent.store_transition(
        *make_transition(
            2
        )
    )

    assert (
        agent.total_transitions_stored
        == 2
    )


def test_store_invalid_action(
    agent,
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

        agent.store_transition(
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
    agent,
):

    assert (
        agent.ready_to_update()
        is False
    )


def test_ready_after_batch_size(
    agent,
):

    fill_buffer(
        agent,
        8,
    )

    assert (
        agent.ready_to_update()
        is True
    )


# ============================================================
# CONDITIONAL UPDATE
# ============================================================

def test_update_if_not_ready(
    agent,
):

    result = (
        agent.update_if_ready()
    )

    assert isinstance(
        result,
        LocalTrainingResult,
    )

    assert result.updated is False

    assert result.sac_result is None


def test_update_if_ready(
    agent,
):

    fill_buffer(
        agent,
        16,
    )

    result = (
        agent.update_if_ready()
    )

    assert result.updated is True

    assert isinstance(
        result.sac_result,
        SACUpdateResult,
    )


def test_update_counter(
    agent,
):

    fill_buffer(
        agent,
        16,
    )

    agent.update()

    assert (
        agent.total_training_updates
        == 1
    )


# ============================================================
# TRAINING RESULT
# ============================================================

def test_training_result_not_updated():

    result = LocalTrainingResult(
        microgrid_id=1,
        updated=False,
        sac_result=None,
    )

    result.validate()

    summary = result.summary()

    assert summary[
        "updated"
    ] is False


def test_invalid_training_result():

    with pytest.raises(
        ValueError
    ):

        LocalTrainingResult(
            microgrid_id=1,
            updated=True,
            sac_result=None,
        ).validate()


# ============================================================
# MODES
# ============================================================

def test_initial_training_mode(
    agent,
):

    assert agent.training is True


def test_eval_mode(
    agent,
):

    agent.eval_mode()

    assert agent.training is False


def test_train_mode(
    agent,
):

    agent.eval_mode()

    agent.train_mode()

    assert agent.training is True


def test_set_training_mode_interface(
    agent,
):

    agent.set_training_mode(
        False
    )

    assert agent.training is False

    agent.set_training_mode(
        True
    )

    assert agent.training is True


# ============================================================
# RESET
# ============================================================

def test_reset_last_state_action(
    agent,
):

    agent.select_action(
        make_state()
    )

    assert (
        agent.last_state
        is not None
    )

    assert (
        agent.last_action
        is not None
    )

    agent.reset()

    assert (
        agent.last_state
        is None
    )

    assert (
        agent.last_action
        is None
    )


def test_reset_preserves_memory(
    agent,
):

    agent.store_transition(
        *make_transition(
            1
        )
    )

    agent.reset()

    assert len(
        agent.replay_buffer
    ) == 1


# ============================================================
# SAVE / LOAD
# ============================================================

def test_save_and_load(
    agent,
    tmp_path,
):

    fill_buffer(
        agent,
        16,
    )

    agent.update()

    path = (
        tmp_path
        / "local_agent_1.pt"
    )

    agent.save(
        path
    )

    second = LocalAgent(
        agent.config
    )

    second.load(
        path
    )

    for first_parameter, second_parameter in zip(
        agent.actor.parameters(),
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
    agent,
    tmp_path,
):

    with pytest.raises(
        FileNotFoundError
    ):

        agent.load(
            tmp_path
            / "missing.pt"
        )


# ============================================================
# SUMMARY
# ============================================================

def test_summary(
    agent,
):

    summary = agent.summary()

    assert (
        summary[
            "agent_type"
        ]
        == "local_microgrid"
    )

    assert (
        summary[
            "microgrid_id"
        ]
        == 1
    )

    assert (
        summary[
            "state_dimension"
        ]
        == 10
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

def test_build_five_local_agents():

    agents = build_local_agents(
        number_of_microgrids=5,

        state_dimension=10,

        action_dimension=2,

        hidden_dimensions=(
            16,
            16,
        ),

        replay_buffer_capacity=50,

        batch_size=4,

        seed=42,
    )

    assert len(
        agents
    ) == 5


def test_factory_microgrid_ids():

    agents = build_local_agents(
        number_of_microgrids=5,

        state_dimension=10,

        action_dimension=2,

        hidden_dimensions=(
            16,
            16,
        ),

        replay_buffer_capacity=50,

        batch_size=4,
    )

    ids = [
        agent.microgrid_id
        for agent in agents
    ]

    assert ids == [
        1,
        2,
        3,
        4,
        5,
    ]


def test_factory_agents_are_independent():

    agents = build_local_agents(
        number_of_microgrids=2,

        state_dimension=10,

        action_dimension=2,

        hidden_dimensions=(
            16,
            16,
        ),

        replay_buffer_capacity=50,

        batch_size=4,
    )

    assert (
        agents[
            0
        ].actor
        is not
        agents[
            1
        ].actor
    )


def test_factory_unique_seed():

    agents = build_local_agents(
        number_of_microgrids=2,

        state_dimension=10,

        action_dimension=2,

        hidden_dimensions=(
            16,
            16,
        ),

        replay_buffer_capacity=50,

        batch_size=4,

        seed=100,
    )

    assert (
        agents[
            0
        ].config.seed
        == 100
    )

    assert (
        agents[
            1
        ].config.seed
        == 101
    )


def test_factory_invalid_number():

    with pytest.raises(
        ValueError
    ):

        build_local_agents(
            number_of_microgrids=0,
            state_dimension=10,
            action_dimension=2,
        )


# ============================================================
# HIERARCHICAL CONTROLLER COMPATIBILITY
# ============================================================

def test_hierarchical_select_action_interface(
    agent,
):

    state = make_state()

    action = agent.select_action(
        state,
        deterministic=True,
    )

    assert isinstance(
        action,
        np.ndarray,
    )

    assert action.ndim == 1


def test_hierarchical_reset_interface(
    agent,
):

    agent.reset()


def test_hierarchical_training_mode_interface(
    agent,
):

    agent.set_training_mode(
        False
    )

    assert agent.training is False
