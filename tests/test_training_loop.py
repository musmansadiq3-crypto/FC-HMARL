import numpy as np
import pytest
from agents.local_agent import (
    LocalAgent,
    LocalAgentConfig,
)
from agents.coordinator_agent import (
    CoordinatorAgent,
    CoordinatorAgentConfig,
)
from marl.training_loop import (
    TrainingLoopConfig,
    TrainingObservation,
    EnvironmentStepResult,
    HierarchicalActionBundle,
    EpisodeStatistics,
    TrainingHistory,
    FCHMARLTrainingLoop,
)
# ============================================================
# TEST DIMENSIONS
# ============================================================

NUMBER_OF_LOCAL_AGENTS = 5

LOCAL_STATE_DIMENSION = 6

LOCAL_ACTION_DIMENSION = 2

COORDINATOR_STATE_DIMENSION = 4

COORDINATOR_ACTION_DIMENSION = 2

# ============================================================
# FACTORIES
# ============================================================

def build_test_local_agents():

    agents = []

    for index in range(
        NUMBER_OF_LOCAL_AGENTS
    ):

        config = LocalAgentConfig(
            microgrid_id=index + 1,

            state_dimension=(
                LOCAL_STATE_DIMENSION
            ),

            action_dimension=(
                LOCAL_ACTION_DIMENSION
            ),

            hidden_dimensions=(
                16,
                16,
            ),

            action_low=[
                -1.0,
                -1.0,
            ],

            action_high=[
                1.0,
                1.0,
            ],

            replay_buffer_capacity=100,

            batch_size=4,

            seed=100 + index,
        )

        agents.append(
            LocalAgent(
                config
            )
        )
    return agents
def build_test_coordinator():

    config = CoordinatorAgentConfig(
        coordinator_id=1,

        state_dimension=(
            COORDINATOR_STATE_DIMENSION
        ),

        action_dimension=(
            COORDINATOR_ACTION_DIMENSION
        ),

        hidden_dimensions=(
            16,
            16,
        ),

        action_low=[
            -1.0,
            -1.0,
        ],

        action_high=[
            1.0,
            1.0,
        ],

        replay_buffer_capacity=100,

        batch_size=4,

        seed=999,
    )

    return CoordinatorAgent(
        config
    )


def make_observation(
    value=0.0,
):

    local_states = [
        np.full(
            LOCAL_STATE_DIMENSION,
            value + index,
            dtype=np.float32,
        )
        for index
        in range(
            NUMBER_OF_LOCAL_AGENTS
        )
    ]

    coordinator_state = np.full(
        COORDINATOR_STATE_DIMENSION,
        value,
        dtype=np.float32,
    )

    return TrainingObservation(
        local_states=local_states,
        coordinator_state=(
            coordinator_state
        ),
    )


def make_reset_function():

    def reset_function(
        episode,
    ):

        return make_observation(
            float(
                episode
            )
        )

    return reset_function


def make_step_function(
    done_after=None,
):

    def step_function(
        episode,
        time_step,
        actions,
    ):

        assert isinstance(
            actions,
            HierarchicalActionBundle,
        )

        next_observation = (
            make_observation(
                float(
                    episode
                    + time_step
                    + 1
                )
            )
        )

        local_rewards = [
            1.0
            for _ in range(
                NUMBER_OF_LOCAL_AGENTS
            )
        ]

        coordinator_reward = 2.0

        if done_after is None:

            done = False

        else:

            done = (
                time_step
                >= done_after
            )

        return EnvironmentStepResult(
            next_observation=(
                next_observation
            ),

            local_rewards=(
                local_rewards
            ),

            coordinator_reward=(
                coordinator_reward
            ),

            done=done,

            info={
                "episode": episode,
                "time_step": time_step,
            },
        )

    return step_function


def build_training_loop(
    maximum_episodes=2,
    steps_per_episode=3,
    done_after=None,
    enable_checkpointing=False,
):

    local_agents = (
        build_test_local_agents()
    )

    coordinator = (
        build_test_coordinator()
    )

    config = TrainingLoopConfig(
        maximum_episodes=(
            maximum_episodes
        ),

        steps_per_episode=(
            steps_per_episode
        ),

        updates_per_step=1,

        checkpoint_interval=10,

        enable_checkpointing=(
            enable_checkpointing
        ),

        verbose=False,
    )

    return FCHMARLTrainingLoop(
        local_agents=local_agents,

        coordinator_agent=(
            coordinator
        ),

        reset_function=(
            make_reset_function()
        ),

        step_function=(
            make_step_function(
                done_after
            )
        ),

        config=config,
    )
# ============================================================
# CONFIGURATION
# ============================================================

def test_training_config_defaults():

    config = (
        TrainingLoopConfig()
    )

    assert (
        config.maximum_episodes
        == 5000
    )
    assert (
        config.steps_per_episode
        == 24
    )
def test_valid_training_config():

    TrainingLoopConfig().validate()
def test_invalid_episodes():

    with pytest.raises(
        ValueError
    ):

        TrainingLoopConfig(
            maximum_episodes=0
        ).validate()


def test_invalid_steps():

    with pytest.raises(
        ValueError
    ):

        TrainingLoopConfig(
            steps_per_episode=0
        ).validate()


def test_invalid_updates_per_step():

    with pytest.raises(
        ValueError
    ):

        TrainingLoopConfig(
            updates_per_step=0
        ).validate()


def test_invalid_checkpoint_interval():

    with pytest.raises(
        ValueError
    ):

        TrainingLoopConfig(
            checkpoint_interval=0
        ).validate()


# ============================================================
# OBSERVATION
# ============================================================

def test_valid_observation():

    observation = (
        make_observation()
    )

    observation.validate(
        NUMBER_OF_LOCAL_AGENTS
    )


def test_wrong_number_local_states():

    observation = (
        TrainingObservation(
            local_states=[
                np.zeros(
                    LOCAL_STATE_DIMENSION
                )
            ],

            coordinator_state=np.zeros(
                COORDINATOR_STATE_DIMENSION
            ),
        )
    )

    with pytest.raises(
        ValueError
    ):

        observation.validate(
            NUMBER_OF_LOCAL_AGENTS
        )


def test_invalid_local_state_rank():

    observation = (
        make_observation()
    )

    observation.local_states[
        0
    ] = np.zeros(
        (
            1,
            LOCAL_STATE_DIMENSION,
        )
    )

    with pytest.raises(
        ValueError
    ):

        observation.validate(
            NUMBER_OF_LOCAL_AGENTS
        )


def test_invalid_coordinator_state():

    observation = (
        make_observation()
    )

    observation.coordinator_state[
        0
    ] = np.nan

    with pytest.raises(
        ValueError
    ):

        observation.validate(
            NUMBER_OF_LOCAL_AGENTS
        )

# ============================================================
# ENVIRONMENT RESULT
# ============================================================

def test_valid_environment_result():

    result = EnvironmentStepResult(
        next_observation=(
            make_observation()
        ),

        local_rewards=[
            1.0
            for _ in range(
                NUMBER_OF_LOCAL_AGENTS
            )
        ],

        coordinator_reward=2.0,

        done=False,
    )

    result.validate(
        NUMBER_OF_LOCAL_AGENTS
    )


def test_wrong_reward_count():

    result = EnvironmentStepResult(
        next_observation=(
            make_observation()
        ),

        local_rewards=[
            1.0
        ],

        coordinator_reward=2.0,

        done=False,
    )

    with pytest.raises(
        ValueError
    ):

        result.validate(
            NUMBER_OF_LOCAL_AGENTS
        )


def test_nan_reward_rejected():

    result = EnvironmentStepResult(
        next_observation=(
            make_observation()
        ),

        local_rewards=[
            np.nan,
            1.0,
            1.0,
            1.0,
            1.0,
        ],

        coordinator_reward=2.0,

        done=False,
    )

    with pytest.raises(
        ValueError
    ):

        result.validate(
            NUMBER_OF_LOCAL_AGENTS
        )


# ============================================================
# CONSTRUCTION
# ============================================================

def test_training_loop_creation():

    loop = build_training_loop()

    assert isinstance(
        loop,
        FCHMARLTrainingLoop,
    )


def test_five_local_agents():

    loop = build_training_loop()

    assert (
        loop.number_of_local_agents
        == 5
    )


def test_initial_global_step():

    loop = build_training_loop()

    assert loop.global_step == 0


# ============================================================
# MODES
# ============================================================

def test_training_mode():

    loop = build_training_loop()

    loop.set_training_mode(
        True
    )

    assert all(
        agent.training
        for agent
        in loop.local_agents
    )

    assert (
        loop.coordinator_agent
        .training
    )


def test_evaluation_mode():

    loop = build_training_loop()

    loop.set_training_mode(
        False
    )

    assert all(
        not agent.training
        for agent
        in loop.local_agents
    )

    assert not (
        loop.coordinator_agent
        .training
    )


# ============================================================
# ACTION SELECTION
# ============================================================

def test_select_actions():

    loop = build_training_loop()

    observation = (
        make_observation()
    )

    actions = loop.select_actions(
        observation
    )

    assert isinstance(
        actions,
        HierarchicalActionBundle,
    )

    assert len(
        actions.local_actions
    ) == 5

    assert (
        actions.coordinator_action.shape
        == (
            COORDINATOR_ACTION_DIMENSION,
        )
    )


def test_local_action_dimensions():

    loop = build_training_loop()

    actions = loop.select_actions(
        make_observation()
    )

    for action in (
        actions.local_actions
    ):

        assert action.shape == (
            LOCAL_ACTION_DIMENSION,
        )

# ============================================================
# STORE TRANSITIONS
# ============================================================

def test_store_transitions():

    loop = build_training_loop()

    observation = (
        make_observation()
    )

    actions = loop.select_actions(
        observation
    )

    result = (
        make_step_function()(
            1,
            0,
            actions,
        )
    )

    loop.store_transitions(
        observation,
        actions,
        result,
    )

    for agent in (
        loop.local_agents
    ):

        assert len(
            agent.replay_buffer
        ) == 1

    assert len(
        loop.coordinator_agent
        .replay_buffer
    ) == 1


# ============================================================
# ONE STEP
# ============================================================

def test_run_step():

    loop = build_training_loop()

    observation = (
        make_observation()
    )

    (
        next_observation,
        statistics,
    ) = loop.run_step(
        episode=1,
        time_step=0,
        observation=observation,
    )

    assert isinstance(
        next_observation,
        TrainingObservation,
    )

    assert (
        statistics.episode
        == 1
    )

    assert (
        statistics.time_step
        == 0
    )

    assert (
        statistics.total_reward
        == pytest.approx(
            7.0
        )
    )

def test_global_step_increments():

    loop = build_training_loop()

    loop.run_step(
        episode=1,
        time_step=0,
        observation=(
            make_observation()
        ),
    )

    assert (
        loop.global_step
        == 1
    )


# ============================================================
# AGENT UPDATES
# ============================================================

def test_agents_not_updated_before_batch():

    loop = build_training_loop(
        steps_per_episode=1
    )

    statistics = (
        loop.run_episode(
            episode=1
        )
    )

    assert (
        statistics
        .local_update_count
        == 0
    )

    assert (
        statistics
        .coordinator_update_count
        == 0
    )


def test_agents_update_after_replay_ready():

    loop = build_training_loop(
        steps_per_episode=5
    )

    statistics = (
        loop.run_episode(
            episode=1
        )
    )

    # Batch size is 4.
    # By the fourth/fifth interaction,
    # optimization should be possible.

    assert (
        statistics
        .local_update_count
        > 0
    )

    assert (
        statistics
        .coordinator_update_count
        > 0
    )


# ============================================================
# EPISODE
# ============================================================

def test_episode_steps():

    loop = build_training_loop(
        steps_per_episode=3
    )

    statistics = (
        loop.run_episode(
            episode=1
        )
    )

    assert isinstance(
        statistics,
        EpisodeStatistics,
    )

    assert statistics.steps == 3


def test_episode_return():

    loop = build_training_loop(
        steps_per_episode=3
    )

    statistics = (
        loop.run_episode(
            episode=1
        )
    )

    # 5 local agents * reward 1
    # + coordinator reward 2
    # = 7 per step
    #
    # 3 steps => 21

    assert (
        statistics.total_return
        == pytest.approx(
            21.0
        )
    )


def test_local_returns():

    loop = build_training_loop(
        steps_per_episode=3
    )

    statistics = (
        loop.run_episode(
            episode=1
        )
    )

    assert (
        statistics.local_returns
        == pytest.approx(
            [
                3.0,
                3.0,
                3.0,
                3.0,
                3.0,
            ]
        )
    )


def test_coordinator_return():

    loop = build_training_loop(
        steps_per_episode=3
    )

    statistics = (
        loop.run_episode(
            episode=1
        )
    )

    assert (
        statistics.coordinator_return
        == pytest.approx(
            6.0
        )
    )


def test_early_termination():

    loop = build_training_loop(
        steps_per_episode=10,
        done_after=2,
    )

    statistics = (
        loop.run_episode(
            episode=1
        )
    )

    assert statistics.steps == 3

    assert (
        statistics.terminated_early
        is True
    )


# ============================================================
# HISTORY
# ============================================================

def test_empty_history():

    history = TrainingHistory()

    assert (
        history.number_of_episodes
        == 0
    )

    assert history.total_steps == 0

    assert (
        history.final_return
        is None
    )


def test_history_properties():

    loop = build_training_loop(
        maximum_episodes=2,
        steps_per_episode=2,
    )

    history = loop.train()

    assert (
        history.number_of_episodes
        == 2
    )

    assert history.total_steps == 4

    assert len(
        history.total_returns
    ) == 2


# ============================================================
# COMPLETE TRAINING
# ============================================================

def test_complete_training():

    loop = build_training_loop(
        maximum_episodes=3,
        steps_per_episode=2,
    )

    history = loop.train()

    assert isinstance(
        history,
        TrainingHistory,
    )

    assert (
        history.number_of_episodes
        == 3
    )


def test_training_global_steps():

    loop = build_training_loop(
        maximum_episodes=3,
        steps_per_episode=2,
    )

    loop.train()

    assert (
        loop.global_step
        == 6
    )


# ============================================================
# EXTERNAL CONVERGENCE
# ============================================================

def test_stop_function():

    local_agents = (
        build_test_local_agents()
    )

    coordinator = (
        build_test_coordinator()
    )

    config = TrainingLoopConfig(
        maximum_episodes=10,

        steps_per_episode=1,

        enable_checkpointing=False,

        verbose=False,
    )

    def stop_function(
        history,
    ):

        return (
            history.number_of_episodes
            >= 3
        )

    loop = FCHMARLTrainingLoop(
        local_agents=local_agents,

        coordinator_agent=(
            coordinator
        ),

        reset_function=(
            make_reset_function()
        ),

        step_function=(
            make_step_function()
        ),

        config=config,

        stop_function=(
            stop_function
        ),
    )

    history = loop.train()

    assert (
        history.number_of_episodes
        == 3
    )


# ============================================================
# SUMMARY
# ============================================================

def test_loop_summary():

    loop = build_training_loop(
        maximum_episodes=2,
        steps_per_episode=2,
    )

    loop.train()

    summary = loop.summary()

    assert (
        summary[
            "number_of_local_agents"
        ]
        == 5
    )

    assert (
        summary[
            "completed_episodes"
        ]
        == 2
    )

    assert (
        summary[
            "global_step"
        ]
        == 4
    )


def test_history_summary():

    loop = build_training_loop(
        maximum_episodes=2,
        steps_per_episode=2,
    )

    history = loop.train()

    summary = (
        history.summary()
    )

    assert (
        summary[
            "number_of_episodes"
        ]
        == 2
    )

    assert (
        summary[
            "final_return"
        ]
        == pytest.approx(
            14.0
        )
    )
