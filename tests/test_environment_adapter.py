"""
Tests for marl/environment_adapter.py.
"""

import numpy as np
import pytest

from marl.environment_adapter import (
    AdapterStepRecord,
    EnvironmentAdapterConfig,
    FCHMARLEnvironmentAdapter,
    RewardOutput,
)

from marl.training_loop import (
    EnvironmentStepResult,
    HierarchicalActionBundle,
    TrainingObservation,
)


NUMBER_OF_MICROGRIDS = 5

LOCAL_STATE_DIMENSION = 6

COORDINATOR_STATE_DIMENSION = 4


# ============================================================
# DUMMY PHYSICAL ENVIRONMENT
# ============================================================

class DummyPhysicalEnvironment:
    """
    Small deterministic physical-environment surrogate used
    only for software testing.

    This is NOT a manuscript physical model.
    """

    def __init__(self):

        self.episode = None

        self.time_step = 0

        self.last_action = None

    def reset(
        self,
        episode,
    ):

        self.episode = episode

        self.time_step = 0

        self.last_action = None

        return self._make_state()

    def step(
        self,
        action,
    ):

        self.last_action = action

        self.time_step += 1

        return self._make_state()

    def _make_state(
        self,
    ):

        return {
            "episode":
                self.episode,

            "time_step":
                self.time_step,

            "local_values": [
                float(
                    self.time_step
                    + index
                )
                for index
                in range(
                    NUMBER_OF_MICROGRIDS
                )
            ],

            "coordinator_value":
                float(
                    self.time_step
                ),

            "physical_done":
                False,
        }


# ============================================================
# BUILDERS
# ============================================================

def observation_builder(
    raw_state,
):

    local_states = []

    for value in (
        raw_state[
            "local_values"
        ]
    ):

        local_states.append(
            np.full(
                LOCAL_STATE_DIMENSION,
                value,
                dtype=np.float32,
            )
        )

    coordinator_state = np.full(
        COORDINATOR_STATE_DIMENSION,
        raw_state[
            "coordinator_value"
        ],
        dtype=np.float32,
    )

    return TrainingObservation(
        local_states=(
            local_states
        ),

        coordinator_state=(
            coordinator_state
        ),
    )


def action_converter(
    actions,
):

    return {
        "local_actions": [
            np.asarray(
                action,
                dtype=np.float32,
            ).copy()
            for action
            in actions.local_actions
        ],

        "coordinator_action":
            np.asarray(
                actions
                .coordinator_action,
                dtype=np.float32,
            ).copy(),
    }


def reward_builder(
    previous_raw_state,
    next_raw_state,
    hierarchical_actions,
    physical_action,
):

    del (
        previous_raw_state,
        hierarchical_actions,
        physical_action,
    )

    local_rewards = [
        1.0
        for _ in range(
            NUMBER_OF_MICROGRIDS
        )
    ]

    coordinator_reward = 2.0

    return RewardOutput(
        local_rewards=(
            local_rewards
        ),

        coordinator_reward=(
            coordinator_reward
        ),

        components={
            "time_step":
                next_raw_state[
                    "time_step"
                ]
        },
    )


def done_extractor(
    raw_state,
):

    return bool(
        raw_state[
            "physical_done"
        ]
    )


def info_extractor(
    raw_state,
):

    return {
        "physical_time_step":
            raw_state[
                "time_step"
            ]
    }


def make_actions():

    local_actions = [
        np.array(
            [
                0.1,
                -0.1,
            ],
            dtype=np.float32,
        )
        for _ in range(
            NUMBER_OF_MICROGRIDS
        )
    ]

    coordinator_action = (
        np.array(
            [
                0.2,
                -0.2,
            ],
            dtype=np.float32,
        )
    )

    return HierarchicalActionBundle(
        local_actions=local_actions,
        coordinator_action=(
            coordinator_action
        ),
    )


def build_adapter(
    horizon=24,
):

    environment = (
        DummyPhysicalEnvironment()
    )

    config = EnvironmentAdapterConfig(
        number_of_microgrids=5,

        strict_validation=True,

        terminate_after_steps=(
            horizon
        ),
    )

    adapter = FCHMARLEnvironmentAdapter(
        environment_reset_function=(
            environment.reset
        ),

        environment_step_function=(
            environment.step
        ),

        observation_builder=(
            observation_builder
        ),

        action_converter=(
            action_converter
        ),

        reward_builder=(
            reward_builder
        ),

        done_extractor=(
            done_extractor
        ),

        info_extractor=(
            info_extractor
        ),

        config=config,
    )

    return (
        adapter,
        environment,
    )


# ============================================================
# CONFIG
# ============================================================

def test_default_configuration():

    config = (
        EnvironmentAdapterConfig()
    )

    assert (
        config.number_of_microgrids
        == 5
    )

    assert (
        config.terminate_after_steps
        == 24
    )


def test_valid_configuration():

    EnvironmentAdapterConfig().validate()


def test_invalid_number_of_microgrids():

    with pytest.raises(
        ValueError
    ):

        EnvironmentAdapterConfig(
            number_of_microgrids=0
        ).validate()


def test_invalid_horizon():

    with pytest.raises(
        ValueError
    ):

        EnvironmentAdapterConfig(
            terminate_after_steps=0
        ).validate()


# ============================================================
# REWARD OUTPUT
# ============================================================

def test_valid_reward_output():

    reward = RewardOutput(
        local_rewards=[
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
        ],

        coordinator_reward=6.0,
    )

    reward.validate(
        number_of_microgrids=5
    )


def test_reward_wrong_count():

    reward = RewardOutput(
        local_rewards=[
            1.0
        ],

        coordinator_reward=2.0,
    )

    with pytest.raises(
        ValueError
    ):

        reward.validate(
            number_of_microgrids=5
        )


def test_reward_nan():

    reward = RewardOutput(
        local_rewards=[
            1.0,
            1.0,
            np.nan,
            1.0,
            1.0,
        ],

        coordinator_reward=2.0,
    )

    with pytest.raises(
        ValueError
    ):

        reward.validate(
            number_of_microgrids=5
        )


def test_coordinator_reward_nan():

    reward = RewardOutput(
        local_rewards=[
            1.0
            for _ in range(
                5
            )
        ],

        coordinator_reward=np.nan,
    )

    with pytest.raises(
        ValueError
    ):

        reward.validate(
            number_of_microgrids=5
        )


# ============================================================
# CONSTRUCTION
# ============================================================

def test_adapter_creation():

    adapter, _ = (
        build_adapter()
    )

    assert isinstance(
        adapter,
        FCHMARLEnvironmentAdapter,
    )


def test_invalid_reset_function():

    with pytest.raises(
        TypeError
    ):

        FCHMARLEnvironmentAdapter(
            environment_reset_function=None,

            environment_step_function=lambda x: x,

            observation_builder=lambda x: x,

            action_converter=lambda x: x,

            reward_builder=lambda a, b, c, d: d,
        )


# ============================================================
# RESET
# ============================================================

def test_reset_returns_training_observation():

    adapter, _ = (
        build_adapter()
    )

    observation = (
        adapter.reset(
            episode=1
        )
    )

    assert isinstance(
        observation,
        TrainingObservation,
    )


def test_reset_has_five_local_states():

    adapter, _ = (
        build_adapter()
    )

    observation = (
        adapter.reset(
            episode=1
        )
    )

    assert len(
        observation.local_states
    ) == 5


def test_reset_counter():

    adapter, _ = (
        build_adapter()
    )

    adapter.reset(
        episode=1
    )

    adapter.reset(
        episode=2
    )

    assert (
        adapter.total_resets
        == 2
    )


def test_reset_episode():

    adapter, _ = (
        build_adapter()
    )

    adapter.reset(
        episode=7
    )

    assert (
        adapter.current_episode
        == 7
    )


def test_invalid_episode():

    adapter, _ = (
        build_adapter()
    )

    with pytest.raises(
        ValueError
    ):

        adapter.reset(
            episode=0
        )


# ============================================================
# ACTION CONVERSION
# ============================================================

def test_action_converter():

    actions = make_actions()

    physical_action = (
        action_converter(
            actions
        )
    )

    assert isinstance(
        physical_action,
        dict,
    )

    assert len(
        physical_action[
            "local_actions"
        ]
    ) == 5

    assert (
        physical_action[
            "coordinator_action"
        ].shape
        == (
            2,
        )
    )


# ============================================================
# STEP
# ============================================================

def test_step_requires_reset():

    adapter, _ = (
        build_adapter()
    )

    with pytest.raises(
        RuntimeError
    ):

        adapter.step(
            episode=1,
            time_step=0,
            actions=(
                make_actions()
            ),
        )


def test_step_returns_environment_result():

    adapter, _ = (
        build_adapter()
    )

    adapter.reset(
        episode=1
    )

    result = adapter.step(
        episode=1,
        time_step=0,
        actions=make_actions(),
    )

    assert isinstance(
        result,
        EnvironmentStepResult,
    )


def test_step_local_rewards():

    adapter, _ = (
        build_adapter()
    )

    adapter.reset(
        episode=1
    )

    result = adapter.step(
        episode=1,
        time_step=0,
        actions=make_actions(),
    )

    assert result.local_rewards == (
        pytest.approx(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ]
        )
    )


def test_step_coordinator_reward():

    adapter, _ = (
        build_adapter()
    )

    adapter.reset(
        episode=1
    )

    result = adapter.step(
        episode=1,
        time_step=0,
        actions=make_actions(),
    )

    assert (
        result.coordinator_reward
        == pytest.approx(
            2.0
        )
    )


def test_step_updates_time():

    adapter, _ = (
        build_adapter()
    )

    adapter.reset(
        episode=1
    )

    adapter.step(
        episode=1,
        time_step=0,
        actions=make_actions(),
    )

    assert (
        adapter.current_time_step
        == 1
    )

    assert (
        adapter.total_steps
        == 1
    )


def test_step_stores_physical_action():

    adapter, environment = (
        build_adapter()
    )

    adapter.reset(
        episode=1
    )

    adapter.step(
        episode=1,
        time_step=0,
        actions=make_actions(),
    )

    assert (
        environment.last_action
        is not None
    )


def test_wrong_episode_rejected():

    adapter, _ = (
        build_adapter()
    )

    adapter.reset(
        episode=1
    )

    with pytest.raises(
        ValueError
    ):

        adapter.step(
            episode=2,
            time_step=0,
            actions=make_actions(),
        )


def test_wrong_time_step_rejected():

    adapter, _ = (
        build_adapter()
    )

    adapter.reset(
        episode=1
    )

    with pytest.raises(
        ValueError
    ):

        adapter.step(
            episode=1,
            time_step=5,
            actions=make_actions(),
        )


# ============================================================
# HORIZON TERMINATION
# ============================================================

def test_not_done_before_horizon():

    adapter, _ = (
        build_adapter(
            horizon=3
        )
    )

    adapter.reset(
        episode=1
    )

    result = adapter.step(
        episode=1,
        time_step=0,
        actions=make_actions(),
    )

    assert result.done is False


def test_done_at_horizon():

    adapter, _ = (
        build_adapter(
            horizon=3
        )
    )

    adapter.reset(
        episode=1
    )

    adapter.step(
        episode=1,
        time_step=0,
        actions=make_actions(),
    )

    adapter.step(
        episode=1,
        time_step=1,
        actions=make_actions(),
    )

    result = adapter.step(
        episode=1,
        time_step=2,
        actions=make_actions(),
    )

    assert result.done is True


# ============================================================
# INFO
# ============================================================

def test_info_extraction():

    adapter, _ = (
        build_adapter()
    )

    adapter.reset(
        episode=1
    )

    result = adapter.step(
        episode=1,
        time_step=0,
        actions=make_actions(),
    )

    assert (
        result.info[
            "physical_time_step"
        ]
        == 1
    )

    assert (
        result.info[
            "episode"
        ]
        == 1
    )


# ============================================================
# RECORD
# ============================================================

def test_last_record():

    adapter, _ = (
        build_adapter()
    )

    adapter.reset(
        episode=1
    )

    adapter.step(
        episode=1,
        time_step=0,
        actions=make_actions(),
    )

    assert isinstance(
        adapter.last_record,
        AdapterStepRecord,
    )

    assert (
        adapter.last_record
        .episode
        == 1
    )


# ============================================================
# CALLBACK COMPATIBILITY
# ============================================================

def test_reset_function_callback():

    adapter, _ = (
        build_adapter()
    )

    observation = (
        adapter.reset_function(
            1
        )
    )

    assert isinstance(
        observation,
        TrainingObservation,
    )


def test_step_function_callback():

    adapter, _ = (
        build_adapter()
    )

    adapter.reset_function(
        1
    )

    result = (
        adapter.step_function(
            1,
            0,
            make_actions(),
        )
    )

    assert isinstance(
        result,
        EnvironmentStepResult,
    )


# ============================================================
# SUMMARY
# ============================================================

def test_summary_before_reset():

    adapter, _ = (
        build_adapter()
    )

    summary = (
        adapter.summary()
    )

    assert (
        summary[
            "number_of_microgrids"
        ]
        == 5
    )

    assert (
        summary[
            "total_steps"
        ]
        == 0
    )


def test_summary_after_step():

    adapter, _ = (
        build_adapter()
    )

    adapter.reset(
        episode=1
    )

    adapter.step(
        episode=1,
        time_step=0,
        actions=make_actions(),
    )

    summary = (
        adapter.summary()
    )

    assert (
        summary[
            "current_episode"
        ]
        == 1
    )

    assert (
        summary[
            "current_time_step"
        ]
        == 1
    )

    assert (
        summary[
            "total_steps"
        ]
        == 1
    )

    assert (
        summary[
            "has_last_record"
        ]
        is True
    )