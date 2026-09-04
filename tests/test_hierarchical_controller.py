"""
Tests for marl/hierarchical_controller.py.
"""

import numpy as np
import pytest

from marl.hierarchical_controller import (
    HierarchicalActionResult,
    HierarchicalController,
    HierarchicalControllerConfig,
    HierarchicalControllerState,
    flatten_hierarchical_actions,
    request_agent_action,
    validate_action_vector,
    validate_agent_interface,
    validate_state_vector,
)


# ============================================================
# TEST AGENTS
# ============================================================

class DummyAgent:
    """
    Simple test-only agent.
    """

    def __init__(
        self,
        action,
    ):

        self.action = np.asarray(
            action,
            dtype=np.float32,
        )

        self.last_state = None

        self.last_deterministic = None

        self.reset_called = False

        self.training_mode = None

    def select_action(
        self,
        state,
        deterministic=False,
    ):

        self.last_state = np.asarray(
            state
        ).copy()

        self.last_deterministic = (
            deterministic
        )

        return self.action.copy()

    def reset(
        self,
    ):

        self.reset_called = True

    def set_training_mode(
        self,
        training,
    ):

        self.training_mode = training


class SimpleInterfaceAgent:
    """
    Agent that accepts only select_action(state).
    """

    def __init__(
        self,
        action,
    ):

        self.action = np.asarray(
            action,
            dtype=np.float32,
        )

    def select_action(
        self,
        state,
    ):

        return self.action.copy()


class InvalidAgent:

    pass


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def config():

    return HierarchicalControllerConfig(
        number_of_microgrids=3,
        deterministic_evaluation=True,
        validate_actions=True,
        dtype="float32",
    )


@pytest.fixture
def local_agents():

    return [
        DummyAgent(
            [
                0.1,
                0.2,
            ]
        ),
        DummyAgent(
            [
                0.3,
                0.4,
            ]
        ),
        DummyAgent(
            [
                0.5,
                0.6,
            ]
        ),
    ]


@pytest.fixture
def coordinator_agent():

    return DummyAgent(
        [
            0.7,
            0.8,
            0.9,
        ]
    )


@pytest.fixture
def local_states():

    return [
        np.array(
            [
                1.0,
                2.0,
                3.0,
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                4.0,
                5.0,
                6.0,
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                7.0,
                8.0,
                9.0,
            ],
            dtype=np.float32,
        ),
    ]


@pytest.fixture
def coordinator_state():

    return np.array(
        [
            10.0,
            11.0,
            12.0,
            13.0,
        ],
        dtype=np.float32,
    )


# ============================================================
# CONFIGURATION
# ============================================================

def test_default_microgrid_count():

    config = (
        HierarchicalControllerConfig()
    )

    assert config.number_of_microgrids == 5


def test_default_deterministic_evaluation():

    config = (
        HierarchicalControllerConfig()
    )

    assert (
        config.deterministic_evaluation
        is True
    )


def test_valid_config():

    (
        HierarchicalControllerConfig()
        .validate()
    )


def test_invalid_microgrid_count():

    config = (
        HierarchicalControllerConfig(
            number_of_microgrids=0
        )
    )

    with pytest.raises(ValueError):

        config.validate()


def test_invalid_dtype():

    config = (
        HierarchicalControllerConfig(
            dtype="int32"
        )
    )

    with pytest.raises(ValueError):

        config.validate()


# ============================================================
# STATE VALIDATION
# ============================================================

def test_valid_state_vector():

    state = validate_state_vector(
        [
            1,
            2,
            3,
        ],
        "state",
    )

    assert state.shape == (
        3,
    )


def test_state_vector_wrong_rank():

    with pytest.raises(ValueError):

        validate_state_vector(
            [
                [
                    1,
                    2,
                ],
                [
                    3,
                    4,
                ],
            ],
            "state",
        )


def test_state_vector_nan():

    with pytest.raises(ValueError):

        validate_state_vector(
            [
                1,
                np.nan,
            ],
            "state",
        )


# ============================================================
# ACTION VALIDATION
# ============================================================

def test_valid_action_vector():

    action = validate_action_vector(
        [
            0.1,
            0.2,
        ],
        "action",
    )

    assert action.shape == (
        2,
    )


def test_action_wrong_rank():

    with pytest.raises(ValueError):

        validate_action_vector(
            [
                [
                    1,
                    2,
                ]
            ],
            "action",
        )


def test_action_nan():

    with pytest.raises(ValueError):

        validate_action_vector(
            [
                0.1,
                np.nan,
            ],
            "action",
        )


# ============================================================
# AGENT INTERFACE
# ============================================================

def test_valid_agent_interface():

    agent = DummyAgent(
        [
            1,
        ]
    )

    validate_agent_interface(
        agent,
        "agent",
    )


def test_invalid_agent_interface():

    with pytest.raises(TypeError):

        validate_agent_interface(
            InvalidAgent(),
            "agent",
        )


# ============================================================
# SINGLE AGENT ACTION
# ============================================================

def test_request_agent_action():

    agent = DummyAgent(
        [
            0.2,
            0.4,
        ]
    )

    action = request_agent_action(
        agent,
        np.array(
            [
                1,
                2,
                3,
            ]
        ),
        deterministic=True,
    )

    assert np.allclose(
        action,
        [
            0.2,
            0.4,
        ],
    )

    assert (
        agent.last_deterministic
        is True
    )


def test_simple_agent_interface_supported():

    agent = SimpleInterfaceAgent(
        [
            0.3,
        ]
    )

    action = request_agent_action(
        agent,
        [
            1,
            2,
        ],
        deterministic=True,
    )

    assert np.allclose(
        action,
        [
            0.3,
        ],
    )


# ============================================================
# ACTION FLATTENING
# ============================================================

def test_flatten_actions():

    local_actions = [
        np.array(
            [
                1,
                2,
            ]
        ),
        np.array(
            [
                3,
                4,
            ]
        ),
    ]

    coordinator = np.array(
        [
            5,
            6,
            7,
        ]
    )

    flat = flatten_hierarchical_actions(
        local_actions,
        coordinator,
    )

    assert np.allclose(
        flat,
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
        ],
    )


def test_flatten_action_dtype():

    flat = flatten_hierarchical_actions(
        [
            np.array(
                [
                    1,
                    2,
                ]
            )
        ],
        np.array(
            [
                3,
            ]
        ),
        dtype="float32",
    )

    assert flat.dtype == np.float32


def test_empty_local_actions():

    with pytest.raises(ValueError):

        flatten_hierarchical_actions(
            [],
            np.array(
                [
                    1,
                ]
            ),
        )


# ============================================================
# CONTROLLER STATE
# ============================================================

def test_controller_state_validation(
    local_states,
    coordinator_state,
):

    state = (
        HierarchicalControllerState(
            local_states=local_states,
            coordinator_state=(
                coordinator_state
            ),
        )
    )

    state.validate(
        3
    )


def test_controller_state_wrong_count(
    local_states,
    coordinator_state,
):

    state = (
        HierarchicalControllerState(
            local_states=(
                local_states[:2]
            ),
            coordinator_state=(
                coordinator_state
            ),
        )
    )

    with pytest.raises(ValueError):

        state.validate(
            3
        )


# ============================================================
# CONTROLLER CREATION
# ============================================================

def test_controller_creation(
    config,
    local_agents,
    coordinator_agent,
):

    controller = (
        HierarchicalController(
            local_agents=(
                local_agents
            ),
            coordinator_agent=(
                coordinator_agent
            ),
            config=config,
        )
    )

    assert len(
        controller.local_agents
    ) == 3


def test_controller_wrong_agent_count(
    config,
    local_agents,
    coordinator_agent,
):

    with pytest.raises(ValueError):

        HierarchicalController(
            local_agents=(
                local_agents[:2]
            ),
            coordinator_agent=(
                coordinator_agent
            ),
            config=config,
        )


def test_controller_invalid_local_agent(
    config,
    local_agents,
    coordinator_agent,
):

    agents = list(
        local_agents
    )

    agents[
        0
    ] = InvalidAgent()

    with pytest.raises(TypeError):

        HierarchicalController(
            agents,
            coordinator_agent,
            config,
        )


# ============================================================
# DETERMINISTIC MODE
# ============================================================

def test_training_defaults_stochastic(
    config,
    local_agents,
    coordinator_agent,
):

    controller = (
        HierarchicalController(
            local_agents,
            coordinator_agent,
            config,
        )
    )

    result = (
        controller
        .resolve_deterministic_mode(
            training=True
        )
    )

    assert result is False


def test_evaluation_defaults_deterministic(
    config,
    local_agents,
    coordinator_agent,
):

    controller = (
        HierarchicalController(
            local_agents,
            coordinator_agent,
            config,
        )
    )

    result = (
        controller
        .resolve_deterministic_mode(
            training=False
        )
    )

    assert result is True


def test_explicit_deterministic_override(
    config,
    local_agents,
    coordinator_agent,
):

    controller = (
        HierarchicalController(
            local_agents,
            coordinator_agent,
            config,
        )
    )

    result = (
        controller
        .resolve_deterministic_mode(
            training=True,
            deterministic=True,
        )
    )

    assert result is True


# ============================================================
# LOCAL ACTION SELECTION
# ============================================================

def test_select_local_actions(
    config,
    local_agents,
    coordinator_agent,
    local_states,
):

    controller = (
        HierarchicalController(
            local_agents,
            coordinator_agent,
            config,
        )
    )

    actions = (
        controller.select_local_actions(
            local_states,
            deterministic=False,
        )
    )

    assert len(
        actions
    ) == 3

    assert np.allclose(
        actions[
            0
        ],
        [
            0.1,
            0.2,
        ],
    )


def test_local_agents_receive_states(
    config,
    local_agents,
    coordinator_agent,
    local_states,
):

    controller = (
        HierarchicalController(
            local_agents,
            coordinator_agent,
            config,
        )
    )

    controller.select_local_actions(
        local_states
    )

    assert np.allclose(
        local_agents[
            1
        ].last_state,
        local_states[
            1
        ],
    )


# ============================================================
# COORDINATOR ACTION
# ============================================================

def test_select_coordinator_action(
    config,
    local_agents,
    coordinator_agent,
    coordinator_state,
):

    controller = (
        HierarchicalController(
            local_agents,
            coordinator_agent,
            config,
        )
    )

    action = (
        controller
        .select_coordinator_action(
            coordinator_state
        )
    )

    assert np.allclose(
        action,
        [
            0.7,
            0.8,
            0.9,
        ],
    )


# ============================================================
# COMPLETE ACTION SELECTION
# ============================================================

def test_complete_action_selection(
    config,
    local_agents,
    coordinator_agent,
    local_states,
    coordinator_state,
):

    controller = (
        HierarchicalController(
            local_agents,
            coordinator_agent,
            config,
        )
    )

    result = controller.select_actions(
        local_states=local_states,
        coordinator_state=(
            coordinator_state
        ),
        training=True,
    )

    assert isinstance(
        result,
        HierarchicalActionResult,
    )


def test_complete_local_action_count(
    config,
    local_agents,
    coordinator_agent,
    local_states,
    coordinator_state,
):

    controller = (
        HierarchicalController(
            local_agents,
            coordinator_agent,
            config,
        )
    )

    result = controller.select_actions(
        local_states,
        coordinator_state,
    )

    assert len(
        result.local_actions
    ) == 3


def test_complete_flat_action(
    config,
    local_agents,
    coordinator_agent,
    local_states,
    coordinator_state,
):

    controller = (
        HierarchicalController(
            local_agents,
            coordinator_agent,
            config,
        )
    )

    result = controller.select_actions(
        local_states,
        coordinator_state,
    )

    assert np.allclose(
        result.flat_action,
        [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
        ],
    )


def test_training_actions_marked_nondeterministic(
    config,
    local_agents,
    coordinator_agent,
    local_states,
    coordinator_state,
):

    controller = (
        HierarchicalController(
            local_agents,
            coordinator_agent,
            config,
        )
    )

    result = controller.select_actions(
        local_states,
        coordinator_state,
        training=True,
    )

    assert (
        result.deterministic
        is False
    )


def test_evaluation_actions_marked_deterministic(
    config,
    local_agents,
    coordinator_agent,
    local_states,
    coordinator_state,
):

    controller = (
        HierarchicalController(
            local_agents,
            coordinator_agent,
            config,
        )
    )

    result = controller.select_actions(
        local_states,
        coordinator_state,
        training=False,
    )

    assert (
        result.deterministic
        is True
    )


def test_deterministic_flag_reaches_agents(
    config,
    local_agents,
    coordinator_agent,
    local_states,
    coordinator_state,
):

    controller = (
        HierarchicalController(
            local_agents,
            coordinator_agent,
            config,
        )
    )

    controller.select_actions(
        local_states,
        coordinator_state,
        training=False,
    )

    for agent in local_agents:

        assert (
            agent.last_deterministic
            is True
        )

    assert (
        coordinator_agent
        .last_deterministic
        is True
    )


# ============================================================
# RESULT
# ============================================================

def test_action_result_summary():

    result = (
        HierarchicalActionResult(
            local_actions=[
                np.array(
                    [
                        1,
                        2,
                    ]
                ),
                np.array(
                    [
                        3,
                    ]
                ),
            ],
            coordinator_action=np.array(
                [
                    4,
                    5,
                ]
            ),
            flat_action=np.array(
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                ]
            ),
            deterministic=True,
        )
    )

    result.validate(
        number_of_microgrids=2
    )

    summary = result.summary()

    assert summary[
        "number_of_local_agents"
    ] == 2

    assert summary[
        "total_action_dimension"
    ] == 5


# ============================================================
# RESET
# ============================================================

def test_controller_reset(
    config,
    local_agents,
    coordinator_agent,
):

    controller = (
        HierarchicalController(
            local_agents,
            coordinator_agent,
            config,
        )
    )

    controller.reset()

    for agent in local_agents:

        assert (
            agent.reset_called
            is True
        )

    assert (
        coordinator_agent.reset_called
        is True
    )


# ============================================================
# TRAINING MODE PROPAGATION
# ============================================================

def test_training_mode_true(
    config,
    local_agents,
    coordinator_agent,
):

    controller = (
        HierarchicalController(
            local_agents,
            coordinator_agent,
            config,
        )
    )

    controller.set_training_mode(
        True
    )

    for agent in local_agents:

        assert (
            agent.training_mode
            is True
        )

    assert (
        coordinator_agent.training_mode
        is True
    )


def test_training_mode_false(
    config,
    local_agents,
    coordinator_agent,
):

    controller = (
        HierarchicalController(
            local_agents,
            coordinator_agent,
            config,
        )
    )

    controller.set_training_mode(
        False
    )

    for agent in local_agents:

        assert (
            agent.training_mode
            is False
        )

    assert (
        coordinator_agent.training_mode
        is False
    )


# ============================================================
# SUMMARY
# ============================================================

def test_controller_summary(
    config,
    local_agents,
    coordinator_agent,
):

    controller = (
        HierarchicalController(
            local_agents,
            coordinator_agent,
            config,
        )
    )

    summary = controller.summary()

    assert summary[
        "number_of_microgrids"
    ] == 3

    assert summary[
        "number_of_local_agents"
    ] == 3

    assert summary[
        "has_coordinator"
    ] is True