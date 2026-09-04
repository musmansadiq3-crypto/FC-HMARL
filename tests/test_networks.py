"""
Tests for agents/networks.py.
"""

import numpy as np
import pytest
import torch
from torch import nn

from agents.networks import (
    GaussianPolicyNetwork,
    QNetwork,
    SACNetworkConfig,
    TwinQNetwork,
    build_activation,
    build_mlp,
    build_sac_networks,
    validate_action_tensor,
    validate_state_tensor,
)


# ============================================================
# FIXTURE
# ============================================================

@pytest.fixture
def config():

    return SACNetworkConfig(
        state_dimension=10,
        action_dimension=3,
        hidden_dimensions=(
            32,
            16,
        ),
        activation="relu",
        action_low=-1.0,
        action_high=1.0,
    )


@pytest.fixture
def actor(
    config,
):

    torch.manual_seed(
        42
    )

    return GaussianPolicyNetwork(
        config
    )


@pytest.fixture
def critic(
    config,
):

    torch.manual_seed(
        42
    )

    return TwinQNetwork(
        config
    )


# ============================================================
# CONFIGURATION
# ============================================================

def test_valid_config(
    config,
):

    config.validate()


def test_invalid_state_dimension():

    with pytest.raises(ValueError):

        SACNetworkConfig(
            state_dimension=0,
            action_dimension=2,
        ).validate()


def test_invalid_action_dimension():

    with pytest.raises(ValueError):

        SACNetworkConfig(
            state_dimension=4,
            action_dimension=0,
        ).validate()


def test_empty_hidden_dimensions():

    with pytest.raises(ValueError):

        SACNetworkConfig(
            state_dimension=4,
            action_dimension=2,
            hidden_dimensions=(),
        ).validate()


def test_invalid_hidden_dimension():

    with pytest.raises(ValueError):

        SACNetworkConfig(
            state_dimension=4,
            action_dimension=2,
            hidden_dimensions=(
                64,
                0,
            ),
        ).validate()


def test_invalid_activation():

    with pytest.raises(ValueError):

        SACNetworkConfig(
            state_dimension=4,
            action_dimension=2,
            activation="invalid",
        ).validate()


def test_invalid_log_std_range():

    with pytest.raises(ValueError):

        SACNetworkConfig(
            state_dimension=4,
            action_dimension=2,
            log_std_min=2.0,
            log_std_max=-2.0,
        ).validate()


def test_invalid_action_bounds():

    with pytest.raises(ValueError):

        SACNetworkConfig(
            state_dimension=4,
            action_dimension=2,
            action_low=1.0,
            action_high=-1.0,
        ).validate()


def test_vector_action_bounds():

    config = SACNetworkConfig(
        state_dimension=4,
        action_dimension=2,
        action_low=[
            -2.0,
            0.0,
        ],
        action_high=[
            2.0,
            5.0,
        ],
    )

    config.validate()


# ============================================================
# ACTIVATION
# ============================================================

def test_relu_activation():

    assert isinstance(
        build_activation(
            "relu"
        ),
        nn.ReLU,
    )


def test_gelu_activation():

    assert isinstance(
        build_activation(
            "gelu"
        ),
        nn.GELU,
    )


def test_elu_activation():

    assert isinstance(
        build_activation(
            "elu"
        ),
        nn.ELU,
    )


def test_tanh_activation():

    assert isinstance(
        build_activation(
            "tanh"
        ),
        nn.Tanh,
    )


# ============================================================
# MLP
# ============================================================

def test_build_mlp():

    network, output_dimension = (
        build_mlp(
            input_dimension=10,
            hidden_dimensions=(
                32,
                16,
            ),
            activation="relu",
        )
    )

    assert isinstance(
        network,
        nn.Sequential,
    )

    assert output_dimension == 16


# ============================================================
# STATE VALIDATION
# ============================================================

def test_valid_state_tensor():

    state = torch.zeros(
        4,
        10,
    )

    validate_state_tensor(
        state,
        10,
    )


def test_invalid_state_rank():

    state = torch.zeros(
        10
    )

    with pytest.raises(ValueError):

        validate_state_tensor(
            state,
            10,
        )


def test_invalid_state_dimension_tensor():

    state = torch.zeros(
        4,
        9,
    )

    with pytest.raises(ValueError):

        validate_state_tensor(
            state,
            10,
        )


def test_state_nan_rejected():

    state = torch.zeros(
        4,
        10,
    )

    state[
        0,
        0
    ] = float(
        "nan"
    )

    with pytest.raises(ValueError):

        validate_state_tensor(
            state,
            10,
        )


# ============================================================
# ACTION VALIDATION
# ============================================================

def test_valid_action_tensor():

    action = torch.zeros(
        4,
        3,
    )

    validate_action_tensor(
        action,
        3,
    )


def test_invalid_action_dimension_tensor():

    action = torch.zeros(
        4,
        2,
    )

    with pytest.raises(ValueError):

        validate_action_tensor(
            action,
            3,
        )


# ============================================================
# ACTOR CONSTRUCTION
# ============================================================

def test_actor_construction(
    actor,
):

    assert isinstance(
        actor,
        GaussianPolicyNetwork,
    )


def test_actor_has_parameters(
    actor,
):

    assert (
        actor.parameter_count()
        > 0
    )


# ============================================================
# ACTOR FORWARD
# ============================================================

def test_actor_forward_shapes(
    actor,
):

    state = torch.randn(
        5,
        10,
    )

    mean, log_std = actor(
        state
    )

    assert mean.shape == (
        5,
        3,
    )

    assert log_std.shape == (
        5,
        3,
    )


def test_actor_log_std_bounds(
    actor,
):

    state = torch.randn(
        10,
        10,
    )

    _, log_std = actor(
        state
    )

    assert torch.all(
        log_std
        >= actor.config.log_std_min
    )

    assert torch.all(
        log_std
        <= actor.config.log_std_max
    )


# ============================================================
# ACTOR SAMPLING
# ============================================================

def test_actor_sample_shapes(
    actor,
):

    state = torch.randn(
        6,
        10,
    )

    (
        action,
        log_probability,
        deterministic_action,
    ) = actor.sample(
        state
    )

    assert action.shape == (
        6,
        3,
    )

    assert log_probability.shape == (
        6,
        1,
    )

    assert deterministic_action.shape == (
        6,
        3,
    )


def test_actor_sample_finite(
    actor,
):

    state = torch.randn(
        6,
        10,
    )

    action, log_probability, _ = (
        actor.sample(
            state
        )
    )

    assert torch.isfinite(
        action
    ).all()

    assert torch.isfinite(
        log_probability
    ).all()


def test_actor_action_bounds(
    actor,
):

    state = torch.randn(
        100,
        10,
    )

    action, _, _ = actor.sample(
        state
    )

    assert torch.all(
        action >= -1.0
    )

    assert torch.all(
        action <= 1.0
    )


def test_actor_deterministic_shape(
    actor,
):

    state = torch.randn(
        4,
        10,
    )

    action = actor.deterministic(
        state
    )

    assert action.shape == (
        4,
        3,
    )


# ============================================================
# CUSTOM ACTION BOUNDS
# ============================================================

def test_custom_action_bounds():

    config = SACNetworkConfig(
        state_dimension=5,
        action_dimension=2,
        hidden_dimensions=(
            16,
            16,
        ),
        action_low=[
            -2.0,
            10.0,
        ],
        action_high=[
            2.0,
            20.0,
        ],
    )

    actor = GaussianPolicyNetwork(
        config
    )

    state = torch.randn(
        50,
        5,
    )

    action, _, _ = actor.sample(
        state
    )

    assert torch.all(
        action[
            :,
            0
        ] >= -2.0
    )

    assert torch.all(
        action[
            :,
            0
        ] <= 2.0
    )

    assert torch.all(
        action[
            :,
            1
        ] >= 10.0
    )

    assert torch.all(
        action[
            :,
            1
        ] <= 20.0
    )


# ============================================================
# Q NETWORK
# ============================================================

def test_single_q_network(
    config,
):

    q_network = QNetwork(
        config
    )

    state = torch.randn(
        5,
        10,
    )

    action = torch.randn(
        5,
        3,
    )

    q = q_network(
        state,
        action,
    )

    assert q.shape == (
        5,
        1,
    )


def test_q_batch_mismatch(
    config,
):

    q_network = QNetwork(
        config
    )

    state = torch.randn(
        5,
        10,
    )

    action = torch.randn(
        4,
        3,
    )

    with pytest.raises(ValueError):

        q_network(
            state,
            action,
        )


# ============================================================
# TWIN CRITIC
# ============================================================

def test_twin_critic_construction(
    critic,
):

    assert isinstance(
        critic,
        TwinQNetwork,
    )


def test_twin_critic_shapes(
    critic,
):

    state = torch.randn(
        5,
        10,
    )

    action = torch.randn(
        5,
        3,
    )

    q1, q2 = critic(
        state,
        action,
    )

    assert q1.shape == (
        5,
        1,
    )

    assert q2.shape == (
        5,
        1,
    )


def test_twin_critic_minimum(
    critic,
):

    state = torch.randn(
        5,
        10,
    )

    action = torch.randn(
        5,
        3,
    )

    q1, q2 = critic(
        state,
        action,
    )

    minimum = critic.minimum(
        state,
        action,
    )

    assert torch.allclose(
        minimum,
        torch.minimum(
            q1,
            q2,
        ),
    )


def test_twin_q_are_independent(
    critic,
):

    q1_parameters = list(
        critic.q1.parameters()
    )

    q2_parameters = list(
        critic.q2.parameters()
    )

    assert (
        q1_parameters[
            0
        ]
        is not
        q2_parameters[
            0
        ]
    )


# ============================================================
# BACKPROPAGATION
# ============================================================

def test_actor_backpropagation(
    actor,
):

    state = torch.randn(
        8,
        10,
    )

    action, log_probability, _ = (
        actor.sample(
            state
        )
    )

    loss = (
        action.mean()
        + log_probability.mean()
    )

    loss.backward()

    gradients = [
        parameter.grad
        for parameter
        in actor.parameters()
        if parameter.requires_grad
    ]

    assert any(
        gradient is not None
        for gradient
        in gradients
    )


def test_critic_backpropagation(
    critic,
):

    state = torch.randn(
        8,
        10,
    )

    action = torch.randn(
        8,
        3,
    )

    q1, q2 = critic(
        state,
        action,
    )

    loss = (
        q1.mean()
        + q2.mean()
    )

    loss.backward()

    gradients = [
        parameter.grad
        for parameter
        in critic.parameters()
        if parameter.requires_grad
    ]

    assert any(
        gradient is not None
        for gradient
        in gradients
    )


# ============================================================
# FACTORY
# ============================================================

def test_build_sac_networks(
    config,
):

    actor, critic, target = (
        build_sac_networks(
            config
        )
    )

    assert isinstance(
        actor,
        GaussianPolicyNetwork,
    )

    assert isinstance(
        critic,
        TwinQNetwork,
    )

    assert isinstance(
        target,
        TwinQNetwork,
    )


def test_target_matches_critic_initially(
    config,
):

    _, critic, target = (
        build_sac_networks(
            config
        )
    )

    for critic_parameter, target_parameter in zip(
        critic.parameters(),
        target.parameters(),
    ):

        assert torch.allclose(
            critic_parameter,
            target_parameter,
        )


def test_target_has_no_gradients(
    config,
):

    _, _, target = (
        build_sac_networks(
            config
        )
    )

    for parameter in target.parameters():

        assert (
            parameter.requires_grad
            is False
        )


# ============================================================
# SUMMARY
# ============================================================

def test_actor_summary(
    actor,
):

    summary = actor.summary()

    assert summary[
        "network"
    ] == "GaussianPolicyNetwork"

    assert summary[
        "state_dimension"
    ] == 10

    assert summary[
        "action_dimension"
    ] == 3


def test_critic_summary(
    critic,
):

    summary = critic.summary()

    assert summary[
        "network"
    ] == "TwinQNetwork"

    assert summary[
        "state_dimension"
    ] == 10

    assert summary[
        "action_dimension"
    ] == 3