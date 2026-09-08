import numpy as np
import pytest
import torch
from torch import nn
from agents.replay_buffer import ReplayBatch
from agents.sac_agent import (
    SACAgent,
    SACAgentConfig,
    SACUpdateResult,
    freeze_network,
    hard_update,
    soft_update,
    torch_dtype_from_name,
    unfreeze_network,
)
# ============================================================
# FIXTURES
# ============================================================
@pytest.fixture
def config():

    return SACAgentConfig(
        state_dimension=6,
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
        discount_factor=0.99,
        learning_rate=1e-4,
        replay_buffer_capacity=100,
        batch_size=8,
        soft_update_coefficient=0.005,
        entropy_coefficient=0.20,
        initial_exploration_noise=0.20,
        exploration_noise_decay=0.999,
        seed=42,
        device="cpu",
    )

@pytest.fixture
def agent(
    config,
):

    return SACAgent(
        config
    )

def random_transition(
    index=0,
):

    rng = np.random.default_rng(
        100 + index
    )

    state = rng.normal(
        size=6
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
        size=6
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
            *random_transition(
                index
            )
        )

# ============================================================
# CONFIGURATION
# ============================================================

def test_manuscript_default_gamma():

    config = SACAgentConfig(
        state_dimension=4,
        action_dimension=2,
    )

    assert config.discount_factor == pytest.approx(
        0.99
    )

def test_manuscript_default_learning_rate():

    config = SACAgentConfig(
        state_dimension=4,
        action_dimension=2,
    )

    assert config.learning_rate == pytest.approx(
        1e-4
    )

def test_manuscript_default_buffer_capacity():

    config = SACAgentConfig(
        state_dimension=4,
        action_dimension=2,
    )

    assert config.replay_buffer_capacity == 1_000_000

def test_manuscript_default_batch_size():

    config = SACAgentConfig(
        state_dimension=4,
        action_dimension=2,
    )

    assert config.batch_size == 512

def test_manuscript_default_tau():

    config = SACAgentConfig(
        state_dimension=4,
        action_dimension=2,
    )

    assert config.soft_update_coefficient == pytest.approx(
        0.005
    )

def test_manuscript_default_exploration_noise():

    config = SACAgentConfig(
        state_dimension=4,
        action_dimension=2,
    )

    assert config.initial_exploration_noise == pytest.approx(
        0.20
    )

def test_manuscript_default_noise_decay():

    config = SACAgentConfig(
        state_dimension=4,
        action_dimension=2,
    )

    assert config.exploration_noise_decay == pytest.approx(
        0.999
    )

def test_valid_config(
    config,
):

    config.validate()

def test_invalid_gamma():

    with pytest.raises(ValueError):

        SACAgentConfig(
            state_dimension=4,
            action_dimension=2,
            discount_factor=1.1,
        ).validate()

def test_invalid_learning_rate():

    with pytest.raises(ValueError):

        SACAgentConfig(
            state_dimension=4,
            action_dimension=2,
            learning_rate=0.0,
        ).validate()

def test_invalid_batch_size():

    with pytest.raises(ValueError):

        SACAgentConfig(
            state_dimension=4,
            action_dimension=2,
            batch_size=0,
        ).validate()

def test_batch_larger_than_buffer():

    with pytest.raises(ValueError):

        SACAgentConfig(
            state_dimension=4,
            action_dimension=2,
            replay_buffer_capacity=10,
            batch_size=20,
        ).validate()

def test_invalid_tau():

    with pytest.raises(ValueError):

        SACAgentConfig(
            state_dimension=4,
            action_dimension=2,
            soft_update_coefficient=0.0,
        ).validate()

def test_invalid_entropy_coefficient():

    with pytest.raises(ValueError):

        SACAgentConfig(
            state_dimension=4,
            action_dimension=2,
            entropy_coefficient=-0.1,
        ).validate()

def test_invalid_exploration_decay():

    with pytest.raises(ValueError):

        SACAgentConfig(
            state_dimension=4,
            action_dimension=2,
            exploration_noise_decay=1.5,
        ).validate()

# ============================================================
# DTYPE
# ============================================================

def test_float32_dtype():

    assert (
        torch_dtype_from_name(
            "float32"
        )
        == torch.float32
    )

def test_float64_dtype():

    assert (
        torch_dtype_from_name(
            "float64"
        )
        == torch.float64
    )

def test_invalid_dtype():

    with pytest.raises(ValueError):

        torch_dtype_from_name(
            "int32"
        )
# ============================================================
# AGENT CONSTRUCTION
# ============================================================

def test_agent_creation(
    agent,
):

    assert isinstance(
        agent,
        SACAgent,
    )


def test_actor_created(
    agent,
):

    assert agent.actor is not None


def test_critic_created(
    agent,
):

    assert agent.critic is not None


def test_target_critic_created(
    agent,
):

    assert agent.target_critic is not None


def test_replay_buffer_created(
    agent,
):

    assert (
        agent.replay_buffer.capacity
        == 100
    )


def test_adam_optimizers(
    agent,
):

    assert (
        agent.actor_optimizer
        .__class__.__name__
        == "Adam"
    )

    assert (
        agent.critic_optimizer
        .__class__.__name__
        == "Adam"
    )


# ============================================================
# TARGET INITIALIZATION
# ============================================================

def test_target_initially_matches_critic(
    agent,
):

    for critic_parameter, target_parameter in zip(
        agent.critic.parameters(),
        agent.target_critic.parameters(),
    ):

        assert torch.allclose(
            critic_parameter,
            target_parameter,
        )


def test_target_is_frozen(
    agent,
):

    assert all(
        parameter.requires_grad
        is False
        for parameter
        in agent.target_critic.parameters()
    )

# ============================================================
# FREEZE / UNFREEZE
# ============================================================

def test_freeze_network():

    network = nn.Linear(
        3,
        2,
    )

    freeze_network(
        network
    )

    assert all(
        parameter.requires_grad
        is False
        for parameter in network.parameters()
    )


def test_unfreeze_network():

    network = nn.Linear(
        3,
        2,
    )

    freeze_network(
        network
    )

    unfreeze_network(
        network
    )

    assert all(
        parameter.requires_grad
        is True
        for parameter in network.parameters()
    )

# ============================================================
# HARD UPDATE
# ============================================================

def test_hard_update():

    source = nn.Linear(
        3,
        2,
    )

    target = nn.Linear(
        3,
        2,
    )

    with torch.no_grad():

        source.weight.fill_(
            2.0
        )

    hard_update(
        target,
        source,
    )

    assert torch.allclose(
        target.weight,
        source.weight,
    )

# ============================================================
# SOFT UPDATE
# ============================================================

def test_soft_update():

    source = nn.Linear(
        1,
        1,
    )

    target = nn.Linear(
        1,
        1,
    )

    with torch.no_grad():

        source.weight.fill_(
            1.0
        )

        target.weight.fill_(
            0.0
        )

        source.bias.fill_(
            1.0
        )

        target.bias.fill_(
            0.0
        )

    soft_update(
        target,
        source,
        tau=0.5,
    )

    assert target.weight.item() == pytest.approx(
        0.5
    )

    assert target.bias.item() == pytest.approx(
        0.5
    )


def test_invalid_soft_update_tau():

    source = nn.Linear(
        1,
        1,
    )

    target = nn.Linear(
        1,
        1,
    )

    with pytest.raises(ValueError):

        soft_update(
            target,
            source,
            tau=0.0,
        )

# ============================================================
# ACTION SELECTION
# ============================================================

def test_deterministic_action_shape(
    agent,
):

    state = np.zeros(
        6,
        dtype=np.float32,
    )

    action = agent.select_action(
        state,
        deterministic=True,
    )

    assert action.shape == (
        2,
    )


def test_stochastic_action_shape(
    agent,
):

    state = np.zeros(
        6,
        dtype=np.float32,
    )

    action = agent.select_action(
        state,
        deterministic=False,
    )

    assert action.shape == (
        2,
    )


def test_action_within_bounds(
    agent,
):

    state = np.random.default_rng(
        1
    ).normal(
        size=6
    )

    for _ in range(
        20
    ):

        action = agent.select_action(
            state,
            deterministic=False,
        )

        assert action[
            0
        ] >= -1.0

        assert action[
            0
        ] <= 1.0

        assert action[
            1
        ] >= -2.0

        assert action[
            1
        ] <= 2.0


def test_deterministic_action_reproducible(
    agent,
):

    state = np.ones(
        6
    )

    action1 = agent.select_action(
        state,
        deterministic=True,
    )

    action2 = agent.select_action(
        state,
        deterministic=True,
    )

    assert np.allclose(
        action1,
        action2,
    )


def test_invalid_state_dimension(
    agent,
):

    with pytest.raises(ValueError):

        agent.select_action(
            np.zeros(
                5
            )
        )


# ============================================================
# EXPERIENCE STORAGE
# ============================================================

def test_store_transition(
    agent,
):

    transition = random_transition(
        1
    )

    index = agent.store_transition(
        *transition
    )

    assert index == 0

    assert len(
        agent.replay_buffer
    ) == 1


def test_environment_step_increases(
    agent,
):

    agent.store_transition(
        *random_transition(
            1
        )
    )

    assert agent.environment_step == 1


def test_not_ready_initially(
    agent,
):

    assert agent.ready_to_update() is False


def test_ready_after_batch_size(
    agent,
):

    fill_buffer(
        agent,
        count=8,
    )

    assert agent.ready_to_update() is True


# ============================================================
# TARGET Q
# ============================================================

def test_target_q_shape(
    agent,
):

    batch_size = 4

    rewards = torch.zeros(
        batch_size,
        1,
    )

    next_states = torch.randn(
        batch_size,
        6,
    )

    dones = torch.zeros(
        batch_size,
        1,
    )

    target = agent.calculate_target_q(
        rewards,
        next_states,
        dones,
    )

    assert target.shape == (
        batch_size,
        1,
    )


def test_terminal_target_equals_reward(
    agent,
):

    rewards = torch.tensor(
        [
            [
                3.5
            ]
        ],
        dtype=torch.float32,
    )

    next_states = torch.randn(
        1,
        6,
    )

    dones = torch.ones(
        1,
        1,
    )

    target = agent.calculate_target_q(
        rewards,
        next_states,
        dones,
    )

    assert target.item() == pytest.approx(
        3.5,
        abs=1e-6,
    )


# ============================================================
# FULL SAC UPDATE
# ============================================================

def test_update_requires_enough_data(
    agent,
):

    with pytest.raises(RuntimeError):

        agent.update()


def test_complete_sac_update(
    agent,
):

    fill_buffer(
        agent,
        count=16,
    )

    result = agent.update()

    assert isinstance(
        result,
        SACUpdateResult,
    )


def test_update_result_finite(
    agent,
):

    fill_buffer(
        agent,
        count=16,
    )

    result = agent.update()

    result.validate()

    assert np.isfinite(
        result.critic_loss
    )

    assert np.isfinite(
        result.actor_loss
    )


def test_update_step_increments(
    agent,
):

    fill_buffer(
        agent,
        count=16,
    )

    agent.update()

    assert agent.update_step == 1

    agent.update()

    assert agent.update_step == 2


def test_actor_parameters_change(
    agent,
):

    fill_buffer(
        agent,
        count=16,
    )

    before = [
        parameter.detach()
        .clone()
        for parameter
        in agent.actor.parameters()
    ]

    agent.update()

    after = list(
        agent.actor.parameters()
    )

    changed = any(
        not torch.allclose(
            old,
            new,
        )
        for old, new in zip(
            before,
            after,
        )
    )

    assert changed


def test_critic_parameters_change(
    agent,
):

    fill_buffer(
        agent,
        count=16,
    )

    before = [
        parameter.detach()
        .clone()
        for parameter
        in agent.critic.parameters()
    ]

    agent.update()

    after = list(
        agent.critic.parameters()
    )

    changed = any(
        not torch.allclose(
            old,
            new,
        )
        for old, new in zip(
            before,
            after,
        )
    )

    assert changed


def test_target_parameters_soft_update(
    agent,
):

    fill_buffer(
        agent,
        count=16,
    )

    before = [
        parameter.detach()
        .clone()
        for parameter
        in agent.target_critic.parameters()
    ]

    agent.update()

    after = list(
        agent.target_critic.parameters()
    )

    changed = any(
        not torch.allclose(
            old,
            new,
        )
        for old, new in zip(
            before,
            after,
        )
    )

    assert changed


# ============================================================
# EXPLORATION NOISE
# ============================================================

def test_initial_exploration_noise(
    agent,
):

    assert (
        agent.current_exploration_noise
        == pytest.approx(
            0.20
        )
    )


def test_decay_exploration_noise(
    agent,
):

    value = agent.decay_exploration_noise()

    assert value == pytest.approx(
        0.20 * 0.999
    )


def test_reset_exploration_noise(
    agent,
):

    agent.decay_exploration_noise()

    agent.reset_exploration_noise()

    assert (
        agent.current_exploration_noise
        == pytest.approx(
            0.20
        )
    )


def test_update_decays_exploration_noise(
    agent,
):

    fill_buffer(
        agent,
        16,
    )

    initial = (
        agent.current_exploration_noise
    )

    agent.update()

    assert (
        agent.current_exploration_noise
        < initial
    )


# ============================================================
# TRAIN / EVAL
# ============================================================

def test_agent_initially_training(
    agent,
):

    assert agent.training is True


def test_eval_mode(
    agent,
):

    agent.eval_mode()

    assert agent.training is False

    assert agent.actor.training is False


def test_train_mode(
    agent,
):

    agent.eval_mode()

    agent.train_mode(
        True
    )

    assert agent.training is True

    assert agent.actor.training is True


def test_controller_training_interface(
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

def test_reset_preserves_replay_memory(
    agent,
):

    fill_buffer(
        agent,
        3,
    )

    agent.decay_exploration_noise()

    agent.reset()

    assert len(
        agent.replay_buffer
    ) == 3

    assert (
        agent.current_exploration_noise
        == pytest.approx(
            0.20
        )
    )


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
        / "sac_agent.pt"
    )

    agent.save(
        path
    )

    second_agent = SACAgent(
        agent.config
    )

    second_agent.load(
        path
    )

    for first, second in zip(
        agent.actor.parameters(),
        second_agent.actor.parameters(),
    ):

        assert torch.allclose(
            first,
            second,
        )

    assert (
        second_agent.update_step
        == agent.update_step
    )


def test_load_missing_checkpoint(
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
# UPDATE RESULT
# ============================================================

def test_update_result_summary():

    result = SACUpdateResult(
        critic_loss=1.0,
        actor_loss=-0.5,
        q1_mean=2.0,
        q2_mean=2.1,
        target_q_mean=2.2,
        log_probability_mean=-1.0,
        entropy_estimate=1.0,
        update_step=1,
        exploration_noise=0.1,
    )

    result.validate()

    summary = result.summary()

    assert summary[
        "critic_loss"
    ] == pytest.approx(
        1.0
    )

    assert summary[
        "update_step"
    ] == 1


# ============================================================
# SUMMARY
# ============================================================

def test_agent_summary(
    agent,
):

    summary = agent.summary()

    assert summary[
        "agent"
    ] == "SACAgent"

    assert summary[
        "state_dimension"
    ] == 6

    assert summary[
        "action_dimension"
    ] == 2

    assert summary[
        "discount_factor"
    ] == pytest.approx(
        0.99
    )

    assert summary[
        "batch_size"
    ] == 8
