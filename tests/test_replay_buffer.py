import numpy as np
import pytest
from agents.replay_buffer import (
    ReplayBatch,
    ReplayBuffer,
    ReplayBufferConfig,
    normalize_done,
    validate_scalar,
    validate_vector,
)
# ============================================================
# FIXTURE
# ============================================================
@pytest.fixture
def config():

    return ReplayBufferConfig(
        capacity=5,
        state_dimension=3,
        action_dimension=2,
        dtype="float32",
        seed=42,
    )
@pytest.fixture
def buffer(
    config,
):

    return ReplayBuffer(
        config
    )


def add_transition(
    buffer,
    value,
    done=False,
):

    state = np.array(
        [
            value,
            value + 1,
            value + 2,
        ],
        dtype=np.float32,
    )

    action = np.array(
        [
            value,
            -value,
        ],
        dtype=np.float32,
    )

    next_state = (
        state + 0.5
    )

    reward = float(
        value
    )

    return buffer.add(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        done=done,
    )


# ============================================================
# CONFIG
# ============================================================

def test_default_capacity():

    config = ReplayBufferConfig()

    assert config.capacity == 1_000_000


def test_valid_config(
    config,
):

    config.validate()


def test_invalid_capacity():

    with pytest.raises(ValueError):

        ReplayBufferConfig(
            capacity=0
        ).validate()


def test_invalid_state_dimension():

    with pytest.raises(ValueError):

        ReplayBufferConfig(
            state_dimension=0
        ).validate()


def test_invalid_action_dimension():

    with pytest.raises(ValueError):

        ReplayBufferConfig(
            action_dimension=0
        ).validate()


def test_invalid_dtype():

    with pytest.raises(ValueError):

        ReplayBufferConfig(
            dtype="int32"
        ).validate()


# ============================================================
# VECTOR VALIDATION
# ============================================================

def test_validate_vector():

    result = validate_vector(
        [
            1,
            2,
            3,
        ],
        expected_dimension=3,
        name="state",
        dtype=np.float32,
    )

    assert result.shape == (
        3,
    )


def test_validate_vector_wrong_dimension():

    with pytest.raises(ValueError):

        validate_vector(
            [
                1,
                2,
            ],
            expected_dimension=3,
            name="state",
            dtype=np.float32,
        )


def test_validate_vector_nan():

    with pytest.raises(ValueError):

        validate_vector(
            [
                1,
                np.nan,
                3,
            ],
            expected_dimension=3,
            name="state",
            dtype=np.float32,
        )


# ============================================================
# SCALAR VALIDATION
# ============================================================

def test_validate_scalar():

    value = validate_scalar(
        2.5,
        "reward",
    )

    assert value == pytest.approx(
        2.5
    )


def test_validate_scalar_array_rejected():

    with pytest.raises(ValueError):

        validate_scalar(
            [
                1,
                2,
            ],
            "reward",
        )


# ============================================================
# DONE NORMALIZATION
# ============================================================

def test_done_false():

    assert normalize_done(
        False
    ) == 0.0


def test_done_true():

    assert normalize_done(
        True
    ) == 1.0


def test_done_numeric():

    assert normalize_done(
        1
    ) == 1.0


def test_invalid_done():

    with pytest.raises(ValueError):

        normalize_done(
            0.5
        )


# ============================================================
# CONSTRUCTION
# ============================================================

def test_buffer_creation(
    buffer,
):

    assert len(
        buffer
    ) == 0


def test_buffer_initially_empty(
    buffer,
):

    assert buffer.is_empty is True


def test_buffer_initially_not_full(
    buffer,
):

    assert buffer.is_full is False


def test_storage_shapes(
    buffer,
):

    assert buffer.states.shape == (
        5,
        3,
    )

    assert buffer.actions.shape == (
        5,
        2,
    )


# ============================================================
# ADD
# ============================================================

def test_add_transition(
    buffer,
):

    index = add_transition(
        buffer,
        1.0,
    )

    assert index == 0

    assert len(
        buffer
    ) == 1


def test_add_state_value(
    buffer,
):

    add_transition(
        buffer,
        1.0,
    )

    assert np.allclose(
        buffer.states[
            0
        ],
        [
            1,
            2,
            3,
        ],
    )


def test_add_action_value(
    buffer,
):

    add_transition(
        buffer,
        2.0,
    )

    assert np.allclose(
        buffer.actions[
            0
        ],
        [
            2,
            -2,
        ],
    )


def test_add_reward(
    buffer,
):

    add_transition(
        buffer,
        3.0,
    )

    assert buffer.rewards[
        0,
        0
    ] == pytest.approx(
        3.0
    )


def test_add_done(
    buffer,
):

    add_transition(
        buffer,
        1.0,
        done=True,
    )

    assert buffer.dones[
        0,
        0
    ] == pytest.approx(
        1.0
    )


def test_total_added(
    buffer,
):

    add_transition(
        buffer,
        1
    )

    add_transition(
        buffer,
        2
    )

    assert buffer.total_added == 2


# ============================================================
# CIRCULAR MEMORY
# ============================================================

def test_buffer_becomes_full(
    buffer,
):

    for value in range(
        5
    ):

        add_transition(
            buffer,
            value,
        )

    assert buffer.is_full is True

    assert len(
        buffer
    ) == 5


def test_buffer_does_not_exceed_capacity(
    buffer,
):

    for value in range(
        10
    ):

        add_transition(
            buffer,
            value,
        )

    assert len(
        buffer
    ) == 5


def test_oldest_transition_overwritten(
    buffer,
):

    for value in range(
        6
    ):

        add_transition(
            buffer,
            value,
        )

    # sixth insertion wraps to index 0

    assert np.allclose(
        buffer.states[
            0
        ],
        [
            5,
            6,
            7,
        ],
    )


def test_position_wraps(
    buffer,
):

    for value in range(
        5
    ):

        add_transition(
            buffer,
            value,
        )

    assert buffer.position == 0

    add_transition(
        buffer,
        5
    )

    assert buffer.position == 1


# ============================================================
# GET
# ============================================================

def test_get_transition(
    buffer,
):

    add_transition(
        buffer,
        2.0,
    )

    transition = buffer.get(
        0
    )

    assert np.allclose(
        transition[
            "state"
        ],
        [
            2,
            3,
            4,
        ],
    )

    assert transition[
        "reward"
    ] == pytest.approx(
        2.0
    )


def test_get_invalid_index(
    buffer,
):

    with pytest.raises(IndexError):

        buffer.get(
            0
        )


# ============================================================
# CAN SAMPLE
# ============================================================

def test_can_sample_false(
    buffer,
):

    add_transition(
        buffer,
        1
    )

    assert buffer.can_sample(
        2
    ) is False


def test_can_sample_true(
    buffer,
):

    add_transition(
        buffer,
        1
    )

    add_transition(
        buffer,
        2
    )

    assert buffer.can_sample(
        2
    ) is True


# ============================================================
# SAMPLE
# ============================================================

def test_sample_batch(
    buffer,
):

    for value in range(
        5
    ):

        add_transition(
            buffer,
            value,
        )

    batch = buffer.sample(
        3
    )

    assert isinstance(
        batch,
        ReplayBatch,
    )

    assert batch.batch_size == 3


def test_sample_state_shape(
    buffer,
):

    for value in range(
        5
    ):

        add_transition(
            buffer,
            value,
        )

    batch = buffer.sample(
        4
    )

    assert batch.states.shape == (
        4,
        3,
    )


def test_sample_action_shape(
    buffer,
):

    for value in range(
        5
    ):

        add_transition(
            buffer,
            value,
        )

    batch = buffer.sample(
        4
    )

    assert batch.actions.shape == (
        4,
        2,
    )


def test_sample_reward_shape(
    buffer,
):

    for value in range(
        5
    ):

        add_transition(
            buffer,
            value,
        )

    batch = buffer.sample(
        4
    )

    assert batch.rewards.shape == (
        4,
        1,
    )


def test_sample_done_shape(
    buffer,
):

    for value in range(
        5
    ):

        add_transition(
            buffer,
            value,
        )

    batch = buffer.sample(
        4
    )

    assert batch.dones.shape == (
        4,
        1,
    )


def test_sample_without_replacement_unique(
    buffer,
):

    for value in range(
        5
    ):

        add_transition(
            buffer,
            value,
        )

    batch = buffer.sample(
        5,
        replace=False,
    )

    assert len(
        np.unique(
            batch.indices
        )
    ) == 5


def test_sample_too_large_rejected(
    buffer,
):

    add_transition(
        buffer,
        1
    )

    with pytest.raises(ValueError):

        buffer.sample(
            2,
            replace=False,
        )


def test_sample_empty_rejected(
    buffer,
):

    with pytest.raises(ValueError):

        buffer.sample(
            1
        )


# ============================================================
# REPRODUCIBILITY
# ============================================================

def test_sampling_seed_reproducibility(
    config,
):

    buffer1 = ReplayBuffer(
        config
    )

    buffer2 = ReplayBuffer(
        config
    )

    for value in range(
        5
    ):

        add_transition(
            buffer1,
            value,
        )

        add_transition(
            buffer2,
            value,
        )

    batch1 = buffer1.sample(
        3
    )

    batch2 = buffer2.sample(
        3
    )

    assert np.array_equal(
        batch1.indices,
        batch2.indices,
    )


# ============================================================
# ADD BATCH
# ============================================================

def test_add_batch(
    buffer,
):

    states = np.array(
        [
            [
                1,
                2,
                3,
            ],
            [
                4,
                5,
                6,
            ],
        ]
    )

    actions = np.array(
        [
            [
                0.1,
                0.2,
            ],
            [
                0.3,
                0.4,
            ],
        ]
    )

    rewards = np.array(
        [
            1.0,
            2.0,
        ]
    )

    next_states = (
        states + 1
    )

    dones = np.array(
        [
            0,
            1,
        ]
    )

    count = buffer.add_batch(
        states,
        actions,
        rewards,
        next_states,
        dones,
    )

    assert count == 2

    assert len(
        buffer
    ) == 2


def test_add_batch_mismatched_sizes(
    buffer,
):

    states = np.zeros(
        (
            2,
            3,
        )
    )

    actions = np.zeros(
        (
            1,
            2,
        )
    )

    with pytest.raises(ValueError):

        buffer.add_batch(
            states=states,
            actions=actions,
            rewards=[
                1,
                2,
            ],
            next_states=states,
            dones=[
                0,
                0,
            ],
        )


# ============================================================
# CLEAR
# ============================================================

def test_clear(
    buffer,
):

    add_transition(
        buffer,
        1
    )

    add_transition(
        buffer,
        2
    )

    buffer.clear()

    assert len(
        buffer
    ) == 0

    assert buffer.position == 0

    assert buffer.total_added == 0

    assert buffer.is_empty is True


# ============================================================
# SAVE / LOAD
# ============================================================

def test_save_and_load(
    buffer,
    tmp_path,
):

    for value in range(
        3
    ):

        add_transition(
            buffer,
            value,
            done=(
                value == 2
            ),
        )

    path = tmp_path / (
        "replay_buffer.npz"
    )

    buffer.save(
        path
    )

    new_buffer = ReplayBuffer(
        ReplayBufferConfig(
            capacity=5,
            state_dimension=3,
            action_dimension=2,
            seed=42,
        )
    )

    new_buffer.load(
        path
    )

    assert len(
        new_buffer
    ) == 3

    assert np.allclose(
        new_buffer.states[
            :3
        ],
        buffer.states[
            :3
        ],
    )

    assert np.allclose(
        new_buffer.actions[
            :3
        ],
        buffer.actions[
            :3
        ],
    )


def test_load_missing_file(
    buffer,
    tmp_path,
):

    with pytest.raises(
        FileNotFoundError
    ):

        buffer.load(
            tmp_path
            / "missing.npz"
        )


# ============================================================
# BATCH RESULT
# ============================================================

def test_batch_summary(
    buffer,
):

    for value in range(
        4
    ):

        add_transition(
            buffer,
            value,
        )

    batch = buffer.sample(
        2
    )

    summary = batch.summary()

    assert summary[
        "batch_size"
    ] == 2

    assert summary[
        "state_shape"
    ] == (
        2,
        3,
    )

    assert summary[
        "action_shape"
    ] == (
        2,
        2,
    )


# ============================================================
# BUFFER SUMMARY
# ============================================================

def test_buffer_summary(
    buffer,
):

    add_transition(
        buffer,
        1
    )

    summary = buffer.summary()

    assert summary[
        "capacity"
    ] == 5

    assert summary[
        "size"
    ] == 1

    assert summary[
        "state_dimension"
    ] == 3

    assert summary[
        "action_dimension"
    ] == 2

    assert summary[
        "is_empty"
    ] is False
