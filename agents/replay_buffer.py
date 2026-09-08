from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import numpy as np
# ============================================================
# CONFIGURATION
# ============================================================
@dataclass
class ReplayBufferConfig:
    """
    Replay-buffer configuration.
    """

    capacity: int = 1_000_000

    state_dimension: int = 1

    action_dimension: int = 1

    dtype: str = "float32"

    seed: Optional[int] = None

    def validate(self) -> None:

        if not isinstance(
            self.capacity,
            int,
        ):
            raise TypeError(
                "capacity must be an integer."
            )

        if self.capacity <= 0:
            raise ValueError(
                "capacity must be positive."
            )

        if not isinstance(
            self.state_dimension,
            int,
        ):
            raise TypeError(
                "state_dimension must be an integer."
            )

        if self.state_dimension <= 0:
            raise ValueError(
                "state_dimension must be positive."
            )

        if not isinstance(
            self.action_dimension,
            int,
        ):
            raise TypeError(
                "action_dimension must be an integer."
            )

        if self.action_dimension <= 0:
            raise ValueError(
                "action_dimension must be positive."
            )

        if self.dtype not in {
            "float32",
            "float64",
        }:
            raise ValueError(
                "dtype must be 'float32' or 'float64'."
            )

        if (
            self.seed is not None
            and not isinstance(
                self.seed,
                int,
            )
        ):
            raise TypeError(
                "seed must be an integer or None."
            )
# ============================================================
# BATCH CONTAINER
# ============================================================
@dataclass
class ReplayBatch:
    """
    Mini-batch sampled from replay memory.
    """

    states: np.ndarray

    actions: np.ndarray

    rewards: np.ndarray

    next_states: np.ndarray

    dones: np.ndarray

    indices: np.ndarray

    def validate(
        self,
    ) -> None:

        arrays = {
            "states":
                self.states,

            "actions":
                self.actions,

            "rewards":
                self.rewards,

            "next_states":
                self.next_states,

            "dones":
                self.dones,

            "indices":
                self.indices,
        }

        for name, array in (
            arrays.items()
        ):

            if not isinstance(
                array,
                np.ndarray,
            ):
                raise TypeError(
                    f"{name} must be a NumPy array."
                )

        if self.states.ndim != 2:
            raise ValueError(
                "states must be two-dimensional."
            )

        if self.next_states.ndim != 2:
            raise ValueError(
                "next_states must be two-dimensional."
            )

        if self.actions.ndim != 2:
            raise ValueError(
                "actions must be two-dimensional."
            )

        if self.rewards.ndim != 2:
            raise ValueError(
                "rewards must have shape (batch, 1)."
            )

        if self.dones.ndim != 2:
            raise ValueError(
                "dones must have shape (batch, 1)."
            )

        if self.indices.ndim != 1:
            raise ValueError(
                "indices must be one-dimensional."
            )

        batch_size = (
            self.states.shape[
                0
            ]
        )

        expected_sizes = [
            self.actions.shape[0],
            self.rewards.shape[0],
            self.next_states.shape[0],
            self.dones.shape[0],
            self.indices.shape[0],
        ]

        if any(
            size != batch_size
            for size in expected_sizes
        ):
            raise ValueError(
                "All batch arrays must have the same "
                "number of samples."
            )

        if self.rewards.shape[1] != 1:
            raise ValueError(
                "rewards must have second dimension 1."
            )

        if self.dones.shape[1] != 1:
            raise ValueError(
                "dones must have second dimension 1."
            )

        for name in (
            "states",
            "actions",
            "rewards",
            "next_states",
            "dones",
        ):

            array = arrays[
                name
            ]

            if not np.isfinite(
                array
            ).all():
                raise ValueError(
                    f"{name} contains NaN or Inf."
                )

    @property
    def batch_size(
        self,
    ) -> int:

        return int(
            self.states.shape[
                0
            ]
        )

    def summary(
        self,
    ) -> Dict[str, object]:

        return {
            "batch_size":
                self.batch_size,

            "state_shape":
                tuple(
                    self.states.shape
                ),

            "action_shape":
                tuple(
                    self.actions.shape
                ),

            "reward_shape":
                tuple(
                    self.rewards.shape
                ),

            "next_state_shape":
                tuple(
                    self.next_states.shape
                ),

            "done_shape":
                tuple(
                    self.dones.shape
                ),
        }


# ============================================================
# VALIDATION HELPERS
# ============================================================

def _numpy_dtype(
    dtype: str,
):

    if dtype == "float32":
        return np.float32

    if dtype == "float64":
        return np.float64

    raise ValueError(
        "Unsupported dtype."
    )


def validate_vector(
    value,
    expected_dimension: int,
    name: str,
    dtype,
) -> np.ndarray:
    """
    Validate state or action vector.
    """

    array = np.asarray(
        value,
        dtype=dtype,
    )

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if array.size != expected_dimension:
        raise ValueError(
            f"{name} dimension mismatch. "
            f"Expected {expected_dimension}, "
            f"received {array.size}."
        )

    if not np.isfinite(
        array
    ).all():
        raise ValueError(
            f"{name} contains NaN or Inf."
        )

    return array


def validate_scalar(
    value,
    name: str,
) -> float:
    """
    Validate scalar transition quantity.
    """

    array = np.asarray(
        value,
        dtype=np.float64,
    )

    if array.ndim != 0:
        raise ValueError(
            f"{name} must be a scalar."
        )

    value = float(
        array
    )

    if not np.isfinite(
        value
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    return value


def normalize_done(
    done,
) -> float:
    """
    Convert terminal flag to SAC-compatible 0/1 float.
    """

    if isinstance(
        done,
        (bool, np.bool_),
    ):
        return float(
            done
        )

    value = validate_scalar(
        done,
        "done",
    )

    if value not in {
        0.0,
        1.0,
    }:
        raise ValueError(
            "done must be bool, 0, or 1."
        )

    return value


# ============================================================
# REPLAY BUFFER
# ============================================================

class ReplayBuffer:
    """
    Fixed-capacity circular experience replay memory.
    """

    def __init__(
        self,
        config: ReplayBufferConfig,
    ) -> None:

        config.validate()

        self.config = config

        self.dtype = _numpy_dtype(
            config.dtype
        )

        self.states = np.zeros(
            (
                config.capacity,
                config.state_dimension,
            ),
            dtype=self.dtype,
        )

        self.actions = np.zeros(
            (
                config.capacity,
                config.action_dimension,
            ),
            dtype=self.dtype,
        )

        self.rewards = np.zeros(
            (
                config.capacity,
                1,
            ),
            dtype=self.dtype,
        )

        self.next_states = np.zeros(
            (
                config.capacity,
                config.state_dimension,
            ),
            dtype=self.dtype,
        )

        self.dones = np.zeros(
            (
                config.capacity,
                1,
            ),
            dtype=self.dtype,
        )

        self.position = 0

        self.size = 0

        self.total_added = 0

        self.rng = (
            np.random.default_rng(
                config.seed
            )
        )

    # ========================================================
    # BASIC PROPERTIES
    # ========================================================

    def __len__(
        self,
    ) -> int:

        return self.size

    @property
    def capacity(
        self,
    ) -> int:

        return self.config.capacity

    @property
    def is_full(
        self,
    ) -> bool:

        return (
            self.size
            == self.capacity
        )

    @property
    def is_empty(
        self,
    ) -> bool:

        return (
            self.size == 0
        )

    # ========================================================
    # ADD TRANSITION
    # ========================================================

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ) -> int:
        """
        Add one transition.

        Returns
        -------
        index:
            Memory slot where the transition was stored.
        """

        state = validate_vector(
            state,
            self.config.state_dimension,
            "state",
            self.dtype,
        )

        action = validate_vector(
            action,
            self.config.action_dimension,
            "action",
            self.dtype,
        )

        reward = validate_scalar(
            reward,
            "reward",
        )

        next_state = validate_vector(
            next_state,
            self.config.state_dimension,
            "next_state",
            self.dtype,
        )

        done = normalize_done(
            done
        )

        index = self.position

        self.states[
            index
        ] = state

        self.actions[
            index
        ] = action

        self.rewards[
            index,
            0
        ] = reward

        self.next_states[
            index
        ] = next_state

        self.dones[
            index,
            0
        ] = done

        self.position = (
            self.position + 1
        ) % self.capacity

        self.size = min(
            self.size + 1,
            self.capacity,
        )

        self.total_added += 1

        return int(
            index
        )

    # ========================================================
    # BATCH ADD
    # ========================================================

    def add_batch(
        self,
        states,
        actions,
        rewards,
        next_states,
        dones,
    ) -> int:
        """
        Add multiple transitions.

        Returns number of transitions inserted.
        """

        states = np.asarray(
            states
        )

        actions = np.asarray(
            actions
        )

        rewards = np.asarray(
            rewards
        )

        next_states = np.asarray(
            next_states
        )

        dones = np.asarray(
            dones
        )

        if states.ndim != 2:
            raise ValueError(
                "states must be two-dimensional."
            )

        batch_size = states.shape[
            0
        ]

        if batch_size == 0:
            raise ValueError(
                "Batch cannot be empty."
            )

        if actions.ndim != 2:
            raise ValueError(
                "actions must be two-dimensional."
            )

        if next_states.ndim != 2:
            raise ValueError(
                "next_states must be two-dimensional."
            )

        rewards = rewards.reshape(
            -1
        )

        dones = dones.reshape(
            -1
        )

        sizes = [
            actions.shape[0],
            rewards.shape[0],
            next_states.shape[0],
            dones.shape[0],
        ]

        if any(
            size != batch_size
            for size in sizes
        ):
            raise ValueError(
                "All transition arrays must have equal "
                "batch size."
            )

        for index in range(
            batch_size
        ):

            self.add(
                state=states[
                    index
                ],
                action=actions[
                    index
                ],
                reward=rewards[
                    index
                ],
                next_state=next_states[
                    index
                ],
                done=dones[
                    index
                ],
            )

        return int(
            batch_size
        )

    # ========================================================
    # SAMPLING
    # ========================================================

    def can_sample(
        self,
        batch_size: int,
    ) -> bool:

        if not isinstance(
            batch_size,
            int,
        ):
            raise TypeError(
                "batch_size must be an integer."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        return (
            self.size
            >= batch_size
        )

    def sample(
        self,
        batch_size: int,
        replace: bool = False,
    ) -> ReplayBatch:
        """
        Randomly sample replay transitions.

        Default behavior samples without replacement.
        """

        if not isinstance(
            batch_size,
            int,
        ):
            raise TypeError(
                "batch_size must be an integer."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if not isinstance(
            replace,
            bool,
        ):
            raise TypeError(
                "replace must be boolean."
            )

        if self.size == 0:
            raise ValueError(
                "Cannot sample from an empty replay buffer."
            )

        if (
            not replace
            and batch_size > self.size
        ):
            raise ValueError(
                "batch_size cannot exceed current replay "
                "size when replace=False."
            )

        indices = self.rng.choice(
            self.size,
            size=batch_size,
            replace=replace,
        )

        batch = ReplayBatch(
            states=self.states[
                indices
            ].copy(),

            actions=self.actions[
                indices
            ].copy(),

            rewards=self.rewards[
                indices
            ].copy(),

            next_states=self.next_states[
                indices
            ].copy(),

            dones=self.dones[
                indices
            ].copy(),

            indices=np.asarray(
                indices,
                dtype=np.int64,
            ),
        )

        batch.validate()

        return batch

    # ========================================================
    # GET TRANSITION
    # ========================================================

    def get(
        self,
        index: int,
    ) -> Dict[str, object]:
        """
        Retrieve one valid memory slot.
        """

        if not isinstance(
            index,
            int,
        ):
            raise TypeError(
                "index must be an integer."
            )

        if (
            index < 0
            or index >= self.size
        ):
            raise IndexError(
                "Replay-buffer index out of range."
            )

        return {
            "state":
                self.states[
                    index
                ].copy(),

            "action":
                self.actions[
                    index
                ].copy(),

            "reward":
                float(
                    self.rewards[
                        index,
                        0
                    ]
                ),

            "next_state":
                self.next_states[
                    index
                ].copy(),

            "done":
                float(
                    self.dones[
                        index,
                        0
                    ]
                ),
        }

    # ========================================================
    # CLEAR
    # ========================================================

    def clear(
        self,
    ) -> None:
        """
        Reset replay memory.
        """

        self.states.fill(
            0
        )

        self.actions.fill(
            0
        )

        self.rewards.fill(
            0
        )

        self.next_states.fill(
            0
        )

        self.dones.fill(
            0
        )

        self.position = 0

        self.size = 0

        self.total_added = 0

    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        path,
    ) -> Path:
        """
        Save current valid replay memory to compressed NPZ.
        """

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            path,
            states=self.states[
                :self.size
            ],

            actions=self.actions[
                :self.size
            ],

            rewards=self.rewards[
                :self.size
            ],

            next_states=self.next_states[
                :self.size
            ],

            dones=self.dones[
                :self.size
            ],

            size=np.array(
                [
                    self.size
                ],
                dtype=np.int64,
            ),

            position=np.array(
                [
                    self.position
                ],
                dtype=np.int64,
            ),

            total_added=np.array(
                [
                    self.total_added
                ],
                dtype=np.int64,
            ),
        )

        return path

    # ========================================================
    # LOAD
    # ========================================================

    def load(
        self,
        path,
    ) -> None:
        """
        Load replay data from NPZ.

        The loaded dimensions must match this buffer.
        """

        path = Path(
            path
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        with np.load(
            path
        ) as data:

            states = data[
                "states"
            ]

            actions = data[
                "actions"
            ]

            rewards = data[
                "rewards"
            ]

            next_states = data[
                "next_states"
            ]

            dones = data[
                "dones"
            ]

            if states.ndim != 2:
                raise ValueError(
                    "Saved states have invalid shape."
                )

            saved_size = states.shape[
                0
            ]

            if saved_size > self.capacity:
                raise ValueError(
                    "Saved replay memory exceeds buffer capacity."
                )

            if (
                states.shape[
                    1
                ]
                != self.config.state_dimension
            ):
                raise ValueError(
                    "Saved state dimension mismatch."
                )

            if (
                actions.ndim != 2
                or actions.shape[
                    1
                ]
                != self.config.action_dimension
            ):
                raise ValueError(
                    "Saved action dimension mismatch."
                )

            if next_states.shape != states.shape:
                raise ValueError(
                    "Saved next_states shape mismatch."
                )

            if rewards.shape != (
                saved_size,
                1,
            ):
                raise ValueError(
                    "Saved rewards shape mismatch."
                )

            if dones.shape != (
                saved_size,
                1,
            ):
                raise ValueError(
                    "Saved dones shape mismatch."
                )

            self.clear()

            self.states[
                :saved_size
            ] = states.astype(
                self.dtype
            )

            self.actions[
                :saved_size
            ] = actions.astype(
                self.dtype
            )

            self.rewards[
                :saved_size
            ] = rewards.astype(
                self.dtype
            )

            self.next_states[
                :saved_size
            ] = next_states.astype(
                self.dtype
            )

            self.dones[
                :saved_size
            ] = dones.astype(
                self.dtype
            )

            self.size = saved_size

            if (
                "position"
                in data
            ):
                saved_position = int(
                    data[
                        "position"
                    ][
                        0
                    ]
                )

                self.position = (
                    saved_position
                    % self.capacity
                )

            else:
                self.position = (
                    saved_size
                    % self.capacity
                )

            if (
                "total_added"
                in data
            ):
                self.total_added = int(
                    data[
                        "total_added"
                    ][
                        0
                    ]
                )

            else:
                self.total_added = (
                    saved_size
                )

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(
        self,
    ) -> Dict[str, object]:

        return {
            "capacity":
                self.capacity,

            "size":
                self.size,

            "position":
                self.position,

            "total_added":
                self.total_added,

            "state_dimension":
                self.config.state_dimension,

            "action_dimension":
                self.config.action_dimension,

            "dtype":
                self.config.dtype,

            "is_full":
                self.is_full,

            "is_empty":
                self.is_empty,
        }
