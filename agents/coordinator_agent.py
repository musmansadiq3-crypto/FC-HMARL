from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from agents.sac_agent import (
    SACAgent,
    SACAgentConfig,
    SACUpdateResult,
)
# CONFIGURATION
@dataclass
class CoordinatorAgentConfig:
    """
    Configuration for the global VPP coordinator.
    """

    state_dimension: int

    action_dimension: int

    coordinator_id: int = 1

    hidden_dimensions: Tuple[int, ...] = (
        256,
        256,
    )

    activation: str = "relu"

    action_low: float | Sequence[float] = -1.0

    action_high: float | Sequence[float] = 1.0

    discount_factor: float = 0.99

    learning_rate: float = 1e-4

    replay_buffer_capacity: int = 1_000_000

    batch_size: int = 512

    soft_update_coefficient: float = 0.005

    entropy_coefficient: float = 0.20

    initial_exploration_noise: float = 0.20

    exploration_noise_decay: float = 0.999

    minimum_exploration_noise: float = 0.0

    gradient_clip_norm: Optional[float] = None

    seed: Optional[int] = None

    device: str = "cpu"

    dtype: str = "float32"

    def validate(self) -> None:
        """
        Validate coordinator configuration.
        """

        if not isinstance(
            self.coordinator_id,
            int,
        ):
            raise TypeError(
                "coordinator_id must be an integer."
            )

        if self.coordinator_id <= 0:
            raise ValueError(
                "coordinator_id must be positive."
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

        self.to_sac_config().validate()

    def to_sac_config(
        self,
    ) -> SACAgentConfig:
        """
        Convert coordinator configuration into SAC configuration.
        """

        return SACAgentConfig(
            state_dimension=self.state_dimension,

            action_dimension=self.action_dimension,

            hidden_dimensions=self.hidden_dimensions,

            activation=self.activation,

            action_low=self.action_low,

            action_high=self.action_high,

            discount_factor=self.discount_factor,

            learning_rate=self.learning_rate,

            replay_buffer_capacity=(
                self.replay_buffer_capacity
            ),

            batch_size=self.batch_size,

            soft_update_coefficient=(
                self.soft_update_coefficient
            ),

            entropy_coefficient=(
                self.entropy_coefficient
            ),

            initial_exploration_noise=(
                self.initial_exploration_noise
            ),

            exploration_noise_decay=(
                self.exploration_noise_decay
            ),

            minimum_exploration_noise=(
                self.minimum_exploration_noise
            ),

            gradient_clip_norm=(
                self.gradient_clip_norm
            ),

            seed=self.seed,

            device=self.device,

            dtype=self.dtype,
        )

# ACTION RESULT
@dataclass
class CoordinatorActionResult:
    """
    Coordinator action with metadata.
    """

    coordinator_id: int

    action: np.ndarray

    deterministic: bool

    def validate(
        self,
        expected_dimension: Optional[int] = None,
    ) -> None:

        if not isinstance(
            self.coordinator_id,
            int,
        ):
            raise TypeError(
                "coordinator_id must be an integer."
            )

        if self.coordinator_id <= 0:
            raise ValueError(
                "coordinator_id must be positive."
            )

        if not isinstance(
            self.action,
            np.ndarray,
        ):
            raise TypeError(
                "action must be a NumPy array."
            )

        if self.action.ndim != 1:
            raise ValueError(
                "Coordinator action must be one-dimensional."
            )

        if (
            expected_dimension
            is not None
            and self.action.size
            != expected_dimension
        ):
            raise ValueError(
                "Coordinator action dimension mismatch."
            )

        if not np.isfinite(
            self.action
        ).all():
            raise ValueError(
                "Coordinator action contains NaN or Inf."
            )

        if not isinstance(
            self.deterministic,
            bool,
        ):
            raise TypeError(
                "deterministic must be boolean."
            )

    def summary(
        self,
    ) -> Dict[str, object]:

        return {
            "coordinator_id":
                self.coordinator_id,

            "action_dimension":
                int(
                    self.action.size
                ),

            "deterministic":
                self.deterministic,

            "action":
                self.action.copy(),
        }


# ============================================================
# TRAINING RESULT
# ============================================================

@dataclass
class CoordinatorTrainingResult:
    """
    Result of one coordinator training call.
    """

    coordinator_id: int

    updated: bool

    sac_result: Optional[
        SACUpdateResult
    ] = None

    def validate(
        self,
    ) -> None:

        if not isinstance(
            self.coordinator_id,
            int,
        ):
            raise TypeError(
                "coordinator_id must be an integer."
            )

        if self.coordinator_id <= 0:
            raise ValueError(
                "coordinator_id must be positive."
            )

        if not isinstance(
            self.updated,
            bool,
        ):
            raise TypeError(
                "updated must be boolean."
            )

        if (
            self.updated
            and self.sac_result is None
        ):
            raise ValueError(
                "sac_result is required when updated=True."
            )

        if (
            not self.updated
            and self.sac_result is not None
        ):
            raise ValueError(
                "sac_result must be None when updated=False."
            )

        if self.sac_result is not None:
            self.sac_result.validate()

    def summary(
        self,
    ) -> Dict[str, object]:

        output = {
            "coordinator_id":
                self.coordinator_id,

            "updated":
                self.updated,
        }

        if self.sac_result is not None:

            output.update(
                self.sac_result.summary()
            )

        return output


# ============================================================
# COORDINATOR AGENT
# ============================================================

class CoordinatorAgent:
    """
    SAC-based global VPP coordinator.

    The class provides the interface required by the
    HierarchicalController while preserving the SAC training
    engine implemented in agents/sac_agent.py.
    """

    def __init__(
        self,
        config: CoordinatorAgentConfig,
    ) -> None:

        config.validate()

        self.config = config

        self.coordinator_id = (
            config.coordinator_id
        )

        self.sac_agent = SACAgent(
            config.to_sac_config()
        )

        self.last_state: Optional[
            np.ndarray
        ] = None

        self.last_action: Optional[
            np.ndarray
        ] = None

        self.total_actions_selected = 0

        self.total_transitions_stored = 0

        self.total_training_updates = 0

    # ========================================================
    # PROPERTIES
    # ========================================================

    @property
    def actor(
        self,
    ):

        return self.sac_agent.actor

    @property
    def critic(
        self,
    ):

        return self.sac_agent.critic

    @property
    def target_critic(
        self,
    ):

        return self.sac_agent.target_critic

    @property
    def replay_buffer(
        self,
    ):

        return self.sac_agent.replay_buffer

    @property
    def training(
        self,
    ) -> bool:

        return self.sac_agent.training

    @property
    def state_dimension(
        self,
    ) -> int:

        return self.config.state_dimension

    @property
    def action_dimension(
        self,
    ) -> int:

        return self.config.action_dimension

    # ========================================================
    # STATE VALIDATION
    # ========================================================

    def validate_state(
        self,
        state,
    ) -> np.ndarray:

        dtype = (
            np.float32
            if self.config.dtype
            == "float32"
            else np.float64
        )

        array = np.asarray(
            state,
            dtype=dtype,
        )

        if array.ndim != 1:
            raise ValueError(
                "Coordinator state must be one-dimensional."
            )

        if (
            array.size
            != self.state_dimension
        ):
            raise ValueError(
                "Coordinator state dimension mismatch. "
                f"Expected {self.state_dimension}, "
                f"received {array.size}."
            )

        if not np.isfinite(
            array
        ).all():
            raise ValueError(
                "Coordinator state contains NaN or Inf."
            )

        return array

    # ========================================================
    # ACTION VALIDATION
    # ========================================================

    def validate_action(
        self,
        action,
    ) -> np.ndarray:

        dtype = (
            np.float32
            if self.config.dtype
            == "float32"
            else np.float64
        )

        array = np.asarray(
            action,
            dtype=dtype,
        )

        if array.ndim != 1:
            raise ValueError(
                "Coordinator action must be one-dimensional."
            )

        if (
            array.size
            != self.action_dimension
        ):
            raise ValueError(
                "Coordinator action dimension mismatch. "
                f"Expected {self.action_dimension}, "
                f"received {array.size}."
            )

        if not np.isfinite(
            array
        ).all():
            raise ValueError(
                "Coordinator action contains NaN or Inf."
            )

        low = self.sac_agent.action_low

        high = self.sac_agent.action_high

        if np.any(
            array < low
        ):
            raise ValueError(
                "Coordinator action is below configured "
                "lower bound."
            )

        if np.any(
            array > high
        ):
            raise ValueError(
                "Coordinator action exceeds configured "
                "upper bound."
            )

        return array

    # ========================================================
    # ACTION SELECTION
    # ========================================================

    def select_action(
        self,
        state,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Select coordinator action.

        The returned array is directly compatible with
        marl/hierarchical_controller.py.
        """

        if not isinstance(
            deterministic,
            bool,
        ):
            raise TypeError(
                "deterministic must be boolean."
            )

        state_array = (
            self.validate_state(
                state
            )
        )

        action = (
            self.sac_agent.select_action(
                state_array,
                deterministic=deterministic,
            )
        )

        action = (
            self.validate_action(
                action
            )
        )

        self.last_state = (
            state_array.copy()
        )

        self.last_action = (
            action.copy()
        )

        self.total_actions_selected += 1

        return action

    def select_action_result(
        self,
        state,
        deterministic: bool = False,
    ) -> CoordinatorActionResult:

        action = self.select_action(
            state,
            deterministic=deterministic,
        )

        result = CoordinatorActionResult(
            coordinator_id=(
                self.coordinator_id
            ),

            action=action.copy(),

            deterministic=(
                deterministic
            ),
        )

        result.validate(
            expected_dimension=(
                self.action_dimension
            )
        )

        return result

     # TRANSITION STORAGE
 
    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ) -> int:

        state_array = (
            self.validate_state(
                state
            )
        )

        action_array = (
            self.validate_action(
                action
            )
        )

        next_state_array = (
            self.validate_state(
                next_state
            )
        )

        index = (
            self.sac_agent.store_transition(
                state=state_array,
                action=action_array,
                reward=reward,
                next_state=next_state_array,
                done=done,
            )
        )

        self.total_transitions_stored += 1

        return index

 
    # TRAINING
  

    def ready_to_update(
        self,
    ) -> bool:

        return (
            self.sac_agent
            .ready_to_update()
        )

    def update(
        self,
    ) -> SACUpdateResult:

        result = (
            self.sac_agent.update()
        )

        self.total_training_updates += 1

        return result

    def update_if_ready(
        self,
    ) -> CoordinatorTrainingResult:

        if not self.ready_to_update():

            result = CoordinatorTrainingResult(
                coordinator_id=(
                    self.coordinator_id
                ),

                updated=False,

                sac_result=None,
            )

            result.validate()

            return result

        sac_result = self.update()

        result = CoordinatorTrainingResult(
            coordinator_id=(
                self.coordinator_id
            ),

            updated=True,

            sac_result=sac_result,
        )

        result.validate()

        return result

    # TRAIN/EVAL MODES
  

    def set_training_mode(
        self,
        training: bool,
    ) -> None:

        if not isinstance(
            training,
            bool,
        ):
            raise TypeError(
                "training must be boolean."
            )

        self.sac_agent.set_training_mode(
            training
        )

    def train_mode(
        self,
    ) -> None:

        self.set_training_mode(
            True
        )

    def eval_mode(
        self,
    ) -> None:

        self.set_training_mode(
            False
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
    ) -> None:
        """
        Reset episode-specific coordinator state.

        Learned parameters and replay memory remain unchanged.
        """

        self.last_state = None

        self.last_action = None

        self.sac_agent.reset()

    # ========================================================
    # CHECKPOINTS
    # ========================================================

    def save(
        self,
        path,
    ) -> Path:

        return self.sac_agent.save(
            path
        )

    def load(
        self,
        path,
        load_optimizers: bool = True,
    ) -> None:

        self.sac_agent.load(
            path,
            load_optimizers=(
                load_optimizers
            ),
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(
        self,
    ) -> Dict[str, object]:

        return {
            "agent_type":
                "vpp_coordinator",

            "coordinator_id":
                self.coordinator_id,

            "state_dimension":
                self.state_dimension,

            "action_dimension":
                self.action_dimension,

            "training":
                self.training,

            "replay_buffer_size":
                len(
                    self.replay_buffer
                ),

            "total_actions_selected":
                self.total_actions_selected,

            "total_transitions_stored":
                self.total_transitions_stored,

            "total_training_updates":
                self.total_training_updates,

            "current_exploration_noise":
                (
                    self.sac_agent
                    .current_exploration_noise
                ),
        }


# ============================================================
# FACTORY
# ============================================================

def build_coordinator_agent(
    state_dimension: int,
    action_dimension: int,
    hidden_dimensions: Tuple[int, ...] = (
        256,
        256,
    ),
    action_low=-1.0,
    action_high=1.0,
    discount_factor: float = 0.99,
    learning_rate: float = 1e-4,
    replay_buffer_capacity: int = 1_000_000,
    batch_size: int = 512,
    soft_update_coefficient: float = 0.005,
    entropy_coefficient: float = 0.20,
    initial_exploration_noise: float = 0.20,
    exploration_noise_decay: float = 0.999,
    seed: Optional[int] = None,
    device: str = "cpu",
) -> CoordinatorAgent:
    """
    Build the single global VPP coordinator.
    """

    config = CoordinatorAgentConfig(
        coordinator_id=1,

        state_dimension=(
            state_dimension
        ),

        action_dimension=(
            action_dimension
        ),

        hidden_dimensions=(
            hidden_dimensions
        ),

        action_low=action_low,

        action_high=action_high,

        discount_factor=(
            discount_factor
        ),

        learning_rate=(
            learning_rate
        ),

        replay_buffer_capacity=(
            replay_buffer_capacity
        ),

        batch_size=batch_size,

        soft_update_coefficient=(
            soft_update_coefficient
        ),

        entropy_coefficient=(
            entropy_coefficient
        ),

        initial_exploration_noise=(
            initial_exploration_noise
        ),

        exploration_noise_decay=(
            exploration_noise_decay
        ),

        seed=seed,

        device=device,
    )

    return CoordinatorAgent(
        config
    )
