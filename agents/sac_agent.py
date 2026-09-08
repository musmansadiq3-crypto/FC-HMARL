from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from agents.networks import (
    GaussianPolicyNetwork,
    SACNetworkConfig,
    TwinQNetwork,
    build_sac_networks,
)
from agents.replay_buffer import (
    ReplayBatch,
    ReplayBuffer,
    ReplayBufferConfig,
)
# ============================================================
# CONFIGURATION
# ============================================================
@dataclass
class SACAgentConfig:
    """
    Configuration of one SAC agent.
    """

    state_dimension: int
    action_dimension: int

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

        if not isinstance(
            self.hidden_dimensions,
            tuple,
        ):
            raise TypeError(
                "hidden_dimensions must be a tuple."
            )

        if len(
            self.hidden_dimensions
        ) == 0:
            raise ValueError(
                "hidden_dimensions cannot be empty."
            )

        for dimension in (
            self.hidden_dimensions
        ):

            if not isinstance(
                dimension,
                int,
            ):
                raise TypeError(
                    "Each hidden dimension must be an integer."
                )

            if dimension <= 0:
                raise ValueError(
                    "Hidden dimensions must be positive."
                )

        if not (
            0.0
            <= self.discount_factor
            <= 1.0
        ):
            raise ValueError(
                "discount_factor must lie in [0, 1]."
            )

        if (
            not np.isfinite(
                self.learning_rate
            )
            or self.learning_rate <= 0
        ):
            raise ValueError(
                "learning_rate must be positive and finite."
            )

        if (
            not isinstance(
                self.replay_buffer_capacity,
                int,
            )
            or self.replay_buffer_capacity <= 0
        ):
            raise ValueError(
                "replay_buffer_capacity must be a "
                "positive integer."
            )

        if (
            not isinstance(
                self.batch_size,
                int,
            )
            or self.batch_size <= 0
        ):
            raise ValueError(
                "batch_size must be a positive integer."
            )

        if (
            self.batch_size
            > self.replay_buffer_capacity
        ):
            raise ValueError(
                "batch_size cannot exceed "
                "replay_buffer_capacity."
            )

        if not (
            0.0
            < self.soft_update_coefficient
            <= 1.0
        ):
            raise ValueError(
                "soft_update_coefficient must lie in (0, 1]."
            )

        if (
            not np.isfinite(
                self.entropy_coefficient
            )
            or self.entropy_coefficient < 0
        ):
            raise ValueError(
                "entropy_coefficient must be "
                "nonnegative and finite."
            )

        if (
            not np.isfinite(
                self.initial_exploration_noise
            )
            or self.initial_exploration_noise < 0
        ):
            raise ValueError(
                "initial_exploration_noise must be nonnegative."
            )

        if not (
            0.0
            < self.exploration_noise_decay
            <= 1.0
        ):
            raise ValueError(
                "exploration_noise_decay must lie in (0, 1]."
            )

        if (
            not np.isfinite(
                self.minimum_exploration_noise
            )
            or self.minimum_exploration_noise < 0
        ):
            raise ValueError(
                "minimum_exploration_noise must be nonnegative."
            )

        if (
            self.minimum_exploration_noise
            > self.initial_exploration_noise
        ):
            raise ValueError(
                "minimum_exploration_noise cannot exceed "
                "initial_exploration_noise."
            )

        if (
            self.gradient_clip_norm
            is not None
        ):

            if (
                not np.isfinite(
                    self.gradient_clip_norm
                )
                or self.gradient_clip_norm <= 0
            ):
                raise ValueError(
                    "gradient_clip_norm must be positive."
                )

        if self.dtype not in {
            "float32",
            "float64",
        }:
            raise ValueError(
                "dtype must be 'float32' or 'float64'."
            )


# ============================================================
# TRAINING RESULT
# ============================================================

@dataclass
class SACUpdateResult:
    """
    Metrics returned after one SAC gradient update.
    """

    critic_loss: float
    actor_loss: float

    q1_mean: float
    q2_mean: float

    target_q_mean: float

    log_probability_mean: float

    entropy_estimate: float

    update_step: int

    exploration_noise: float

    def validate(self) -> None:

        numerical_values = [
            self.critic_loss,
            self.actor_loss,
            self.q1_mean,
            self.q2_mean,
            self.target_q_mean,
            self.log_probability_mean,
            self.entropy_estimate,
            self.exploration_noise,
        ]

        if not np.isfinite(
            numerical_values
        ).all():
            raise ValueError(
                "SAC update result contains NaN or Inf."
            )

        if self.update_step < 1:
            raise ValueError(
                "update_step must be positive."
            )

    def summary(
        self,
    ) -> Dict[str, float | int]:

        return {
            "critic_loss":
                self.critic_loss,

            "actor_loss":
                self.actor_loss,

            "q1_mean":
                self.q1_mean,

            "q2_mean":
                self.q2_mean,

            "target_q_mean":
                self.target_q_mean,

            "log_probability_mean":
                self.log_probability_mean,

            "entropy_estimate":
                self.entropy_estimate,

            "update_step":
                self.update_step,

            "exploration_noise":
                self.exploration_noise,
        }
# ============================================================
# HELPER FUNCTIONS
# ============================================================

def torch_dtype_from_name(
    dtype: str,
) -> torch.dtype:

    if dtype == "float32":
        return torch.float32

    if dtype == "float64":
        return torch.float64

    raise ValueError(
        f"Unsupported dtype: {dtype}"
    )
def resolve_device(
    device: str,
) -> torch.device:
    """
    Resolve requested PyTorch device.
    """

    requested = str(
        device
    ).lower()

    if requested == "cpu":

        return torch.device(
            "cpu"
        )

    if requested.startswith(
        "cuda"
    ):

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA device requested but CUDA is unavailable."
            )

        return torch.device(
            requested
        )

    raise ValueError(
        "device must be 'cpu' or a CUDA device."
    )


def hard_update(
    target: nn.Module,
    source: nn.Module,
) -> None:
    """
    Copy all source-network parameters to target network.
    """

    target.load_state_dict(
        source.state_dict()
    )
@torch.no_grad()
def soft_update(
    target: nn.Module,
    source: nn.Module,
    tau: float,
) -> None:
    """
    Polyak / soft target-network update.

        theta_target <-
            (1 - tau) theta_target
            + tau theta_source
    """

    if not (
        0.0
        < tau
        <= 1.0
    ):
        raise ValueError(
            "tau must lie in (0, 1]."
        )

    target_parameters = list(
        target.parameters()
    )

    source_parameters = list(
        source.parameters()
    )

    if (
        len(
            target_parameters
        )
        != len(
            source_parameters
        )
    ):
        raise ValueError(
            "Target and source parameter counts differ."
        )

    for (
        target_parameter,
        source_parameter,
    ) in zip(
        target_parameters,
        source_parameters,
    ):

        target_parameter.data.mul_(
            1.0 - tau
        )

        target_parameter.data.add_(
            source_parameter.data,
            alpha=tau,
        )


def freeze_network(
    network: nn.Module,
) -> None:

    for parameter in (
        network.parameters()
    ):

        parameter.requires_grad_(
            False
        )


def unfreeze_network(
    network: nn.Module,
) -> None:

    for parameter in (
        network.parameters()
    ):

        parameter.requires_grad_(
            True
        )
# ============================================================
# SAC AGENT
# ============================================================

class SACAgent:
    """
    Standard continuous-action Soft Actor-Critic agent.

    One instance may represent either:

        - one local microgrid agent, or
        - the global VPP coordinator.

    The hierarchical roles are added by local_agent.py and
    coordinator_agent.py.
    """
    def __init__(
        self,
        config: SACAgentConfig,
    ) -> None:

        config.validate()

        self.config = config

        self.device = resolve_device(
            config.device
        )
        self.torch_dtype = (
            torch_dtype_from_name(
                config.dtype
            )
        )
        if config.seed is not None:

            np.random.seed(
                config.seed
            )

            torch.manual_seed(
                config.seed
            )
            if torch.cuda.is_available():

                torch.cuda.manual_seed_all(
                    config.seed
                )
        self.rng = np.random.default_rng(
            config.seed
        )

        network_config = SACNetworkConfig(
            state_dimension=(
                config.state_dimension
            ),

            action_dimension=(
                config.action_dimension
            ),

            hidden_dimensions=(
                config.hidden_dimensions
            ),

            activation=(
                config.activation
            ),

            action_low=(
                config.action_low
            ),

            action_high=(
                config.action_high
            ),
        )

        (
            self.actor,
            self.critic,
            self.target_critic,
        ) = build_sac_networks(
            network_config
        )

        self.actor = (
            self.actor.to(
                device=self.device,
                dtype=self.torch_dtype,
            )
        )

        self.critic = (
            self.critic.to(
                device=self.device,
                dtype=self.torch_dtype,
            )
        )

        self.target_critic = (
            self.target_critic.to(
                device=self.device,
                dtype=self.torch_dtype,
            )
        )

        freeze_network(
            self.target_critic
        )

        self.actor_optimizer = Adam(
            self.actor.parameters(),
            lr=config.learning_rate,
        )

        self.critic_optimizer = Adam(
            self.critic.parameters(),
            lr=config.learning_rate,
        )

        replay_config = ReplayBufferConfig(
            capacity=(
                config.replay_buffer_capacity
            ),

            state_dimension=(
                config.state_dimension
            ),

            action_dimension=(
                config.action_dimension
            ),

            dtype=(
                config.dtype
            ),

            seed=(
                config.seed
            ),
        )

        self.replay_buffer = ReplayBuffer(
            replay_config
        )

        self.current_exploration_noise = float(
            config.initial_exploration_noise
        )

        self.update_step = 0

        self.environment_step = 0

        self.training = True

    # ========================================================
    # ARRAY / TENSOR CONVERSION
    # ========================================================

    def _state_array(
        self,
        state,
    ) -> np.ndarray:

        state = np.asarray(
            state,
            dtype=np.float32
            if self.config.dtype == "float32"
            else np.float64,
        )

        if state.ndim != 1:
            raise ValueError(
                "State must be one-dimensional."
            )

        if (
            state.size
            != self.config.state_dimension
        ):
            raise ValueError(
                "State dimension mismatch."
            )

        if not np.isfinite(
            state
        ).all():
            raise ValueError(
                "State contains NaN or Inf."
            )

        return state

    def _state_tensor(
        self,
        state,
    ) -> torch.Tensor:

        state = self._state_array(
            state
        )

        return torch.as_tensor(
            state,
            device=self.device,
            dtype=self.torch_dtype,
        ).unsqueeze(
            0
        )

    def _batch_to_tensors(
        self,
        batch: ReplayBatch,
    ):

        batch.validate()

        states = torch.as_tensor(
            batch.states,
            device=self.device,
            dtype=self.torch_dtype,
        )

        actions = torch.as_tensor(
            batch.actions,
            device=self.device,
            dtype=self.torch_dtype,
        )

        rewards = torch.as_tensor(
            batch.rewards,
            device=self.device,
            dtype=self.torch_dtype,
        )

        next_states = torch.as_tensor(
            batch.next_states,
            device=self.device,
            dtype=self.torch_dtype,
        )

        dones = torch.as_tensor(
            batch.dones,
            device=self.device,
            dtype=self.torch_dtype,
        )

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
        )

    # ========================================================
    # ACTION BOUNDS
    # ========================================================

    @property
    def action_low(
        self,
    ) -> np.ndarray:

        return (
            self.actor.action_low
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

    @property
    def action_high(
        self,
    ) -> np.ndarray:

        return (
            self.actor.action_high
            .detach()
            .cpu()
            .numpy()
            .copy()
        )
    # ========================================================
    # ACTION SELECTION
    # ========================================================

    @torch.no_grad()
    def select_action(
        self,
        state,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Select one bounded physical action.

        deterministic=True
            Uses the actor mean policy.

        deterministic=False
            Uses SAC stochastic sampling and, during training,
            optional decaying external exploration noise.
        """

        state_tensor = (
            self._state_tensor(
                state
            )
        )

        if deterministic:

            action_tensor = (
                self.actor.deterministic(
                    state_tensor
                )
            )

        else:

            (
                action_tensor,
                _,
                _,
            ) = self.actor.sample(
                state_tensor
            )

        action = (
            action_tensor[
                0
            ]
            .detach()
            .cpu()
            .numpy()
        )

        if (
            not deterministic
            and self.training
            and self.current_exploration_noise > 0
        ):

            noise = self.rng.normal(
                loc=0.0,
                scale=(
                    self.current_exploration_noise
                ),
                size=(
                    self.config.action_dimension
                ),
            )

            action = (
                action
                + noise
            )

        action = np.clip(
            action,
            self.action_low,
            self.action_high,
        )

        dtype = (
            np.float32
            if self.config.dtype
            == "float32"
            else np.float64
        )

        return action.astype(
            dtype,
            copy=False,
        )

    # ========================================================
    # EXPERIENCE STORAGE
    # ========================================================

    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ) -> int:

        index = self.replay_buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        self.environment_step += 1

        return index

    # ========================================================
    # SAMPLING
    # ========================================================

    def ready_to_update(
        self,
    ) -> bool:

        return self.replay_buffer.can_sample(
            self.config.batch_size
        )

    # ========================================================
    # CRITIC TARGET
    # ========================================================

    @torch.no_grad()
    def calculate_target_q(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        SAC target:

        y = r + gamma(1-d)
            [min(Q1_target,Q2_target)
             - alpha log pi(a'|s')]
        """

        (
            next_actions,
            next_log_probability,
            _,
        ) = self.actor.sample(
            next_states
        )

        target_q = (
            self.target_critic.minimum(
                next_states,
                next_actions,
            )
        )

        soft_target_q = (
            target_q
            - self.config.entropy_coefficient
            * next_log_probability
        )

        backup = (
            rewards
            + self.config.discount_factor
            * (
                1.0
                - dones
            )
            * soft_target_q
        )

        return backup

    # ========================================================
    # CRITIC UPDATE
    # ========================================================

    def update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        target_q: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        q1, q2 = self.critic(
            states,
            actions,
        )

        q1_loss = nn.functional.mse_loss(
            q1,
            target_q,
        )

        q2_loss = nn.functional.mse_loss(
            q2,
            target_q,
        )

        critic_loss = (
            q1_loss
            + q2_loss
        )

        self.critic_optimizer.zero_grad(
            set_to_none=True
        )

        critic_loss.backward()

        if (
            self.config.gradient_clip_norm
            is not None
        ):

            nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                self.config.gradient_clip_norm,
            )

        self.critic_optimizer.step()

        return (
            critic_loss.detach(),
            q1.detach(),
            q2.detach(),
        )

    # ========================================================
    # ACTOR UPDATE
    # ========================================================

    def update_actor(
        self,
        states: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        # Critic parameters are temporarily frozen so that actor
        # optimization does not accumulate critic gradients.
        freeze_network(
            self.critic
        )

        try:

            (
                actions,
                log_probability,
                _,
            ) = self.actor.sample(
                states
            )

            q_value = (
                self.critic.minimum(
                    states,
                    actions,
                )
            )

            actor_loss = (
                (
                    self.config.entropy_coefficient
                    * log_probability
                )
                - q_value
            ).mean()

            self.actor_optimizer.zero_grad(
                set_to_none=True
            )

            actor_loss.backward()

            if (
                self.config.gradient_clip_norm
                is not None
            ):

                nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self.config.gradient_clip_norm,
                )

            self.actor_optimizer.step()

        finally:

            unfreeze_network(
                self.critic
            )

        return (
            actor_loss.detach(),
            log_probability.detach(),
        )

    # ========================================================
    # FULL UPDATE
    # ========================================================

    def update(
        self,
        batch: Optional[
            ReplayBatch
        ] = None,
    ) -> SACUpdateResult:
        """
        Perform one complete SAC optimization step.

        Sequence:

            1. sample replay mini-batch
            2. compute Bellman target
            3. update twin critics
            4. update actor
            5. soft-update target critics
            6. decay exploration noise
        """

        if batch is None:

            if not self.ready_to_update():

                raise RuntimeError(
                    "Replay buffer does not contain enough "
                    "samples for one SAC update."
                )

            batch = self.replay_buffer.sample(
                self.config.batch_size
            )

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self._batch_to_tensors(
            batch
        )

        target_q = (
            self.calculate_target_q(
                rewards=rewards,
                next_states=next_states,
                dones=dones,
            )
        )

        (
            critic_loss,
            q1,
            q2,
        ) = self.update_critic(
            states=states,
            actions=actions,
            target_q=target_q,
        )

        (
            actor_loss,
            log_probability,
        ) = self.update_actor(
            states=states
        )

        soft_update(
            target=self.target_critic,
            source=self.critic,
            tau=(
                self.config
                .soft_update_coefficient
            ),
        )

        self.update_step += 1

        self.decay_exploration_noise()

        entropy_estimate = float(
            (
                -log_probability.mean()
            )
            .detach()
            .cpu()
            .item()
        )

        result = SACUpdateResult(
            critic_loss=float(
                critic_loss.cpu().item()
            ),

            actor_loss=float(
                actor_loss.cpu().item()
            ),

            q1_mean=float(
                q1.mean().cpu().item()
            ),

            q2_mean=float(
                q2.mean().cpu().item()
            ),

            target_q_mean=float(
                target_q.mean()
                .detach()
                .cpu()
                .item()
            ),

            log_probability_mean=float(
                log_probability.mean()
                .cpu()
                .item()
            ),

            entropy_estimate=(
                entropy_estimate
            ),

            update_step=(
                self.update_step
            ),

            exploration_noise=float(
                self.current_exploration_noise
            ),
        )

        result.validate()

        return result

    # ========================================================
    # EXPLORATION
    # ========================================================

    def decay_exploration_noise(
        self,
    ) -> float:

        updated = (
            self.current_exploration_noise
            * self.config.exploration_noise_decay
        )

        self.current_exploration_noise = max(
            self.config.minimum_exploration_noise,
            updated,
        )

        return float(
            self.current_exploration_noise
        )

    def reset_exploration_noise(
        self,
    ) -> None:

        self.current_exploration_noise = float(
            self.config.initial_exploration_noise
        )

    # ========================================================
    # TRAIN / EVAL MODES
    # ========================================================

    def train_mode(
        self,
        mode: bool = True,
    ) -> None:

        if not isinstance(
            mode,
            bool,
        ):
            raise TypeError(
                "mode must be boolean."
            )

        self.training = mode

        self.actor.train(
            mode
        )

        self.critic.train(
            mode
        )

        self.target_critic.train(
            mode
        )

    def eval_mode(
        self,
    ) -> None:

        self.train_mode(
            False
        )

    # Interface used by hierarchical_controller.py
    def set_training_mode(
        self,
        training: bool,
    ) -> None:

        self.train_mode(
            training
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
    ) -> None:
        """
        Reset episode-level exploration state.

        Learned parameters and replay memory are preserved.
        """

        self.reset_exploration_noise()

    # ========================================================
    # CHECKPOINT SAVE
    # ========================================================

    def save(
        self,
        path,
    ) -> Path:
        """
        Save complete SAC training state.

        The main .pt checkpoint stores:
            - actor
            - critic
            - target critic
            - actor optimizer
            - critic optimizer
            - training counters
            - exploration state
            - agent NumPy RNG state
            - replay-buffer RNG state

        Replay-memory arrays are stored in a separate compressed
        NPZ file next to the main checkpoint.

        Example
        -------
        local_agent_1_episode_100.pt
        local_agent_1_episode_100.replay.npz
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------------
        # Replay-buffer sidecar file
        # --------------------------------------------------------

        replay_path = path.with_suffix(
            ".replay.npz"
        )

        self.replay_buffer.save(
            replay_path
        )

        # --------------------------------------------------------
        # RNG states
        # --------------------------------------------------------

        agent_rng_state = None

        if hasattr(
            self,
            "rng",
        ):
            agent_rng_state = (
                self.rng.bit_generator.state
            )


        replay_rng_state = None

        if hasattr(
            self.replay_buffer,
            "rng",
        ):
            replay_rng_state = (
                self.replay_buffer
                .rng
                .bit_generator
                .state
            )


        # --------------------------------------------------------
        # Main SAC checkpoint
        # --------------------------------------------------------

        checkpoint = {

            "checkpoint_version":
                2,

            "actor":
                self.actor.state_dict(),

            "critic":
                self.critic.state_dict(),

            "target_critic":
                self.target_critic.state_dict(),

            "actor_optimizer":
                self.actor_optimizer.state_dict(),

            "critic_optimizer":
                self.critic_optimizer.state_dict(),

            "update_step":
                self.update_step,

            "environment_step":
                self.environment_step,

            "current_exploration_noise":
                self.current_exploration_noise,

            "agent_rng_state":
                agent_rng_state,

            "replay_rng_state":
                replay_rng_state,

            "replay_buffer_file":
                replay_path.name,

            "replay_buffer_size":
                int(
                    self.replay_buffer.size
                ),

            "replay_buffer_position":
                int(
                    self.replay_buffer.position
                ),

            "replay_buffer_total_added":
                int(
                    self.replay_buffer.total_added
                ),
        }


        torch.save(
            checkpoint,
            path,
        )

        return path

    # ========================================================
    # CHECKPOINT LOAD
    # ========================================================

    def load(
        self,
        path,
        load_optimizers: bool = True,
    ) -> None:
        """
        Load SAC checkpoint.

        Version-2 checkpoints restore:
            - neural networks
            - optimizer states
            - counters
            - exploration state
            - replay memory
            - NumPy RNG states

        Older checkpoints remain loadable. If an old checkpoint
        has no replay-buffer sidecar, the replay buffer remains
        empty.
        """

        path = Path(path)

        if not path.exists():

            raise FileNotFoundError(
                path
            )


        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )


        # ========================================================
        # NETWORKS
        # ========================================================

        self.actor.load_state_dict(
            checkpoint[
                "actor"
            ]
        )

        self.critic.load_state_dict(
            checkpoint[
                "critic"
            ]
        )

        self.target_critic.load_state_dict(
            checkpoint[
                "target_critic"
            ]
        )


        freeze_network(
            self.target_critic
        )


        # ========================================================
        # OPTIMIZERS
        # ========================================================

        if load_optimizers:

            self.actor_optimizer.load_state_dict(
                checkpoint[
                    "actor_optimizer"
                ]
            )

            self.critic_optimizer.load_state_dict(
                checkpoint[
                    "critic_optimizer"
                ]
            )


        # ========================================================
        # TRAINING COUNTERS
        # ========================================================

        self.update_step = int(
            checkpoint.get(
                "update_step",
                0,
            )
        )

        self.environment_step = int(
            checkpoint.get(
                "environment_step",
                0,
            )
        )

        self.current_exploration_noise = float(
            checkpoint.get(
                "current_exploration_noise",
                self.config.initial_exploration_noise,
            )
        )


        # ========================================================
        # AGENT RNG STATE
        # ========================================================

        agent_rng_state = checkpoint.get(
            "agent_rng_state",
            None,
        )

        if (
            agent_rng_state is not None
            and hasattr(
                self,
                "rng",
            )
        ):

            self.rng.bit_generator.state = (
                agent_rng_state
            )


        # ========================================================
        # REPLAY MEMORY
        # ========================================================

        replay_filename = checkpoint.get(
            "replay_buffer_file",
            None,
        )


        if replay_filename is not None:

            replay_path = (
                path.parent
                / replay_filename
            )


            if not replay_path.exists():

                raise FileNotFoundError(
                    "Checkpoint expects replay-buffer "
                    f"file but it was not found:\n"
                    f"{replay_path}"
                )


            self.replay_buffer.load(
                replay_path
            )


            # ----------------------------------------------------
            # Restore replay-buffer RNG state
            # ----------------------------------------------------

            replay_rng_state = checkpoint.get(
                "replay_rng_state",
                None,
            )


            if (
                replay_rng_state is not None
                and hasattr(
                    self.replay_buffer,
                    "rng",
                )
            ):

                self.replay_buffer.rng.bit_generator.state = (
                    replay_rng_state
                )
        # ========================================================
        # BACKWARD COMPATIBILITY
        # ========================================================
        #
        # Existing episode-100 / episode-200 checkpoints were
        # created before replay-buffer persistence was introduced.
        #
        # They will still load correctly, but replay memory will
        # remain empty.
        # ========================================================

        self.training = True

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(
        self,
    ) -> Dict[str, object]:

        return {
            "agent":
                "SACAgent",

            "state_dimension":
                self.config.state_dimension,

            "action_dimension":
                self.config.action_dimension,

            "discount_factor":
                self.config.discount_factor,

            "learning_rate":
                self.config.learning_rate,

            "batch_size":
                self.config.batch_size,

            "replay_buffer_capacity":
                self.config.replay_buffer_capacity,

            "replay_buffer_size":
                len(
                    self.replay_buffer
                ),

            "soft_update_coefficient":
                self.config.soft_update_coefficient,

            "entropy_coefficient":
                self.config.entropy_coefficient,

            "current_exploration_noise":
                self.current_exploration_noise,

            "update_step":
                self.update_step,

            "environment_step":
                self.environment_step,

            "device":
                str(
                    self.device
                ),

            "training":
                self.training,
        }
