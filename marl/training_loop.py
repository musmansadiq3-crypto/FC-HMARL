"""
Offline FC-HMARL training loop.

This module implements the training sequence described in
Algorithm 1 of the manuscript.

Manuscript-supported procedure
------------------------------
1. Initialize local and coordinator networks.
2. Initialize replay memory and target networks.
3. Generate forecasts.
4. Calculate forecasting confidence.
5. Construct confidence-aware predictive states.
6. Observe local and coordinator states.
7. Execute local actions.
8. Execute coordinator actions.
9. Operate the VPP.
10. Calculate rewards.
11. Store transitions.
12. Sample mini-batches.
13. Update local SAC agents.
14. Update coordinator SAC agent.
15. Soft-update target networks.
16. Repeat until convergence.

Important implementation note
-----------------------------
The manuscript specifies the training workflow but does not
specify the exact software API between the environment,
forecasting module, state builder, and agents.

This implementation therefore uses reset_function and
step_function callbacks. These callbacks will later be connected
to the concrete VPPEnvironment in train.py.
"""

from __future__ import annotations

import random
import numpy as np
import torch

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
)

from agents.local_agent import (
    LocalAgent,
    LocalTrainingResult,
)

from agents.coordinator_agent import (
    CoordinatorAgent,
    CoordinatorTrainingResult,
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

@dataclass
class TrainingLoopConfig:
    """
    FC-HMARL offline training configuration.

    Manuscript-supported defaults
    -----------------------------
    maximum_episodes = 5000
    steps_per_episode = 24

    The manuscript describes a 24-hour operating episode and
    maximum training of 5000 episodes.

    Other software-specific options are reconstruction choices.
    """

    maximum_episodes: int = 5000

    steps_per_episode: int = 24

    updates_per_step: int = 1

    checkpoint_interval: int = 100

    enable_checkpointing: bool = True

    checkpoint_directory: str = (
        "outputs/checkpoints"
    )

    deterministic_actions: bool = False

    verbose: bool = True

    def validate(self) -> None:

        if not isinstance(
            self.maximum_episodes,
            int,
        ):
            raise TypeError(
                "maximum_episodes must be an integer."
            )

        if self.maximum_episodes <= 0:
            raise ValueError(
                "maximum_episodes must be positive."
            )

        if not isinstance(
            self.steps_per_episode,
            int,
        ):
            raise TypeError(
                "steps_per_episode must be an integer."
            )

        if self.steps_per_episode <= 0:
            raise ValueError(
                "steps_per_episode must be positive."
            )

        if not isinstance(
            self.updates_per_step,
            int,
        ):
            raise TypeError(
                "updates_per_step must be an integer."
            )

        if self.updates_per_step <= 0:
            raise ValueError(
                "updates_per_step must be positive."
            )

        if not isinstance(
            self.checkpoint_interval,
            int,
        ):
            raise TypeError(
                "checkpoint_interval must be an integer."
            )

        if self.checkpoint_interval <= 0:
            raise ValueError(
                "checkpoint_interval must be positive."
            )

        if not isinstance(
            self.enable_checkpointing,
            bool,
        ):
            raise TypeError(
                "enable_checkpointing must be boolean."
            )

        if not isinstance(
            self.deterministic_actions,
            bool,
        ):
            raise TypeError(
                "deterministic_actions must be boolean."
            )

        if not isinstance(
            self.verbose,
            bool,
        ):
            raise TypeError(
                "verbose must be boolean."
            )


# ============================================================
# OBSERVATION
# ============================================================

@dataclass
class TrainingObservation:
    """
    Hierarchical FC-HMARL observation.

    local_states:
        one local state vector for every microgrid agent.

    coordinator_state:
        global VPP coordinator state vector.

    predictive/confidence information is expected to already
    be embedded by the StateBuilder before this object is
    supplied to the training loop.
    """

    local_states: Sequence[np.ndarray]

    coordinator_state: np.ndarray

    def validate(
        self,
        expected_local_agents: Optional[int] = None,
    ) -> None:

        if not isinstance(
            self.local_states,
            Sequence,
        ):
            raise TypeError(
                "local_states must be a sequence."
            )

        if (
            expected_local_agents
            is not None
            and len(self.local_states)
            != expected_local_agents
        ):
            raise ValueError(
                "Number of local states does not match "
                "number of local agents."
            )

        for index, state in enumerate(
            self.local_states
        ):

            array = np.asarray(
                state
            )

            if array.ndim != 1:
                raise ValueError(
                    f"Local state {index} must be "
                    "one-dimensional."
                )

            if not np.isfinite(
                array
            ).all():
                raise ValueError(
                    f"Local state {index} contains "
                    "NaN or Inf."
                )

        coordinator_array = np.asarray(
            self.coordinator_state
        )

        if coordinator_array.ndim != 1:
            raise ValueError(
                "coordinator_state must be "
                "one-dimensional."
            )

        if not np.isfinite(
            coordinator_array
        ).all():
            raise ValueError(
                "coordinator_state contains NaN or Inf."
            )


# ============================================================
# ENVIRONMENT STEP RESULT
# ============================================================

@dataclass
class EnvironmentStepResult:
    """
    Result returned by the VPP step adapter.
    """

    next_observation: TrainingObservation

    local_rewards: Sequence[float]

    coordinator_reward: float

    done: bool

    info: Dict[str, Any] = field(
        default_factory=dict
    )

    def validate(
        self,
        expected_local_agents: Optional[int] = None,
    ) -> None:
        """
        Validate one environment transition.

        local_rewards may originate from NumPy-based physical and
        reward calculations.  Normalize them to a plain Python
        list[float] before performing the training-loop checks.
        """

        # --------------------------------------------------------
        # NEXT OBSERVATION
        # --------------------------------------------------------

        self.next_observation.validate(
            expected_local_agents
        )

        # --------------------------------------------------------
        # NORMALIZE LOCAL REWARDS
        # --------------------------------------------------------

        try:
            reward_array = np.asarray(
                self.local_rewards,
                dtype=np.float64,
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise TypeError(
                "local_rewards must be convertible "
                "to a one-dimensional numeric sequence."
            ) from exc

        if reward_array.ndim == 0:
            raise TypeError(
                "local_rewards must contain one reward "
                "for every local agent."
            )

        if reward_array.ndim != 1:
            raise ValueError(
                "local_rewards must be one-dimensional."
            )

        # Critical normalization:
        # np.ndarray -> genuine Python list[float]
        self.local_rewards = [
            float(reward)
            for reward
            in reward_array.tolist()
        ]

        # --------------------------------------------------------
        # NUMBER OF LOCAL REWARDS
        # --------------------------------------------------------

        if (
            expected_local_agents
            is not None
            and
            len(
                self.local_rewards
            )
            != expected_local_agents
        ):
            raise ValueError(
                "Number of local rewards does not match "
                "number of local agents. "
                f"Expected {expected_local_agents}, "
                f"received {len(self.local_rewards)}."
            )

        # --------------------------------------------------------
        # FINITE LOCAL REWARDS
        # --------------------------------------------------------

        for (
            index,
            reward,
        ) in enumerate(
            self.local_rewards
        ):

            if not np.isfinite(
                float(
                    reward
                )
            ):
                raise ValueError(
                    f"Local reward {index} contains "
                    "NaN or Inf."
                )

        # --------------------------------------------------------
        # COORDINATOR REWARD
        # --------------------------------------------------------

        try:
            self.coordinator_reward = float(
                self.coordinator_reward
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise TypeError(
                "coordinator_reward must be numeric."
            ) from exc

        if not np.isfinite(
            self.coordinator_reward
        ):
            raise ValueError(
                "Coordinator reward contains NaN or Inf."
            )

        # --------------------------------------------------------
        # DONE FLAG
        # --------------------------------------------------------

        if not isinstance(
            self.done,
            (
                bool,
                np.bool_,
            ),
        ):
            raise TypeError(
                "done must be boolean."
            )

        self.done = bool(
            self.done
        )

        # --------------------------------------------------------
        # INFO
        # --------------------------------------------------------

        if not isinstance(
            self.info,
            dict,
        ):
            raise TypeError(
                "info must be a dictionary."
            )


# ============================================================
# ACTION BUNDLE
# ============================================================

@dataclass
class HierarchicalActionBundle:
    """
    Actions selected at one FC-HMARL time step.
    """

    local_actions: List[np.ndarray]

    coordinator_action: np.ndarray

    def validate(
        self,
        expected_local_agents: Optional[int] = None,
    ) -> None:

        if (
            expected_local_agents
            is not None
            and len(self.local_actions)
            != expected_local_agents
        ):
            raise ValueError(
                "Number of local actions does not match "
                "number of local agents."
            )

        for index, action in enumerate(
            self.local_actions
        ):

            array = np.asarray(
                action
            )

            if array.ndim != 1:
                raise ValueError(
                    f"Local action {index} must be "
                    "one-dimensional."
                )

            if not np.isfinite(
                array
            ).all():
                raise ValueError(
                    f"Local action {index} contains "
                    "NaN or Inf."
                )

        coordinator_array = np.asarray(
            self.coordinator_action
        )

        if coordinator_array.ndim != 1:
            raise ValueError(
                "Coordinator action must be "
                "one-dimensional."
            )

        if not np.isfinite(
            coordinator_array
        ).all():
            raise ValueError(
                "Coordinator action contains NaN or Inf."
            )


# ============================================================
# STEP STATISTICS
# ============================================================

@dataclass
class TrainingStepStatistics:
    """
    Statistics collected for one operating interval.
    """

    episode: int

    time_step: int

    local_rewards: List[float]

    coordinator_reward: float

    total_reward: float

    local_agents_updated: int

    coordinator_updated: bool

    done: bool


# ============================================================
# EPISODE STATISTICS
# ============================================================

@dataclass
class EpisodeStatistics:
    """
    Summary of one 24-hour FC-HMARL episode.
    """

    episode: int

    steps: int

    local_returns: List[float]

    coordinator_return: float

    total_return: float

    local_update_count: int

    coordinator_update_count: int

    terminated_early: bool

    step_statistics: List[
        TrainingStepStatistics
    ] = field(
        default_factory=list
    )

    def summary(
        self,
    ) -> Dict[str, Any]:

        return {
            "episode":
                self.episode,

            "steps":
                self.steps,

            "local_returns":
                self.local_returns.copy(),

            "coordinator_return":
                self.coordinator_return,

            "total_return":
                self.total_return,

            "local_update_count":
                self.local_update_count,

            "coordinator_update_count":
                self.coordinator_update_count,

            "terminated_early":
                self.terminated_early,
        }


# ============================================================
# TRAINING HISTORY
# ============================================================

@dataclass
class TrainingHistory:
    """
    Full FC-HMARL training history.
    """

    episodes: List[
        EpisodeStatistics
    ] = field(
        default_factory=list
    )

    @property
    def number_of_episodes(
        self,
    ) -> int:

        return len(
            self.episodes
        )

    @property
    def total_steps(
        self,
    ) -> int:

        return int(
            sum(
                episode.steps
                for episode
                in self.episodes
            )
        )

    @property
    def total_returns(
        self,
    ) -> List[float]:

        return [
            episode.total_return
            for episode
            in self.episodes
        ]

    @property
    def final_return(
        self,
    ) -> Optional[float]:

        if not self.episodes:
            return None

        return self.episodes[
            -1
        ].total_return

    def summary(
        self,
    ) -> Dict[str, Any]:

        if not self.episodes:

            return {
                "number_of_episodes": 0,
                "total_steps": 0,
                "final_return": None,
                "mean_return": None,
            }

        returns = np.asarray(
            self.total_returns,
            dtype=np.float64,
        )

        return {
            "number_of_episodes":
                self.number_of_episodes,

            "total_steps":
                self.total_steps,

            "final_return":
                self.final_return,

            "mean_return":
                float(
                    np.mean(
                        returns
                    )
                ),

            "maximum_return":
                float(
                    np.max(
                        returns
                    )
                ),

            "minimum_return":
                float(
                    np.min(
                        returns
                    )
                ),
        }


# ============================================================
# CALLBACK TYPES
# ============================================================

ResetFunction = Callable[
    [int],
    TrainingObservation,
]

StepFunction = Callable[
    [
        int,
        int,
        HierarchicalActionBundle,
    ],
    EnvironmentStepResult,
]

StopFunction = Callable[
    [TrainingHistory],
    bool,
]


# ============================================================
# FC-HMARL TRAINING LOOP
# ============================================================

class FCHMARLTrainingLoop:
    """
    Offline training procedure for FC-HMARL.

    The loop is intentionally independent of the exact
    environment software API.

    reset_function
    --------------
    Receives the episode index and returns the initial
    TrainingObservation.

    step_function
    -------------
    Receives:

        episode index
        time-step index
        HierarchicalActionBundle

    and returns EnvironmentStepResult.

    This design lets train.py later connect:

        forecasting
        confidence calculation
        StateBuilder
        VPPEnvironment
        reward functions

    without duplicating SAC training logic.
    """

    def __init__(
        self,
        local_agents: Sequence[
            LocalAgent
        ],
        coordinator_agent: CoordinatorAgent,
        reset_function: ResetFunction,
        step_function: StepFunction,
        config: Optional[
            TrainingLoopConfig
        ] = None,
        stop_function: Optional[
            StopFunction
        ] = None,
    ) -> None:

        if config is None:
            config = TrainingLoopConfig()

        config.validate()

        if not isinstance(
            local_agents,
            Sequence,
        ):
            raise TypeError(
                "local_agents must be a sequence."
            )

        if len(
            local_agents
        ) == 0:
            raise ValueError(
                "At least one local agent is required."
            )

        for agent in local_agents:

            if not isinstance(
                agent,
                LocalAgent,
            ):
                raise TypeError(
                    "Every local agent must be "
                    "an instance of LocalAgent."
                )

        if not isinstance(
            coordinator_agent,
            CoordinatorAgent,
        ):
            raise TypeError(
                "coordinator_agent must be "
                "CoordinatorAgent."
            )

        if not callable(
            reset_function
        ):
            raise TypeError(
                "reset_function must be callable."
            )

        if not callable(
            step_function
        ):
            raise TypeError(
                "step_function must be callable."
            )

        if (
            stop_function is not None
            and not callable(
                stop_function
            )
        ):
            raise TypeError(
                "stop_function must be callable."
            )

        self.local_agents = list(
            local_agents
        )

        self.coordinator_agent = (
            coordinator_agent
        )

        self.reset_function = (
            reset_function
        )

        self.step_function = (
            step_function
        )

        self.stop_function = (
            stop_function
        )

        self.config = config

        self.history = (
            TrainingHistory()
        )

        self.global_step = 0

    # ========================================================
    # PROPERTIES
    # ========================================================

    @property
    def number_of_local_agents(
        self,
    ) -> int:

        return len(
            self.local_agents
        )

    # ========================================================
    # MODES
    # ========================================================

    def set_training_mode(
        self,
        training: bool,
    ) -> None:

        for agent in self.local_agents:

            agent.set_training_mode(
                training
            )

        self.coordinator_agent.set_training_mode(
            training
        )

    def reset_agents(
        self,
    ) -> None:

        for agent in self.local_agents:

            agent.reset()

        self.coordinator_agent.reset()

    # ========================================================
    # ACTION SELECTION
    # ========================================================

    def select_actions(
        self,
        observation: TrainingObservation,
    ) -> HierarchicalActionBundle:

        observation.validate(
            self.number_of_local_agents
        )

        local_actions = []

        for agent, local_state in zip(
            self.local_agents,
            observation.local_states,
        ):

            action = agent.select_action(
                local_state,
                deterministic=(
                    self.config
                    .deterministic_actions
                ),
            )

            local_actions.append(
                np.asarray(
                    action
                ).copy()
            )

        coordinator_action = (
            self.coordinator_agent
            .select_action(
                observation.coordinator_state,
                deterministic=(
                    self.config
                    .deterministic_actions
                ),
            )
        )

        bundle = HierarchicalActionBundle(
            local_actions=local_actions,

            coordinator_action=(
                np.asarray(
                    coordinator_action
                ).copy()
            ),
        )

        bundle.validate(
            self.number_of_local_agents
        )

        return bundle

    # ========================================================
    # STORE EXPERIENCE
    # ========================================================

    def store_transitions(
        self,
        observation: TrainingObservation,
        actions: HierarchicalActionBundle,
        result: EnvironmentStepResult,
    ) -> None:

        observation.validate(
            self.number_of_local_agents
        )

        actions.validate(
            self.number_of_local_agents
        )

        result.validate(
            self.number_of_local_agents
        )

        for index, agent in enumerate(
            self.local_agents
        ):

            agent.store_transition(
                state=(
                    observation
                    .local_states[
                        index
                    ]
                ),

                action=(
                    actions
                    .local_actions[
                        index
                    ]
                ),

                reward=(
                    result
                    .local_rewards[
                        index
                    ]
                ),

                next_state=(
                    result
                    .next_observation
                    .local_states[
                        index
                    ]
                ),

                done=result.done,
            )

        self.coordinator_agent.store_transition(
            state=(
                observation
                .coordinator_state
            ),

            action=(
                actions
                .coordinator_action
            ),

            reward=(
                result
                .coordinator_reward
            ),

            next_state=(
                result
                .next_observation
                .coordinator_state
            ),

            done=result.done,
        )

    # ========================================================
    # NETWORK UPDATES
    # ========================================================

    def update_agents(
        self,
    ) -> tuple[
        List[LocalTrainingResult],
        CoordinatorTrainingResult,
    ]:

        local_results = []

        coordinator_result = None

        for _ in range(
            self.config.updates_per_step
        ):

            current_local_results = []

            for agent in self.local_agents:

                result = (
                    agent
                    .update_if_ready()
                )

                current_local_results.append(
                    result
                )

            coordinator_result = (
                self.coordinator_agent
                .update_if_ready()
            )

            local_results = (
                current_local_results
            )

        assert (
            coordinator_result
            is not None
        )

        return (
            local_results,
            coordinator_result,
        )

    # ========================================================
    # ONE TRAINING STEP
    # ========================================================

    def run_step(
        self,
        episode: int,
        time_step: int,
        observation: TrainingObservation,
    ) -> tuple[
        TrainingObservation,
        TrainingStepStatistics,
    ]:

        actions = self.select_actions(
            observation
        )

        result = self.step_function(
            episode,
            time_step,
            actions,
        )

        if not isinstance(
            result,
            EnvironmentStepResult,
        ):
            raise TypeError(
                "step_function must return "
                "EnvironmentStepResult."
            )

        result.validate(
            self.number_of_local_agents
        )

        self.store_transitions(
            observation,
            actions,
            result,
        )

        (
            local_update_results,
            coordinator_update_result,
        ) = self.update_agents()

        local_agents_updated = sum(
            1
            for update_result
            in local_update_results
            if update_result.updated
        )

        coordinator_updated = (
            coordinator_update_result.updated
        )

        local_rewards = [
            float(reward)
            for reward
            in result.local_rewards
        ]

        coordinator_reward = float(
            result.coordinator_reward
        )

        total_reward = (
            float(
                np.sum(
                    local_rewards
                )
            )
            + coordinator_reward
        )

        statistics = (
            TrainingStepStatistics(
                episode=episode,

                time_step=time_step,

                local_rewards=(
                    local_rewards
                ),

                coordinator_reward=(
                    coordinator_reward
                ),

                total_reward=(
                    total_reward
                ),

                local_agents_updated=(
                    local_agents_updated
                ),

                coordinator_updated=(
                    coordinator_updated
                ),

                done=bool(
                    result.done
                ),
            )
        )

        self.global_step += 1

        return (
            result.next_observation,
            statistics,
        )

    # ========================================================
    # ONE EPISODE
    # ========================================================

    def run_episode(
        self,
        episode: int,
    ) -> EpisodeStatistics:

        self.reset_agents()

        observation = (
            self.reset_function(
                episode
            )
        )

        if not isinstance(
            observation,
            TrainingObservation,
        ):
            raise TypeError(
                "reset_function must return "
                "TrainingObservation."
            )

        observation.validate(
            self.number_of_local_agents
        )

        local_returns = [
            0.0
            for _ in range(
                self.number_of_local_agents
            )
        ]

        coordinator_return = 0.0

        local_update_count = 0

        coordinator_update_count = 0

        step_statistics = []

        terminated_early = False

        for time_step in range(
            self.config.steps_per_episode
        ):

            (
                next_observation,
                statistics,
            ) = self.run_step(
                episode=episode,
                time_step=time_step,
                observation=observation,
            )

            step_statistics.append(
                statistics
            )

            for index, reward in enumerate(
                statistics.local_rewards
            ):

                local_returns[
                    index
                ] += float(
                    reward
                )

            coordinator_return += float(
                statistics
                .coordinator_reward
            )

            local_update_count += (
                statistics
                .local_agents_updated
            )

            if (
                statistics
                .coordinator_updated
            ):

                coordinator_update_count += 1

            observation = (
                next_observation
            )

            if statistics.done:

                terminated_early = (
                    time_step
                    < (
                        self.config
                        .steps_per_episode
                        - 1
                    )
                )

                break

        total_return = (
            float(
                np.sum(
                    local_returns
                )
            )
            + float(
                coordinator_return
            )
        )

        episode_statistics = (
            EpisodeStatistics(
                episode=episode,

                steps=len(
                    step_statistics
                ),

                local_returns=(
                    local_returns
                ),

                coordinator_return=(
                    coordinator_return
                ),

                total_return=(
                    total_return
                ),

                local_update_count=(
                    local_update_count
                ),

                coordinator_update_count=(
                    coordinator_update_count
                ),

                terminated_early=(
                    terminated_early
                ),

                step_statistics=(
                    step_statistics
                ),
            )
        )

        return episode_statistics

    # ========================================================
    # CHECKPOINTS
    # ========================================================

    def save_checkpoint(
        self,
        episode: int,
    ) -> Dict[str, Path]:
        """
        Save a complete FC-HMARL checkpoint.

        Each checkpoint contains:

            1. Five local-agent SAC checkpoints
            2. Five local-agent replay-buffer sidecars
            3. Coordinator SAC checkpoint
            4. Coordinator replay-buffer sidecar
            5. Global training RNG state

        The RNG-state file allows stochastic training to continue
        from the same random state after interruption.
        """

        episode = int(
            episode
        )

        directory = Path(
            self.config.checkpoint_directory
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


        saved_paths = {}


        # ========================================================
        # LOCAL AGENTS
        # ========================================================

        for agent in self.local_agents:

            path = (
                directory
                / (
                    f"local_agent_"
                    f"{agent.microgrid_id}_"
                    f"episode_{episode}.pt"
                )
            )

            saved_paths[
                f"local_{agent.microgrid_id}"
            ] = agent.save(
                path
            )


        # ========================================================
        # COORDINATOR
        # ========================================================

        coordinator_path = (
            directory
            / (
                "coordinator_"
                f"episode_{episode}.pt"
            )
        )

        saved_paths[
            "coordinator"
        ] = (
            self.coordinator_agent.save(
                coordinator_path
            )
        )


        # ========================================================
        # GLOBAL TRAINING STATE
        # ========================================================

        training_state_path = (
            directory
            / (
                "training_state_"
                f"episode_{episode}.pt"
            )
        )


        training_state = {

            "checkpoint_version":
                2,

            "episode":
                episode,

            "python_random_state":
                random.getstate(),

            "numpy_random_state":
                np.random.get_state(),

            "torch_cpu_rng_state":
                torch.get_rng_state(),

            "torch_cuda_rng_state_all":
                (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),

            "number_of_local_agents":
                self.number_of_local_agents,

            "maximum_episodes":
                self.config.maximum_episodes,

            "steps_per_episode":
                self.config.steps_per_episode,

            "checkpoint_interval":
                self.config.checkpoint_interval,
        }


        torch.save(
            training_state,
            training_state_path,
        )


        saved_paths[
            "training_state"
        ] = (
            training_state_path
        )


        return saved_paths

    # ========================================================
    # FULL TRAINING
    # ========================================================

    def train(
        self,
        start_episode: int = 1,
    ) -> TrainingHistory:
        """
        Execute FC-HMARL training.

        Parameters
        ----------
        start_episode:
            Absolute first episode number to execute.

            Default = 1 for normal training.

            Example:
                start_episode=201
                maximum_episodes=500

            runs episodes:
                201, 202, ..., 500

        This preserves absolute episode numbering when training
        is resumed from a complete checkpoint.
        """

        start_episode = int(
            start_episode
        )

        if start_episode <= 0:

            raise ValueError(
                "start_episode must be positive."
            )

        if (
            start_episode
            > self.config.maximum_episodes
        ):

            raise ValueError(
                "start_episode cannot exceed "
                "maximum_episodes."
            )


        self.set_training_mode(
            True
        )


        for episode in range(
            start_episode,
            self.config.maximum_episodes
            + 1,
        ):

            statistics = (
                self.run_episode(
                    episode
                )
            )


            self.history.episodes.append(
                statistics
            )


            if self.config.verbose:

                print(
                    f"Episode "
                    f"{episode:5d}/"
                    f"{self.config.maximum_episodes} "
                    f"| Steps: "
                    f"{statistics.steps:2d} "
                    f"| Return: "
                    f"{statistics.total_return:.6f} "
                    f"| Local updates: "
                    f"{statistics.local_update_count} "
                    f"| Coordinator updates: "
                    f"{statistics.coordinator_update_count}"
                )


            if (
                self.config.enable_checkpointing
                and episode
                % self.config.checkpoint_interval
                == 0
            ):

                self.save_checkpoint(
                    episode
                )


            if self.stop_function is not None:

                if self.stop_function(
                    self.history
                ):

                    break


        return self.history

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(
        self,
    ) -> Dict[str, Any]:

        return {
            "number_of_local_agents":
                self.number_of_local_agents,

            "maximum_episodes":
                self.config.maximum_episodes,

            "steps_per_episode":
                self.config.steps_per_episode,

            "updates_per_step":
                self.config.updates_per_step,

            "global_step":
                self.global_step,

            "completed_episodes":
                self.history.number_of_episodes,

            "training_history":
                self.history.summary(),
        }