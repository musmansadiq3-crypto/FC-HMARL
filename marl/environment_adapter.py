from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
)
import numpy as np

from marl.training_loop import (
    EnvironmentStepResult,
    HierarchicalActionBundle,
    TrainingObservation,
)

# ============================================================
# TYPE ALIASES
# ============================================================

EnvironmentResetFunction = Callable[
    [int],
    Any,
]

EnvironmentStepFunction = Callable[
    [Any],
    Any,
]

ObservationBuilderFunction = Callable[
    [Any],
    TrainingObservation,
]

ActionConverterFunction = Callable[
    [HierarchicalActionBundle],
    Any,
]

RewardBuilderFunction = Callable[
    [
        Any,
        Any,
        HierarchicalActionBundle,
        Any,
    ],
    "RewardOutput",
]

DoneExtractorFunction = Callable[
    [Any],
    bool,
]

InfoExtractorFunction = Callable[
    [Any],
    Dict[str, Any],
]


# ============================================================
# ADAPTER CONFIGURATION
# ============================================================

@dataclass
class EnvironmentAdapterConfig:
    """
    Configuration for the VPP-to-FC-HMARL adapter.

    number_of_microgrids = 5 is manuscript-supported.

    Other fields are software-interface options.
    """

    number_of_microgrids: int = 5

    strict_validation: bool = True

    terminate_after_steps: Optional[int] = 24

    def validate(self) -> None:

        if not isinstance(
            self.number_of_microgrids,
            int,
        ):
            raise TypeError(
                "number_of_microgrids must be an integer."
            )

        if self.number_of_microgrids <= 0:
            raise ValueError(
                "number_of_microgrids must be positive."
            )

        if not isinstance(
            self.strict_validation,
            bool,
        ):
            raise TypeError(
                "strict_validation must be boolean."
            )

        if self.terminate_after_steps is not None:

            if not isinstance(
                self.terminate_after_steps,
                int,
            ):
                raise TypeError(
                    "terminate_after_steps must be "
                    "an integer or None."
                )

            if self.terminate_after_steps <= 0:
                raise ValueError(
                    "terminate_after_steps must be positive."
                )


# ============================================================
# REWARD OUTPUT
# ============================================================

@dataclass
class RewardOutput:
    """
    Hierarchical reward result.

    local_rewards:
        one reward for each microgrid agent.

    coordinator_reward:
        global VPP reward.

    components:
        optional detailed reward information for logging.
    """

    local_rewards: Sequence[float]

    coordinator_reward: float

    components: Dict[str, Any] = field(
        default_factory=dict
    )

    def validate(
        self,
        number_of_microgrids: Optional[int] = None,
    ) -> None:

        if not isinstance(
            self.local_rewards,
            Sequence,
        ):
            raise TypeError(
                "local_rewards must be a sequence."
            )

        if (
            number_of_microgrids is not None
            and len(self.local_rewards)
            != number_of_microgrids
        ):
            raise ValueError(
                "Number of local rewards does not match "
                "number_of_microgrids."
            )

        for index, reward in enumerate(
            self.local_rewards
        ):

            value = float(
                reward
            )

            if not np.isfinite(
                value
            ):
                raise ValueError(
                    f"Local reward {index} contains "
                    "NaN or Inf."
                )

        coordinator_value = float(
            self.coordinator_reward
        )

        if not np.isfinite(
            coordinator_value
        ):
            raise ValueError(
                "coordinator_reward contains NaN or Inf."
            )

        if not isinstance(
            self.components,
            dict,
        ):
            raise TypeError(
                "components must be a dictionary."
            )


# ============================================================
# ADAPTER STEP RECORD
# ============================================================

@dataclass
class AdapterStepRecord:
    """
    Diagnostic information for one adapter step.
    """

    episode: int

    time_step: int

    physical_action: Any

    raw_environment_result: Any

    reward_output: RewardOutput

    done: bool


# ============================================================
# ENVIRONMENT ADAPTER
# ============================================================

class FCHMARLEnvironmentAdapter:

    def __init__(
        self,
        environment_reset_function: EnvironmentResetFunction,
        environment_step_function: EnvironmentStepFunction,
        observation_builder: ObservationBuilderFunction,
        action_converter: ActionConverterFunction,
        reward_builder: RewardBuilderFunction,
        config: Optional[
            EnvironmentAdapterConfig
        ] = None,
        done_extractor: Optional[
            DoneExtractorFunction
        ] = None,
        info_extractor: Optional[
            InfoExtractorFunction
        ] = None,
    ) -> None:

        if config is None:
            config = EnvironmentAdapterConfig()

        config.validate()

        required_functions = {
            "environment_reset_function":
                environment_reset_function,

            "environment_step_function":
                environment_step_function,

            "observation_builder":
                observation_builder,

            "action_converter":
                action_converter,

            "reward_builder":
                reward_builder,
        }

        for name, function in (
            required_functions.items()
        ):

            if not callable(
                function
            ):
                raise TypeError(
                    f"{name} must be callable."
                )

        if (
            done_extractor is not None
            and not callable(
                done_extractor
            )
        ):
            raise TypeError(
                "done_extractor must be callable."
            )

        if (
            info_extractor is not None
            and not callable(
                info_extractor
            )
        ):
            raise TypeError(
                "info_extractor must be callable."
            )

        self.environment_reset_function = (
            environment_reset_function
        )

        self.environment_step_function = (
            environment_step_function
        )

        self.observation_builder = (
            observation_builder
        )

        self.action_converter = (
            action_converter
        )

        self.reward_builder = (
            reward_builder
        )

        self.done_extractor = (
            done_extractor
        )

        self.info_extractor = (
            info_extractor
        )

        self.config = config

        self.current_episode: Optional[
            int
        ] = None

        self.current_time_step = 0

        self.current_raw_state: Any = None

        self.current_observation: Optional[
            TrainingObservation
        ] = None

        self.last_record: Optional[
            AdapterStepRecord
        ] = None

        self.total_resets = 0

        self.total_steps = 0

    # ========================================================
    # OBSERVATION VALIDATION
    # ========================================================

    def _validate_observation(
        self,
        observation: TrainingObservation,
    ) -> TrainingObservation:

        if not isinstance(
            observation,
            TrainingObservation,
        ):
            raise TypeError(
                "observation_builder must return "
                "TrainingObservation."
            )

        if self.config.strict_validation:

            observation.validate(
                expected_local_agents=(
                    self.config
                    .number_of_microgrids
                )
            )

        return observation

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        episode: int,
    ) -> TrainingObservation:
        """
        Reset physical environment for a new episode.

        This method has exactly the callback signature required
        by FCHMARLTrainingLoop.reset_function.
        """

        if not isinstance(
            episode,
            int,
        ):
            raise TypeError(
                "episode must be an integer."
            )

        if episode <= 0:
            raise ValueError(
                "episode must be positive."
            )

        raw_state = (
            self.environment_reset_function(
                episode
            )
        )

        observation = (
            self.observation_builder(
                raw_state
            )
        )

        observation = (
            self._validate_observation(
                observation
            )
        )

        self.current_episode = episode

        self.current_time_step = 0

        self.current_raw_state = (
            raw_state
        )

        self.current_observation = (
            observation
        )

        self.last_record = None

        self.total_resets += 1

        return observation

    # ========================================================
    # DONE
    # ========================================================

    def _extract_done(
        self,
        raw_result: Any,
        time_step: int,
    ) -> bool:

        physical_done = False

        if self.done_extractor is not None:

            physical_done = bool(
                self.done_extractor(
                    raw_result
                )
            )

        horizon_done = False

        if (
            self.config
            .terminate_after_steps
            is not None
        ):

            horizon_done = (
                time_step + 1
                >= self.config
                .terminate_after_steps
            )

        return bool(
            physical_done
            or horizon_done
        )

    # ========================================================
    # INFO
    # ========================================================

    def _extract_info(
        self,
        raw_result: Any,
    ) -> Dict[str, Any]:

        if self.info_extractor is None:

            return {}

        info = (
            self.info_extractor(
                raw_result
            )
        )

        if not isinstance(
            info,
            dict,
        ):
            raise TypeError(
                "info_extractor must return a dictionary."
            )

        return dict(
            info
        )

    # ========================================================
    # STEP
    # ========================================================

    def step(
        self,
        episode: int,
        time_step: int,
        actions: HierarchicalActionBundle,
    ) -> EnvironmentStepResult:
        """
        Execute one physical FC-HMARL operating interval.

        This method has exactly the callback signature required
        by FCHMARLTrainingLoop.step_function.
        """

        if self.current_episode is None:
            raise RuntimeError(
                "Environment adapter must be reset before step."
            )

        if episode != self.current_episode:
            raise ValueError(
                "Episode does not match currently active "
                "environment episode."
            )

        if not isinstance(
            time_step,
            int,
        ):
            raise TypeError(
                "time_step must be an integer."
            )

        if time_step < 0:
            raise ValueError(
                "time_step cannot be negative."
            )

        if time_step != self.current_time_step:
            raise ValueError(
                "Unexpected time_step. "
                f"Expected {self.current_time_step}, "
                f"received {time_step}."
            )

        if not isinstance(
            actions,
            HierarchicalActionBundle,
        ):
            raise TypeError(
                "actions must be HierarchicalActionBundle."
            )

        if self.config.strict_validation:

            actions.validate(
                expected_local_agents=(
                    self.config
                    .number_of_microgrids
                )
            )

        previous_raw_state = (
            self.current_raw_state
        )

        physical_action = (
            self.action_converter(
                actions
            )
        )

        raw_result = (
            self.environment_step_function(
                physical_action
            )
        )

        next_observation = (
            self.observation_builder(
                raw_result
            )
        )

        next_observation = (
            self._validate_observation(
                next_observation
            )
        )

        reward_output = (
            self.reward_builder(
                previous_raw_state,
                raw_result,
                actions,
                physical_action,
            )
        )

        if not isinstance(
            reward_output,
            RewardOutput,
        ):
            raise TypeError(
                "reward_builder must return RewardOutput."
            )

        reward_output.validate(
            number_of_microgrids=(
                self.config
                .number_of_microgrids
            )
        )

        done = self._extract_done(
            raw_result=raw_result,
            time_step=time_step,
        )

        info = self._extract_info(
            raw_result
        )

        info.update(
            {
                "episode":
                    episode,

                "time_step":
                    time_step,

                "adapter_total_steps":
                    self.total_steps + 1,
            }
        )

        result = EnvironmentStepResult(
            next_observation=(
                next_observation
            ),

            local_rewards=[
                float(value)
                for value in (
                    reward_output
                    .local_rewards
                )
            ],

            coordinator_reward=float(
                reward_output
                .coordinator_reward
            ),

            done=done,

            info=info,
        )

        if self.config.strict_validation:

            result.validate(
                expected_local_agents=(
                    self.config
                    .number_of_microgrids
                )
            )

        self.current_raw_state = (
            raw_result
        )

        self.current_observation = (
            next_observation
        )

        self.current_time_step += 1

        self.total_steps += 1

        self.last_record = (
            AdapterStepRecord(
                episode=episode,

                time_step=time_step,

                physical_action=(
                    physical_action
                ),

                raw_environment_result=(
                    raw_result
                ),

                reward_output=(
                    reward_output
                ),

                done=done,
            )
        )

        return result

    # ========================================================
    # TRAINING-LOOP CALLBACKS
    # ========================================================

    @property
    def reset_function(
        self,
    ) -> EnvironmentResetFunction:
        """
        Callback passed directly to FCHMARLTrainingLoop.
        """

        return self.reset

    @property
    def step_function(
        self,
    ):
        """
        Callback passed directly to FCHMARLTrainingLoop.
        """

        return self.step

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(
        self,
    ) -> Dict[str, Any]:

        return {
            "number_of_microgrids":
                self.config
                .number_of_microgrids,

            "strict_validation":
                self.config
                .strict_validation,

            "episode_horizon":
                self.config
                .terminate_after_steps,

            "current_episode":
                self.current_episode,

            "current_time_step":
                self.current_time_step,

            "total_resets":
                self.total_resets,

            "total_steps":
                self.total_steps,

            "has_current_observation":
                (
                    self.current_observation
                    is not None
                ),

            "has_last_record":
                (
                    self.last_record
                    is not None
                ),
        }
