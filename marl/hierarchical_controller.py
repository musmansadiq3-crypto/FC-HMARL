

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class HierarchicalControllerConfig:
    """
    Configuration for hierarchical action coordination.
    """
    number_of_microgrids: int = 5

    deterministic_evaluation: bool = True

    validate_actions: bool = True

    dtype: str = "float32"

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
            self.deterministic_evaluation,
            bool,
        ):
            raise TypeError(
                "deterministic_evaluation must be boolean."
            )

        if not isinstance(
            self.validate_actions,
            bool,
        ):
            raise TypeError(
                "validate_actions must be boolean."
            )

        if self.dtype not in {
            "float32",
            "float64",
        }:
            raise ValueError(
                "dtype must be 'float32' or 'float64'."
            )


# ============================================================
# GENERAL HELPERS
# ============================================================

def _resolve_dtype(
    dtype: str,
):

    if dtype == "float32":
        return np.float32

    if dtype == "float64":
        return np.float64

    raise ValueError(
        "Unsupported dtype."
    )


def validate_state_vector(
    state,
    name: str,
) -> np.ndarray:
    """
    Validate one neural-network state vector.
    """

    state = np.asarray(
        state,
        dtype=np.float64,
    )

    if state.ndim != 1:

        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if state.size == 0:

        raise ValueError(
            f"{name} cannot be empty."
        )

    if not np.isfinite(
        state
    ).all():

        raise ValueError(
            f"{name} contains NaN or Inf."
        )

    return state


def validate_action_vector(
    action,
    name: str,
) -> np.ndarray:
    """
    Validate one agent action vector.
    """

    action = np.asarray(
        action,
        dtype=np.float64,
    )

    if action.ndim != 1:

        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if action.size == 0:

        raise ValueError(
            f"{name} cannot be empty."
        )

    if not np.isfinite(
        action
    ).all():

        raise ValueError(
            f"{name} contains NaN or Inf."
        )

    return action


def validate_agent_interface(
    agent,
    name: str,
) -> None:
    """
    Ensure that an agent implements select_action().
    """

    if not hasattr(
        agent,
        "select_action",
    ):

        raise TypeError(
            f"{name} must implement select_action()."
        )

    if not callable(
        agent.select_action
    ):

        raise TypeError(
            f"{name}.select_action must be callable."
        )


# ============================================================
# HIERARCHICAL ACTION RESULT
# ============================================================

@dataclass
class HierarchicalActionResult:
    local_actions: List[
        np.ndarray
    ]

    coordinator_action: np.ndarray

    flat_action: np.ndarray

    deterministic: bool

    def validate(
        self,
        number_of_microgrids: Optional[
            int
        ] = None,
    ) -> None:

        if not isinstance(
            self.local_actions,
            list,
        ):

            raise TypeError(
                "local_actions must be a list."
            )

        if len(
            self.local_actions
        ) == 0:

            raise ValueError(
                "local_actions cannot be empty."
            )

        if (
            number_of_microgrids
            is not None
            and len(
                self.local_actions
            )
            != number_of_microgrids
        ):

            raise ValueError(
                "local action count does not match "
                "number_of_microgrids."
            )

        for index, action in enumerate(
            self.local_actions
        ):

            validate_action_vector(
                action,
                f"local_actions[{index}]",
            )

        validate_action_vector(
            self.coordinator_action,
            "coordinator_action",
        )

        validate_action_vector(
            self.flat_action,
            "flat_action",
        )

        expected_size = (
            sum(
                action.size
                for action in self.local_actions
            )
            + self.coordinator_action.size
        )

        if self.flat_action.size != expected_size:

            raise ValueError(
                "flat_action dimension does not match "
                "local and coordinator actions."
            )

    @property
    def number_of_local_agents(
        self,
    ) -> int:

        return len(
            self.local_actions
        )

    @property
    def total_action_dimension(
        self,
    ) -> int:

        return int(
            self.flat_action.size
        )

    def summary(
        self,
    ) -> Dict[str, object]:

        return {
            "number_of_local_agents":
                self.number_of_local_agents,

            "local_action_dimensions":
                [
                    int(
                        action.size
                    )
                    for action
                    in self.local_actions
                ],

            "coordinator_action_dimension":
                int(
                    self.coordinator_action.size
                ),

            "total_action_dimension":
                self.total_action_dimension,

            "deterministic":
                self.deterministic,
        }


# ============================================================
# HIERARCHICAL STATE INPUT
# ============================================================

@dataclass
class HierarchicalControllerState:
    """
    State information supplied to the controller.
    """

    local_states: Sequence[
        np.ndarray
    ]

    coordinator_state: np.ndarray

    def validate(
        self,
        number_of_microgrids: int,
    ) -> None:

        if len(
            self.local_states
        ) != number_of_microgrids:

            raise ValueError(
                "local_states length does not match "
                "number_of_microgrids."
            )

        for index, state in enumerate(
            self.local_states
        ):

            validate_state_vector(
                state,
                f"local_states[{index}]",
            )

        validate_state_vector(
            self.coordinator_state,
            "coordinator_state",
        )


# ============================================================
# ACTION FLATTENING
# ============================================================

def flatten_hierarchical_actions(
    local_actions: Sequence[
        np.ndarray
    ],
    coordinator_action,
    dtype: str = "float32",
) -> np.ndarray:
    """
    Concatenate all local and coordinator actions.

    Manuscript concept:

        A(t) = [
            a_1(t),
            a_2(t),
            ...,
            a_N(t),
            a_VPP(t)
        ]

    Reconstruction implementation:
        all actions are flattened into one software vector.
    """

    if len(
        local_actions
    ) == 0:

        raise ValueError(
            "local_actions cannot be empty."
        )

    validated = []

    for index, action in enumerate(
        local_actions
    ):

        action = validate_action_vector(
            action,
            f"local_actions[{index}]",
        )

        validated.append(
            action.reshape(
                -1
            )
        )

    coordinator_action = (
        validate_action_vector(
            coordinator_action,
            "coordinator_action",
        )
    )

    flat = np.concatenate(
        validated
        + [
            coordinator_action.reshape(
                -1
            )
        ]
    )

    return flat.astype(
        _resolve_dtype(
            dtype
        ),
        copy=False,
    )


# ============================================================
# SINGLE AGENT ACTION CALL
# ============================================================

def request_agent_action(
    agent,
    state,
    deterministic: bool = False,
):
    """
    Request an action from one agent.

    Preferred interface:

        agent.select_action(
            state,
            deterministic=...
        )

    For compatibility with simpler agent implementations,
    the function also supports:

        agent.select_action(state)

    when the deterministic keyword is not accepted.
    """

    validate_agent_interface(
        agent,
        "agent",
    )

    state = validate_state_vector(
        state,
        "state",
    )

    try:

        action = agent.select_action(
            state,
            deterministic=deterministic,
        )

    except TypeError:

        action = agent.select_action(
            state
        )

    return validate_action_vector(
        action,
        "agent_action",
    )


# ============================================================
# MAIN CONTROLLER
# ============================================================

class HierarchicalController:
    """
    Coordinates local agents and the VPP coordinator.
    """

    def __init__(
        self,
        local_agents: Sequence,
        coordinator_agent,
        config: Optional[
            HierarchicalControllerConfig
        ] = None,
    ) -> None:

        if config is None:

            config = (
                HierarchicalControllerConfig()
            )

        config.validate()

        if len(
            local_agents
        ) != config.number_of_microgrids:

            raise ValueError(
                "Number of local agents does not match "
                "configured number_of_microgrids. "
                f"Expected {config.number_of_microgrids}, "
                f"received {len(local_agents)}."
            )

        for index, agent in enumerate(
            local_agents
        ):

            validate_agent_interface(
                agent,
                f"local_agents[{index}]",
            )

        validate_agent_interface(
            coordinator_agent,
            "coordinator_agent",
        )

        self.local_agents = list(
            local_agents
        )

        self.coordinator_agent = (
            coordinator_agent
        )

        self.config = config

    # ========================================================
    # MODE RESOLUTION
    # ========================================================

    def resolve_deterministic_mode(
        self,
        training: bool,
        deterministic: Optional[
            bool
        ] = None,
    ) -> bool:
        if not isinstance(
            training,
            bool,
        ):
            raise TypeError(
                "training must be boolean."
            )
        if deterministic is not None:

            if not isinstance(
                deterministic,
                bool,
            ):
                raise TypeError(
                    "deterministic must be boolean or None."
                )
            return deterministic

        if training:

            return False

        return (
            self.config
            .deterministic_evaluation
        )

    # ========================================================
    # LOCAL ACTIONS
    # ========================================================

    def select_local_actions(
        self,
        local_states: Sequence[
            np.ndarray
        ],
        deterministic: bool = False,
    ) -> List[np.ndarray]:
        """
        Request one action from every local microgrid agent.
        """

        if len(
            local_states
        ) != self.config.number_of_microgrids:

            raise ValueError(
                "local_states length does not match "
                "number_of_microgrids."
            )

        actions = []

        for index, (
            agent,
            state,
        ) in enumerate(
            zip(
                self.local_agents,
                local_states,
            )
        ):

            state = validate_state_vector(
                state,
                f"local_states[{index}]",
            )

            action = request_agent_action(
                agent=agent,
                state=state,
                deterministic=(
                    deterministic
                ),
            )

            actions.append(
                action.astype(
                    _resolve_dtype(
                        self.config.dtype
                    ),
                    copy=False,
                )
            )

        return actions

    # ========================================================
    # COORDINATOR ACTION
    # ========================================================

    def select_coordinator_action(
        self,
        coordinator_state,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Request the upper-level VPP coordinator action.
        """

        state = validate_state_vector(
            coordinator_state,
            "coordinator_state",
        )

        action = request_agent_action(
            agent=self.coordinator_agent,
            state=state,
            deterministic=(
                deterministic
            ),
        )

        return action.astype(
            _resolve_dtype(
                self.config.dtype
            ),
            copy=False,
        )

    # ========================================================
    # FULL HIERARCHICAL ACTION
    # ========================================================

    def select_actions(
        self,
        local_states: Sequence[
            np.ndarray
        ],
        coordinator_state,
        training: bool = True,
        deterministic: Optional[
            bool
        ] = None,
    ) -> HierarchicalActionResult:
        """
        Select all hierarchical actions for one environment step.
        """

        controller_state = (
            HierarchicalControllerState(
                local_states=(
                    local_states
                ),
                coordinator_state=(
                    coordinator_state
                ),
            )
        )

        controller_state.validate(
            self.config.number_of_microgrids
        )

        deterministic_mode = (
            self.resolve_deterministic_mode(
                training=training,
                deterministic=deterministic,
            )
        )

        local_actions = (
            self.select_local_actions(
                local_states=(
                    controller_state
                    .local_states
                ),
                deterministic=(
                    deterministic_mode
                ),
            )
        )

        coordinator_action = (
            self.select_coordinator_action(
                coordinator_state=(
                    controller_state
                    .coordinator_state
                ),
                deterministic=(
                    deterministic_mode
                ),
            )
        )

        flat_action = (
            flatten_hierarchical_actions(
                local_actions=(
                    local_actions
                ),
                coordinator_action=(
                    coordinator_action
                ),
                dtype=self.config.dtype,
            )
        )

        result = (
            HierarchicalActionResult(
                local_actions=(
                    local_actions
                ),

                coordinator_action=(
                    coordinator_action
                ),

                flat_action=(
                    flat_action
                ),

                deterministic=(
                    deterministic_mode
                ),
            )
        )

        if self.config.validate_actions:

            result.validate(
                number_of_microgrids=(
                    self.config
                    .number_of_microgrids
                )
            )

        return result

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
    ) -> None:
        """
        Reset episode-specific state in all agents if supported.

        Agents without reset() are left unchanged.
        """

        for agent in self.local_agents:

            if hasattr(
                agent,
                "reset",
            ) and callable(
                agent.reset
            ):

                agent.reset()

        if hasattr(
            self.coordinator_agent,
            "reset",
        ) and callable(
            self.coordinator_agent.reset
        ):

            self.coordinator_agent.reset()

    # ========================================================
    # TRAIN / EVAL PROPAGATION
    # ========================================================

    def set_training_mode(
        self,
        training: bool,
    ) -> None:
        """
        Propagate training/evaluation mode when agents support it.

        Supported interfaces:

            agent.train()
            agent.eval()

        or

            agent.set_training_mode(bool)
        """

        if not isinstance(
            training,
            bool,
        ):

            raise TypeError(
                "training must be boolean."
            )

        agents = (
            self.local_agents
            + [
                self.coordinator_agent
            ]
        )

        for agent in agents:

            if hasattr(
                agent,
                "set_training_mode",
            ) and callable(
                agent.set_training_mode
            ):

                agent.set_training_mode(
                    training
                )

                continue

            if training:

                if hasattr(
                    agent,
                    "train",
                ) and callable(
                    agent.train
                ):

                    agent.train()

            else:

                if hasattr(
                    agent,
                    "eval",
                ) and callable(
                    agent.eval
                ):

                    agent.eval()

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(
        self,
    ) -> Dict[str, object]:

        return {
            "number_of_microgrids":
                self.config
                .number_of_microgrids,

            "number_of_local_agents":
                len(
                    self.local_agents
                ),

            "has_coordinator":
                self.coordinator_agent
                is not None,

            "deterministic_evaluation":
                self.config
                .deterministic_evaluation,

            "validate_actions":
                self.config
                .validate_actions,

            "dtype":
                self.config.dtype,
        }
