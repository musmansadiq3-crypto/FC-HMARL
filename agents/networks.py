"""
Neural-network components for the FC-HMARL SAC agents.

This module provides:

    1. Gaussian stochastic policy / actor network
    2. Twin Q critic network
    3. Deterministic action evaluation
    4. Reparameterized stochastic sampling
    5. Tanh action squashing
    6. Action scaling to physical action bounds
    7. Parameter initialization utilities

Manuscript-supported concept
----------------------------
The FC-HMARL framework uses Soft Actor-Critic (SAC) as the
learning backbone for:

    local microgrid agents
    VPP coordinator agent

The manuscript also describes actor-network updates, critic-network
updates, and soft target-network updates.

Reconstruction choices
----------------------
The manuscript does not specify the exact neural-network layer sizes
or complete implementation architecture.

Therefore, the following defaults are implementation choices:

    hidden dimensions = (256, 256)
    activation        = ReLU
    log_std range     = [-20, 2]

These are standard SAC-compatible choices and remain configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class SACNetworkConfig:
    """
    Configuration for actor and critic networks.
    """

    state_dimension: int

    action_dimension: int

    hidden_dimensions: Tuple[int, ...] = (
        256,
        256,
    )

    activation: str = "relu"

    log_std_min: float = -20.0

    log_std_max: float = 2.0

    action_low: float | Sequence[float] = -1.0

    action_high: float | Sequence[float] = 1.0

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

        for value in self.hidden_dimensions:

            if not isinstance(
                value,
                int,
            ):
                raise TypeError(
                    "Every hidden dimension must be an integer."
                )

            if value <= 0:
                raise ValueError(
                    "Hidden dimensions must be positive."
                )

        if self.activation.lower() not in {
            "relu",
            "gelu",
            "elu",
            "tanh",
        }:
            raise ValueError(
                "Unsupported activation."
            )

        if not np.isfinite(
            self.log_std_min
        ):
            raise ValueError(
                "log_std_min must be finite."
            )

        if not np.isfinite(
            self.log_std_max
        ):
            raise ValueError(
                "log_std_max must be finite."
            )

        if (
            self.log_std_min
            >= self.log_std_max
        ):
            raise ValueError(
                "log_std_min must be smaller than log_std_max."
            )

        low = _expand_action_bound(
            self.action_low,
            self.action_dimension,
            "action_low",
        )

        high = _expand_action_bound(
            self.action_high,
            self.action_dimension,
            "action_high",
        )

        if np.any(
            low >= high
        ):
            raise ValueError(
                "Every action_low value must be smaller "
                "than the corresponding action_high value."
            )


# ============================================================
# HELPERS
# ============================================================

def _expand_action_bound(
    bound,
    action_dimension: int,
    name: str,
) -> np.ndarray:
    """
    Convert scalar/vector action bound into a 1-D array.
    """

    array = np.asarray(
        bound,
        dtype=np.float32,
    )

    if array.ndim == 0:

        array = np.full(
            (
                action_dimension,
            ),
            float(array),
            dtype=np.float32,
        )

    elif array.ndim == 1:

        if array.size != action_dimension:

            raise ValueError(
                f"{name} length must equal action_dimension."
            )

    else:

        raise ValueError(
            f"{name} must be a scalar or one-dimensional vector."
        )

    if not np.isfinite(
        array
    ).all():

        raise ValueError(
            f"{name} contains NaN or Inf."
        )

    return array


def build_activation(
    name: str,
) -> nn.Module:
    """
    Construct activation module.
    """

    name = name.lower()

    if name == "relu":
        return nn.ReLU()

    if name == "gelu":
        return nn.GELU()

    if name == "elu":
        return nn.ELU()

    if name == "tanh":
        return nn.Tanh()

    raise ValueError(
        f"Unsupported activation: {name}"
    )


def initialize_linear_layer(
    layer: nn.Linear,
    gain: float = np.sqrt(2.0),
) -> None:
    """
    Xavier initialization for Linear layers.
    """

    nn.init.xavier_uniform_(
        layer.weight,
        gain=float(
            gain
        ),
    )

    nn.init.zeros_(
        layer.bias
    )


def build_mlp(
    input_dimension: int,
    hidden_dimensions: Sequence[int],
    activation: str,
) -> tuple[
    nn.Sequential,
    int,
]:
    """
    Build shared feed-forward feature extractor.

    Returns
    -------
    network
        Sequential MLP.

    output_dimension
        Width of final hidden layer.
    """

    if input_dimension <= 0:

        raise ValueError(
            "input_dimension must be positive."
        )

    if len(
        hidden_dimensions
    ) == 0:

        raise ValueError(
            "hidden_dimensions cannot be empty."
        )

    layers = []

    previous_dimension = (
        input_dimension
    )

    for hidden_dimension in (
        hidden_dimensions
    ):

        if hidden_dimension <= 0:

            raise ValueError(
                "Hidden dimensions must be positive."
            )

        linear = nn.Linear(
            previous_dimension,
            hidden_dimension,
        )

        initialize_linear_layer(
            linear
        )

        layers.append(
            linear
        )

        layers.append(
            build_activation(
                activation
            )
        )

        previous_dimension = (
            hidden_dimension
        )

    return (
        nn.Sequential(
            *layers
        ),
        previous_dimension,
    )


def validate_state_tensor(
    state: torch.Tensor,
    expected_dimension: int,
) -> None:
    """
    Validate actor/critic state tensor.
    """

    if not isinstance(
        state,
        torch.Tensor,
    ):
        raise TypeError(
            "state must be a torch.Tensor."
        )

    if state.ndim != 2:

        raise ValueError(
            "state must have shape "
            "(batch, state_dimension)."
        )

    if state.shape[
        1
    ] != expected_dimension:

        raise ValueError(
            "State dimension mismatch."
        )

    if not torch.isfinite(
        state
    ).all():

        raise ValueError(
            "state contains NaN or Inf."
        )


def validate_action_tensor(
    action: torch.Tensor,
    expected_dimension: int,
) -> None:
    """
    Validate critic action tensor.
    """

    if not isinstance(
        action,
        torch.Tensor,
    ):
        raise TypeError(
            "action must be a torch.Tensor."
        )

    if action.ndim != 2:

        raise ValueError(
            "action must have shape "
            "(batch, action_dimension)."
        )

    if action.shape[
        1
    ] != expected_dimension:

        raise ValueError(
            "Action dimension mismatch."
        )

    if not torch.isfinite(
        action
    ).all():

        raise ValueError(
            "action contains NaN or Inf."
        )


# ============================================================
# GAUSSIAN ACTOR
# ============================================================

class GaussianPolicyNetwork(
    nn.Module
):
    """
    SAC stochastic Gaussian actor.

    The network predicts:

        mean
        log_standard_deviation

    A reparameterized Gaussian sample is passed through tanh
    and then mapped to the configured physical action bounds.
    """

    def __init__(
        self,
        config: SACNetworkConfig,
    ) -> None:

        super().__init__()

        config.validate()

        self.config = config

        (
            self.feature_network,
            feature_dimension,
        ) = build_mlp(
            input_dimension=(
                config.state_dimension
            ),
            hidden_dimensions=(
                config.hidden_dimensions
            ),
            activation=(
                config.activation
            ),
        )

        self.mean_layer = nn.Linear(
            feature_dimension,
            config.action_dimension,
        )

        self.log_std_layer = nn.Linear(
            feature_dimension,
            config.action_dimension,
        )

        initialize_linear_layer(
            self.mean_layer,
            gain=0.01,
        )

        initialize_linear_layer(
            self.log_std_layer,
            gain=0.01,
        )

        action_low = (
            _expand_action_bound(
                config.action_low,
                config.action_dimension,
                "action_low",
            )
        )

        action_high = (
            _expand_action_bound(
                config.action_high,
                config.action_dimension,
                "action_high",
            )
        )

        action_scale = (
            action_high
            - action_low
        ) / 2.0

        action_bias = (
            action_high
            + action_low
        ) / 2.0

        self.register_buffer(
            "action_low",
            torch.tensor(
                action_low,
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "action_high",
            torch.tensor(
                action_high,
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "action_scale",
            torch.tensor(
                action_scale,
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "action_bias",
            torch.tensor(
                action_bias,
                dtype=torch.float32,
            ),
        )

    # ========================================================
    # DISTRIBUTION PARAMETERS
    # ========================================================

    def forward(
        self,
        state: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        validate_state_tensor(
            state,
            self.config.state_dimension,
        )

        features = self.feature_network(
            state
        )

        mean = self.mean_layer(
            features
        )

        log_std = (
            self.log_std_layer(
                features
            )
        )

        log_std = torch.clamp(
            log_std,
            min=self.config.log_std_min,
            max=self.config.log_std_max,
        )

        return (
            mean,
            log_std,
        )

    # ========================================================
    # ACTION SCALING
    # ========================================================

    def scale_action(
        self,
        normalized_action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Map tanh-normalized action [-1, 1] to physical bounds.
        """

        return (
            normalized_action
            * self.action_scale
            + self.action_bias
        )

    # ========================================================
    # SAMPLING
    # ========================================================

    def sample(
        self,
        state: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Reparameterized SAC policy sample.

        Returns
        -------
        action
            Squashed/scaled stochastic action.

        log_probability
            Corrected log probability after tanh transform.

        deterministic_action
            Squashed/scaled mean action.
        """

        if epsilon <= 0:

            raise ValueError(
                "epsilon must be positive."
            )

        mean, log_std = self.forward(
            state
        )

        std = torch.exp(
            log_std
        )

        distribution = Normal(
            mean,
            std,
        )

        pre_tanh = (
            distribution.rsample()
        )

        normalized_action = (
            torch.tanh(
                pre_tanh
            )
        )

        action = self.scale_action(
            normalized_action
        )

        log_probability = (
            distribution.log_prob(
                pre_tanh
            )
        )

        correction = torch.log(
            self.action_scale
            * (
                1.0
                - normalized_action.pow(
                    2
                )
            )
            + epsilon
        )

        log_probability = (
            log_probability
            - correction
        )

        log_probability = (
            log_probability.sum(
                dim=-1,
                keepdim=True,
            )
        )

        deterministic_action = (
            self.scale_action(
                torch.tanh(
                    mean
                )
            )
        )

        return (
            action,
            log_probability,
            deterministic_action,
        )

    # ========================================================
    # DETERMINISTIC
    # ========================================================

    def deterministic(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return deterministic mean-policy action.
        """

        mean, _ = self.forward(
            state
        )

        normalized = torch.tanh(
            mean
        )

        return self.scale_action(
            normalized
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    def parameter_count(
        self,
    ) -> int:

        return sum(
            parameter.numel()
            for parameter
            in self.parameters()
        )

    def summary(
        self,
    ) -> Dict[str, object]:

        return {
            "network":
                "GaussianPolicyNetwork",

            "state_dimension":
                self.config.state_dimension,

            "action_dimension":
                self.config.action_dimension,

            "hidden_dimensions":
                self.config.hidden_dimensions,

            "activation":
                self.config.activation,

            "parameter_count":
                self.parameter_count(),
        }


# ============================================================
# SINGLE Q NETWORK
# ============================================================

class QNetwork(
    nn.Module
):
    """
    State-action Q-value network.

    Input:

        [state, action]

    Output:

        Q(s, a)
    """

    def __init__(
        self,
        config: SACNetworkConfig,
    ) -> None:

        super().__init__()

        config.validate()

        self.config = config

        input_dimension = (
            config.state_dimension
            + config.action_dimension
        )

        (
            self.feature_network,
            feature_dimension,
        ) = build_mlp(
            input_dimension=(
                input_dimension
            ),
            hidden_dimensions=(
                config.hidden_dimensions
            ),
            activation=(
                config.activation
            ),
        )

        self.output_layer = nn.Linear(
            feature_dimension,
            1,
        )

        initialize_linear_layer(
            self.output_layer,
            gain=1.0,
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:

        validate_state_tensor(
            state,
            self.config.state_dimension,
        )

        validate_action_tensor(
            action,
            self.config.action_dimension,
        )

        if (
            state.shape[
                0
            ]
            != action.shape[
                0
            ]
        ):

            raise ValueError(
                "state and action batch sizes must match."
            )

        state_action = torch.cat(
            [
                state,
                action,
            ],
            dim=-1,
        )

        features = self.feature_network(
            state_action
        )

        q_value = self.output_layer(
            features
        )

        return q_value

    def parameter_count(
        self,
    ) -> int:

        return sum(
            parameter.numel()
            for parameter
            in self.parameters()
        )


# ============================================================
# TWIN Q CRITIC
# ============================================================

class TwinQNetwork(
    nn.Module
):
    """
    Twin critic required by SAC.

    Returns:

        Q1(s,a)
        Q2(s,a)

    Using two critics reduces positive value-estimation bias.
    """

    def __init__(
        self,
        config: SACNetworkConfig,
    ) -> None:

        super().__init__()

        config.validate()

        self.config = config

        self.q1 = QNetwork(
            config
        )

        self.q2 = QNetwork(
            config
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        q1 = self.q1(
            state,
            action,
        )

        q2 = self.q2(
            state,
            action,
        )

        return (
            q1,
            q2,
        )

    def minimum(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return min(Q1, Q2), as used by SAC targets/policy updates.
        """

        q1, q2 = self.forward(
            state,
            action,
        )

        return torch.minimum(
            q1,
            q2,
        )

    def parameter_count(
        self,
    ) -> int:

        return sum(
            parameter.numel()
            for parameter
            in self.parameters()
        )

    def summary(
        self,
    ) -> Dict[str, object]:

        return {
            "network":
                "TwinQNetwork",

            "state_dimension":
                self.config.state_dimension,

            "action_dimension":
                self.config.action_dimension,

            "hidden_dimensions":
                self.config.hidden_dimensions,

            "activation":
                self.config.activation,

            "parameter_count":
                self.parameter_count(),
        }


# ============================================================
# NETWORK FACTORY
# ============================================================

def build_sac_networks(
    config: SACNetworkConfig,
) -> tuple[
    GaussianPolicyNetwork,
    TwinQNetwork,
    TwinQNetwork,
]:
    """
    Build standard SAC neural networks.

    Returns
    -------
    actor

    critic

    target_critic

    Target critic initially receives an exact copy of critic weights.
    """

    config.validate()

    actor = GaussianPolicyNetwork(
        config
    )

    critic = TwinQNetwork(
        config
    )

    target_critic = TwinQNetwork(
        config
    )

    target_critic.load_state_dict(
        critic.state_dict()
    )

    for parameter in (
        target_critic.parameters()
    ):

        parameter.requires_grad = False

    return (
        actor,
        critic,
        target_critic,
    )