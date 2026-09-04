"""
State construction utilities for the FC-HMARL framework.

This module connects:

    physical VPP observations
    forecasting outputs
    forecast confidence information

to the state vectors consumed by the hierarchical MARL agents.

Manuscript-supported local state
--------------------------------

For microgrid i:

    s_i(t) = [
        SOC_i(t),
        P_i^PV(t),
        P_i^Load(t),
        P_i^EV(t),
        P_i^grid(t),
        S^pred(t)
    ]^T

Manuscript-supported coordinator state
--------------------------------------

    S_VPP(t) = [
        P^VPP(t),
        E^share(t),
        lambda(t),
        S^pred(t)
    ]^T

The global state contains all local states plus the coordinator state.

Reconstruction choice
---------------------
S^pred may contain multi-horizon information.

For neural-network input, this implementation flattens the predictive
state in chronological order.

For the default forecasting arrangement:

    24 forecast hours
    4 forecast variables

the predictive vector contains:

    24 * 4 = 96 elements.

Therefore:

    local state dimension       = 5 + 96 = 101
    coordinator state dimension = 3 + 96 = 99

for the default configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class StateBuilderConfig:
    """
    Configuration for FC-HMARL state construction.
    """

    number_of_microgrids: int = 5

    forecast_horizon: int = 24

    forecast_features: int = 4

    flatten_predictive_state: bool = True

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
            self.forecast_horizon,
            int,
        ):
            raise TypeError(
                "forecast_horizon must be an integer."
            )

        if self.forecast_horizon <= 0:
            raise ValueError(
                "forecast_horizon must be positive."
            )

        if not isinstance(
            self.forecast_features,
            int,
        ):
            raise TypeError(
                "forecast_features must be an integer."
            )

        if self.forecast_features <= 0:
            raise ValueError(
                "forecast_features must be positive."
            )

        supported_dtypes = {
            "float32",
            "float64",
        }

        if self.dtype not in supported_dtypes:
            raise ValueError(
                "dtype must be 'float32' or 'float64'."
            )

    @property
    def predictive_state_dimension(
        self,
    ) -> int:

        return (
            self.forecast_horizon
            * self.forecast_features
        )

    @property
    def local_state_dimension(
        self,
    ) -> int:

        return (
            5
            + self.predictive_state_dimension
        )

    @property
    def coordinator_state_dimension(
        self,
    ) -> int:

        return (
            3
            + self.predictive_state_dimension
        )

    @property
    def global_state_dimension(
        self,
    ) -> int:

        return (
            self.number_of_microgrids
            * self.local_state_dimension
            + self.coordinator_state_dimension
        )


# ============================================================
# GENERAL VALIDATION
# ============================================================

def _validate_scalar(
    value,
    name: str,
) -> float:
    """
    Validate one scalar physical state value.
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


# ============================================================
# PREDICTIVE STATE
# ============================================================

def validate_predictive_state(
    predictive_state,
    forecast_horizon: Optional[int] = None,
    forecast_features: Optional[int] = None,
) -> np.ndarray:
    """
    Validate confidence-aware predictive information.

    Accepted forms
    --------------

    Matrix:
        (forecast_horizon, forecast_features)

    or already flattened vector:
        (forecast_horizon * forecast_features,)

    The matrix form is preferred.
    """

    predictive_state = np.asarray(
        predictive_state,
        dtype=np.float64,
    )

    if predictive_state.size == 0:

        raise ValueError(
            "predictive_state cannot be empty."
        )

    if not np.isfinite(
        predictive_state
    ).all():

        raise ValueError(
            "predictive_state contains NaN or Inf."
        )

    if predictive_state.ndim not in {
        1,
        2,
    }:

        raise ValueError(
            "predictive_state must be a "
            "1-D vector or 2-D matrix."
        )

    if (
        forecast_horizon is not None
        and forecast_features is not None
    ):

        expected_size = (
            forecast_horizon
            * forecast_features
        )

        if predictive_state.size != expected_size:

            raise ValueError(
                "predictive_state size does not match "
                "forecast_horizon * forecast_features. "
                f"Expected {expected_size}, "
                f"received {predictive_state.size}."
            )

        if predictive_state.ndim == 2:

            expected_shape = (
                forecast_horizon,
                forecast_features,
            )

            if predictive_state.shape != expected_shape:

                raise ValueError(
                    "predictive_state matrix shape does "
                    f"not match {expected_shape}."
                )

    return predictive_state


def flatten_predictive_state(
    predictive_state,
    forecast_horizon: Optional[int] = None,
    forecast_features: Optional[int] = None,
    dtype: str = "float32",
) -> np.ndarray:
    """
    Flatten predictive information for neural-network input.

    Chronological ordering is preserved:

        hour 1: all features
        hour 2: all features
        ...
        hour H: all features
    """

    predictive_state = (
        validate_predictive_state(
            predictive_state,
            forecast_horizon=(
                forecast_horizon
            ),
            forecast_features=(
                forecast_features
            ),
        )
    )

    return predictive_state.reshape(
        -1
    ).astype(
        _resolve_dtype(
            dtype
        ),
        copy=False,
    )


# ============================================================
# LOCAL MICROGRID STATE
# ============================================================

def build_local_state(
    soc,
    pv_power,
    load_power,
    ev_power,
    grid_exchange,
    predictive_state,
    config: Optional[
        StateBuilderConfig
    ] = None,
) -> np.ndarray:
    """
    Construct one local microgrid state.

    Manuscript form:

        s_i(t) = [
            SOC_i,
            P_i^PV,
            P_i^Load,
            P_i^EV,
            P_i^grid,
            S^pred
        ]^T
    """

    if config is None:

        config = (
            StateBuilderConfig()
        )

    config.validate()

    scalar_state = np.array(
        [
            _validate_scalar(
                soc,
                "soc",
            ),

            _validate_scalar(
                pv_power,
                "pv_power",
            ),

            _validate_scalar(
                load_power,
                "load_power",
            ),

            _validate_scalar(
                ev_power,
                "ev_power",
            ),

            _validate_scalar(
                grid_exchange,
                "grid_exchange",
            ),
        ],
        dtype=_resolve_dtype(
            config.dtype
        ),
    )

    predictive_vector = (
        flatten_predictive_state(
            predictive_state,
            forecast_horizon=(
                config.forecast_horizon
            ),
            forecast_features=(
                config.forecast_features
            ),
            dtype=config.dtype,
        )
    )

    state = np.concatenate(
        [
            scalar_state,
            predictive_vector,
        ]
    )

    expected_dimension = (
        config.local_state_dimension
    )

    if state.shape != (
        expected_dimension,
    ):

        raise RuntimeError(
            "Unexpected local state dimension. "
            f"Expected {expected_dimension}, "
            f"received {state.shape[0]}."
        )

    return state


# ============================================================
# COORDINATOR STATE
# ============================================================

def build_coordinator_state(
    vpp_power,
    energy_sharing,
    market_price,
    predictive_state,
    config: Optional[
        StateBuilderConfig
    ] = None,
) -> np.ndarray:
    """
    Construct the upper-level VPP coordinator state.

    Manuscript form:

        S_VPP(t) = [
            P^VPP(t),
            E^share(t),
            lambda(t),
            S^pred(t)
        ]^T
    """

    if config is None:

        config = (
            StateBuilderConfig()
        )

    config.validate()

    aggregate_state = np.array(
        [
            _validate_scalar(
                vpp_power,
                "vpp_power",
            ),

            _validate_scalar(
                energy_sharing,
                "energy_sharing",
            ),

            _validate_scalar(
                market_price,
                "market_price",
            ),
        ],
        dtype=_resolve_dtype(
            config.dtype
        ),
    )

    predictive_vector = (
        flatten_predictive_state(
            predictive_state,
            forecast_horizon=(
                config.forecast_horizon
            ),
            forecast_features=(
                config.forecast_features
            ),
            dtype=config.dtype,
        )
    )

    state = np.concatenate(
        [
            aggregate_state,
            predictive_vector,
        ]
    )

    expected_dimension = (
        config.coordinator_state_dimension
    )

    if state.shape != (
        expected_dimension,
    ):

        raise RuntimeError(
            "Unexpected coordinator state dimension. "
            f"Expected {expected_dimension}, "
            f"received {state.shape[0]}."
        )

    return state


# ============================================================
# GLOBAL HIERARCHICAL STATE
# ============================================================

def build_global_state(
    local_states: Sequence[
        np.ndarray
    ],
    coordinator_state: np.ndarray,
    config: Optional[
        StateBuilderConfig
    ] = None,
) -> np.ndarray:
    """
    Construct full hierarchical VPP state.

    Manuscript concept:

        S(t) = [
            s_1(t),
            s_2(t),
            ...,
            s_NMG(t),
            S_VPP(t)
        ]

    All state components are flattened into one vector for
    centralized storage/training interfaces.
    """

    if config is None:

        config = (
            StateBuilderConfig()
        )

    config.validate()

    if len(
        local_states
    ) != config.number_of_microgrids:

        raise ValueError(
            "Number of local states does not "
            "match number_of_microgrids. "
            f"Expected {config.number_of_microgrids}, "
            f"received {len(local_states)}."
        )

    validated_local_states = []

    for index, state in enumerate(
        local_states
    ):

        state = np.asarray(
            state,
            dtype=_resolve_dtype(
                config.dtype
            ),
        )

        if state.ndim != 1:

            raise ValueError(
                f"local_states[{index}] must "
                "be one-dimensional."
            )

        if state.shape[
            0
        ] != config.local_state_dimension:

            raise ValueError(
                f"local_states[{index}] has "
                "incorrect dimension."
            )

        if not np.isfinite(
            state
        ).all():

            raise ValueError(
                f"local_states[{index}] contains "
                "NaN or Inf."
            )

        validated_local_states.append(
            state
        )

    coordinator_state = np.asarray(
        coordinator_state,
        dtype=_resolve_dtype(
            config.dtype
        ),
    )

    if coordinator_state.ndim != 1:

        raise ValueError(
            "coordinator_state must be "
            "one-dimensional."
        )

    if (
        coordinator_state.shape[
            0
        ]
        != config.coordinator_state_dimension
    ):

        raise ValueError(
            "coordinator_state has incorrect "
            "dimension."
        )

    if not np.isfinite(
        coordinator_state
    ).all():

        raise ValueError(
            "coordinator_state contains "
            "NaN or Inf."
        )

    global_state = np.concatenate(
        validated_local_states
        + [
            coordinator_state
        ]
    )

    if (
        global_state.shape[
            0
        ]
        != config.global_state_dimension
    ):

        raise RuntimeError(
            "Unexpected global state dimension."
        )

    return global_state


# ============================================================
# AGGREGATE VPP POWER
# ============================================================

def calculate_vpp_power(
    grid_exchanges,
) -> float:
    """
    Calculate aggregate VPP grid power.

    Reconstruction implementation:
        aggregate power is the sum of local grid exchanges.

    Sign convention follows the environment:
        positive/negative meaning is inherited from each
        microgrid's grid-exchange convention.
    """

    grid_exchanges = np.asarray(
        grid_exchanges,
        dtype=np.float64,
    )

    if grid_exchanges.ndim != 1:

        raise ValueError(
            "grid_exchanges must be one-dimensional."
        )

    if grid_exchanges.size == 0:

        raise ValueError(
            "grid_exchanges cannot be empty."
        )

    if not np.isfinite(
        grid_exchanges
    ).all():

        raise ValueError(
            "grid_exchanges contain NaN or Inf."
        )

    return float(
        np.sum(
            grid_exchanges
        )
    )


# ============================================================
# TOTAL ENERGY-SHARING ACTIVITY
# ============================================================

def calculate_total_sharing_activity(
    sharing_matrix,
) -> float:
    """
    Calculate total inter-microgrid sharing activity.

    The manuscript coordinator state contains total
    energy-sharing activity but does not specify the exact
    software reduction operation.

    Reconstruction choice
    ---------------------
    Sum the non-negative directed sharing powers contained
    in the sharing matrix.

    Diagonal self-sharing values are ignored.
    """

    matrix = np.asarray(
        sharing_matrix,
        dtype=np.float64,
    )

    if matrix.ndim != 2:

        raise ValueError(
            "sharing_matrix must be two-dimensional."
        )

    if matrix.shape[
        0
    ] != matrix.shape[
        1
    ]:

        raise ValueError(
            "sharing_matrix must be square."
        )

    if matrix.size == 0:

        raise ValueError(
            "sharing_matrix cannot be empty."
        )

    if not np.isfinite(
        matrix
    ).all():

        raise ValueError(
            "sharing_matrix contains NaN or Inf."
        )

    if np.any(
        matrix < 0
    ):

        raise ValueError(
            "sharing_matrix cannot contain "
            "negative directed sharing powers."
        )

    matrix = matrix.copy()

    np.fill_diagonal(
        matrix,
        0.0,
    )

    return float(
        np.sum(
            matrix
        )
    )


# ============================================================
# COMPLETE STATE RESULT
# ============================================================

@dataclass
class HierarchicalStateResult:
    """
    Complete FC-HMARL state bundle.
    """

    local_states: List[
        np.ndarray
    ]

    coordinator_state: np.ndarray

    global_state: np.ndarray

    predictive_state: np.ndarray

    def validate(
        self,
        config: StateBuilderConfig,
    ) -> None:

        if len(
            self.local_states
        ) != config.number_of_microgrids:

            raise ValueError(
                "Incorrect number of local states."
            )

        for state in self.local_states:

            if state.shape != (
                config.local_state_dimension,
            ):

                raise ValueError(
                    "Invalid local state shape."
                )

        if self.coordinator_state.shape != (
            config.coordinator_state_dimension,
        ):

            raise ValueError(
                "Invalid coordinator state shape."
            )

        if self.global_state.shape != (
            config.global_state_dimension,
        ):

            raise ValueError(
                "Invalid global state shape."
            )

    def summary(
        self,
    ) -> Dict[str, object]:

        return {
            "number_of_local_agents":
                len(
                    self.local_states
                ),

            "local_state_dimension":
                int(
                    self.local_states[
                        0
                    ].shape[
                        0
                    ]
                ),

            "coordinator_state_dimension":
                int(
                    self.coordinator_state.shape[
                        0
                    ]
                ),

            "global_state_dimension":
                int(
                    self.global_state.shape[
                        0
                    ]
                ),

            "predictive_state_dimension":
                int(
                    self.predictive_state.size
                ),
        }


# ============================================================
# MAIN STATE BUILDER
# ============================================================

class HierarchicalStateBuilder:
    """
    Main interface for FC-HMARL state construction.
    """

    def __init__(
        self,
        config: Optional[
            StateBuilderConfig
        ] = None,
    ) -> None:

        if config is None:

            config = (
                StateBuilderConfig()
            )

        config.validate()

        self.config = config

    def build(
        self,
        microgrid_observations:
        Sequence[
            Dict[str, float]
        ],
        predictive_state,
        market_price,
        sharing_matrix,
    ) -> HierarchicalStateResult:
        """
        Build local, coordinator, and global states.

        Each microgrid observation dictionary must contain:

            soc
            pv_power
            load_power
            ev_power
            grid_exchange
        """

        if len(
            microgrid_observations
        ) != self.config.number_of_microgrids:

            raise ValueError(
                "microgrid_observations length "
                "does not match configured "
                "number_of_microgrids."
            )

        predictive_matrix = (
            validate_predictive_state(
                predictive_state,
                forecast_horizon=(
                    self.config.forecast_horizon
                ),
                forecast_features=(
                    self.config.forecast_features
                ),
            )
        )

        predictive_vector = (
            flatten_predictive_state(
                predictive_matrix,
                forecast_horizon=(
                    self.config.forecast_horizon
                ),
                forecast_features=(
                    self.config.forecast_features
                ),
                dtype=self.config.dtype,
            )
        )

        required_keys = {
            "soc",
            "pv_power",
            "load_power",
            "ev_power",
            "grid_exchange",
        }

        local_states = []

        grid_exchanges = []

        for index, observation in enumerate(
            microgrid_observations
        ):

            missing = (
                required_keys
                - set(
                    observation.keys()
                )
            )

            if missing:

                raise KeyError(
                    f"Microgrid {index} is missing "
                    f"required keys: "
                    f"{sorted(missing)}"
                )

            local_state = build_local_state(
                soc=observation[
                    "soc"
                ],
                pv_power=observation[
                    "pv_power"
                ],
                load_power=observation[
                    "load_power"
                ],
                ev_power=observation[
                    "ev_power"
                ],
                grid_exchange=observation[
                    "grid_exchange"
                ],
                predictive_state=(
                    predictive_matrix
                ),
                config=self.config,
            )

            local_states.append(
                local_state
            )

            grid_exchanges.append(
                observation[
                    "grid_exchange"
                ]
            )

        vpp_power = (
            calculate_vpp_power(
                grid_exchanges
            )
        )

        sharing_activity = (
            calculate_total_sharing_activity(
                sharing_matrix
            )
        )

        coordinator_state = (
            build_coordinator_state(
                vpp_power=vpp_power,
                energy_sharing=(
                    sharing_activity
                ),
                market_price=market_price,
                predictive_state=(
                    predictive_matrix
                ),
                config=self.config,
            )
        )

        global_state = build_global_state(
            local_states=local_states,
            coordinator_state=(
                coordinator_state
            ),
            config=self.config,
        )

        result = (
            HierarchicalStateResult(
                local_states=(
                    local_states
                ),
                coordinator_state=(
                    coordinator_state
                ),
                global_state=(
                    global_state
                ),
                predictive_state=(
                    predictive_vector
                ),
            )
        )

        result.validate(
            self.config
        )

        return result