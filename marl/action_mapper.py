from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from environment.vpp_env import VPPEnvironment
from marl.environment_adapter import (
    HierarchicalActionBundle,
)
from marl.vpp_training_bridge import (
    VPPExogenousInput,
    VPPPhysicalAction,
)
# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class ActionMapperConfig:
    use_dynamic_bess_feasibility: bool = True

    coordinator_controls_grid: bool = False

    coordinator_controls_reserve: bool = True

    coordinator_controls_sharing: bool = True

    reserve_fraction_of_bess_rating: float = 1.0

    reserve_duration_hours: float = 1.0

    action_tolerance: float = 1e-8

    def validate(
        self,
    ) -> None:

        if (
            self.reserve_fraction_of_bess_rating
            < 0.0
        ):
            raise ValueError(
                "reserve_fraction_of_bess_rating "
                "cannot be negative."
            )

        if (
            not np.isfinite(
                self.reserve_duration_hours
            )
            or self.reserve_duration_hours <= 0.0
        ):
            raise ValueError(
                "reserve_duration_hours must be finite "
                "and greater than zero."
            )

        if self.action_tolerance < 0.0:
            raise ValueError(
                "action_tolerance cannot be negative."
            )
# ============================================================
# BASIC HELPERS
# ============================================================

def _to_action_vector(
    action,
    name: str,
) -> np.ndarray:

    array = np.asarray(
        action,
        dtype=np.float64,
    )

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if array.size == 0:
        raise ValueError(
            f"{name} cannot be empty."
        )

    if not np.isfinite(
        array
    ).all():
        raise ValueError(
            f"{name} contains NaN or Inf."
        )

    return array


def _clip_normalized_action(
    value: float,
    tolerance: float,
) -> float:

    value = float(
        value
    )

    if not np.isfinite(
        value
    ):
        raise ValueError(
            "Normalized action must be finite."
        )

    if (
        value
        < -1.0 - tolerance
        or value
        > 1.0 + tolerance
    ):
        raise ValueError(
            "Normalized action must lie in [-1, 1]."
        )

    return float(
        np.clip(
            value,
            -1.0,
            1.0,
        )
    )


# ============================================================
# BESS MAPPING
# ============================================================

def map_bess_action(
    normalized_action: float,
    bess,
    use_dynamic_feasibility: bool = True,
    tolerance: float = 1e-8,
) -> float:
    """
    Convert normalized SAC action to requested physical BESS power.

    Sign convention:
        +1 -> maximum discharge
        -1 -> maximum charge
         0 -> idle
    """

    value = _clip_normalized_action(
        normalized_action,
        tolerance,
    )

    if value == 0.0:
        return 0.0

    if use_dynamic_feasibility:

        if value > 0.0:

            maximum_power = float(
                bess
                .maximum_feasible_discharge_power_kw()
            )

        else:

            maximum_power = float(
                bess
                .maximum_feasible_charge_power_kw()
            )

    else:

        maximum_power = float(
            bess.rated_power_kw
        )

    return float(
        value
        * maximum_power
    )


# ============================================================
# LOCAL SHARING MAPPING
# ============================================================

def build_local_sharing_matrix(
    local_actions,
    environment: VPPEnvironment,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """
    Decode sharing commands from every local agent.

    Reconstruction encoding:
        local_action[0]
            BESS command

        local_action[1:]
            directional sharing requests from MG i to all other MGs
            in ascending destination-index order.

    For N microgrids, each complete local action therefore has:

        1 + (N - 1) = N

    action values.

    Example for N=5:

        MG0 action:
            [BESS, 0->1, 0->2, 0->3, 0->4]

        MG2 action:
            [BESS, 2->0, 2->1, 2->3, 2->4]
    """

    n = int(
        environment.num_microgrids
    )

    if len(
        local_actions
    ) != n:
        raise ValueError(
            "Number of local actions does not "
            "match environment microgrid count."
        )

    matrix = np.zeros(
        (n, n),
        dtype=np.float64,
    )

    maximum_matrix = np.asarray(
        environment
        .energy_sharing_network
        .maximum_power_matrix_kw,
        dtype=np.float64,
    )

    connectivity = np.asarray(
        environment
        .energy_sharing_network
        .connectivity_matrix,
        dtype=np.float64,
    )

    for source_index in range(
        n
    ):

        action = _to_action_vector(
            local_actions[
                source_index
            ],
            f"local_actions[{source_index}]",
        )

        expected_dimension = n

        if action.shape != (
            expected_dimension,
        ):
            raise ValueError(
                f"local_actions[{source_index}] "
                f"must have dimension "
                f"{expected_dimension}. "
                "Expected [BESS + one sharing "
                "command per other microgrid]."
            )

        sharing_components = (
            action[
                1:
            ]
        )

        destination_indices = [
            destination
            for destination in range(
                n
            )
            if destination
            != source_index
        ]

        for component_index, destination in enumerate(
            destination_indices
        ):

            normalized = (
                _clip_normalized_action(
                    sharing_components[
                        component_index
                    ],
                    tolerance,
                )
            )

            # Directional transfer cannot be negative.
            #
            # SAC output <= 0 means no transfer.
            positive_request = max(
                normalized,
                0.0,
            )

            if connectivity[
                source_index,
                destination,
            ] <= 0.0:

                matrix[
                    source_index,
                    destination,
                ] = 0.0

                continue

            maximum_power = float(
                maximum_matrix[
                    source_index,
                    destination,
                ]
            )

            matrix[
                source_index,
                destination,
            ] = (
                positive_request
                * maximum_power
            )

    np.fill_diagonal(
        matrix,
        0.0,
    )

    return matrix


# ============================================================
# COORDINATOR SHARING MODULATION
# ============================================================

def apply_coordinator_sharing_multiplier(
    sharing_matrix: np.ndarray,
    coordinator_action,
    enabled: bool = True,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """
    Apply coordinator-level global sharing modulation.

    Reconstruction:
        coordinator_action[2] in [-1, 1]

    is transformed into:

        multiplier = (a + 1) / 2

    therefore:

        -1 -> disable sharing
         0 -> 50% local requested sharing
        +1 -> 100% local requested sharing
    """

    matrix = np.asarray(
        sharing_matrix,
        dtype=np.float64,
    ).copy()

    if not enabled:
        return matrix

    action = _to_action_vector(
        coordinator_action,
        "coordinator_action",
    )

    if action.size < 3:
        return matrix

    normalized = (
        _clip_normalized_action(
            action[
                2
            ],
            tolerance,
        )
    )

    multiplier = (
        normalized
        + 1.0
    ) / 2.0

    matrix *= multiplier

    return matrix


# ============================================================
# RESERVE ACTION
# ============================================================

def map_reserve_actions(
    coordinator_action,
    environment: VPPEnvironment,
    bess_actions_kw=None,
    reserve_fraction_of_bess_rating: float = 1.0,
    reserve_duration_hours: float = 1.0,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """
    Convert the coordinator reserve command into feasible per-MG reserve.

    Coordinator action[1] is transformed from [-1, 1] to [0, 1].

    The raw reserve request for MG i is:

        participation
        * reserve_fraction_of_bess_rating
        * P_i_rated

    The raw request is then clipped by the BESS physical capability:

        P_res <= P_rated - max(P_BESS, 0)

    and by energy available above SOC_min for reserve_duration_hours.

    This coupling prevents simultaneous BESS discharge and reserve
    commitments from double-counting the same converter/energy capacity.
    """

    action = _to_action_vector(
        coordinator_action,
        "coordinator_action",
    )

    n = int(
        environment.num_microgrids
    )

    if (
        not np.isfinite(
            reserve_duration_hours
        )
        or reserve_duration_hours <= 0.0
    ):
        raise ValueError(
            "reserve_duration_hours must be finite "
            "and greater than zero."
        )

    if action.size < 2:
        return np.zeros(
            n,
            dtype=np.float64,
        )

    if bess_actions_kw is None:
        scheduled_bess = np.zeros(
            n,
            dtype=np.float64,
        )
    else:
        scheduled_bess = np.asarray(
            bess_actions_kw,
            dtype=np.float64,
        )

        if scheduled_bess.shape != (n,):
            raise ValueError(
                "bess_actions_kw must have one value "
                "per microgrid."
            )

        if not np.isfinite(
            scheduled_bess
        ).all():
            raise ValueError(
                "bess_actions_kw contains NaN or Inf."
            )

    normalized = (
        _clip_normalized_action(
            action[1],
            tolerance,
        )
    )

    participation = (
        normalized
        + 1.0
    ) / 2.0

    reserve = np.zeros(
        n,
        dtype=np.float64,
    )

    for index, microgrid in enumerate(
        environment.microgrids
    ):

        bess = microgrid.bess

        rated_power = float(
            bess.rated_power_kw
        )

        raw_request_kw = (
            participation
            * reserve_fraction_of_bess_rating
            * rated_power
        )

        physical_limit_kw = float(
            bess.maximum_feasible_upward_reserve_power_kw(
                scheduled_power_kw=(
                    scheduled_bess[index]
                ),
                reserve_duration_hours=(
                    reserve_duration_hours
                ),
            )
        )

        reserve[index] = min(
            raw_request_kw,
            physical_limit_kw,
        )

    return reserve


# ============================================================
# OPTIONAL GRID ACTION
# ============================================================

def map_grid_actions(
    coordinator_action,
    environment: VPPEnvironment,
    enabled: bool = False,
    tolerance: float = 1e-8,
) -> Optional[np.ndarray]:
    """
    Optional explicit grid-power control.

    By default this is disabled because VPPEnvironment can close each
    microgrid's physical balance automatically.

    If enabled, coordinator action[0] is interpreted as a normalized
    aggregate import/export command and distributed according to each
    microgrid transformer rating.

    Positive -> import
    Negative -> export
    """

    if not enabled:
        return None

    action = _to_action_vector(
        coordinator_action,
        "coordinator_action",
    )

    if action.size < 1:
        raise ValueError(
            "Coordinator action requires at least "
            "one component for grid control."
        )

    normalized = (
        _clip_normalized_action(
            action[
                0
            ],
            tolerance,
        )
    )

    ratings = np.asarray(
        [
            float(
                microgrid
                .transformer_rating_kva
            )
            for microgrid
            in environment.microgrids
        ],
        dtype=np.float64,
    )

    if np.any(
        ratings <= 0.0
    ):
        raise ValueError(
            "Transformer ratings must be positive."
        )

    return (
        normalized
        * ratings
    )


# ============================================================
# MAIN ACTION MAPPER
# ============================================================

class FCHMARLActionMapper:
    """
    Production action mapper for the real VPP environment.
    """

    def __init__(
        self,
        config: Optional[
            ActionMapperConfig
        ] = None,
    ) -> None:

        if config is None:
            config = ActionMapperConfig()

        config.validate()

        self.config = config

    def __call__(
        self,
        actions:
        HierarchicalActionBundle,
        exogenous:
        VPPExogenousInput,
        environment:
        VPPEnvironment,
    ) -> VPPPhysicalAction:

        if not isinstance(
            actions,
            HierarchicalActionBundle,
        ):
            raise TypeError(
                "actions must be a "
                "HierarchicalActionBundle."
            )

        if not isinstance(
            environment,
            VPPEnvironment,
        ):
            raise TypeError(
                "environment must be a "
                "VPPEnvironment."
            )

        if len(
            actions.local_actions
        ) != environment.num_microgrids:
            raise ValueError(
                "Local action count does not match "
                "number of microgrids."
            )

        # ----------------------------------------------------
        # LOCAL BESS ACTIONS
        # ----------------------------------------------------

        bess_actions = np.zeros(
            environment.num_microgrids,
            dtype=np.float64,
        )

        for index, microgrid in enumerate(
            environment.microgrids
        ):

            local_action = (
                _to_action_vector(
                    actions.local_actions[
                        index
                    ],
                    f"local_actions[{index}]",
                )
            )

            if local_action.size < 1:
                raise ValueError(
                    "Each local action must contain "
                    "a BESS command."
                )

            bess_actions[
                index
            ] = map_bess_action(
                normalized_action=(
                    local_action[
                        0
                    ]
                ),

                bess=microgrid.bess,

                use_dynamic_feasibility=(
                    self.config
                    .use_dynamic_bess_feasibility
                ),

                tolerance=(
                    self.config
                    .action_tolerance
                ),
            )

        # ----------------------------------------------------
        # LOCAL SHARING
        # ----------------------------------------------------

        sharing_matrix = (
            build_local_sharing_matrix(
                local_actions=(
                    actions.local_actions
                ),

                environment=environment,

                tolerance=(
                    self.config
                    .action_tolerance
                ),
            )
        )

        # ----------------------------------------------------
        # COORDINATOR ACTION
        # ----------------------------------------------------

        coordinator_action = (
            _to_action_vector(
                actions.coordinator_action,
                "coordinator_action",
            )
        )

        sharing_matrix = (
            apply_coordinator_sharing_multiplier(
                sharing_matrix=(
                    sharing_matrix
                ),

                coordinator_action=(
                    coordinator_action
                ),

                enabled=(
                    self.config
                    .coordinator_controls_sharing
                ),

                tolerance=(
                    self.config
                    .action_tolerance
                ),
            )
        )

        if (
            self.config
            .coordinator_controls_reserve
        ):

            reserve_actions = (
                map_reserve_actions(
                    coordinator_action=(
                        coordinator_action
                    ),

                    environment=environment,

                    bess_actions_kw=(
                        bess_actions
                    ),

                    reserve_fraction_of_bess_rating=(
                        self.config
                        .reserve_fraction_of_bess_rating
                    ),

                    reserve_duration_hours=(
                        self.config
                        .reserve_duration_hours
                    ),

                    tolerance=(
                        self.config
                        .action_tolerance
                    ),
                )
            )

        else:

            reserve_actions = np.zeros(
                environment.num_microgrids,
                dtype=np.float64,
            )

        grid_actions = (
            map_grid_actions(
                coordinator_action=(
                    coordinator_action
                ),

                environment=environment,

                enabled=(
                    self.config
                    .coordinator_controls_grid
                ),

                tolerance=(
                    self.config
                    .action_tolerance
                ),
            )
        )

        physical_action = (
            VPPPhysicalAction(
                bess_actions_kw=(
                    bess_actions
                ),

                sharing_matrix_kw=(
                    sharing_matrix
                ),

                reserve_actions_kw=(
                    reserve_actions
                ),

                grid_power_actions_kw=(
                    grid_actions
                ),

                ev_requested_charging_powers_kw=(
                    exogenous
                    .ev_requested_charging_powers_kw
                ),
            )
        )

        physical_action.validate(
            environment.num_microgrids
        )

        return physical_action
