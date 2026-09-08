from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


# ============================================================
# PARAMETERS
# ============================================================

@dataclass
class EnergySharingParameters:
    """
    Parameters of the VPP internal energy-sharing network.

    Parameters
    ----------
    number_of_microgrids:
        Number of interconnected microgrids.

    efficiency:
        Energy-sharing transfer efficiency.

        Manuscript operational parameter:
            0.98

    tolerance_kw:
        Numerical tolerance used in conservation checks.
    """

    number_of_microgrids: int
    efficiency: float = 0.98
    tolerance_kw: float = 1e-6

    def validate(self) -> None:
        """Validate energy-sharing parameters."""

        if not isinstance(
            self.number_of_microgrids,
            int,
        ):
            raise TypeError(
                "number_of_microgrids must be an integer."
            )

        if self.number_of_microgrids <= 1:
            raise ValueError(
                "number_of_microgrids must be greater than one."
            )

        if not 0 < self.efficiency <= 1:
            raise ValueError(
                "efficiency must satisfy 0 < efficiency <= 1."
            )

        if self.tolerance_kw < 0:
            raise ValueError(
                "tolerance_kw must be non-negative."
            )


# ============================================================
# ENERGY SHARING NETWORK
# ============================================================

class EnergySharingNetwork:
    """
    Directional internal power-sharing network.

    Matrices
    --------
    connectivity[i, j]
        1 if MG i can send power to MG j.
        0 otherwise.

    maximum_power_kw[i, j]
        Maximum scheduled transfer from MG i to MG j.

    scheduled_power_kw[i, j]
        Current scheduled power sent from MG i to MG j.
    """

    def __init__(
        self,
        parameters: EnergySharingParameters,
        connectivity_matrix,
        maximum_power_matrix_kw,
        name: str = "VPP_Energy_Sharing",
    ) -> None:

        parameters.validate()

        self.parameters = parameters
        self.name = name

        self.connectivity_matrix = np.asarray(
            connectivity_matrix,
            dtype=int,
        )

        self.maximum_power_matrix_kw = np.asarray(
            maximum_power_matrix_kw,
            dtype=float,
        )

        self._validate_network()

        n = self.number_of_microgrids

        self.scheduled_power_matrix_kw = np.zeros(
            (n, n),
            dtype=float,
        )

    # ========================================================
    # BASIC PROPERTIES
    # ========================================================

    @property
    def number_of_microgrids(self) -> int:
        return self.parameters.number_of_microgrids

    @property
    def efficiency(self) -> float:
        return self.parameters.efficiency

    @property
    def tolerance_kw(self) -> float:
        return self.parameters.tolerance_kw

    # ========================================================
    # NETWORK VALIDATION
    # ========================================================

    def _validate_network(self) -> None:
        """Validate topology and sharing-capacity matrices."""

        n = self.number_of_microgrids

        expected_shape = (
            n,
            n,
        )

        if (
            self.connectivity_matrix.shape
            != expected_shape
        ):
            raise ValueError(
                "connectivity_matrix must have shape "
                f"{expected_shape}."
            )

        if (
            self.maximum_power_matrix_kw.shape
            != expected_shape
        ):
            raise ValueError(
                "maximum_power_matrix_kw must have shape "
                f"{expected_shape}."
            )

        if np.any(
            ~np.isin(
                self.connectivity_matrix,
                [0, 1],
            )
        ):
            raise ValueError(
                "connectivity_matrix must contain only 0 or 1."
            )

        if np.any(
            self.maximum_power_matrix_kw < 0
        ):
            raise ValueError(
                "Maximum sharing powers cannot be negative."
            )

        # No self-transfer.
        if np.any(
            np.diag(
                self.connectivity_matrix
            ) != 0
        ):
            raise ValueError(
                "Diagonal connectivity values must be zero."
            )

        if np.any(
            np.diag(
                self.maximum_power_matrix_kw
            ) != 0
        ):
            raise ValueError(
                "Diagonal maximum sharing powers must be zero."
            )

        # Capacity must be zero where no connection exists.
        invalid_capacity = (
            (self.connectivity_matrix == 0)
            & (self.maximum_power_matrix_kw > 0)
        )

        if np.any(
            invalid_capacity
        ):
            raise ValueError(
                "Sharing capacity must be zero where "
                "connectivity is zero."
            )

    # ========================================================
    # CONNECTIVITY
    # ========================================================

    def _validate_microgrid_index(
        self,
        index: int,
    ) -> None:
        """Validate one microgrid zero-based index."""

        if not isinstance(
            index,
            (int, np.integer),
        ):
            raise TypeError(
                "Microgrid index must be an integer."
            )

        if not (
            0 <= index < self.number_of_microgrids
        ):
            raise IndexError(
                f"Microgrid index {index} is outside "
                f"0..{self.number_of_microgrids - 1}."
            )

    def is_connected(
        self,
        sender: int,
        receiver: int,
    ) -> bool:
        """Return whether sender can transfer to receiver."""

        self._validate_microgrid_index(
            sender
        )

        self._validate_microgrid_index(
            receiver
        )

        if sender == receiver:
            return False

        return bool(
            self.connectivity_matrix[
                sender,
                receiver,
            ]
        )

    def maximum_transfer_kw(
        self,
        sender: int,
        receiver: int,
    ) -> float:
        """
        Return maximum scheduled directional sharing power.
        """

        self._validate_microgrid_index(
            sender
        )

        self._validate_microgrid_index(
            receiver
        )

        return float(
            self.maximum_power_matrix_kw[
                sender,
                receiver,
            ]
        )

    # ========================================================
    # MANUSCRIPT SHARING CONSTRAINT
    # ========================================================

    def constrain_transfer(
        self,
        sender: int,
        receiver: int,
        requested_power_kw: float,
    ) -> float:
        """
        Apply:

            0 <= P_ij^sh
               <= a_ij * P_ij^sh,max
        """

        self._validate_microgrid_index(
            sender
        )

        self._validate_microgrid_index(
            receiver
        )

        requested_power_kw = float(
            requested_power_kw
        )

        if not np.isfinite(
            requested_power_kw
        ):
            raise ValueError(
                "requested_power_kw must be finite."
            )

        if requested_power_kw < 0:
            raise ValueError(
                "Directional sharing power cannot be negative."
            )

        if sender == receiver:
            return 0.0

        if not self.is_connected(
            sender,
            receiver,
        ):
            return 0.0

        max_power = self.maximum_transfer_kw(
            sender,
            receiver,
        )

        return float(
            min(
                requested_power_kw,
                max_power,
            )
        )

    # ========================================================
    # SET ONE TRANSFER
    # ========================================================

    def set_transfer(
        self,
        sender: int,
        receiver: int,
        requested_power_kw: float,
    ) -> Dict[str, float]:
        """
        Schedule directional power from sender to receiver.

        Returns scheduled, delivered, and lost power.
        """

        feasible_power = self.constrain_transfer(
            sender=sender,
            receiver=receiver,
            requested_power_kw=requested_power_kw,
        )

        self.scheduled_power_matrix_kw[
            sender,
            receiver,
        ] = feasible_power

        received_power = (
            feasible_power
            * self.efficiency
        )

        transfer_loss = (
            feasible_power
            - received_power
        )

        return {
            "sender":
                sender,

            "receiver":
                receiver,

            "requested_power_kw":
                float(requested_power_kw),

            "scheduled_power_kw":
                feasible_power,

            "received_power_kw":
                received_power,

            "transfer_loss_kw":
                transfer_loss,
        }

    # ========================================================
    # CLEAR TRANSFER
    # ========================================================

    def clear_transfer(
        self,
        sender: int,
        receiver: int,
    ) -> None:
        """Set one directional transfer to zero."""

        self._validate_microgrid_index(
            sender
        )

        self._validate_microgrid_index(
            receiver
        )

        self.scheduled_power_matrix_kw[
            sender,
            receiver,
        ] = 0.0

    # ========================================================
    # FULL ACTION MATRIX
    # ========================================================

    def set_transfer_matrix(
        self,
        requested_matrix_kw,
    ) -> np.ndarray:
        """
        Apply all topology/capacity constraints to a complete
        directional sharing matrix.
        """

        requested = np.asarray(
            requested_matrix_kw,
            dtype=float,
        )

        n = self.number_of_microgrids

        if requested.shape != (
            n,
            n,
        ):
            raise ValueError(
                f"requested_matrix_kw must have shape {(n, n)}."
            )

        if np.any(
            ~np.isfinite(requested)
        ):
            raise ValueError(
                "Sharing matrix contains non-finite values."
            )

        if np.any(
            requested < 0
        ):
            raise ValueError(
                "Directional sharing matrix cannot "
                "contain negative power."
            )

        feasible = np.zeros(
            (n, n),
            dtype=float,
        )

        for sender in range(n):

            for receiver in range(n):

                feasible[
                    sender,
                    receiver,
                ] = self.constrain_transfer(
                    sender=sender,
                    receiver=receiver,
                    requested_power_kw=requested[
                        sender,
                        receiver,
                    ],
                )

        self.scheduled_power_matrix_kw = (
            feasible
        )

        return feasible.copy()

    # ========================================================
    # OUTGOING POWER
    # ========================================================

    def outgoing_power_kw(
        self,
        microgrid: int,
    ) -> float:
        """
        Total scheduled power sent by one microgrid.
        """

        self._validate_microgrid_index(
            microgrid
        )

        return float(
            np.sum(
                self.scheduled_power_matrix_kw[
                    microgrid,
                    :
                ]
            )
        )

    # ========================================================
    # INCOMING POWER
    # ========================================================

    def incoming_scheduled_power_kw(
        self,
        microgrid: int,
    ) -> float:
        """
        Total scheduled incoming transfer before losses.

        This quantity is useful for checking manuscript
        scheduled-power conservation.
        """

        self._validate_microgrid_index(
            microgrid
        )

        return float(
            np.sum(
                self.scheduled_power_matrix_kw[
                    :,
                    microgrid,
                ]
            )
        )

    def incoming_received_power_kw(
        self,
        microgrid: int,
    ) -> float:
        """
        Total actually received power after transfer efficiency.
        """

        incoming_scheduled = (
            self.incoming_scheduled_power_kw(
                microgrid
            )
        )

        return float(
            incoming_scheduled
            * self.efficiency
        )

    # ========================================================
    # TOTAL NETWORK SHARING
    # ========================================================

    def total_scheduled_outgoing_kw(
        self,
    ) -> float:
        """Total power scheduled across all directed links."""

        return float(
            np.sum(
                self.scheduled_power_matrix_kw
            )
        )

    def total_scheduled_incoming_kw(
        self,
    ) -> float:
        """
        Total scheduled incoming power.

        Algebraically this is identical to total outgoing
        because every P_ij appears once in each network accounting.
        """

        column_totals = np.sum(
            self.scheduled_power_matrix_kw,
            axis=0,
        )

        return float(
            np.sum(
                column_totals
            )
        )

    def total_received_power_kw(
        self,
    ) -> float:
        """Total power received after sharing efficiency."""

        return float(
            self.total_scheduled_outgoing_kw()
            * self.efficiency
        )

    def total_transfer_loss_kw(
        self,
    ) -> float:
        """Total network sharing loss."""

        return float(
            self.total_scheduled_outgoing_kw()
            - self.total_received_power_kw()
        )

    # ========================================================
    # CONSERVATION
    # ========================================================

    def scheduled_power_is_conserved(
        self,
    ) -> bool:
        """
        Check manuscript scheduled exchange conservation.

        Sum of all scheduled outgoing transfers must equal
        sum of all scheduled incoming transfers.
        """

        outgoing = (
            self.total_scheduled_outgoing_kw()
        )

        incoming = (
            self.total_scheduled_incoming_kw()
        )

        return bool(
            abs(
                outgoing
                - incoming
            )
            <= self.tolerance_kw
        )

    # ========================================================
    # SURPLUS CONSTRAINT
    # ========================================================

    def constrain_outgoing_by_surplus(
        self,
        microgrid: int,
        available_surplus_kw: float,
    ) -> Dict[str, float]:
        """
        Ensure that total outgoing sharing from one microgrid
        does not exceed its currently available power surplus.

        If current outgoing sharing exceeds available surplus,
        all outgoing transfers from that MG are proportionally
        scaled down.

        This is an implementation-level physical feasibility
        mechanism used when combining sharing with the
        microgrid power balance.
        """

        self._validate_microgrid_index(
            microgrid
        )

        available_surplus_kw = float(
            available_surplus_kw
        )

        if available_surplus_kw < 0:
            raise ValueError(
                "available_surplus_kw must be non-negative."
            )

        current_outgoing = (
            self.outgoing_power_kw(
                microgrid
            )
        )

        if current_outgoing <= 0:

            return {
                "original_outgoing_kw":
                    0.0,

                "constrained_outgoing_kw":
                    0.0,

                "scaling_factor":
                    1.0,
            }

        if (
            current_outgoing
            <= available_surplus_kw
        ):

            return {
                "original_outgoing_kw":
                    current_outgoing,

                "constrained_outgoing_kw":
                    current_outgoing,

                "scaling_factor":
                    1.0,
            }

        scaling_factor = (
            available_surplus_kw
            / current_outgoing
        )

        self.scheduled_power_matrix_kw[
            microgrid,
            :
        ] *= scaling_factor

        constrained_outgoing = (
            self.outgoing_power_kw(
                microgrid
            )
        )

        return {
            "original_outgoing_kw":
                current_outgoing,

            "constrained_outgoing_kw":
                constrained_outgoing,

            "scaling_factor":
                scaling_factor,
        }

    # ========================================================
    # NETWORK STATE
    # ========================================================

    def get_state(
        self,
    ) -> Dict[str, object]:
        """Return current sharing-network state."""

        n = self.number_of_microgrids

        outgoing = np.array(
            [
                self.outgoing_power_kw(i)
                for i in range(n)
            ],
            dtype=float,
        )

        incoming_scheduled = np.array(
            [
                self.incoming_scheduled_power_kw(i)
                for i in range(n)
            ],
            dtype=float,
        )

        incoming_received = np.array(
            [
                self.incoming_received_power_kw(i)
                for i in range(n)
            ],
            dtype=float,
        )

        return {
            "scheduled_power_matrix_kw":
                self.scheduled_power_matrix_kw.copy(),

            "outgoing_power_kw":
                outgoing,

            "incoming_scheduled_power_kw":
                incoming_scheduled,

            "incoming_received_power_kw":
                incoming_received,

            "total_scheduled_power_kw":
                self.total_scheduled_outgoing_kw(),

            "total_received_power_kw":
                self.total_received_power_kw(),

            "total_transfer_loss_kw":
                self.total_transfer_loss_kw(),

            "scheduled_power_conserved":
                self.scheduled_power_is_conserved(),
        }

    # ========================================================
    # RESET
    # ========================================================

    def reset(self) -> None:
        """Reset all scheduled internal transfers to zero."""

        self.scheduled_power_matrix_kw.fill(
            0.0
        )

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(
        self,
    ) -> str:

        return (
            f"EnergySharingNetwork("
            f"name='{self.name}', "
            f"microgrids={self.number_of_microgrids}, "
            f"efficiency={self.efficiency:.4f})"
        )
