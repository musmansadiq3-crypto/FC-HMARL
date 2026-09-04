"""
Electric Vehicle (EV) fleet aggregation model for FC-HMARL.

The implementation follows the EV formulation in the manuscript.

Manuscript equations
--------------------

EV availability:

    u_i,n^EV(t) = 1,  if T_arr,n <= t <= T_dep,n
                  0,  otherwise

Aggregated EV charging demand:

    P_i^EV(t)
        = sum_n u_i,n^EV(t) * P_i,n^ch(t)

Aggregate charging constraint:

    0 <= P_i^EV(t) <= P_i^EV,max


Important reconstruction notes
------------------------------
1. The manuscript gives the number of EVs in each microgrid.

2. The manuscript does NOT provide a numerical value for the
   maximum aggregate EV charging power P_i^EV,max in the recovered
   configuration table.

3. The manuscript does NOT define individual EV battery SOC dynamics
   in this EV-demand model.

4. Therefore:
   - individual arrival times,
   - departure times,
   - individual charging powers, and
   - aggregate charging capacity

   are supplied explicitly to this class.

5. No V2G discharging is introduced here because the recovered
   manuscript equation represents aggregated EV charging demand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


# ============================================================
# INDIVIDUAL EV RECORD
# ============================================================

@dataclass
class EVRecord:
    """
    Operating information for one EV.

    Parameters
    ----------
    arrival_time:
        EV arrival time expressed in simulation-hour coordinates.

    departure_time:
        EV departure time expressed in simulation-hour coordinates.

    charging_power_kw:
        Maximum or assigned charging power of this EV [kW].

    ev_id:
        Optional vehicle identifier.
    """

    arrival_time: float
    departure_time: float
    charging_power_kw: float
    ev_id: Optional[str] = None

    def validate(self) -> None:
        """
        Validate one EV record.
        """

        if self.arrival_time < 0:
            raise ValueError(
                "arrival_time must be non-negative."
            )

        if self.departure_time < 0:
            raise ValueError(
                "departure_time must be non-negative."
            )

        # The manuscript availability equation is written directly as:
        #
        # T_arr <= t <= T_dep
        #
        # Therefore this implementation requires departure >= arrival.
        # Cross-midnight wrapping is not silently introduced.
        if self.departure_time < self.arrival_time:
            raise ValueError(
                "departure_time must be greater than or equal "
                "to arrival_time for the manuscript interval model."
            )

        if self.charging_power_kw < 0:
            raise ValueError(
                "charging_power_kw must be non-negative."
            )


# ============================================================
# FLEET PARAMETERS
# ============================================================

@dataclass
class EVFleetParameters:
    """
    Parameters of one aggregated EV fleet.

    Parameters
    ----------
    number_of_evs:
        Number N_i^EV of participating EVs in the microgrid.

    maximum_aggregate_charging_power_kw:
        Maximum allowable aggregated charging demand
        P_i^EV,max [kW].

    time_step_hours:
        Simulation time step [h].
    """

    number_of_evs: int
    maximum_aggregate_charging_power_kw: float
    time_step_hours: float = 1.0

    def validate(self) -> None:
        """
        Validate fleet parameters.
        """

        if not isinstance(self.number_of_evs, int):
            raise TypeError(
                "number_of_evs must be an integer."
            )

        if self.number_of_evs <= 0:
            raise ValueError(
                "number_of_evs must be greater than zero."
            )

        if self.maximum_aggregate_charging_power_kw < 0:
            raise ValueError(
                "maximum_aggregate_charging_power_kw "
                "must be non-negative."
            )

        if self.time_step_hours <= 0:
            raise ValueError(
                "time_step_hours must be greater than zero."
            )


# ============================================================
# EV FLEET
# ============================================================

class EVFleet:
    """
    Aggregated EV charging-demand model.

    The fleet evaluates:

        individual availability
                ↓
        available charging powers
                ↓
        aggregate EV charging demand
                ↓
        microgrid EV charging capacity constraint
    """

    def __init__(
        self,
        parameters: EVFleetParameters,
        vehicles: Optional[Sequence[EVRecord]] = None,
        name: str = "EV_Fleet",
    ) -> None:

        parameters.validate()

        self.parameters = parameters
        self.name = name

        self.vehicles: List[EVRecord] = []

        if vehicles is not None:
            self.set_vehicles(vehicles)

        self.current_time = 0.0

        self.last_unconstrained_power_kw = 0.0
        self.last_aggregate_power_kw = 0.0
        self.last_available_evs = 0
        self.last_availability_vector = np.array(
            [],
            dtype=float,
        )

    # ========================================================
    # BASIC PROPERTIES
    # ========================================================

    @property
    def number_of_evs(self) -> int:
        return self.parameters.number_of_evs

    @property
    def maximum_aggregate_charging_power_kw(self) -> float:
        return (
            self.parameters
            .maximum_aggregate_charging_power_kw
        )

    @property
    def time_step_hours(self) -> float:
        return self.parameters.time_step_hours

    # ========================================================
    # VEHICLE MANAGEMENT
    # ========================================================

    def set_vehicles(
        self,
        vehicles: Sequence[EVRecord],
    ) -> None:
        """
        Assign individual EV records to the fleet.
        """

        vehicles = list(vehicles)

        if len(vehicles) != self.number_of_evs:
            raise ValueError(
                f"Expected {self.number_of_evs} EV records, "
                f"but received {len(vehicles)}."
            )

        for vehicle in vehicles:
            if not isinstance(vehicle, EVRecord):
                raise TypeError(
                    "Every vehicle must be an EVRecord instance."
                )

            vehicle.validate()

        self.vehicles = vehicles

    def _require_vehicles(self) -> None:
        """
        Ensure that individual EV data have been supplied.
        """

        if len(self.vehicles) != self.number_of_evs:
            raise RuntimeError(
                "EV records have not been fully initialized."
            )

    # ========================================================
    # AVAILABILITY FUNCTION
    # ========================================================

    @staticmethod
    def vehicle_is_available(
        vehicle: EVRecord,
        time: float,
    ) -> bool:
        """
        Evaluate manuscript EV availability equation.

        Returns True when:

            T_arr <= t <= T_dep
        """

        time = float(time)

        if time < 0:
            raise ValueError(
                "Simulation time must be non-negative."
            )

        return bool(
            vehicle.arrival_time
            <= time
            <= vehicle.departure_time
        )

    def availability_vector(
        self,
        time: float,
    ) -> np.ndarray:
        """
        Return u_i,n^EV(t) for all EVs.

        1 -> EV is available
        0 -> EV is unavailable
        """

        self._require_vehicles()

        availability = np.array(
            [
                1.0
                if self.vehicle_is_available(
                    vehicle,
                    time,
                )
                else 0.0
                for vehicle in self.vehicles
            ],
            dtype=float,
        )

        return availability

    def number_available(
        self,
        time: float,
    ) -> int:
        """
        Return number of connected/available EVs.
        """

        availability = self.availability_vector(
            time
        )

        return int(
            np.sum(availability)
        )

    # ========================================================
    # INDIVIDUAL CHARGING POWER
    # ========================================================

    def rated_charging_power_vector(
        self,
    ) -> np.ndarray:
        """
        Return individual EV charging powers.
        """

        self._require_vehicles()

        return np.array(
            [
                vehicle.charging_power_kw
                for vehicle in self.vehicles
            ],
            dtype=float,
        )

    def validate_requested_charging_powers(
        self,
        requested_charging_powers_kw: Iterable[float],
    ) -> np.ndarray:
        """
        Validate requested individual EV charging powers.

        Each requested power is constrained to:

            0 <= P_i,n^ch <= EV charging-power rating
        """

        self._require_vehicles()

        requested = np.asarray(
            list(requested_charging_powers_kw),
            dtype=float,
        )

        if requested.ndim != 1:
            raise ValueError(
                "requested_charging_powers_kw "
                "must be one-dimensional."
            )

        if len(requested) != self.number_of_evs:
            raise ValueError(
                f"Expected {self.number_of_evs} requested "
                f"charging powers, received {len(requested)}."
            )

        if np.any(~np.isfinite(requested)):
            raise ValueError(
                "Requested charging powers must be finite."
            )

        if np.any(requested < 0):
            raise ValueError(
                "EV charging powers cannot be negative "
                "in the manuscript charging-demand model."
            )

        ratings = self.rated_charging_power_vector()

        return np.minimum(
            requested,
            ratings,
        )

    # ========================================================
    # AGGREGATED EV DEMAND
    # ========================================================

    def unconstrained_aggregate_power(
        self,
        time: float,
        requested_charging_powers_kw: Optional[
            Iterable[float]
        ] = None,
    ) -> float:
        """
        Evaluate:

            sum_n u_i,n^EV(t) * P_i,n^ch(t)

        before applying the microgrid aggregate charging limit.

        If requested_charging_powers_kw is omitted, each available
        EV is assumed to request its assigned charging-power value.
        """

        self._require_vehicles()

        availability = self.availability_vector(
            time
        )

        if requested_charging_powers_kw is None:

            charging_power = (
                self.rated_charging_power_vector()
            )

        else:

            charging_power = (
                self.validate_requested_charging_powers(
                    requested_charging_powers_kw
                )
            )

        aggregate = float(
            np.sum(
                availability
                * charging_power
            )
        )

        return aggregate

    def constrain_aggregate_power(
        self,
        aggregate_power_kw: float,
    ) -> float:
        """
        Apply manuscript aggregate EV charging constraint:

            0 <= P_i^EV <= P_i^EV,max
        """

        aggregate_power_kw = float(
            aggregate_power_kw
        )

        if not np.isfinite(aggregate_power_kw):
            raise ValueError(
                "aggregate_power_kw must be finite."
            )

        return float(
            np.clip(
                aggregate_power_kw,
                0.0,
                self.maximum_aggregate_charging_power_kw,
            )
        )

    def aggregate_power(
        self,
        time: float,
        requested_charging_powers_kw: Optional[
            Iterable[float]
        ] = None,
    ) -> float:
        """
        Calculate physically feasible aggregated EV charging demand.
        """

        unconstrained = (
            self.unconstrained_aggregate_power(
                time=time,
                requested_charging_powers_kw=(
                    requested_charging_powers_kw
                ),
            )
        )

        return self.constrain_aggregate_power(
            unconstrained
        )

    # ========================================================
    # MAIN ENVIRONMENT STEP
    # ========================================================

    def step(
        self,
        time: float,
        requested_charging_powers_kw: Optional[
            Iterable[float]
        ] = None,
    ) -> Dict[str, object]:
        """
        Evaluate EV charging demand for one scheduling instant.

        Returns information required later by the microgrid and
        FC-HMARL state construction.
        """

        self._require_vehicles()

        availability = self.availability_vector(
            time
        )

        unconstrained_power = (
            self.unconstrained_aggregate_power(
                time=time,
                requested_charging_powers_kw=(
                    requested_charging_powers_kw
                ),
            )
        )

        aggregate_power = (
            self.constrain_aggregate_power(
                unconstrained_power
            )
        )

        available_evs = int(
            np.sum(availability)
        )

        self.current_time = float(time)

        self.last_availability_vector = (
            availability.copy()
        )

        self.last_available_evs = available_evs

        self.last_unconstrained_power_kw = (
            unconstrained_power
        )

        self.last_aggregate_power_kw = (
            aggregate_power
        )

        return {
            "time":
                self.current_time,

            "number_of_evs":
                self.number_of_evs,

            "available_evs":
                available_evs,

            "availability_vector":
                availability.copy(),

            "unconstrained_ev_power_kw":
                unconstrained_power,

            "ev_power_kw":
                aggregate_power,

            "maximum_ev_power_kw":
                self.maximum_aggregate_charging_power_kw,
        }

    # ========================================================
    # STATE
    # ========================================================

    def get_state(self) -> Dict[str, object]:
        """
        Return current fleet state.
        """

        return {
            "time":
                self.current_time,

            "number_of_evs":
                self.number_of_evs,

            "available_evs":
                self.last_available_evs,

            "availability_vector":
                self.last_availability_vector.copy(),

            "unconstrained_ev_power_kw":
                self.last_unconstrained_power_kw,

            "ev_power_kw":
                self.last_aggregate_power_kw,

            "maximum_ev_power_kw":
                self.maximum_aggregate_charging_power_kw,
        }

    # ========================================================
    # RESET
    # ========================================================

    def reset(self) -> None:
        """
        Reset dynamic fleet outputs.

        Individual EV arrival/departure/charging records remain loaded.
        """

        self.current_time = 0.0

        self.last_unconstrained_power_kw = 0.0
        self.last_aggregate_power_kw = 0.0
        self.last_available_evs = 0

        self.last_availability_vector = np.zeros(
            self.number_of_evs,
            dtype=float,
        )

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(self) -> str:
        return (
            f"EVFleet("
            f"name='{self.name}', "
            f"number_of_evs={self.number_of_evs}, "
            f"maximum_power="
            f"{self.maximum_aggregate_charging_power_kw:.2f} kW)"
        )