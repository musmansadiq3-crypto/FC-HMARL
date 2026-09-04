"""
Battery Energy Storage System (BESS) model for FC-HMARL.

The implementation follows the BESS formulation in the manuscript:

SOC(t+1)
    = (1 - self_discharge) * SOC(t)
      + eta_ch * P_ch(t) * dt / E_capacity
      - P_dis(t) * dt / (eta_dis * E_capacity)

Net BESS power:

P_BESS(t) = P_dis(t) - P_ch(t)

Sign convention used by the manuscript:
    P_BESS > 0  -> battery is discharging
    P_BESS < 0  -> battery is charging

The environment clips infeasible requested actions so that:
    SOC_min <= SOC <= SOC_max
    0 <= P_ch <= P_rated
    0 <= P_dis <= P_rated
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class BESSParameters:
    """
    Physical and operational parameters of one BESS.

    Parameters
    ----------
    capacity_kwh:
        Rated battery energy capacity.

    rated_power_kw:
        BESS rated charging/discharging power.

        Reconstruction choice:
        The manuscript Table 2 provides one BESS rated power per
        microgrid. Therefore, this implementation uses the same value
        as both maximum charging and maximum discharging power.

    charging_efficiency:
        Charging efficiency.

    discharging_efficiency:
        Discharging efficiency.

    self_discharge_rate:
        Fractional self-discharge rate per simulation step.

    minimum_soc:
        Minimum permitted state of charge.

    maximum_soc:
        Maximum permitted state of charge.

    time_step_hours:
        Simulation interval in hours.
    """

    capacity_kwh: float
    rated_power_kw: float

    charging_efficiency: float = 0.95
    discharging_efficiency: float = 0.95
    self_discharge_rate: float = 0.001

    minimum_soc: float = 0.20
    maximum_soc: float = 0.95

    time_step_hours: float = 1.0

    def validate(self) -> None:
        """Validate BESS parameters."""

        if self.capacity_kwh <= 0:
            raise ValueError("capacity_kwh must be greater than zero.")

        if self.rated_power_kw <= 0:
            raise ValueError("rated_power_kw must be greater than zero.")

        if not 0 < self.charging_efficiency <= 1:
            raise ValueError(
                "charging_efficiency must be within (0, 1]."
            )

        if not 0 < self.discharging_efficiency <= 1:
            raise ValueError(
                "discharging_efficiency must be within (0, 1]."
            )

        if not 0 <= self.self_discharge_rate < 1:
            raise ValueError(
                "self_discharge_rate must be within [0, 1)."
            )

        if not 0 <= self.minimum_soc < self.maximum_soc <= 1:
            raise ValueError(
                "SOC limits must satisfy "
                "0 <= minimum_soc < maximum_soc <= 1."
            )

        if self.time_step_hours <= 0:
            raise ValueError(
                "time_step_hours must be greater than zero."
            )


class BatteryEnergyStorageSystem:
    """
    Dynamic BESS model used by each local microgrid.

    An RL agent may request either charging or discharging through
    signed battery power:

        requested_power_kw > 0 : discharge
        requested_power_kw < 0 : charge
        requested_power_kw = 0 : idle

    The class converts that signed request into physically feasible
    charging/discharging powers and updates SOC.
    """

    def __init__(
        self,
        parameters: BESSParameters,
        initial_soc: Optional[float] = None,
        name: str = "BESS",
    ) -> None:

        parameters.validate()

        self.parameters = parameters
        self.name = name

        # ----------------------------------------------------
        # Initial SOC
        # ----------------------------------------------------
        # The manuscript gives SOC operating limits but does not
        # specify one universal initial SOC value.
        #
        # Therefore, if initial_soc is not supplied, the midpoint
        # between minimum and maximum SOC is used as an explicit
        # reconstruction choice.
        # ----------------------------------------------------

        if initial_soc is None:
            initial_soc = (
                parameters.minimum_soc
                + parameters.maximum_soc
            ) / 2.0

        self._validate_initial_soc(initial_soc)

        self.initial_soc = float(initial_soc)
        self.soc = float(initial_soc)

        self.last_requested_power_kw = 0.0
        self.last_net_power_kw = 0.0
        self.last_charging_power_kw = 0.0
        self.last_discharging_power_kw = 0.0
        self.last_energy_throughput_kwh = 0.0

    # ========================================================
    # BASIC PROPERTIES
    # ========================================================

    @property
    def capacity_kwh(self) -> float:
        return self.parameters.capacity_kwh

    @property
    def rated_power_kw(self) -> float:
        return self.parameters.rated_power_kw

    @property
    def minimum_soc(self) -> float:
        return self.parameters.minimum_soc

    @property
    def maximum_soc(self) -> float:
        return self.parameters.maximum_soc

    @property
    def charging_efficiency(self) -> float:
        return self.parameters.charging_efficiency

    @property
    def discharging_efficiency(self) -> float:
        return self.parameters.discharging_efficiency

    @property
    def self_discharge_rate(self) -> float:
        return self.parameters.self_discharge_rate

    @property
    def time_step_hours(self) -> float:
        return self.parameters.time_step_hours

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_initial_soc(self, soc: float) -> None:
        """Check that initial SOC is inside the operating range."""

        if not self.minimum_soc <= soc <= self.maximum_soc:
            raise ValueError(
                f"Initial SOC {soc:.4f} is outside "
                f"[{self.minimum_soc:.4f}, "
                f"{self.maximum_soc:.4f}]."
            )

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        soc: Optional[float] = None,
    ) -> float:
        """
        Reset the battery state.

        If SOC is omitted, the original initial SOC is restored.
        """

        if soc is None:
            soc = self.initial_soc

        self._validate_initial_soc(float(soc))

        self.soc = float(soc)

        self.last_requested_power_kw = 0.0
        self.last_net_power_kw = 0.0
        self.last_charging_power_kw = 0.0
        self.last_discharging_power_kw = 0.0
        self.last_energy_throughput_kwh = 0.0

        return self.soc

    # ========================================================
    # AVAILABLE POWER
    # ========================================================

    def maximum_feasible_charge_power_kw(self) -> float:
        """
        Calculate the maximum charging power allowed by both:

        1. BESS rated power
        2. Remaining SOC headroom
        """

        dt = self.time_step_hours

        # SOC after self-discharge before charging.
        soc_after_loss = (
            1.0 - self.self_discharge_rate
        ) * self.soc

        available_soc_room = max(
            0.0,
            self.maximum_soc - soc_after_loss,
        )

        # From:
        #
        # delta_SOC = eta_ch * P_ch * dt / E_capacity
        #
        # Therefore:
        #
        # P_ch = delta_SOC * E_capacity / (eta_ch * dt)

        soc_limited_power = (
            available_soc_room
            * self.capacity_kwh
            / (
                self.charging_efficiency
                * dt
            )
        )

        return float(
            min(
                self.rated_power_kw,
                soc_limited_power,
            )
        )

    def maximum_feasible_discharge_power_kw(self) -> float:
        """
        Calculate the maximum discharging power allowed by both:

        1. BESS rated power
        2. Energy available above minimum SOC
        """

        dt = self.time_step_hours

        soc_after_loss = (
            1.0 - self.self_discharge_rate
        ) * self.soc

        available_soc_energy = max(
            0.0,
            soc_after_loss - self.minimum_soc,
        )

        # From:
        #
        # delta_SOC =
        # P_dis * dt / (eta_dis * E_capacity)
        #
        # Therefore:
        #
        # P_dis =
        # delta_SOC * eta_dis * E_capacity / dt

        soc_limited_power = (
            available_soc_energy
            * self.discharging_efficiency
            * self.capacity_kwh
            / dt
        )

        return float(
            min(
                self.rated_power_kw,
                soc_limited_power,
            )
        )

    def maximum_feasible_upward_reserve_power_kw(
        self,
        scheduled_power_kw: float = 0.0,
        reserve_duration_hours: float = 1.0,
    ) -> float:
        """
        Return physically feasible upward reserve capability.

        This is a reconstruction added after the Step 7R audit.

        Upward reserve is limited simultaneously by:

        1. instantaneous discharge-power headroom, and
        2. energy available above minimum SOC for the reserve duration.

        The scheduled BESS operating point is included explicitly so
        reserve cannot be sold on top of already-used discharge power.

        Sign convention
        ---------------
        scheduled_power_kw > 0
            Scheduled discharge.

        scheduled_power_kw < 0
            Scheduled charge.

        Notes
        -----
        This implementation is intentionally conservative when the BESS
        is charging: planned charging energy is not credited as future
        reserve energy. Reserve therefore relies only on energy already
        stored above the minimum SOC boundary.
        """

        scheduled_power_kw = float(
            scheduled_power_kw
        )

        reserve_duration_hours = float(
            reserve_duration_hours
        )

        if not np.isfinite(
            scheduled_power_kw
        ):
            raise ValueError(
                "scheduled_power_kw must be finite."
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

        # Re-apply the ordinary BESS feasibility limiter without
        # mutating the state.
        feasible_schedule_kw = (
            self.constrain_power(
                scheduled_power_kw
            )
        )

        scheduled_discharge_kw = max(
            feasible_schedule_kw,
            0.0,
        )

        # ----------------------------------------------------
        # POWER HEADROOM
        # ----------------------------------------------------
        # Additional upward reserve cannot exceed unused discharge
        # inverter / converter capability.
        power_headroom_kw = max(
            0.0,
            self.rated_power_kw
            - scheduled_discharge_kw,
        )

        # ----------------------------------------------------
        # ENERGY HEADROOM
        # ----------------------------------------------------
        # Energy above SOC_min after self-discharge, expressed as
        # deliverable AC-side discharge energy.
        soc_after_loss = (
            1.0
            - self.self_discharge_rate
        ) * self.soc

        stored_soc_margin = max(
            0.0,
            soc_after_loss
            - self.minimum_soc,
        )

        deliverable_energy_kwh = (
            stored_soc_margin
            * self.capacity_kwh
            * self.discharging_efficiency
        )

        # Scheduled discharge already consumes part of that energy
        # during the current operating interval.
        scheduled_discharge_energy_kwh = (
            scheduled_discharge_kw
            * self.time_step_hours
        )

        remaining_deliverable_energy_kwh = max(
            0.0,
            deliverable_energy_kwh
            - scheduled_discharge_energy_kwh,
        )

        energy_limited_reserve_kw = (
            remaining_deliverable_energy_kwh
            / reserve_duration_hours
        )

        return float(
            min(
                power_headroom_kw,
                energy_limited_reserve_kw,
            )
        )


    # ========================================================
    # ACTION CONVERSION
    # ========================================================

    def constrain_power(
        self,
        requested_power_kw: float,
    ) -> float:
        """
        Convert an agent-requested power into a feasible BESS power.

        Manuscript sign convention:

            positive -> discharge
            negative -> charge
        """

        requested_power_kw = float(requested_power_kw)

        if requested_power_kw > 0.0:

            max_discharge = (
                self.maximum_feasible_discharge_power_kw()
            )

            return float(
                min(
                    requested_power_kw,
                    max_discharge,
                )
            )

        if requested_power_kw < 0.0:

            max_charge = (
                self.maximum_feasible_charge_power_kw()
            )

            return float(
                max(
                    requested_power_kw,
                    -max_charge,
                )
            )

        return 0.0

    # ========================================================
    # SOC TRANSITION
    # ========================================================

    def calculate_next_soc(
        self,
        charging_power_kw: float,
        discharging_power_kw: float,
    ) -> float:
        """
        Compute SOC(t+1) using the manuscript BESS equation.
        """

        charging_power_kw = float(charging_power_kw)
        discharging_power_kw = float(discharging_power_kw)

        if charging_power_kw < 0:
            raise ValueError(
                "charging_power_kw must be non-negative."
            )

        if discharging_power_kw < 0:
            raise ValueError(
                "discharging_power_kw must be non-negative."
            )

        # A battery cannot simultaneously charge and discharge.
        if (
            charging_power_kw > 0.0
            and discharging_power_kw > 0.0
        ):
            raise ValueError(
                "Simultaneous charging and discharging "
                "is not permitted."
            )

        dt = self.time_step_hours
        capacity = self.capacity_kwh

        soc_after_self_discharge = (
            1.0 - self.self_discharge_rate
        ) * self.soc

        charging_term = (
            self.charging_efficiency
            * charging_power_kw
            * dt
            / capacity
        )

        discharging_term = (
            discharging_power_kw
            * dt
            / (
                self.discharging_efficiency
                * capacity
            )
        )

        next_soc = (
            soc_after_self_discharge
            + charging_term
            - discharging_term
        )

        return float(next_soc)

    # ========================================================
    # MAIN BESS STEP
    # ========================================================

    def step(
        self,
        requested_power_kw: float,
    ) -> Dict[str, float]:
        """
        Apply one BESS control action.

        Parameters
        ----------
        requested_power_kw:
            Signed BESS power requested by the controller.

            > 0 : discharge
            < 0 : charge
            = 0 : idle

        Returns
        -------
        dict
            Detailed transition information.
        """

        requested_power_kw = float(requested_power_kw)

        old_soc = self.soc

        # Enforce power and SOC feasibility.
        feasible_power_kw = self.constrain_power(
            requested_power_kw
        )

        # Manuscript convention:
        #
        # P_BESS = P_dis - P_ch
        #
        # Therefore:
        #
        # positive signed power => P_dis > 0
        # negative signed power => P_ch > 0

        if feasible_power_kw >= 0.0:

            discharging_power_kw = feasible_power_kw
            charging_power_kw = 0.0

        else:

            charging_power_kw = -feasible_power_kw
            discharging_power_kw = 0.0

        next_soc = self.calculate_next_soc(
            charging_power_kw=charging_power_kw,
            discharging_power_kw=discharging_power_kw,
        )

        # Numerical protection.
        next_soc = float(
            np.clip(
                next_soc,
                self.minimum_soc,
                self.maximum_soc,
            )
        )

        self.soc = next_soc

        net_bess_power_kw = (
            discharging_power_kw
            - charging_power_kw
        )

        energy_throughput_kwh = (
            charging_power_kw
            + discharging_power_kw
        ) * self.time_step_hours

        self.last_requested_power_kw = requested_power_kw
        self.last_net_power_kw = net_bess_power_kw
        self.last_charging_power_kw = charging_power_kw
        self.last_discharging_power_kw = (
            discharging_power_kw
        )
        self.last_energy_throughput_kwh = (
            energy_throughput_kwh
        )

        return {
            "old_soc": old_soc,
            "new_soc": self.soc,

            "requested_power_kw":
                requested_power_kw,

            "feasible_power_kw":
                feasible_power_kw,

            "charging_power_kw":
                charging_power_kw,

            "discharging_power_kw":
                discharging_power_kw,

            "net_bess_power_kw":
                net_bess_power_kw,

            "energy_throughput_kwh":
                energy_throughput_kwh,
        }

    # ========================================================
    # INFORMATION
    # ========================================================

    def get_state(self) -> Dict[str, float]:
        """
        Return the current BESS state.
        """

        return {
            "soc": self.soc,
            "capacity_kwh": self.capacity_kwh,
            "rated_power_kw": self.rated_power_kw,
            "minimum_soc": self.minimum_soc,
            "maximum_soc": self.maximum_soc,
            "last_net_power_kw":
                self.last_net_power_kw,
        }

    def __repr__(self) -> str:
        return (
            f"BatteryEnergyStorageSystem("
            f"name='{self.name}', "
            f"capacity={self.capacity_kwh:.1f} kWh, "
            f"rated_power={self.rated_power_kw:.1f} kW, "
            f"SOC={self.soc:.4f})"
        )