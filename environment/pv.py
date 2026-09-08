from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class PVParameters:
    rated_capacity_kw: float
    efficiency: float
    critical_irradiance_w_m2: float
    stc_irradiance_w_m2: float

    def validate(self) -> None:
        """Validate all physical PV parameters."""

        if self.rated_capacity_kw <= 0:
            raise ValueError(
                "rated_capacity_kw must be greater than zero."
            )

        if not 0 < self.efficiency <= 1:
            raise ValueError(
                "efficiency must satisfy 0 < efficiency <= 1."
            )

        if self.critical_irradiance_w_m2 <= 0:
            raise ValueError(
                "critical_irradiance_w_m2 must be greater than zero."
            )

        if self.stc_irradiance_w_m2 <= 0:
            raise ValueError(
                "stc_irradiance_w_m2 must be greater than zero."
            )

        if (
            self.critical_irradiance_w_m2
            >= self.stc_irradiance_w_m2
        ):
            raise ValueError(
                "critical irradiance must be smaller than "
                "STC irradiance."
            )


class PhotovoltaicSystem:
    """
    Physical photovoltaic-generation model.

    The class evaluates the piecewise irradiance relationship
    described in the manuscript and returns physically bounded
    non-negative PV generation.
    """

    def __init__(
        self,
        parameters: PVParameters,
        name: str = "PV",
    ) -> None:

        parameters.validate()

        self.parameters = parameters
        self.name = name

        self.last_irradiance_w_m2 = 0.0
        self.last_irradiance_ratio = 0.0
        self.last_power_kw = 0.0

    # ========================================================
    # BASIC PROPERTIES
    # ========================================================

    @property
    def rated_capacity_kw(self) -> float:
        return self.parameters.rated_capacity_kw

    @property
    def efficiency(self) -> float:
        return self.parameters.efficiency

    @property
    def critical_irradiance_w_m2(self) -> float:
        return self.parameters.critical_irradiance_w_m2

    @property
    def stc_irradiance_w_m2(self) -> float:
        return self.parameters.stc_irradiance_w_m2

    # ========================================================
    # IRRADIANCE RATIO
    # ========================================================

    def irradiance_ratio(
        self,
        irradiance_w_m2: float,
    ) -> float:
        """
        Compute manuscript irradiance ratio r(t).

        Parameters
        ----------
        irradiance_w_m2:
            Solar irradiance I(t) [W/m^2].

        Returns
        -------
        float
            Piecewise irradiance ratio r(t).
        """

        irradiance = float(irradiance_w_m2)

        if irradiance < 0:
            raise ValueError(
                "Solar irradiance cannot be negative."
            )

        critical = self.critical_irradiance_w_m2
        stc = self.stc_irradiance_w_m2

        # ----------------------------------------------------
        # Region 1: low irradiance
        #
        # r(t) = I(t)^2 / (I_c * I_STC)
        # ----------------------------------------------------

        if irradiance < critical:

            ratio = (
                irradiance ** 2
                / (critical * stc)
            )

        # ----------------------------------------------------
        # Region 2: medium irradiance
        #
        # r(t) = I(t) / I_STC
        # ----------------------------------------------------

        elif irradiance < stc:

            ratio = irradiance / stc

        # ----------------------------------------------------
        # Region 3: irradiance >= STC
        #
        # r(t) = 1
        # ----------------------------------------------------

        else:

            ratio = 1.0

        return float(ratio)

    # ========================================================
    # PV POWER
    # ========================================================

    def power_from_irradiance(
        self,
        irradiance_w_m2: float,
    ) -> float:
        """
        Calculate instantaneous PV generation.

        Manuscript equation:

            P_PV(t)
                = eta_PV
                  * r(t)
                  * P_PV_rated
        """

        ratio = self.irradiance_ratio(
            irradiance_w_m2
        )

        power_kw = (
            self.efficiency
            * ratio
            * self.rated_capacity_kw
        )

        # Numerical protection.
        power_kw = max(0.0, power_kw)

        # Since r(t) <= 1 and efficiency <= 1,
        # this normally cannot exceed rated capacity.
        power_kw = min(
            power_kw,
            self.rated_capacity_kw,
        )

        return float(power_kw)

    # ========================================================
    # MAIN ENVIRONMENT STEP
    # ========================================================

    def step(
        self,
        irradiance_w_m2: float,
    ) -> Dict[str, float]:
        """
        Evaluate PV generation for one scheduling interval.
        """

        ratio = self.irradiance_ratio(
            irradiance_w_m2
        )

        power_kw = self.power_from_irradiance(
            irradiance_w_m2
        )

        self.last_irradiance_w_m2 = float(
            irradiance_w_m2
        )

        self.last_irradiance_ratio = ratio

        self.last_power_kw = power_kw

        return {
            "irradiance_w_m2":
                self.last_irradiance_w_m2,

            "irradiance_ratio":
                self.last_irradiance_ratio,

            "pv_power_kw":
                self.last_power_kw,

            "rated_capacity_kw":
                self.rated_capacity_kw,
        }

    # ========================================================
    # VECTOR / PROFILE CALCULATION
    # ========================================================

    def power_profile(
        self,
        irradiance_profile_w_m2,
    ) -> np.ndarray:
        """
        Convert an irradiance time series into a PV power profile.

        Parameters
        ----------
        irradiance_profile_w_m2:
            Sequence or NumPy array of irradiance values.

        Returns
        -------
        numpy.ndarray
            PV generation profile [kW].
        """

        irradiance_profile = np.asarray(
            irradiance_profile_w_m2,
            dtype=float,
        )

        if np.any(irradiance_profile < 0):
            raise ValueError(
                "Irradiance profile contains negative values."
            )

        output = np.array(
            [
                self.power_from_irradiance(value)
                for value in irradiance_profile
            ],
            dtype=float,
        )

        return output

    # ========================================================
    # HISTORICAL PROFILE SUPPORT
    # ========================================================

    def validate_historical_power(
        self,
        power_kw: float,
    ) -> float:
        """
        Validate an externally supplied historical PV measurement.

        The manuscript states that historical datasets are used for
        forecasting and case studies. Therefore this method allows
        measured PV data to enter the environment while enforcing
        physical non-negativity and installed-capacity limits.

        Values outside the physical range are clipped to:

            0 <= P_PV <= rated_capacity
        """

        power = float(power_kw)

        power = float(
            np.clip(
                power,
                0.0,
                self.rated_capacity_kw,
            )
        )

        return power

    # ========================================================
    # STATE INFORMATION
    # ========================================================

    def get_state(self) -> Dict[str, float]:
        """Return current PV-system information."""

        return {
            "irradiance_w_m2":
                self.last_irradiance_w_m2,

            "irradiance_ratio":
                self.last_irradiance_ratio,

            "pv_power_kw":
                self.last_power_kw,

            "rated_capacity_kw":
                self.rated_capacity_kw,
        }

    def reset(self) -> None:
        """Reset stored PV state."""

        self.last_irradiance_w_m2 = 0.0
        self.last_irradiance_ratio = 0.0
        self.last_power_kw = 0.0

    def __repr__(self) -> str:
        return (
            f"PhotovoltaicSystem("
            f"name='{self.name}', "
            f"rated_capacity="
            f"{self.rated_capacity_kw:.1f} kW)"
        )
