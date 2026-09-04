"""
Local microgrid model for the FC-HMARL VPP environment.

This module integrates:

- photovoltaic generation
- BESS operation
- EV charging demand
- local load
- internal energy sharing
- utility-grid exchange
- transformer / PCC constraints
- local market accounting

Manuscript local net-power formulation
--------------------------------------

    P_net
        = P_PV
          + P_BESS
          - P_load
          - P_EV

Final microgrid balance
-----------------------

    P_balance
        = P_net
          + P_share_in
          - P_share_out
          + P_grid

Feasible operation requires:

    P_balance = 0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from environment.bess import (
    BatteryEnergyStorageSystem,
)

from environment.pv import (
    PhotovoltaicSystem,
)

from environment.ev_fleet import (
    EVFleet,
)

from environment.constraints import (
    ConstraintParameters,
    clip_pcc_exchange,
    evaluate_microgrid_constraints,
    required_grid_power_for_balance,
)

from environment.market import (
    ElectricityMarket,
)


# ============================================================
# MICROGRID PARAMETERS
# ============================================================

@dataclass
class MicrogridParameters:
    """
    Static configuration of one local microgrid.

    Parameters
    ----------
    name:
        Microgrid name.

    peak_load_kw:
        Peak electricity demand.

    transformer_rating_kva:
        Transformer rating from manuscript Table 2.

    power_factor:
        Explicit reconstruction parameter used to convert
        kVA to active-power limit.

    balance_tolerance_kw:
        Numerical tolerance for power balance.
    """

    name: str
    peak_load_kw: float
    transformer_rating_kva: float

    power_factor: float = 1.0
    balance_tolerance_kw: float = 1e-6

    def validate(self) -> None:

        if not self.name:
            raise ValueError(
                "Microgrid name cannot be empty."
            )

        if self.peak_load_kw <= 0:
            raise ValueError(
                "peak_load_kw must be greater than zero."
            )

        if self.transformer_rating_kva <= 0:
            raise ValueError(
                "transformer_rating_kva must be greater than zero."
            )

        if not 0 < self.power_factor <= 1:
            raise ValueError(
                "power_factor must satisfy 0 < power_factor <= 1."
            )

        if self.balance_tolerance_kw < 0:
            raise ValueError(
                "balance_tolerance_kw must be non-negative."
            )


# ============================================================
# MICROGRID MODEL
# ============================================================

class Microgrid:
    """
    Integrated physical/economic model of one microgrid.
    """

    def __init__(
        self,
        parameters: MicrogridParameters,
        pv_system: PhotovoltaicSystem,
        bess: BatteryEnergyStorageSystem,
        ev_fleet: EVFleet,
        market: ElectricityMarket,
    ) -> None:

        parameters.validate()

        self.parameters = parameters
        self.pv_system = pv_system
        self.bess = bess
        self.ev_fleet = ev_fleet
        self.market = market

        self.constraint_parameters = (
            ConstraintParameters(
                transformer_rating_kva=(
                    parameters.transformer_rating_kva
                ),
                power_factor=(
                    parameters.power_factor
                ),
                balance_tolerance_kw=(
                    parameters.balance_tolerance_kw
                ),
            )
        )

        self.current_time = 0.0

        self.last_load_kw = 0.0
        self.last_pv_power_kw = 0.0
        self.last_bess_power_kw = 0.0
        self.last_ev_power_kw = 0.0

        self.last_net_local_power_kw = 0.0

        self.last_incoming_sharing_kw = 0.0
        self.last_outgoing_sharing_kw = 0.0

        self.last_grid_power_kw = 0.0

        self.last_balance_residual_kw = 0.0

        self.last_market_result = {}
        self.last_constraint_result = {}

    # ========================================================
    # BASIC PROPERTIES
    # ========================================================

    @property
    def name(self) -> str:
        return self.parameters.name

    @property
    def peak_load_kw(self) -> float:
        return self.parameters.peak_load_kw

    @property
    def transformer_rating_kva(self) -> float:
        return self.parameters.transformer_rating_kva

    # ========================================================
    # LOAD VALIDATION
    # ========================================================

    def validate_load(
        self,
        load_kw: float,
    ) -> float:
        """
        Validate local load.

        The manuscript provides peak load per microgrid.
        Therefore load is constrained to:

            0 <= load <= peak load
        """

        load_kw = float(load_kw)

        if not np.isfinite(load_kw):
            raise ValueError(
                "load_kw must be finite."
            )

        if load_kw < 0:
            raise ValueError(
                "load_kw cannot be negative."
            )

        if load_kw > self.peak_load_kw:
            raise ValueError(
                f"load_kw={load_kw} exceeds "
                f"peak_load_kw={self.peak_load_kw}."
            )

        return load_kw

    # ========================================================
    # LOCAL NET POWER
    # ========================================================

    @staticmethod
    def calculate_local_net_power(
        pv_power_kw: float,
        bess_power_kw: float,
        load_kw: float,
        ev_power_kw: float,
    ) -> float:
        """
        Manuscript local net-power equation:

            P_net
                = P_PV
                  + P_BESS
                  - P_load
                  - P_EV

        Positive P_net:
            local surplus.

        Negative P_net:
            local deficit.
        """

        return float(
            float(pv_power_kw)
            + float(bess_power_kw)
            - float(load_kw)
            - float(ev_power_kw)
        )

    # ========================================================
    # GRID POWER
    # ========================================================

    @staticmethod
    def calculate_required_grid_power(
        net_local_power_kw: float,
        incoming_sharing_kw: float,
        outgoing_sharing_kw: float,
    ) -> float:
        """
        Calculate grid power required for exact balance.
        """

        return required_grid_power_for_balance(
            net_local_power_kw=net_local_power_kw,
            incoming_sharing_kw=incoming_sharing_kw,
            outgoing_sharing_kw=outgoing_sharing_kw,
        )

    # ========================================================
    # MAIN MICROGRID STEP
    # ========================================================

    def step(
        self,
        time: float,
        load_kw: float,
        irradiance_w_m2: float,
        bess_requested_power_kw: float,
        incoming_sharing_kw: float,
        outgoing_sharing_kw: float,
        buy_price_usd_per_kwh: float,
        sell_price_usd_per_kwh: float,
        reserve_power_kw: float = 0.0,
        ev_requested_charging_powers_kw=None,
        grid_power_kw: Optional[float] = None,
    ) -> Dict[str, object]:
        """
        Execute one microgrid operating interval.

        Sequence
        --------
        1. Validate load.
        2. Evaluate PV output.
        3. Apply BESS action.
        4. Evaluate aggregated EV demand.
        5. Compute local net power.
        6. Determine grid power if not explicitly supplied.
        7. Evaluate power balance and transformer constraint.
        8. Evaluate grid/market economics.
        """

        time = float(time)

        if time < 0:
            raise ValueError(
                "time must be non-negative."
            )

        load_kw = self.validate_load(
            load_kw
        )

        incoming_sharing_kw = float(
            incoming_sharing_kw
        )

        outgoing_sharing_kw = float(
            outgoing_sharing_kw
        )

        if incoming_sharing_kw < 0:
            raise ValueError(
                "incoming_sharing_kw cannot be negative."
            )

        if outgoing_sharing_kw < 0:
            raise ValueError(
                "outgoing_sharing_kw cannot be negative."
            )

        # ----------------------------------------------------
        # PV
        # ----------------------------------------------------

        pv_result = self.pv_system.step(
            irradiance_w_m2
        )

        pv_power_kw = pv_result[
            "pv_power_kw"
        ]

        # ----------------------------------------------------
        # BESS
        # ----------------------------------------------------

        bess_result = self.bess.step(
            bess_requested_power_kw
        )

        bess_power_kw = bess_result[
            "net_bess_power_kw"
        ]

        # ----------------------------------------------------
        # EV FLEET
        # ----------------------------------------------------

        ev_result = self.ev_fleet.step(
            time=time,
            requested_charging_powers_kw=(
                ev_requested_charging_powers_kw
            ),
        )

        ev_power_kw = ev_result[
            "ev_power_kw"
        ]

        # ----------------------------------------------------
        # LOCAL NET POWER
        # ----------------------------------------------------

        net_local_power_kw = (
            self.calculate_local_net_power(
                pv_power_kw=pv_power_kw,
                bess_power_kw=bess_power_kw,
                load_kw=load_kw,
                ev_power_kw=ev_power_kw,
            )
        )

        # ----------------------------------------------------
        # GRID POWER
        # ----------------------------------------------------

        # First determine the requested grid exchange.
        if grid_power_kw is None:

            requested_grid_power_kw = (
                self.calculate_required_grid_power(
                    net_local_power_kw=(
                        net_local_power_kw
                    ),
                    incoming_sharing_kw=(
                        incoming_sharing_kw
                    ),
                    outgoing_sharing_kw=(
                        outgoing_sharing_kw
                    ),
                )
            )

        else:

            requested_grid_power_kw = float(
                grid_power_kw
            )

            if not np.isfinite(
                requested_grid_power_kw
            ):
                raise ValueError(
                    "grid_power_kw must be finite."
                )

        # ----------------------------------------------------
        # TRANSFORMER / PCC FEASIBILITY SAFETY LAYER
        # ----------------------------------------------------
        # The transformer rating is an apparent-power limit.
        # Under the reconstruction power factor, the corresponding
        # active-power magnitude is:
        #
        #     P_grid,max = S_transformer * power_factor
        #
        # The actual utility-grid exchange is clipped to this
        # physical PCC limit. Any remaining deficit/surplus is NOT
        # hidden: it remains in the power-balance residual and is
        # therefore available to the constraint/reward layer.
        #
        # This prevents market settlement of physically impossible
        # imports/exports while preserving a violation signal for RL.
        transformer_active_power_limit_kw = float(
            self.parameters.transformer_rating_kva
            * self.parameters.power_factor
        )

        # IMPORTANT:
        # The reconstructed transformer/PCC constraint is defined on the
        # combined exchange:
        #
        #     |P_grid + P_share,out| <= P_max
        #
        # Therefore the feasible grid interval depends on outgoing sharing.
        # Use the canonical constraint-layer projection instead of clipping
        # P_grid alone to [-P_max, +P_max].
        grid_power_kw = float(
            clip_pcc_exchange(
                grid_power_kw=requested_grid_power_kw,
                outgoing_sharing_kw=outgoing_sharing_kw,
                active_power_limit_kw=(
                    transformer_active_power_limit_kw
                ),
            )
        )

        grid_power_curtailed_kw = float(
            requested_grid_power_kw
            - grid_power_kw
        )

        # ----------------------------------------------------
        # PHYSICAL CONSTRAINTS
        # ----------------------------------------------------

        constraint_result = (
            evaluate_microgrid_constraints(
                net_local_power_kw=(
                    net_local_power_kw
                ),
                incoming_sharing_kw=(
                    incoming_sharing_kw
                ),
                outgoing_sharing_kw=(
                    outgoing_sharing_kw
                ),
                grid_power_kw=(
                    grid_power_kw
                ),
                parameters=(
                    self.constraint_parameters
                ),
            )
        )

        # ----------------------------------------------------
        # MARKET
        # ----------------------------------------------------

        market_result = self.market.step(
            grid_power_kw=grid_power_kw,
            buy_price_usd_per_kwh=(
                buy_price_usd_per_kwh
            ),
            sell_price_usd_per_kwh=(
                sell_price_usd_per_kwh
            ),
            reserve_power_kw=(
                reserve_power_kw
            ),
        )

        # ----------------------------------------------------
        # SAVE STATE
        # ----------------------------------------------------

        self.current_time = time

        self.last_load_kw = load_kw

        self.last_pv_power_kw = (
            pv_power_kw
        )

        self.last_bess_power_kw = (
            bess_power_kw
        )

        self.last_ev_power_kw = (
            ev_power_kw
        )

        self.last_net_local_power_kw = (
            net_local_power_kw
        )

        self.last_incoming_sharing_kw = (
            incoming_sharing_kw
        )

        self.last_outgoing_sharing_kw = (
            outgoing_sharing_kw
        )

        self.last_grid_power_kw = (
            grid_power_kw
        )

        self.last_balance_residual_kw = (
            constraint_result[
                "power_balance_residual_kw"
            ]
        )

        self.last_market_result = (
            market_result.copy()
        )

        self.last_constraint_result = (
            constraint_result.copy()
        )

        return {
            "microgrid":
                self.name,

            "time":
                time,

            "load_kw":
                load_kw,

            "pv":
                pv_result,

            "bess":
                bess_result,

            "ev":
                ev_result,

            "pv_power_kw":
                pv_power_kw,

            "bess_power_kw":
                bess_power_kw,

            "ev_power_kw":
                ev_power_kw,

            "net_local_power_kw":
                net_local_power_kw,

            "incoming_sharing_kw":
                incoming_sharing_kw,

            "outgoing_sharing_kw":
                outgoing_sharing_kw,

            "requested_grid_power_kw":
                requested_grid_power_kw,

            "grid_power_kw":
                grid_power_kw,

            "grid_power_curtailed_kw":
                grid_power_curtailed_kw,

            "transformer_active_power_limit_kw":
                transformer_active_power_limit_kw,

            "constraints":
                constraint_result,

            "market":
                market_result,
        }

    # ========================================================
    # STATE
    # ========================================================

    def get_state(
        self,
    ) -> Dict[str, object]:

        return {
            "microgrid":
                self.name,

            "time":
                self.current_time,

            "load_kw":
                self.last_load_kw,

            "pv_power_kw":
                self.last_pv_power_kw,

            "bess_power_kw":
                self.last_bess_power_kw,

            "bess_soc":
                self.bess.soc,

            "ev_power_kw":
                self.last_ev_power_kw,

            "net_local_power_kw":
                self.last_net_local_power_kw,

            "incoming_sharing_kw":
                self.last_incoming_sharing_kw,

            "outgoing_sharing_kw":
                self.last_outgoing_sharing_kw,

            "grid_power_kw":
                self.last_grid_power_kw,

            "power_balance_residual_kw":
                self.last_balance_residual_kw,

            "constraints":
                self.last_constraint_result.copy(),

            "market":
                self.last_market_result.copy(),
        }

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        bess_soc: Optional[float] = None,
    ) -> None:

        self.pv_system.reset()
        self.ev_fleet.reset()
        self.market.reset()

        if bess_soc is None:
            self.bess.reset()
        else:
            self.bess.reset(
                bess_soc
            )

        self.current_time = 0.0

        self.last_load_kw = 0.0
        self.last_pv_power_kw = 0.0
        self.last_bess_power_kw = 0.0
        self.last_ev_power_kw = 0.0

        self.last_net_local_power_kw = 0.0

        self.last_incoming_sharing_kw = 0.0
        self.last_outgoing_sharing_kw = 0.0

        self.last_grid_power_kw = 0.0
        self.last_balance_residual_kw = 0.0

        self.last_market_result = {}
        self.last_constraint_result = {}

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(
        self,
    ) -> str:

        return (
            f"Microgrid("
            f"name='{self.name}', "
            f"peak_load={self.peak_load_kw:.1f} kW, "
            f"transformer="
            f"{self.transformer_rating_kva:.1f} kVA)"
        )