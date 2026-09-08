from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from environment.microgrid import Microgrid
from environment.energy_sharing import EnergySharingNetwork
# ============================================================
# VPP ENVIRONMENT
# ============================================================

class VPPEnvironment:
    def __init__(
        self,
        microgrids: Sequence[Microgrid],
        energy_sharing_network: EnergySharingNetwork,
        episode_length_hours: int = 24,
        name: str = "FC_HMARL_VPP",
    ) -> None:

        self.microgrids: List[Microgrid] = list(
            microgrids
        )

        if len(self.microgrids) == 0:
            raise ValueError(
                "At least one microgrid is required."
            )

        if len(self.microgrids) != (
            energy_sharing_network.number_of_microgrids
        ):
            raise ValueError(
                "Number of microgrids must match the "
                "energy-sharing network dimension."
            )

        if episode_length_hours <= 0:
            raise ValueError(
                "episode_length_hours must be greater than zero."
            )

        names = [
            mg.name
            for mg in self.microgrids
        ]

        if len(names) != len(set(names)):
            raise ValueError(
                "Microgrid names must be unique."
            )

        self.energy_sharing_network = (
            energy_sharing_network
        )

        self.episode_length_hours = int(
            episode_length_hours
        )

        self.name = name

        self.num_microgrids = len(
            self.microgrids
        )

        self.current_step = 0
        self.current_time = 0.0

        self.last_result: Dict[str, object] = {}

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    def _vector(
        self,
        values: Iterable[float],
        name: str,
    ) -> np.ndarray:
        """
        Convert an input sequence to a finite vector of length N_MG.
        """

        array = np.asarray(
            list(values),
            dtype=float,
        )

        if array.ndim != 1:
            raise ValueError(
                f"{name} must be one-dimensional."
            )

        if len(array) != self.num_microgrids:
            raise ValueError(
                f"{name} must contain exactly "
                f"{self.num_microgrids} values."
            )

        if np.any(~np.isfinite(array)):
            raise ValueError(
                f"{name} contains non-finite values."
            )

        return array

    def _optional_nested_ev_requests(
        self,
        requests,
    ):
        """
        Validate optional EV charging-action container.

        Expected format:

            [
                requests_for_MG1,
                requests_for_MG2,
                ...
            ]

        Each local request is passed to EVFleet.
        """

        if requests is None:
            return [
                None
                for _ in range(
                    self.num_microgrids
                )
            ]

        if len(requests) != self.num_microgrids:
            raise ValueError(
                "EV request container must contain one "
                "entry per microgrid."
            )

        return list(requests)

    # ========================================================
    # SHARING
    # ========================================================

    def apply_sharing_actions(
        self,
        requested_sharing_matrix_kw,
    ) -> Dict[str, object]:
        """
        Apply the VPP coordinator's requested sharing matrix.

        The EnergySharingNetwork performs:

        - topology enforcement
        - directional power-capacity enforcement
        - zero diagonal enforcement

        Returns incoming and outgoing powers for all MGs.
        """

        feasible_matrix = (
            self.energy_sharing_network
            .set_transfer_matrix(
                requested_sharing_matrix_kw
            )
        )

        outgoing = np.array(
            [
                self.energy_sharing_network
                .outgoing_power_kw(i)
                for i in range(
                    self.num_microgrids
                )
            ],
            dtype=float,
        )

        # IMPORTANT:
        # Physical receiving MG obtains sharing after
        # the 98% transfer efficiency.
        incoming_received = np.array(
            [
                self.energy_sharing_network
                .incoming_received_power_kw(i)
                for i in range(
                    self.num_microgrids
                )
            ],
            dtype=float,
        )

        incoming_scheduled = np.array(
            [
                self.energy_sharing_network
                .incoming_scheduled_power_kw(i)
                for i in range(
                    self.num_microgrids
                )
            ],
            dtype=float,
        )

        return {
            "feasible_matrix_kw":
                feasible_matrix,

            "outgoing_power_kw":
                outgoing,

            "incoming_scheduled_power_kw":
                incoming_scheduled,

            "incoming_received_power_kw":
                incoming_received,

            "total_scheduled_sharing_kw":
                self.energy_sharing_network
                .total_scheduled_outgoing_kw(),

            "total_received_sharing_kw":
                self.energy_sharing_network
                .total_received_power_kw(),

            "total_sharing_loss_kw":
                self.energy_sharing_network
                .total_transfer_loss_kw(),
        }
    # ONE VPP STEP
    def step(
        self,
        time: float,
        loads_kw: Iterable[float],
        irradiances_w_m2: Iterable[float],
        bess_actions_kw: Iterable[float],
        sharing_matrix_kw,
        buy_prices_usd_per_kwh: Iterable[float],
        sell_prices_usd_per_kwh: Iterable[float],
        reserve_actions_kw: Optional[
            Iterable[float]
        ] = None,
        ev_requested_charging_powers_kw=None,
        grid_power_actions_kw: Optional[
            Iterable[float]
        ] = None,
    ) -> Dict[str, object]:
        time = float(time)

        if time < 0:
            raise ValueError(
                "time must be non-negative."
            )

        loads = self._vector(
            loads_kw,
            "loads_kw",
        )

        irradiances = self._vector(
            irradiances_w_m2,
            "irradiances_w_m2",
        )

        bess_actions = self._vector(
            bess_actions_kw,
            "bess_actions_kw",
        )

        buy_prices = self._vector(
            buy_prices_usd_per_kwh,
            "buy_prices_usd_per_kwh",
        )

        sell_prices = self._vector(
            sell_prices_usd_per_kwh,
            "sell_prices_usd_per_kwh",
        )

        if reserve_actions_kw is None:

            reserve_actions = np.zeros(
                self.num_microgrids,
                dtype=float,
            )

        else:

            reserve_actions = self._vector(
                reserve_actions_kw,
                "reserve_actions_kw",
            )

        if np.any(
            reserve_actions < 0
        ):
            raise ValueError(
                "Reserve actions cannot be negative."
            )

        # ----------------------------------------------------
        # RESERVE FEASIBILITY SAFETY LAYER
        # ----------------------------------------------------
        # The action mapper normally performs this clipping first.
        # It is repeated here deliberately so direct callers of
        # VPPEnvironment cannot bypass physical BESS reserve limits.
        #
        # Upward reserve shares both discharge-power capability and
        # stored energy with the scheduled BESS operating point.
        requested_reserve_actions = (
            reserve_actions.copy()
        )

        feasible_reserve_actions = np.zeros(
            self.num_microgrids,
            dtype=float,
        )

        for i, microgrid in enumerate(
            self.microgrids
        ):

            scheduled_bess_kw = float(
                microgrid.bess.constrain_power(
                    bess_actions[i]
                )
            )

            reserve_limit_kw = float(
                microgrid
                .bess
                .maximum_feasible_upward_reserve_power_kw(
                    scheduled_power_kw=(
                        scheduled_bess_kw
                    ),
                    reserve_duration_hours=(
                        microgrid
                        .bess
                        .time_step_hours
                    ),
                )
            )

            feasible_reserve_actions[i] = min(
                reserve_actions[i],
                reserve_limit_kw,
            )

        reserve_actions = (
            feasible_reserve_actions
        )

        reserve_curtailed_actions = (
            requested_reserve_actions
            - reserve_actions
        )

        if grid_power_actions_kw is None:

            grid_actions = [
                None
                for _ in range(
                    self.num_microgrids
                )
            ]

        else:

            grid_actions = self._vector(
                grid_power_actions_kw,
                "grid_power_actions_kw",
            )

        ev_requests = (
            self._optional_nested_ev_requests(
                ev_requested_charging_powers_kw
            )
        )

        # ----------------------------------------------------
        # ENERGY SHARING
        # ----------------------------------------------------

        sharing_result = (
            self.apply_sharing_actions(
                sharing_matrix_kw
            )
        )

        incoming = sharing_result[
            "incoming_received_power_kw"
        ]

        outgoing = sharing_result[
            "outgoing_power_kw"
        ]

        # ----------------------------------------------------
        # STEP EACH MICROGRID
        # ----------------------------------------------------

        local_results = []

        for i, microgrid in enumerate(
            self.microgrids
        ):

            result = microgrid.step(
                time=time,
                load_kw=loads[i],
                irradiance_w_m2=(
                    irradiances[i]
                ),
                bess_requested_power_kw=(
                    bess_actions[i]
                ),
                incoming_sharing_kw=(
                    incoming[i]
                ),
                outgoing_sharing_kw=(
                    outgoing[i]
                ),
                buy_price_usd_per_kwh=(
                    buy_prices[i]
                ),
                sell_price_usd_per_kwh=(
                    sell_prices[i]
                ),
                reserve_power_kw=(
                    reserve_actions[i]
                ),
                ev_requested_charging_powers_kw=(
                    ev_requests[i]
                ),
                grid_power_kw=(
                    grid_actions[i]
                ),
            )

            local_results.append(
                result
            )

        # ----------------------------------------------------
        # SYSTEM-LEVEL AGGREGATION
        # ----------------------------------------------------

        total_pv_kw = float(
            sum(
                result["pv_power_kw"]
                for result in local_results
            )
        )

        total_load_kw = float(
            sum(
                result["load_kw"]
                for result in local_results
            )
        )

        total_ev_kw = float(
            sum(
                result["ev_power_kw"]
                for result in local_results
            )
        )

        total_bess_kw = float(
            sum(
                result["bess_power_kw"]
                for result in local_results
            )
        )

        total_grid_power_kw = float(
            sum(
                result["grid_power_kw"]
                for result in local_results
            )
        )

        total_requested_grid_power_kw = float(
            sum(
                result.get(
                    "requested_grid_power_kw",
                    result["grid_power_kw"],
                )
                for result in local_results
            )
        )

        total_grid_power_curtailed_kw = float(
            sum(
                abs(
                    result.get(
                        "grid_power_curtailed_kw",
                        0.0,
                    )
                )
                for result in local_results
            )
        )

        total_grid_import_kw = float(
            sum(
                max(
                    result["grid_power_kw"],
                    0.0,
                )
                for result in local_results
            )
        )

        total_grid_export_kw = float(
            sum(
                max(
                    -result["grid_power_kw"],
                    0.0,
                )
                for result in local_results
            )
        )

        total_purchase_cost_usd = float(
            sum(
                result["market"][
                    "grid_purchase_cost_usd"
                ]
                for result in local_results
            )
        )

        total_sale_revenue_usd = float(
            sum(
                result["market"][
                    "grid_sale_revenue_usd"
                ]
                for result in local_results
            )
        )

        total_requested_reserve_kw = float(
            np.sum(
                requested_reserve_actions
            )
        )

        total_feasible_reserve_kw = float(
            np.sum(
                reserve_actions
            )
        )

        total_curtailed_reserve_kw = float(
            np.sum(
                reserve_curtailed_actions
            )
        )

        total_reserve_revenue_usd = float(
            sum(
                result["market"][
                    "reserve_revenue_usd"
                ]
                for result in local_results
            )
        )

        total_market_revenue_usd = (
            total_sale_revenue_usd
            + total_reserve_revenue_usd
        )

        total_market_profit_usd = (
            total_market_revenue_usd
            - total_purchase_cost_usd
        )

        all_power_balanced = bool(
            all(
                result["constraints"][
                    "power_balance_satisfied"
                ]
                for result in local_results
            )
        )

        all_transformers_feasible = bool(
            all(
                result["constraints"][
                    "transformer_limit_satisfied"
                ]
                for result in local_results
            )
        )

        total_balance_violation_kw = float(
            sum(
                result["constraints"][
                    "power_balance_violation_kw"
                ]
                for result in local_results
            )
        )

        total_transformer_violation_kw = float(
            sum(
                result["constraints"][
                    "transformer_violation_kw"
                ]
                for result in local_results
            )
        )

        # ----------------------------------------------------
        # TIME UPDATE
        # ----------------------------------------------------

        self.current_time = time

        self.current_step += 1

        done = bool(
            self.current_step
            >= self.episode_length_hours
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        result = {
            "time":
                time,

            "step":
                self.current_step,

            "done":
                done,

            "num_microgrids":
                self.num_microgrids,

            "local_results":
                local_results,

            "sharing":
                sharing_result,

            "totals": {
                "pv_power_kw":
                    total_pv_kw,

                "load_power_kw":
                    total_load_kw,

                "ev_power_kw":
                    total_ev_kw,

                "bess_power_kw":
                    total_bess_kw,

                "requested_grid_power_kw":
                    total_requested_grid_power_kw,

                "grid_power_kw":
                    total_grid_power_kw,

                "grid_power_curtailed_kw":
                    total_grid_power_curtailed_kw,

                "grid_import_kw":
                    total_grid_import_kw,

                "grid_export_kw":
                    total_grid_export_kw,

                "grid_purchase_cost_usd":
                    total_purchase_cost_usd,

                "grid_sale_revenue_usd":
                    total_sale_revenue_usd,

                "reserve_requested_kw":
                    total_requested_reserve_kw,

                "reserve_feasible_kw":
                    total_feasible_reserve_kw,

                "reserve_curtailed_kw":
                    total_curtailed_reserve_kw,

                "reserve_revenue_usd":
                    total_reserve_revenue_usd,

                "market_revenue_usd":
                    total_market_revenue_usd,

                "market_profit_usd":
                    total_market_profit_usd,

                "sharing_scheduled_kw":
                    sharing_result[
                        "total_scheduled_sharing_kw"
                    ],

                "sharing_received_kw":
                    sharing_result[
                        "total_received_sharing_kw"
                    ],

                "sharing_loss_kw":
                    sharing_result[
                        "total_sharing_loss_kw"
                    ],
            },

            "constraints": {
                "all_power_balanced":
                    all_power_balanced,

                "all_transformers_feasible":
                    all_transformers_feasible,

                "total_power_balance_violation_kw":
                    total_balance_violation_kw,

                "total_transformer_violation_kw":
                    total_transformer_violation_kw,
            },
        }

        self.last_result = result

        return result

    # ========================================================
    # LOCAL STATES
    # ========================================================

    def get_local_states(
        self,
    ) -> List[Dict[str, object]]:
        """
        Return state of each local microgrid.

        These local states will later feed the five local FC-HMARL
        agents.
        """

        return [
            microgrid.get_state()
            for microgrid in self.microgrids
        ]

    # ========================================================
    # GLOBAL VPP STATE
    # ========================================================

    def get_global_state(
        self,
    ) -> Dict[str, object]:
        """
        Return a system-level VPP state.

        This will later be extended by the forecast-confidence layer
        before being passed to the upper-level coordinator.
        """

        local_states = (
            self.get_local_states()
        )

        return {
            "time":
                self.current_time,

            "step":
                self.current_step,

            "num_microgrids":
                self.num_microgrids,

            "local_states":
                local_states,

            "sharing_state":
                self.energy_sharing_network
                .get_state(),

            "last_result":
                self.last_result,
        }

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        bess_initial_socs: Optional[
            Iterable[float]
        ] = None,
    ) -> Dict[str, object]:
        """
        Reset the complete VPP environment.
        """

        if bess_initial_socs is None:

            socs = [
                None
                for _ in range(
                    self.num_microgrids
                )
            ]

        else:

            socs = list(
                bess_initial_socs
            )

            if len(socs) != self.num_microgrids:
                raise ValueError(
                    "bess_initial_socs must contain one "
                    "SOC value per microgrid."
                )

        for index, microgrid in enumerate(
            self.microgrids
        ):

            microgrid.reset(
                bess_soc=socs[index]
            )

        self.energy_sharing_network.reset()

        self.current_step = 0
        self.current_time = 0.0
        self.last_result = {}

        return self.get_global_state()

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(
        self,
    ) -> str:

        return (
            f"VPPEnvironment("
            f"name='{self.name}', "
            f"microgrids={self.num_microgrids}, "
            f"episode_length={self.episode_length_hours})"
        )
