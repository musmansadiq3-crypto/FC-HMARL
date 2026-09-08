from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

# ============================================================
# MARKET PARAMETERS
# ============================================================

@dataclass
class MarketParameters:
        minimum_buy_price_usd_per_kwh: float = 0.12
    maximum_buy_price_usd_per_kwh: float = 0.32

    minimum_sell_price_usd_per_kwh: float = 0.08
    maximum_sell_price_usd_per_kwh: float = 0.24

    reserve_price_usd_per_kwh: float = 0.05

    time_step_hours: float = 1.0

    def validate(self) -> None:
        """Validate economic parameters."""

        if self.minimum_buy_price_usd_per_kwh < 0:
            raise ValueError(
                "minimum buy price cannot be negative."
            )

        if (
            self.maximum_buy_price_usd_per_kwh
            < self.minimum_buy_price_usd_per_kwh
        ):
            raise ValueError(
                "maximum buy price must be >= minimum buy price."
            )

        if self.minimum_sell_price_usd_per_kwh < 0:
            raise ValueError(
                "minimum sell price cannot be negative."
            )

        if (
            self.maximum_sell_price_usd_per_kwh
            < self.minimum_sell_price_usd_per_kwh
        ):
            raise ValueError(
                "maximum sell price must be >= minimum sell price."
            )

        if self.reserve_price_usd_per_kwh < 0:
            raise ValueError(
                "reserve price cannot be negative."
            )

        if self.time_step_hours <= 0:
            raise ValueError(
                "time_step_hours must be greater than zero."
            )


# ============================================================
# MARKET MODEL
# ============================================================

class ElectricityMarket:
    """
    Utility-grid and reserve-market accounting model.
    """

    def __init__(
        self,
        parameters: MarketParameters,
        name: str = "Electricity_Market",
    ) -> None:

        parameters.validate()

        self.parameters = parameters
        self.name = name

        self.last_buy_price = 0.0
        self.last_sell_price = 0.0
        self.last_grid_power_kw = 0.0

        self.last_grid_purchase_cost = 0.0
        self.last_grid_sale_revenue = 0.0
        self.last_reserve_revenue = 0.0

    # ========================================================
    # BASIC PROPERTIES
    # ========================================================

    @property
    def time_step_hours(self) -> float:
        return self.parameters.time_step_hours

    @property
    def reserve_price_usd_per_kwh(self) -> float:
        return self.parameters.reserve_price_usd_per_kwh

    # ========================================================
    # PRICE VALIDATION
    # ========================================================

    def validate_buy_price(
        self,
        price_usd_per_kwh: float,
    ) -> float:
        """
        Validate buy price inside configured manuscript range.
        """

        price = float(
            price_usd_per_kwh
        )

        if not np.isfinite(price):
            raise ValueError(
                "Buy price must be finite."
            )

        if not (
            self.parameters.minimum_buy_price_usd_per_kwh
            <= price
            <= self.parameters.maximum_buy_price_usd_per_kwh
        ):
            raise ValueError(
                "Buy price is outside the configured range."
            )

        return price

    def validate_sell_price(
        self,
        price_usd_per_kwh: float,
    ) -> float:
        """
        Validate sell price inside configured manuscript range.
        """

        price = float(
            price_usd_per_kwh
        )

        if not np.isfinite(price):
            raise ValueError(
                "Sell price must be finite."
            )

        if not (
            self.parameters.minimum_sell_price_usd_per_kwh
            <= price
            <= self.parameters.maximum_sell_price_usd_per_kwh
        ):
            raise ValueError(
                "Sell price is outside the configured range."
            )

        return price

    # ========================================================
    # GRID PURCHASE
    # ========================================================

    def grid_purchase_cost(
        self,
        import_power_kw: float,
        buy_price_usd_per_kwh: float,
    ) -> float:
        """
        Calculate utility-grid purchase cost.

        Cost =
            import_power
            * time_step
            * buy_price
        """

        import_power_kw = float(
            import_power_kw
        )

        if import_power_kw < 0:
            raise ValueError(
                "import_power_kw must be non-negative."
            )

        buy_price = self.validate_buy_price(
            buy_price_usd_per_kwh
        )

        energy_kwh = (
            import_power_kw
            * self.time_step_hours
        )

        return float(
            energy_kwh
            * buy_price
        )

    # ========================================================
    # GRID SALE
    # ========================================================

    def grid_sale_revenue(
        self,
        export_power_kw: float,
        sell_price_usd_per_kwh: float,
    ) -> float:
        """
        Calculate revenue from exporting energy to the utility grid.
        """

        export_power_kw = float(
            export_power_kw
        )

        if export_power_kw < 0:
            raise ValueError(
                "export_power_kw must be non-negative."
            )

        sell_price = self.validate_sell_price(
            sell_price_usd_per_kwh
        )

        energy_kwh = (
            export_power_kw
            * self.time_step_hours
        )

        return float(
            energy_kwh
            * sell_price
        )

    # ========================================================
    # RESERVE MARKET
    # ========================================================

    def reserve_revenue(
        self,
        reserve_power_kw: float,
    ) -> float:
        """
        Calculate reserve-market revenue from physically feasible reserve.

        Physical BESS power-headroom and SOC/energy feasibility are enforced
        upstream by the FC-HMARL action mapper and again by VPPEnvironment.
        This market object performs accounting only.

        Revenue =
            reserve_power
            * time_step
            * reserve_price
        """

        reserve_power_kw = float(
            reserve_power_kw
        )

        if reserve_power_kw < 0:
            raise ValueError(
                "reserve_power_kw must be non-negative."
            )

        energy_equivalent_kwh = (
            reserve_power_kw
            * self.time_step_hours
        )

        return float(
            energy_equivalent_kwh
            * self.reserve_price_usd_per_kwh
        )

    # ========================================================
    # GRID INTERACTION
    # ========================================================

    def evaluate_grid_interaction(
        self,
        grid_power_kw: float,
        buy_price_usd_per_kwh: float,
        sell_price_usd_per_kwh: float,
    ) -> Dict[str, float]:
        """
        Evaluate grid import/export economics.

        Sign convention:

            grid_power > 0 -> grid import
            grid_power < 0 -> grid export
        """

        grid_power_kw = float(
            grid_power_kw
        )

        if not np.isfinite(
            grid_power_kw
        ):
            raise ValueError(
                "grid_power_kw must be finite."
            )

        buy_price = self.validate_buy_price(
            buy_price_usd_per_kwh
        )

        sell_price = self.validate_sell_price(
            sell_price_usd_per_kwh
        )

        import_power_kw = max(
            grid_power_kw,
            0.0,
        )

        export_power_kw = max(
            -grid_power_kw,
            0.0,
        )

        purchase_cost = self.grid_purchase_cost(
            import_power_kw=import_power_kw,
            buy_price_usd_per_kwh=buy_price,
        )

        sale_revenue = self.grid_sale_revenue(
            export_power_kw=export_power_kw,
            sell_price_usd_per_kwh=sell_price,
        )

        net_grid_cost = (
            purchase_cost
            - sale_revenue
        )

        return {
            "grid_power_kw":
                grid_power_kw,

            "import_power_kw":
                import_power_kw,

            "export_power_kw":
                export_power_kw,

            "buy_price_usd_per_kwh":
                buy_price,

            "sell_price_usd_per_kwh":
                sell_price,

            "grid_purchase_cost_usd":
                purchase_cost,

            "grid_sale_revenue_usd":
                sale_revenue,

            "net_grid_cost_usd":
                net_grid_cost,
        }

    # ========================================================
    # COMPLETE MARKET STEP
    # ========================================================

    def step(
        self,
        grid_power_kw: float,
        buy_price_usd_per_kwh: float,
        sell_price_usd_per_kwh: float,
        reserve_power_kw: float = 0.0,
    ) -> Dict[str, float]:
        """
        Evaluate one market interval.

        Includes:
        - energy purchase
        - energy sale
        - reserve revenue
        """

        grid_result = (
            self.evaluate_grid_interaction(
                grid_power_kw=grid_power_kw,
                buy_price_usd_per_kwh=(
                    buy_price_usd_per_kwh
                ),
                sell_price_usd_per_kwh=(
                    sell_price_usd_per_kwh
                ),
            )
        )

        reserve_income = self.reserve_revenue(
            reserve_power_kw
        )

        total_market_revenue = (
            grid_result[
                "grid_sale_revenue_usd"
            ]
            + reserve_income
        )

        total_market_cost = (
            grid_result[
                "grid_purchase_cost_usd"
            ]
        )

        net_market_profit = (
            total_market_revenue
            - total_market_cost
        )

        self.last_buy_price = float(
            buy_price_usd_per_kwh
        )

        self.last_sell_price = float(
            sell_price_usd_per_kwh
        )

        self.last_grid_power_kw = float(
            grid_power_kw
        )

        self.last_grid_purchase_cost = (
            grid_result[
                "grid_purchase_cost_usd"
            ]
        )

        self.last_grid_sale_revenue = (
            grid_result[
                "grid_sale_revenue_usd"
            ]
        )

        self.last_reserve_revenue = (
            reserve_income
        )

        return {
            **grid_result,

            "reserve_power_kw":
                float(reserve_power_kw),

            "reserve_revenue_usd":
                reserve_income,

            "total_market_revenue_usd":
                total_market_revenue,

            "total_market_cost_usd":
                total_market_cost,

            "net_market_profit_usd":
                net_market_profit,
        }

    # ========================================================
    # RESET
    # ========================================================

    def reset(self) -> None:
        """Reset stored market outputs."""

        self.last_buy_price = 0.0
        self.last_sell_price = 0.0
        self.last_grid_power_kw = 0.0

        self.last_grid_purchase_cost = 0.0
        self.last_grid_sale_revenue = 0.0
        self.last_reserve_revenue = 0.0

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(
        self,
    ) -> str:

        return (
            f"ElectricityMarket("
            f"name='{self.name}', "
            f"buy_range="
            f"[{self.parameters.minimum_buy_price_usd_per_kwh:.2f}, "
            f"{self.parameters.maximum_buy_price_usd_per_kwh:.2f}], "
            f"sell_range="
            f"[{self.parameters.minimum_sell_price_usd_per_kwh:.2f}, "
            f"{self.parameters.maximum_sell_price_usd_per_kwh:.2f}])"
        )
