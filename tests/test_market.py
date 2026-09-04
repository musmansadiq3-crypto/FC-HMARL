"""
Unit tests for the FC-HMARL electricity-market model.

These tests validate:

1. Grid purchase cost
2. Grid sale revenue
3. Reserve-market revenue
4. Import/export sign convention
5. Complete market step
6. Market parameter validation

The tested price ranges match the manuscript configuration.
"""

import pytest

from environment.market import (
    ElectricityMarket,
    MarketParameters,
)


# ============================================================
# FIXTURE
# ============================================================

@pytest.fixture
def market():

    parameters = MarketParameters(
        minimum_buy_price_usd_per_kwh=0.12,
        maximum_buy_price_usd_per_kwh=0.32,
        minimum_sell_price_usd_per_kwh=0.08,
        maximum_sell_price_usd_per_kwh=0.24,
        reserve_price_usd_per_kwh=0.05,
        time_step_hours=1.0,
    )

    return ElectricityMarket(
        parameters=parameters,
        name="Test_Market",
    )


# ============================================================
# PURCHASE COST
# ============================================================

def test_grid_purchase_cost(market):
    """
    100 kW for one hour at $0.20/kWh:

        cost = 100 * 1 * 0.20
             = $20
    """

    cost = market.grid_purchase_cost(
        import_power_kw=100.0,
        buy_price_usd_per_kwh=0.20,
    )

    assert cost == pytest.approx(
        20.0
    )


def test_zero_import_cost(market):

    cost = market.grid_purchase_cost(
        import_power_kw=0.0,
        buy_price_usd_per_kwh=0.20,
    )

    assert cost == pytest.approx(
        0.0
    )


def test_negative_import_rejected(market):

    with pytest.raises(ValueError):

        market.grid_purchase_cost(
            import_power_kw=-10.0,
            buy_price_usd_per_kwh=0.20,
        )


# ============================================================
# GRID SALE
# ============================================================

def test_grid_sale_revenue(market):
    """
    100 kW exported for one hour at $0.10/kWh:

        revenue = $10
    """

    revenue = market.grid_sale_revenue(
        export_power_kw=100.0,
        sell_price_usd_per_kwh=0.10,
    )

    assert revenue == pytest.approx(
        10.0
    )


def test_zero_export_revenue(market):

    revenue = market.grid_sale_revenue(
        export_power_kw=0.0,
        sell_price_usd_per_kwh=0.10,
    )

    assert revenue == pytest.approx(
        0.0
    )


def test_negative_export_rejected(market):

    with pytest.raises(ValueError):

        market.grid_sale_revenue(
            export_power_kw=-10.0,
            sell_price_usd_per_kwh=0.10,
        )


# ============================================================
# RESERVE
# ============================================================

def test_reserve_revenue(market):
    """
    Reserve = 100 kW
    Reserve price = $0.05/kWh

    Revenue = $5 for one hour.
    """

    revenue = market.reserve_revenue(
        reserve_power_kw=100.0
    )

    assert revenue == pytest.approx(
        5.0
    )


def test_zero_reserve_revenue(market):

    assert market.reserve_revenue(
        0.0
    ) == pytest.approx(
        0.0
    )


def test_negative_reserve_rejected(market):

    with pytest.raises(ValueError):

        market.reserve_revenue(
            -1.0
        )


# ============================================================
# IMPORT INTERACTION
# ============================================================

def test_positive_grid_power_means_import(market):

    result = market.evaluate_grid_interaction(
        grid_power_kw=100.0,
        buy_price_usd_per_kwh=0.20,
        sell_price_usd_per_kwh=0.10,
    )

    assert result[
        "import_power_kw"
    ] == pytest.approx(
        100.0
    )

    assert result[
        "export_power_kw"
    ] == pytest.approx(
        0.0
    )


def test_import_purchase_cost(market):

    result = market.evaluate_grid_interaction(
        grid_power_kw=100.0,
        buy_price_usd_per_kwh=0.20,
        sell_price_usd_per_kwh=0.10,
    )

    assert result[
        "grid_purchase_cost_usd"
    ] == pytest.approx(
        20.0
    )

    assert result[
        "grid_sale_revenue_usd"
    ] == pytest.approx(
        0.0
    )


# ============================================================
# EXPORT INTERACTION
# ============================================================

def test_negative_grid_power_means_export(market):

    result = market.evaluate_grid_interaction(
        grid_power_kw=-100.0,
        buy_price_usd_per_kwh=0.20,
        sell_price_usd_per_kwh=0.10,
    )

    assert result[
        "import_power_kw"
    ] == pytest.approx(
        0.0
    )

    assert result[
        "export_power_kw"
    ] == pytest.approx(
        100.0
    )


def test_export_sale_revenue(market):

    result = market.evaluate_grid_interaction(
        grid_power_kw=-100.0,
        buy_price_usd_per_kwh=0.20,
        sell_price_usd_per_kwh=0.10,
    )

    assert result[
        "grid_sale_revenue_usd"
    ] == pytest.approx(
        10.0
    )

    assert result[
        "grid_purchase_cost_usd"
    ] == pytest.approx(
        0.0
    )


# ============================================================
# ZERO GRID INTERACTION
# ============================================================

def test_zero_grid_power(market):

    result = market.evaluate_grid_interaction(
        grid_power_kw=0.0,
        buy_price_usd_per_kwh=0.20,
        sell_price_usd_per_kwh=0.10,
    )

    assert result[
        "import_power_kw"
    ] == pytest.approx(
        0.0
    )

    assert result[
        "export_power_kw"
    ] == pytest.approx(
        0.0
    )

    assert result[
        "net_grid_cost_usd"
    ] == pytest.approx(
        0.0
    )


# ============================================================
# COMPLETE MARKET STEP
# ============================================================

def test_complete_step_import_with_reserve(market):
    """
    Import:
        100 kW * $0.20 = $20 cost

    Reserve:
        50 kW * $0.05 = $2.50 revenue

    Net market profit:
        2.50 - 20 = -17.50
    """

    result = market.step(
        grid_power_kw=100.0,
        buy_price_usd_per_kwh=0.20,
        sell_price_usd_per_kwh=0.10,
        reserve_power_kw=50.0,
    )

    assert result[
        "grid_purchase_cost_usd"
    ] == pytest.approx(
        20.0
    )

    assert result[
        "reserve_revenue_usd"
    ] == pytest.approx(
        2.5
    )

    assert result[
        "net_market_profit_usd"
    ] == pytest.approx(
        -17.5
    )


def test_complete_step_export_with_reserve(market):
    """
    Export:
        100 kW * 0.10 = $10

    Reserve:
        50 kW * 0.05 = $2.50

    Total revenue:
        $12.50
    """

    result = market.step(
        grid_power_kw=-100.0,
        buy_price_usd_per_kwh=0.20,
        sell_price_usd_per_kwh=0.10,
        reserve_power_kw=50.0,
    )

    assert result[
        "total_market_revenue_usd"
    ] == pytest.approx(
        12.5
    )

    assert result[
        "total_market_cost_usd"
    ] == pytest.approx(
        0.0
    )

    assert result[
        "net_market_profit_usd"
    ] == pytest.approx(
        12.5
    )


# ============================================================
# PRICE VALIDATION
# ============================================================

def test_buy_price_at_minimum_allowed(market):

    assert market.validate_buy_price(
        0.12
    ) == pytest.approx(
        0.12
    )


def test_buy_price_at_maximum_allowed(market):

    assert market.validate_buy_price(
        0.32
    ) == pytest.approx(
        0.32
    )


def test_buy_price_below_range_rejected(market):

    with pytest.raises(ValueError):

        market.validate_buy_price(
            0.10
        )


def test_buy_price_above_range_rejected(market):

    with pytest.raises(ValueError):

        market.validate_buy_price(
            0.40
        )


def test_sell_price_at_minimum_allowed(market):

    assert market.validate_sell_price(
        0.08
    ) == pytest.approx(
        0.08
    )


def test_sell_price_at_maximum_allowed(market):

    assert market.validate_sell_price(
        0.24
    ) == pytest.approx(
        0.24
    )


def test_sell_price_below_range_rejected(market):

    with pytest.raises(ValueError):

        market.validate_sell_price(
            0.05
        )


def test_sell_price_above_range_rejected(market):

    with pytest.raises(ValueError):

        market.validate_sell_price(
            0.30
        )


# ============================================================
# PARAMETER VALIDATION
# ============================================================

def test_invalid_buy_price_range():

    parameters = MarketParameters(
        minimum_buy_price_usd_per_kwh=0.40,
        maximum_buy_price_usd_per_kwh=0.20,
    )

    with pytest.raises(ValueError):

        ElectricityMarket(
            parameters
        )


def test_invalid_sell_price_range():

    parameters = MarketParameters(
        minimum_sell_price_usd_per_kwh=0.30,
        maximum_sell_price_usd_per_kwh=0.10,
    )

    with pytest.raises(ValueError):

        ElectricityMarket(
            parameters
        )


def test_negative_reserve_price_rejected():

    parameters = MarketParameters(
        reserve_price_usd_per_kwh=-0.05,
    )

    with pytest.raises(ValueError):

        ElectricityMarket(
            parameters
        )


def test_invalid_time_step():

    parameters = MarketParameters(
        time_step_hours=0.0,
    )

    with pytest.raises(ValueError):

        ElectricityMarket(
            parameters
        )


# ============================================================
# RESET
# ============================================================

def test_reset(market):

    market.step(
        grid_power_kw=100.0,
        buy_price_usd_per_kwh=0.20,
        sell_price_usd_per_kwh=0.10,
        reserve_power_kw=50.0,
    )

    market.reset()

    assert market.last_grid_power_kw == pytest.approx(
        0.0
    )

    assert market.last_grid_purchase_cost == pytest.approx(
        0.0
    )

    assert market.last_grid_sale_revenue == pytest.approx(
        0.0
    )

    assert market.last_reserve_revenue == pytest.approx(
        0.0
    )


# ============================================================
# REPRESENTATION
# ============================================================

def test_repr(market):

    text = repr(
        market
    )

    assert "Test_Market" in text
    assert "0.12" in text
    assert "0.32" in text
    assert "0.08" in text
    assert "0.24" in text