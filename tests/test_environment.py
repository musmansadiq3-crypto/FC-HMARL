import pytest
from environment.bess import (
    BESSParameters,
    BatteryEnergyStorageSystem,
)
from environment.pv import (
    PVParameters,
    PhotovoltaicSystem,
)
from environment.ev_fleet import (
    EVFleet,
    EVFleetParameters,
    EVRecord,
)
from environment.market import (
    ElectricityMarket,
    MarketParameters,
)
from environment.microgrid import (
    Microgrid,
    MicrogridParameters,
)
# ============================================================
# FIXTURE
# ============================================================
@pytest.fixture
def microgrid():

    pv = PhotovoltaicSystem(
        PVParameters(
            rated_capacity_kw=500.0,
            efficiency=1.0,
            critical_irradiance_w_m2=200.0,
            stc_irradiance_w_m2=1000.0,
        ),
        name="MG1_PV",
    )

    bess = BatteryEnergyStorageSystem(
        BESSParameters(
            capacity_kwh=1000.0,
            rated_power_kw=250.0,
            charging_efficiency=0.95,
            discharging_efficiency=0.95,
            self_discharge_rate=0.001,
            minimum_soc=0.20,
            maximum_soc=0.95,
            time_step_hours=1.0,
        ),
        initial_soc=0.60,
        name="MG1_BESS",
    )

    ev_parameters = EVFleetParameters(
        number_of_evs=2,
        maximum_aggregate_charging_power_kw=20.0,
        time_step_hours=1.0,
    )

    evs = [
        EVRecord(
            ev_id="EV1",
            arrival_time=8.0,
            departure_time=12.0,
            charging_power_kw=7.0,
        ),
        EVRecord(
            ev_id="EV2",
            arrival_time=9.0,
            departure_time=15.0,
            charging_power_kw=11.0,
        ),
    ]

    ev_fleet = EVFleet(
        parameters=ev_parameters,
        vehicles=evs,
        name="MG1_EV",
    )

    market = ElectricityMarket(
        MarketParameters(
            minimum_buy_price_usd_per_kwh=0.12,
            maximum_buy_price_usd_per_kwh=0.32,
            minimum_sell_price_usd_per_kwh=0.08,
            maximum_sell_price_usd_per_kwh=0.24,
            reserve_price_usd_per_kwh=0.05,
            time_step_hours=1.0,
        ),
        name="MG1_Market",
    )

    parameters = MicrogridParameters(
        name="MG1",
        peak_load_kw=750.0,
        transformer_rating_kva=1000.0,
        power_factor=1.0,
        balance_tolerance_kw=1e-6,
    )

    return Microgrid(
        parameters=parameters,
        pv_system=pv,
        bess=bess,
        ev_fleet=ev_fleet,
        market=market,
    )
# ============================================================
# BASIC CONFIGURATION
# ============================================================

def test_microgrid_name(microgrid):

    assert microgrid.name == "MG1"
def test_peak_load(microgrid):

    assert microgrid.peak_load_kw == pytest.approx(
        750.0
    )
def test_transformer_rating(microgrid):

    assert (
        microgrid.transformer_rating_kva
        == pytest.approx(1000.0)
    )

# ============================================================
# LOCAL NET POWER EQUATION
# ============================================================

def test_local_net_power_equation():

    result = Microgrid.calculate_local_net_power(
        pv_power_kw=500.0,
        bess_power_kw=100.0,
        load_kw=400.0,
        ev_power_kw=50.0,
    )

    # 500 + 100 - 400 - 50 = 150

    assert result == pytest.approx(
        150.0
    )


def test_local_net_power_deficit():

    result = Microgrid.calculate_local_net_power(
        pv_power_kw=100.0,
        bess_power_kw=0.0,
        load_kw=400.0,
        ev_power_kw=50.0,
    )

    assert result == pytest.approx(
        -350.0
    )


# ============================================================
# GRID POWER FOR BALANCE
# ============================================================

def test_required_grid_power_for_deficit():

    result = Microgrid.calculate_required_grid_power(
        net_local_power_kw=-300.0,
        incoming_sharing_kw=100.0,
        outgoing_sharing_kw=0.0,
    )

    assert result == pytest.approx(
        200.0
    )


def test_required_grid_power_for_surplus():

    result = Microgrid.calculate_required_grid_power(
        net_local_power_kw=300.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=100.0,
    )

    assert result == pytest.approx(
        -200.0
    )


# ============================================================
# COMPLETE MICROGRID STEP
# ============================================================

def test_complete_step_balances_automatically(microgrid):

    result = microgrid.step(
        time=10.0,
        load_kw=400.0,
        irradiance_w_m2=500.0,
        bess_requested_power_kw=0.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=0.0,
        buy_price_usd_per_kwh=0.20,
        sell_price_usd_per_kwh=0.10,
        reserve_power_kw=0.0,
    )

    assert result[
        "constraints"
    ][
        "power_balance_satisfied"
    ]


def test_step_pv_power(microgrid):

    result = microgrid.step(
        time=10.0,
        load_kw=400.0,
        irradiance_w_m2=500.0,
        bess_requested_power_kw=0.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=0.0,
        buy_price_usd_per_kwh=0.20,
        sell_price_usd_per_kwh=0.10,
    )

    assert result[
        "pv_power_kw"
    ] == pytest.approx(
        250.0
    )


def test_step_ev_power(microgrid):

    result = microgrid.step(
        time=10.0,
        load_kw=400.0,
        irradiance_w_m2=500.0,
        bess_requested_power_kw=0.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=0.0,
        buy_price_usd_per_kwh=0.20,
        sell_price_usd_per_kwh=0.10,
    )

    assert result[
        "ev_power_kw"
    ] == pytest.approx(
        18.0
    )


def test_grid_import_for_deficit(microgrid):

    result = microgrid.step(
        time=10.0,
        load_kw=500.0,
        irradiance_w_m2=200.0,
        bess_requested_power_kw=0.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=0.0,
        buy_price_usd_per_kwh=0.20,
        sell_price_usd_per_kwh=0.10,
    )

    assert result[
        "grid_power_kw"
    ] > 0.0


def test_grid_export_for_surplus(microgrid):

    result = microgrid.step(
        time=7.0,
        load_kw=100.0,
        irradiance_w_m2=1000.0,
        bess_requested_power_kw=0.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=0.0,
        buy_price_usd_per_kwh=0.20,
        sell_price_usd_per_kwh=0.10,
    )

    assert result[
        "grid_power_kw"
    ] < 0.0


# ============================================================
# SHARING EFFECT
# ============================================================

def test_incoming_sharing_reduces_grid_import(microgrid):

    no_sharing = microgrid.step(
        time=10.0,
        load_kw=500.0,
        irradiance_w_m2=200.0,
        bess_requested_power_kw=0.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=0.0,
        buy_price_usd_per_kwh=0.20,
        sell_price_usd_per_kwh=0.10,
    )

    grid_without = no_sharing[
        "grid_power_kw"
    ]

    microgrid.reset(
        bess_soc=0.60
    )

    with_sharing = microgrid.step(
        time=10.0,
        load_kw=500.0,
        irradiance_w_m2=200.0,
        bess_requested_power_kw=0.0,
        incoming_sharing_kw=100.0,
        outgoing_sharing_kw=0.0,
        buy_price_usd_per_kwh=0.20,
        sell_price_usd_per_kwh=0.10,
    )

    grid_with = with_sharing[
        "grid_power_kw"
    ]

    assert grid_with < grid_without


def test_outgoing_sharing_increases_required_grid_import(
    microgrid,
):

    base = microgrid.step(
        time=10.0,
        load_kw=500.0,
        irradiance_w_m2=200.0,
        bess_requested_power_kw=0.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=0.0,
        buy_price_usd_per_kwh=0.20,
        sell_price_usd_per_kwh=0.10,
    )

    base_grid = base[
        "grid_power_kw"
    ]

    microgrid.reset(
        bess_soc=0.60
    )

    shared = microgrid.step(
        time=10.0,
        load_kw=500.0,
        irradiance_w_m2=200.0,
        bess_requested_power_kw=0.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=50.0,
        buy_price_usd_per_kwh=0.20,
        sell_price_usd_per_kwh=0.10,
    )

    assert shared[
        "grid_power_kw"
    ] > base_grid


# ============================================================
# BESS INTEGRATION
# ============================================================

def test_bess_discharge_reduces_grid_import(microgrid):

    base = microgrid.step(
        time=10.0,
        load_kw=500.0,
        irradiance_w_m2=200.0,
        bess_requested_power_kw=0.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=0.0,
        buy_price_usd_per_kwh=0.20,
        sell_price_usd_per_kwh=0.10,
    )

    base_grid = base[
        "grid_power_kw"
    ]

    microgrid.reset(
        bess_soc=0.60
    )

    discharge = microgrid.step(
        time=10.0,
        load_kw=500.0,
        irradiance_w_m2=200.0,
        bess_requested_power_kw=100.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=0.0,
        buy_price_usd_per_kwh=0.20,
        sell_price_usd_per_kwh=0.10,
    )

    assert discharge[
        "grid_power_kw"
    ] < base_grid


def test_bess_charge_increases_grid_import(microgrid):

    base = microgrid.step(
        time=10.0,
        load_kw=500.0,
        irradiance_w_m2=200.0,
        bess_requested_power_kw=0.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=0.0,
        buy_price_usd_per_kwh=0.20,
        sell_price_usd_per_kwh=0.10,
    )

    base_grid = base[
        "grid_power_kw"
    ]

    microgrid.reset(
        bess_soc=0.60
    )

    charge = microgrid.step(
        time=10.0,
        load_kw=500.0,
        irradiance_w_m2=200.0,
        bess_requested_power_kw=-100.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=0.0,
        buy_price_usd_per_kwh=0.20,
        sell_price_usd_per_kwh=0.10,
    )

    assert charge[
        "grid_power_kw"
    ] > base_grid


# ============================================================
# MARKET INTEGRATION
# ============================================================

def test_import_creates_purchase_cost(microgrid):

    result = microgrid.step(
        time=10.0,
        load_kw=600.0,
        irradiance_w_m2=200.0,
        bess_requested_power_kw=0.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=0.0,
        buy_price_usd_per_kwh=0.20,
        sell_price_usd_per_kwh=0.10,
    )

    assert result[
        "market"
    ][
        "grid_purchase_cost_usd"
    ] > 0.0


def test_export_creates_sale_revenue(microgrid):

    result = microgrid.step(
        time=7.0,
        load_kw=100.0,
        irradiance_w_m2=1000.0,
        bess_requested_power_kw=0.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=0.0,
        buy_price_usd_per_kwh=0.20,
        sell_price_usd_per_kwh=0.10,
    )

    assert result[
        "market"
    ][
        "grid_sale_revenue_usd"
    ] > 0.0


# ============================================================
# MANUAL GRID POWER
# ============================================================

def test_manual_incorrect_grid_power_detected(microgrid):

    result = microgrid.step(
        time=10.0,
        load_kw=500.0,
        irradiance_w_m2=200.0,
        bess_requested_power_kw=0.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=0.0,
        buy_price_usd_per_kwh=0.20,
        sell_price_usd_per_kwh=0.10,
        grid_power_kw=0.0,
    )

    assert not result[
        "constraints"
    ][
        "power_balance_satisfied"
    ]


# ============================================================
# LOAD VALIDATION
# ============================================================

def test_negative_load_rejected(microgrid):

    with pytest.raises(ValueError):

        microgrid.step(
            time=10.0,
            load_kw=-1.0,
            irradiance_w_m2=500.0,
            bess_requested_power_kw=0.0,
            incoming_sharing_kw=0.0,
            outgoing_sharing_kw=0.0,
            buy_price_usd_per_kwh=0.20,
            sell_price_usd_per_kwh=0.10,
        )


def test_load_above_peak_rejected(microgrid):

    with pytest.raises(ValueError):

        microgrid.step(
            time=10.0,
            load_kw=800.0,
            irradiance_w_m2=500.0,
            bess_requested_power_kw=0.0,
            incoming_sharing_kw=0.0,
            outgoing_sharing_kw=0.0,
            buy_price_usd_per_kwh=0.20,
            sell_price_usd_per_kwh=0.10,
        )


# ============================================================
# SHARING INPUT VALIDATION
# ============================================================

def test_negative_incoming_sharing_rejected(microgrid):

    with pytest.raises(ValueError):

        microgrid.step(
            time=10.0,
            load_kw=400.0,
            irradiance_w_m2=500.0,
            bess_requested_power_kw=0.0,
            incoming_sharing_kw=-1.0,
            outgoing_sharing_kw=0.0,
            buy_price_usd_per_kwh=0.20,
            sell_price_usd_per_kwh=0.10,
        )


def test_negative_outgoing_sharing_rejected(microgrid):

    with pytest.raises(ValueError):

        microgrid.step(
            time=10.0,
            load_kw=400.0,
            irradiance_w_m2=500.0,
            bess_requested_power_kw=0.0,
            incoming_sharing_kw=0.0,
            outgoing_sharing_kw=-1.0,
            buy_price_usd_per_kwh=0.20,
            sell_price_usd_per_kwh=0.10,
        )


# ============================================================
# STATE / RESET
# ============================================================

def test_get_state(microgrid):

    microgrid.step(
        time=10.0,
        load_kw=400.0,
        irradiance_w_m2=500.0,
        bess_requested_power_kw=0.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=0.0,
        buy_price_usd_per_kwh=0.20,
        sell_price_usd_per_kwh=0.10,
    )

    state = microgrid.get_state()

    assert state["microgrid"] == "MG1"

    assert state[
        "power_balance_residual_kw"
    ] == pytest.approx(
        0.0
    )


def test_reset(microgrid):

    microgrid.step(
        time=10.0,
        load_kw=400.0,
        irradiance_w_m2=500.0,
        bess_requested_power_kw=100.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=0.0,
        buy_price_usd_per_kwh=0.20,
        sell_price_usd_per_kwh=0.10,
    )

    microgrid.reset(
        bess_soc=0.60
    )

    state = microgrid.get_state()

    assert state["time"] == pytest.approx(
        0.0
    )

    assert state["grid_power_kw"] == pytest.approx(
        0.0
    )

    assert microgrid.bess.soc == pytest.approx(
        0.60
    )


# ============================================================
# PARAMETER VALIDATION
# ============================================================

def test_invalid_peak_load():

    parameters = MicrogridParameters(
        name="MG1",
        peak_load_kw=-1.0,
        transformer_rating_kva=1000.0,
    )

    with pytest.raises(ValueError):

        parameters.validate()


def test_invalid_transformer_rating():

    parameters = MicrogridParameters(
        name="MG1",
        peak_load_kw=750.0,
        transformer_rating_kva=-1.0,
    )

    with pytest.raises(ValueError):

        parameters.validate()


def test_invalid_power_factor():

    parameters = MicrogridParameters(
        name="MG1",
        peak_load_kw=750.0,
        transformer_rating_kva=1000.0,
        power_factor=1.2,
    )

    with pytest.raises(ValueError):

        parameters.validate()


# ============================================================
# REPRESENTATION
# ============================================================

def test_repr(microgrid):

    text = repr(
        microgrid
    )

    assert "MG1" in text
    assert "750.0" in text
    assert "1000.0" in text
