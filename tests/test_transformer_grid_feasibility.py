import numpy as np
import pytest

from train_real_fc_hmarl_reserve_corrected_v2 import build_real_microgrid, EV_COUNTS


def make_mg():
    # Zero-based index 2 = manuscript MG3:
    # peak load 700 kW, transformer 1000 kVA, BESS 250 kW / 900 kWh.
    return build_real_microgrid(2)


def zero_ev_requests_for_mg3():
    # Isolate PCC/transformer behavior by explicitly requesting zero EV charging.
    return np.zeros(int(EV_COUNTS[2]), dtype=float)


def test_auto_grid_import_is_projected_to_combined_pcc_limit():
    mg = make_mg()

    r = mg.step(
        time=0.0,
        load_kw=700.0,
        irradiance_w_m2=0.0,
        bess_requested_power_kw=-250.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=250.0,
        buy_price_usd_per_kwh=0.2,
        sell_price_usd_per_kwh=0.1,
        ev_requested_charging_powers_kw=zero_ev_requests_for_mg3(),
    )

    # Requested grid closes the unconstrained balance:
    # net_local = -950 kW, outgoing sharing = 250 kW
    # requested grid = 1200 kW.
    assert r["requested_grid_power_kw"] == pytest.approx(1200.0)

    # PCC constraint is |P_grid + P_share,out| <= 1000 kW.
    # With 250 kW outgoing sharing, feasible grid import is therefore 750 kW.
    assert r["grid_power_kw"] == pytest.approx(750.0)
    assert r["grid_power_curtailed_kw"] == pytest.approx(450.0)

    assert r["constraints"]["pcc_exchange_kw"] == pytest.approx(1000.0)
    assert r["constraints"]["transformer_limit_satisfied"]
    assert r["constraints"]["transformer_violation_kw"] == pytest.approx(0.0)


def test_unserved_balance_remains_visible_after_pcc_projection():
    mg = make_mg()

    r = mg.step(
        time=0.0,
        load_kw=700.0,
        irradiance_w_m2=0.0,
        bess_requested_power_kw=-250.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=250.0,
        buy_price_usd_per_kwh=0.2,
        sell_price_usd_per_kwh=0.1,
        ev_requested_charging_powers_kw=zero_ev_requests_for_mg3(),
    )

    assert not r["constraints"]["power_balance_satisfied"]
    assert r["constraints"]["power_balance_violation_kw"] == pytest.approx(450.0)


def test_explicit_grid_import_is_projected_to_pcc_limit():
    mg = make_mg()

    r = mg.step(
        time=0.0,
        load_kw=100.0,
        irradiance_w_m2=0.0,
        bess_requested_power_kw=0.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=0.0,
        buy_price_usd_per_kwh=0.2,
        sell_price_usd_per_kwh=0.1,
        ev_requested_charging_powers_kw=zero_ev_requests_for_mg3(),
        grid_power_kw=1500.0,
    )

    assert r["requested_grid_power_kw"] == pytest.approx(1500.0)
    assert r["grid_power_kw"] == pytest.approx(1000.0)
    assert r["grid_power_curtailed_kw"] == pytest.approx(500.0)
    assert r["constraints"]["transformer_limit_satisfied"]


def test_explicit_grid_export_is_projected_symmetrically():
    mg = make_mg()

    r = mg.step(
        time=0.0,
        load_kw=100.0,
        irradiance_w_m2=1000.0,
        bess_requested_power_kw=0.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=0.0,
        buy_price_usd_per_kwh=0.2,
        sell_price_usd_per_kwh=0.1,
        ev_requested_charging_powers_kw=zero_ev_requests_for_mg3(),
        grid_power_kw=-1500.0,
    )

    assert r["requested_grid_power_kw"] == pytest.approx(-1500.0)
    assert r["grid_power_kw"] == pytest.approx(-1000.0)
    assert r["grid_power_curtailed_kw"] == pytest.approx(-500.0)
    assert r["constraints"]["transformer_limit_satisfied"]


def test_market_settlement_uses_feasible_grid_not_requested_grid():
    mg = make_mg()

    r = mg.step(
        time=0.0,
        load_kw=100.0,
        irradiance_w_m2=0.0,
        bess_requested_power_kw=0.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=0.0,
        buy_price_usd_per_kwh=0.2,
        sell_price_usd_per_kwh=0.1,
        ev_requested_charging_powers_kw=zero_ev_requests_for_mg3(),
        grid_power_kw=1500.0,
    )

    assert r["grid_power_kw"] == pytest.approx(1000.0)
    assert r["market"]["grid_purchase_cost_usd"] == pytest.approx(200.0)


def test_normal_feasible_grid_exchange_is_unchanged():
    mg = make_mg()

    r = mg.step(
        time=0.0,
        load_kw=400.0,
        irradiance_w_m2=0.0,
        bess_requested_power_kw=0.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=0.0,
        buy_price_usd_per_kwh=0.2,
        sell_price_usd_per_kwh=0.1,
        ev_requested_charging_powers_kw=zero_ev_requests_for_mg3(),
    )

    assert r["requested_grid_power_kw"] == pytest.approx(400.0)
    assert r["grid_power_kw"] == pytest.approx(400.0)
    assert r["grid_power_curtailed_kw"] == pytest.approx(0.0)
    assert r["constraints"]["power_balance_satisfied"]
    assert r["constraints"]["transformer_limit_satisfied"]


def test_outgoing_sharing_reduces_available_grid_import_headroom():
    mg = make_mg()

    r = mg.step(
        time=0.0,
        load_kw=100.0,
        irradiance_w_m2=0.0,
        bess_requested_power_kw=0.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=250.0,
        buy_price_usd_per_kwh=0.2,
        sell_price_usd_per_kwh=0.1,
        ev_requested_charging_powers_kw=zero_ev_requests_for_mg3(),
        grid_power_kw=1000.0,
    )

    # Because outgoing sharing is 250 kW, grid import may be at most 750 kW.
    assert r["grid_power_kw"] == pytest.approx(750.0)
    assert r["constraints"]["pcc_exchange_kw"] == pytest.approx(1000.0)
    assert r["constraints"]["transformer_limit_satisfied"]
