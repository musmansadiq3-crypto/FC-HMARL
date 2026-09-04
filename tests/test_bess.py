"""
Unit tests for the FC-HMARL BESS model.
"""

import pytest

from environment.bess import (
    BESSParameters,
    BatteryEnergyStorageSystem,
)


@pytest.fixture
def battery():
    params = BESSParameters(
        capacity_kwh=1000,
        rated_power_kw=250,
        charging_efficiency=0.95,
        discharging_efficiency=0.95,
        self_discharge_rate=0.001,
        minimum_soc=0.20,
        maximum_soc=0.95,
        time_step_hours=1.0,
    )

    return BatteryEnergyStorageSystem(
        parameters=params,
        initial_soc=0.60,
        name="MG1_BESS",
    )


def test_initial_soc(battery):
    assert battery.soc == pytest.approx(0.60)


def test_charging_reduces_signed_power(battery):
    result = battery.step(-100.0)

    assert result["net_bess_power_kw"] < 0
    assert result["charging_power_kw"] > 0
    assert result["discharging_power_kw"] == 0


def test_discharging_positive_power(battery):
    result = battery.step(100.0)

    assert result["net_bess_power_kw"] > 0
    assert result["discharging_power_kw"] > 0
    assert result["charging_power_kw"] == 0


def test_charging_increases_soc(battery):
    old_soc = battery.soc

    result = battery.step(-100.0)

    assert result["new_soc"] > old_soc


def test_discharging_decreases_soc(battery):
    old_soc = battery.soc

    result = battery.step(100.0)

    assert result["new_soc"] < old_soc


def test_soc_never_exceeds_maximum(battery):
    battery.reset(0.94)

    for _ in range(20):
        battery.step(-250.0)

    assert battery.soc <= 0.95


def test_soc_never_falls_below_minimum(battery):
    battery.reset(0.21)

    for _ in range(20):
        battery.step(250.0)

    assert battery.soc >= 0.20


def test_power_does_not_exceed_rating(battery):
    result = battery.step(1000.0)

    assert abs(result["feasible_power_kw"]) <= 250.0


def test_charge_power_does_not_exceed_rating(battery):
    result = battery.step(-1000.0)

    assert abs(result["feasible_power_kw"]) <= 250.0


def test_idle_action(battery):
    old_soc = battery.soc

    result = battery.step(0.0)

    assert result["net_bess_power_kw"] == 0.0
    assert result["new_soc"] <= old_soc


def test_reset(battery):
    battery.step(100.0)

    battery.reset(0.70)

    assert battery.soc == pytest.approx(0.70)


def test_invalid_initial_soc():
    params = BESSParameters(
        capacity_kwh=1000,
        rated_power_kw=250,
        minimum_soc=0.20,
        maximum_soc=0.95,
    )

    with pytest.raises(ValueError):
        BatteryEnergyStorageSystem(
            parameters=params,
            initial_soc=0.99,
        )


def test_no_simultaneous_charge_and_discharge(battery):
    with pytest.raises(ValueError):
        battery.calculate_next_soc(
            charging_power_kw=100.0,
            discharging_power_kw=100.0,
        )