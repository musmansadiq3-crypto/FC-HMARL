"""
Unit tests for the FC-HMARL photovoltaic model.

These tests validate the piecewise PV generation model used in the
manuscript.

Important:
The manuscript provides the PV equation and the PV rated capacities,
but it does not provide numerical values for PV efficiency, critical
irradiance, or STC irradiance in the recovered configuration tables.

Therefore, the values used below for:
    efficiency = 1.0
    critical irradiance = 200 W/m^2
    STC irradiance = 1000 W/m^2

are TEST PARAMETERS only.

They are selected to make the manuscript equations easy to verify
analytically.
"""

import numpy as np
import pytest

from environment.pv import (
    PVParameters,
    PhotovoltaicSystem,
)


# ============================================================
# FIXTURE
# ============================================================

@pytest.fixture
def pv():
    """
    Create an MG1-type PV system for testing.

    Manuscript MG1 PV rated capacity:
        500 kW

    Test-only values:
        efficiency = 1.0
        critical irradiance = 200 W/m^2
        STC irradiance = 1000 W/m^2
    """

    parameters = PVParameters(
        rated_capacity_kw=500.0,
        efficiency=1.0,
        critical_irradiance_w_m2=200.0,
        stc_irradiance_w_m2=1000.0,
    )

    return PhotovoltaicSystem(
        parameters=parameters,
        name="MG1_PV",
    )


# ============================================================
# BASIC PV OUTPUT TESTS
# ============================================================

def test_zero_irradiance_produces_zero_power(pv):
    """
    If solar irradiance is zero, PV output must be zero.
    """

    power = pv.power_from_irradiance(0.0)

    assert power == pytest.approx(0.0)


def test_low_irradiance_ratio(pv):
    """
    Manuscript low-irradiance equation:

        r = I^2 / (I_c * I_STC)

    For:
        I = 100 W/m^2
        I_c = 200 W/m^2
        I_STC = 1000 W/m^2

    Therefore:

        r = 100^2 / (200 * 1000)
          = 10000 / 200000
          = 0.05
    """

    ratio = pv.irradiance_ratio(100.0)

    assert ratio == pytest.approx(0.05)


def test_low_irradiance_power(pv):
    """
    With efficiency = 1.0:

        P_PV = eta * r * P_rated
             = 1.0 * 0.05 * 500
             = 25 kW
    """

    power = pv.power_from_irradiance(100.0)

    expected_power = 25.0

    assert power == pytest.approx(expected_power)


# ============================================================
# CRITICAL IRRADIANCE BOUNDARY
# ============================================================

def test_critical_irradiance_boundary(pv):
    """
    At I = I_c, the second branch is used:

        r = I / I_STC

    Therefore:

        r = 200 / 1000 = 0.20
    """

    ratio = pv.irradiance_ratio(200.0)

    assert ratio == pytest.approx(0.20)


# ============================================================
# MEDIUM IRRADIANCE REGION
# ============================================================

def test_medium_irradiance_ratio(pv):
    """
    Manuscript medium-irradiance equation:

        r = I / I_STC

    For:
        I = 500
        I_STC = 1000

        r = 0.5
    """

    ratio = pv.irradiance_ratio(500.0)

    assert ratio == pytest.approx(0.50)


def test_medium_irradiance_power(pv):
    """
    For:
        eta = 1
        r = 0.5
        P_rated = 500 kW

        P_PV = 250 kW
    """

    power = pv.power_from_irradiance(500.0)

    expected_power = 250.0

    assert power == pytest.approx(expected_power)


# ============================================================
# STC AND HIGH IRRADIANCE REGION
# ============================================================

def test_stc_irradiance_ratio_equals_one(pv):
    """
    At I = I_STC:

        r = 1
    """

    ratio = pv.irradiance_ratio(1000.0)

    assert ratio == pytest.approx(1.0)


def test_above_stc_ratio_remains_one(pv):
    """
    For irradiance above STC:

        r = 1
    """

    ratio = pv.irradiance_ratio(1200.0)

    assert ratio == pytest.approx(1.0)


def test_power_does_not_exceed_rated_capacity(pv):
    """
    PV output must not exceed installed rated capacity.
    """

    power = pv.power_from_irradiance(1500.0)

    assert power <= 500.0


# ============================================================
# PV EFFICIENCY TEST
# ============================================================

def test_efficiency_affects_power():
    """
    Verify the eta_PV factor in:

        P_PV = eta_PV * r * P_rated
    """

    parameters = PVParameters(
        rated_capacity_kw=500.0,
        efficiency=0.90,
        critical_irradiance_w_m2=200.0,
        stc_irradiance_w_m2=1000.0,
    )

    pv_system = PhotovoltaicSystem(
        parameters=parameters,
        name="PV_efficiency_test",
    )

    power = pv_system.power_from_irradiance(
        1000.0
    )

    expected_power = (
        0.90
        * 1.0
        * 500.0
    )

    assert power == pytest.approx(
        expected_power
    )


# ============================================================
# INVALID INPUT TESTS
# ============================================================

def test_negative_irradiance_is_rejected(pv):
    """
    Negative irradiance is physically invalid.
    """

    with pytest.raises(ValueError):
        pv.power_from_irradiance(-1.0)


def test_negative_irradiance_ratio_is_rejected(pv):
    """
    irradiance_ratio() itself must reject negative irradiance.
    """

    with pytest.raises(ValueError):
        pv.irradiance_ratio(-10.0)


# ============================================================
# PV PROFILE TEST
# ============================================================

def test_power_profile(pv):
    """
    Check a full irradiance profile.

    Irradiance:
        0
        100
        200
        500
        1000

    Expected powers:
        0
        25
        100
        250
        500
    """

    irradiance = np.array(
        [
            0.0,
            100.0,
            200.0,
            500.0,
            1000.0,
        ]
    )

    power = pv.power_profile(
        irradiance
    )

    expected = np.array(
        [
            0.0,
            25.0,
            100.0,
            250.0,
            500.0,
        ]
    )

    assert np.allclose(
        power,
        expected,
    )


def test_negative_value_in_profile_is_rejected(pv):
    """
    A profile containing any negative irradiance
    must be rejected.
    """

    irradiance = np.array(
        [
            0.0,
            100.0,
            -20.0,
            500.0,
        ]
    )

    with pytest.raises(ValueError):
        pv.power_profile(
            irradiance
        )


# ============================================================
# HISTORICAL PV DATA VALIDATION
# ============================================================

def test_historical_power_negative_is_clipped(pv):
    """
    Negative measured PV generation is clipped to zero.
    """

    power = pv.validate_historical_power(
        -20.0
    )

    assert power == pytest.approx(
        0.0
    )


def test_historical_power_above_capacity_is_clipped(pv):
    """
    Historical PV power above installed capacity
    is clipped to rated PV capacity.
    """

    power = pv.validate_historical_power(
        700.0
    )

    assert power == pytest.approx(
        500.0
    )


def test_historical_power_inside_range_is_unchanged(pv):
    """
    Valid historical PV power should remain unchanged.
    """

    power = pv.validate_historical_power(
        300.0
    )

    assert power == pytest.approx(
        300.0
    )


# ============================================================
# STEP FUNCTION TEST
# ============================================================

def test_step_returns_correct_information(pv):
    """
    Check that step() returns the complete PV state.
    """

    result = pv.step(
        500.0
    )

    assert result[
        "irradiance_w_m2"
    ] == pytest.approx(
        500.0
    )

    assert result[
        "irradiance_ratio"
    ] == pytest.approx(
        0.50
    )

    assert result[
        "pv_power_kw"
    ] == pytest.approx(
        250.0
    )

    assert result[
        "rated_capacity_kw"
    ] == pytest.approx(
        500.0
    )


# ============================================================
# STATE TEST
# ============================================================

def test_get_state_after_step(pv):
    """
    get_state() should return the most recent values.
    """

    pv.step(
        500.0
    )

    state = pv.get_state()

    assert state[
        "irradiance_w_m2"
    ] == pytest.approx(
        500.0
    )

    assert state[
        "irradiance_ratio"
    ] == pytest.approx(
        0.50
    )

    assert state[
        "pv_power_kw"
    ] == pytest.approx(
        250.0
    )

    assert state[
        "rated_capacity_kw"
    ] == pytest.approx(
        500.0
    )


# ============================================================
# RESET TEST
# ============================================================

def test_reset(pv):
    """
    Reset must return the stored PV state to zero.
    """

    pv.step(
        500.0
    )

    pv.reset()

    state = pv.get_state()

    assert state[
        "irradiance_w_m2"
    ] == pytest.approx(
        0.0
    )

    assert state[
        "irradiance_ratio"
    ] == pytest.approx(
        0.0
    )

    assert state[
        "pv_power_kw"
    ] == pytest.approx(
        0.0
    )


# ============================================================
# PARAMETER VALIDATION TESTS
# ============================================================

def test_invalid_rated_capacity():
    """
    Rated PV capacity must be positive.
    """

    parameters = PVParameters(
        rated_capacity_kw=-500.0,
        efficiency=0.90,
        critical_irradiance_w_m2=200.0,
        stc_irradiance_w_m2=1000.0,
    )

    with pytest.raises(ValueError):
        PhotovoltaicSystem(
            parameters
        )


def test_zero_rated_capacity_is_rejected():
    """
    Zero PV capacity is also invalid.
    """

    parameters = PVParameters(
        rated_capacity_kw=0.0,
        efficiency=0.90,
        critical_irradiance_w_m2=200.0,
        stc_irradiance_w_m2=1000.0,
    )

    with pytest.raises(ValueError):
        PhotovoltaicSystem(
            parameters
        )


def test_invalid_efficiency_above_one():
    """
    Efficiency greater than one is invalid.
    """

    parameters = PVParameters(
        rated_capacity_kw=500.0,
        efficiency=1.20,
        critical_irradiance_w_m2=200.0,
        stc_irradiance_w_m2=1000.0,
    )

    with pytest.raises(ValueError):
        PhotovoltaicSystem(
            parameters
        )


def test_invalid_efficiency_zero():
    """
    Efficiency equal to zero is invalid.
    """

    parameters = PVParameters(
        rated_capacity_kw=500.0,
        efficiency=0.0,
        critical_irradiance_w_m2=200.0,
        stc_irradiance_w_m2=1000.0,
    )

    with pytest.raises(ValueError):
        PhotovoltaicSystem(
            parameters
        )


def test_invalid_critical_irradiance():
    """
    Critical irradiance must be positive.
    """

    parameters = PVParameters(
        rated_capacity_kw=500.0,
        efficiency=0.90,
        critical_irradiance_w_m2=0.0,
        stc_irradiance_w_m2=1000.0,
    )

    with pytest.raises(ValueError):
        PhotovoltaicSystem(
            parameters
        )


def test_invalid_stc_irradiance():
    """
    STC irradiance must be positive.
    """

    parameters = PVParameters(
        rated_capacity_kw=500.0,
        efficiency=0.90,
        critical_irradiance_w_m2=200.0,
        stc_irradiance_w_m2=0.0,
    )

    with pytest.raises(ValueError):
        PhotovoltaicSystem(
            parameters
        )


def test_critical_irradiance_must_be_below_stc():
    """
    The piecewise model requires:

        I_c < I_STC
    """

    parameters = PVParameters(
        rated_capacity_kw=500.0,
        efficiency=0.90,
        critical_irradiance_w_m2=1200.0,
        stc_irradiance_w_m2=1000.0,
    )

    with pytest.raises(ValueError):
        PhotovoltaicSystem(
            parameters
        )


# ============================================================
# REPRESENTATION TEST
# ============================================================

def test_repr_contains_name_and_capacity(pv):
    """
    __repr__ should contain useful identifying information.
    """

    text = repr(pv)

    assert "MG1_PV" in text
    assert "500.0" in text