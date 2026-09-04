"""
Unit tests for the FC-HMARL EV aggregation model.

The tests verify the manuscript equations for:

1. EV arrival/departure availability
2. Aggregated charging demand
3. Aggregate charging-power limit

Numerical individual charging powers and the fleet-level maximum
charging power used here are TEST PARAMETERS only. They are not claimed
to be values reported by the manuscript.
"""

import numpy as np
import pytest

from environment.ev_fleet import (
    EVFleet,
    EVFleetParameters,
    EVRecord,
)


# ============================================================
# TEST FIXTURE
# ============================================================

@pytest.fixture
def fleet():
    """
    Create a simple three-EV test fleet.

    EV1:
        arrival   = 8
        departure = 12
        charging  = 7 kW

    EV2:
        arrival   = 9
        departure = 15
        charging  = 11 kW

    EV3:
        arrival   = 18
        departure = 22
        charging  = 7 kW

    Fleet aggregate limit:
        15 kW

    These are test values only.
    """

    parameters = EVFleetParameters(
        number_of_evs=3,
        maximum_aggregate_charging_power_kw=15.0,
        time_step_hours=1.0,
    )

    vehicles = [
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

        EVRecord(
            ev_id="EV3",
            arrival_time=18.0,
            departure_time=22.0,
            charging_power_kw=7.0,
        ),
    ]

    return EVFleet(
        parameters=parameters,
        vehicles=vehicles,
        name="Test_EV_Fleet",
    )


# ============================================================
# BASIC PARAMETER TESTS
# ============================================================

def test_number_of_evs(fleet):

    assert fleet.number_of_evs == 3


def test_maximum_aggregate_power(fleet):

    assert (
        fleet.maximum_aggregate_charging_power_kw
        == pytest.approx(15.0)
    )


def test_time_step(fleet):

    assert fleet.time_step_hours == pytest.approx(
        1.0
    )


# ============================================================
# AVAILABILITY TESTS
# ============================================================

def test_ev_unavailable_before_arrival(fleet):

    availability = fleet.availability_vector(
        7.0
    )

    assert np.array_equal(
        availability,
        np.array([0.0, 0.0, 0.0]),
    )


def test_ev_available_at_arrival_time(fleet):
    """
    Manuscript uses:

        T_arr <= t <= T_dep

    so arrival time is included.
    """

    availability = fleet.availability_vector(
        8.0
    )

    assert np.array_equal(
        availability,
        np.array([1.0, 0.0, 0.0]),
    )


def test_two_evs_available_at_10(fleet):

    availability = fleet.availability_vector(
        10.0
    )

    assert np.array_equal(
        availability,
        np.array([1.0, 1.0, 0.0]),
    )


def test_ev_available_at_departure_time(fleet):
    """
    Departure time is included because the paper uses <=.
    """

    availability = fleet.availability_vector(
        12.0
    )

    assert availability[0] == pytest.approx(
        1.0
    )


def test_ev_unavailable_after_departure(fleet):

    availability = fleet.availability_vector(
        12.1
    )

    assert availability[0] == pytest.approx(
        0.0
    )


def test_evening_ev_availability(fleet):

    availability = fleet.availability_vector(
        20.0
    )

    assert np.array_equal(
        availability,
        np.array([0.0, 0.0, 1.0]),
    )


def test_number_available(fleet):

    assert fleet.number_available(10.0) == 2


# ============================================================
# RATED CHARGING POWER VECTOR
# ============================================================

def test_rated_charging_power_vector(fleet):

    powers = fleet.rated_charging_power_vector()

    assert np.allclose(
        powers,
        np.array([7.0, 11.0, 7.0]),
    )


# ============================================================
# AGGREGATED CHARGING EQUATION
# ============================================================

def test_unconstrained_power_before_arrivals(fleet):

    power = fleet.unconstrained_aggregate_power(
        time=7.0
    )

    assert power == pytest.approx(
        0.0
    )


def test_unconstrained_power_one_available_ev(fleet):
    """
    At t = 8:

        EV1 available = 1
        EV2 available = 0
        EV3 available = 0

    Therefore:

        P_EV = 1*7 + 0*11 + 0*7
             = 7 kW
    """

    power = fleet.unconstrained_aggregate_power(
        time=8.0
    )

    assert power == pytest.approx(
        7.0
    )


def test_unconstrained_power_two_available_evs(fleet):
    """
    At t = 10:

        EV1 = 7 kW
        EV2 = 11 kW

    Unconstrained aggregate:

        7 + 11 = 18 kW
    """

    power = fleet.unconstrained_aggregate_power(
        time=10.0
    )

    assert power == pytest.approx(
        18.0
    )


# ============================================================
# AGGREGATE CAPACITY CONSTRAINT
# ============================================================

def test_aggregate_power_is_clipped_to_maximum(fleet):
    """
    Manuscript constraint:

        P_EV <= P_EV,max

    Unconstrained = 18 kW
    Maximum       = 15 kW

    Therefore final demand = 15 kW.
    """

    power = fleet.aggregate_power(
        time=10.0
    )

    assert power == pytest.approx(
        15.0
    )


def test_aggregate_power_below_limit_unchanged(fleet):

    power = fleet.aggregate_power(
        time=8.0
    )

    assert power == pytest.approx(
        7.0
    )


def test_constrain_negative_aggregate_to_zero(fleet):

    power = fleet.constrain_aggregate_power(
        -5.0
    )

    assert power == pytest.approx(
        0.0
    )


def test_constrain_above_maximum(fleet):

    power = fleet.constrain_aggregate_power(
        50.0
    )

    assert power == pytest.approx(
        15.0
    )


# ============================================================
# REQUESTED CHARGING POWERS
# ============================================================

def test_requested_charging_power(fleet):
    """
    At t=10 both EV1 and EV2 are available.

    Requests:
        EV1 = 3
        EV2 = 5
        EV3 = 7

    Aggregate:
        3 + 5 = 8 kW
    """

    power = fleet.aggregate_power(
        time=10.0,
        requested_charging_powers_kw=[
            3.0,
            5.0,
            7.0,
        ],
    )

    assert power == pytest.approx(
        8.0
    )


def test_requested_power_is_limited_by_vehicle_rating(fleet):
    """
    Requesting more than an EV's charging rating is clipped.
    """

    requested = fleet.validate_requested_charging_powers(
        [
            100.0,
            100.0,
            100.0,
        ]
    )

    expected = np.array(
        [
            7.0,
            11.0,
            7.0,
        ]
    )

    assert np.allclose(
        requested,
        expected,
    )


def test_negative_requested_charging_power_rejected(fleet):

    with pytest.raises(ValueError):

        fleet.validate_requested_charging_powers(
            [
                7.0,
                -1.0,
                7.0,
            ]
        )


def test_wrong_requested_power_vector_length(fleet):

    with pytest.raises(ValueError):

        fleet.validate_requested_charging_powers(
            [
                7.0,
                11.0,
            ]
        )


# ============================================================
# STEP TESTS
# ============================================================

def test_step_returns_correct_information(fleet):

    result = fleet.step(
        time=10.0
    )

    assert result["time"] == pytest.approx(
        10.0
    )

    assert result["number_of_evs"] == 3

    assert result["available_evs"] == 2

    assert result[
        "unconstrained_ev_power_kw"
    ] == pytest.approx(
        18.0
    )

    assert result[
        "ev_power_kw"
    ] == pytest.approx(
        15.0
    )

    assert result[
        "maximum_ev_power_kw"
    ] == pytest.approx(
        15.0
    )


def test_step_availability_vector(fleet):

    result = fleet.step(
        time=10.0
    )

    assert np.array_equal(
        result["availability_vector"],
        np.array([1.0, 1.0, 0.0]),
    )


# ============================================================
# STATE TESTS
# ============================================================

def test_get_state_after_step(fleet):

    fleet.step(
        time=20.0
    )

    state = fleet.get_state()

    assert state["time"] == pytest.approx(
        20.0
    )

    assert state["available_evs"] == 1

    assert state["ev_power_kw"] == pytest.approx(
        7.0
    )


# ============================================================
# RESET TEST
# ============================================================

def test_reset(fleet):

    fleet.step(
        time=10.0
    )

    fleet.reset()

    state = fleet.get_state()

    assert state["time"] == pytest.approx(
        0.0
    )

    assert state["available_evs"] == 0

    assert state["ev_power_kw"] == pytest.approx(
        0.0
    )

    assert np.array_equal(
        state["availability_vector"],
        np.zeros(3),
    )


# ============================================================
# EV RECORD VALIDATION
# ============================================================

def test_negative_arrival_time_rejected():

    vehicle = EVRecord(
        arrival_time=-1.0,
        departure_time=10.0,
        charging_power_kw=7.0,
    )

    with pytest.raises(ValueError):

        vehicle.validate()


def test_departure_before_arrival_rejected():

    vehicle = EVRecord(
        arrival_time=15.0,
        departure_time=10.0,
        charging_power_kw=7.0,
    )

    with pytest.raises(ValueError):

        vehicle.validate()


def test_negative_vehicle_charging_power_rejected():

    vehicle = EVRecord(
        arrival_time=8.0,
        departure_time=12.0,
        charging_power_kw=-7.0,
    )

    with pytest.raises(ValueError):

        vehicle.validate()


# ============================================================
# FLEET PARAMETER VALIDATION
# ============================================================

def test_zero_number_of_evs_rejected():

    parameters = EVFleetParameters(
        number_of_evs=0,
        maximum_aggregate_charging_power_kw=15.0,
    )

    with pytest.raises(ValueError):

        EVFleet(
            parameters=parameters
        )


def test_negative_maximum_power_rejected():

    parameters = EVFleetParameters(
        number_of_evs=3,
        maximum_aggregate_charging_power_kw=-1.0,
    )

    with pytest.raises(ValueError):

        EVFleet(
            parameters=parameters
        )


def test_wrong_number_of_vehicle_records_rejected():

    parameters = EVFleetParameters(
        number_of_evs=3,
        maximum_aggregate_charging_power_kw=15.0,
    )

    vehicles = [
        EVRecord(
            arrival_time=8.0,
            departure_time=12.0,
            charging_power_kw=7.0,
        ),
    ]

    with pytest.raises(ValueError):

        EVFleet(
            parameters=parameters,
            vehicles=vehicles,
        )


# ============================================================
# UNINITIALIZED FLEET TEST
# ============================================================

def test_fleet_without_vehicle_records_cannot_calculate():

    parameters = EVFleetParameters(
        number_of_evs=3,
        maximum_aggregate_charging_power_kw=15.0,
    )

    fleet = EVFleet(
        parameters=parameters
    )

    with pytest.raises(RuntimeError):

        fleet.aggregate_power(
            time=10.0
        )


# ============================================================
# INVALID TIME
# ============================================================

def test_negative_simulation_time_rejected(fleet):

    with pytest.raises(ValueError):

        fleet.availability_vector(
            -1.0
        )


# ============================================================
# REPRESENTATION
# ============================================================

def test_repr_contains_fleet_information(fleet):

    text = repr(fleet)

    assert "Test_EV_Fleet" in text
    assert "number_of_evs=3" in text
    assert "15.00" in text