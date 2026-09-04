"""
Tests for the FC-HMARL inter-microgrid energy-sharing model.

These tests verify:

1. Connectivity logic
2. Directional sharing limits
3. Manuscript topology constraint
4. Incoming/outgoing accounting
5. Scheduled-power conservation
6. 98% sharing efficiency
7. Transfer loss
8. Surplus-based feasibility
9. Network reset and state

The numerical connectivity/capacity matrices below are TEST VALUES.
They are not claimed to be the lost original topology.
"""

import numpy as np
import pytest

from environment.energy_sharing import (
    EnergySharingNetwork,
    EnergySharingParameters,
)


# ============================================================
# TEST NETWORK
# ============================================================

@pytest.fixture
def sharing():
    """
    Three-MG synthetic network.

    Connectivity:

        MG1 -> MG2
        MG1 -> MG3
        MG2 -> MG1
        MG2 -> MG3
        MG3 -> MG1
        MG3 -> MG2

    All MGs are mutually connected for this unit test.

    Directional capacities:

        MG1->MG2 = 100 kW
        MG1->MG3 = 80 kW

        MG2->MG1 = 100 kW
        MG2->MG3 = 120 kW

        MG3->MG1 = 80 kW
        MG3->MG2 = 120 kW
    """

    parameters = EnergySharingParameters(
        number_of_microgrids=3,
        efficiency=0.98,
        tolerance_kw=1e-6,
    )

    connectivity = np.array(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ],
        dtype=int,
    )

    maximum_power = np.array(
        [
            [0.0, 100.0, 80.0],
            [100.0, 0.0, 120.0],
            [80.0, 120.0, 0.0],
        ],
        dtype=float,
    )

    return EnergySharingNetwork(
        parameters=parameters,
        connectivity_matrix=connectivity,
        maximum_power_matrix_kw=maximum_power,
        name="Test_Sharing_Network",
    )


# ============================================================
# BASIC PARAMETERS
# ============================================================

def test_number_of_microgrids(sharing):

    assert sharing.number_of_microgrids == 3


def test_efficiency(sharing):

    assert sharing.efficiency == pytest.approx(
        0.98
    )


# ============================================================
# CONNECTIVITY
# ============================================================

def test_connected_pair(sharing):

    assert sharing.is_connected(
        0,
        1,
    )


def test_self_connection_is_false(sharing):

    assert not sharing.is_connected(
        0,
        0,
    )


def test_maximum_transfer(sharing):

    assert sharing.maximum_transfer_kw(
        0,
        1,
    ) == pytest.approx(
        100.0
    )


# ============================================================
# TRANSFER CONSTRAINT
# ============================================================

def test_transfer_below_capacity_unchanged(sharing):

    power = sharing.constrain_transfer(
        sender=0,
        receiver=1,
        requested_power_kw=60.0,
    )

    assert power == pytest.approx(
        60.0
    )


def test_transfer_above_capacity_clipped(sharing):

    power = sharing.constrain_transfer(
        sender=0,
        receiver=1,
        requested_power_kw=150.0,
    )

    assert power == pytest.approx(
        100.0
    )


def test_self_transfer_is_zero(sharing):

    power = sharing.constrain_transfer(
        sender=0,
        receiver=0,
        requested_power_kw=50.0,
    )

    assert power == pytest.approx(
        0.0
    )


def test_negative_transfer_rejected(sharing):

    with pytest.raises(ValueError):

        sharing.constrain_transfer(
            sender=0,
            receiver=1,
            requested_power_kw=-10.0,
        )


# ============================================================
# DISCONNECTED PAIR
# ============================================================

def test_disconnected_pair_transfer_is_zero():

    parameters = EnergySharingParameters(
        number_of_microgrids=3,
        efficiency=0.98,
    )

    connectivity = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]
    )

    capacities = np.array(
        [
            [0.0, 100.0, 0.0],
            [100.0, 0.0, 120.0],
            [0.0, 120.0, 0.0],
        ]
    )

    network = EnergySharingNetwork(
        parameters,
        connectivity,
        capacities,
    )

    power = network.constrain_transfer(
        sender=0,
        receiver=2,
        requested_power_kw=50.0,
    )

    assert power == pytest.approx(
        0.0
    )


# ============================================================
# SET TRANSFER
# ============================================================

def test_set_transfer(sharing):

    result = sharing.set_transfer(
        sender=0,
        receiver=1,
        requested_power_kw=50.0,
    )

    assert result[
        "scheduled_power_kw"
    ] == pytest.approx(
        50.0
    )


def test_received_power_uses_efficiency(sharing):

    result = sharing.set_transfer(
        sender=0,
        receiver=1,
        requested_power_kw=100.0,
    )

    assert result[
        "received_power_kw"
    ] == pytest.approx(
        98.0
    )


def test_transfer_loss(sharing):

    result = sharing.set_transfer(
        sender=0,
        receiver=1,
        requested_power_kw=100.0,
    )

    assert result[
        "transfer_loss_kw"
    ] == pytest.approx(
        2.0
    )


# ============================================================
# OUTGOING / INCOMING
# ============================================================

def test_outgoing_power(sharing):

    sharing.set_transfer(
        0,
        1,
        50.0,
    )

    sharing.set_transfer(
        0,
        2,
        30.0,
    )

    assert sharing.outgoing_power_kw(
        0
    ) == pytest.approx(
        80.0
    )


def test_incoming_scheduled_power(sharing):

    sharing.set_transfer(
        0,
        1,
        50.0,
    )

    sharing.set_transfer(
        2,
        1,
        20.0,
    )

    assert sharing.incoming_scheduled_power_kw(
        1
    ) == pytest.approx(
        70.0
    )


def test_incoming_received_power(sharing):

    sharing.set_transfer(
        0,
        1,
        50.0,
    )

    sharing.set_transfer(
        2,
        1,
        20.0,
    )

    assert sharing.incoming_received_power_kw(
        1
    ) == pytest.approx(
        70.0 * 0.98
    )


# ============================================================
# CONSERVATION
# ============================================================

def test_scheduled_power_conservation(sharing):

    sharing.set_transfer(
        0,
        1,
        50.0,
    )

    sharing.set_transfer(
        1,
        2,
        40.0,
    )

    sharing.set_transfer(
        2,
        0,
        30.0,
    )

    assert (
        sharing.total_scheduled_outgoing_kw()
        == pytest.approx(
            120.0
        )
    )

    assert (
        sharing.total_scheduled_incoming_kw()
        == pytest.approx(
            120.0
        )
    )

    assert sharing.scheduled_power_is_conserved()


# ============================================================
# TOTAL RECEIVED AND LOSS
# ============================================================

def test_total_received_power(sharing):

    sharing.set_transfer(
        0,
        1,
        100.0,
    )

    assert sharing.total_received_power_kw() == pytest.approx(
        98.0
    )


def test_total_network_loss(sharing):

    sharing.set_transfer(
        0,
        1,
        100.0,
    )

    assert sharing.total_transfer_loss_kw() == pytest.approx(
        2.0
    )


# ============================================================
# COMPLETE MATRIX
# ============================================================

def test_set_transfer_matrix(sharing):

    requested = np.array(
        [
            [0.0, 50.0, 40.0],
            [30.0, 0.0, 60.0],
            [20.0, 50.0, 0.0],
        ]
    )

    result = sharing.set_transfer_matrix(
        requested
    )

    assert np.allclose(
        result,
        requested,
    )


def test_transfer_matrix_capacity_clipping(sharing):

    requested = np.array(
        [
            [0.0, 500.0, 500.0],
            [500.0, 0.0, 500.0],
            [500.0, 500.0, 0.0],
        ]
    )

    result = sharing.set_transfer_matrix(
        requested
    )

    expected = np.array(
        [
            [0.0, 100.0, 80.0],
            [100.0, 0.0, 120.0],
            [80.0, 120.0, 0.0],
        ]
    )

    assert np.allclose(
        result,
        expected,
    )


def test_negative_matrix_rejected(sharing):

    matrix = np.zeros(
        (3, 3)
    )

    matrix[
        0,
        1
    ] = -10.0

    with pytest.raises(ValueError):

        sharing.set_transfer_matrix(
            matrix
        )


def test_wrong_matrix_shape_rejected(sharing):

    with pytest.raises(ValueError):

        sharing.set_transfer_matrix(
            np.zeros(
                (2, 2)
            )
        )


# ============================================================
# SURPLUS LIMIT
# ============================================================

def test_outgoing_within_surplus_unchanged(sharing):

    sharing.set_transfer(
        0,
        1,
        40.0,
    )

    sharing.set_transfer(
        0,
        2,
        30.0,
    )

    result = (
        sharing.constrain_outgoing_by_surplus(
            microgrid=0,
            available_surplus_kw=100.0,
        )
    )

    assert result[
        "constrained_outgoing_kw"
    ] == pytest.approx(
        70.0
    )

    assert result[
        "scaling_factor"
    ] == pytest.approx(
        1.0
    )


def test_outgoing_exceeding_surplus_scaled(sharing):

    sharing.set_transfer(
        0,
        1,
        80.0,
    )

    sharing.set_transfer(
        0,
        2,
        60.0,
    )

    # Total scheduled outgoing = 140 kW
    # Available surplus = 70 kW
    #
    # scaling factor = 70 / 140 = 0.5

    result = (
        sharing.constrain_outgoing_by_surplus(
            microgrid=0,
            available_surplus_kw=70.0,
        )
    )

    assert result[
        "scaling_factor"
    ] == pytest.approx(
        0.5
    )

    assert result[
        "constrained_outgoing_kw"
    ] == pytest.approx(
        70.0
    )


def test_zero_surplus_removes_outgoing(sharing):

    sharing.set_transfer(
        0,
        1,
        50.0,
    )

    sharing.constrain_outgoing_by_surplus(
        microgrid=0,
        available_surplus_kw=0.0,
    )

    assert sharing.outgoing_power_kw(
        0
    ) == pytest.approx(
        0.0
    )


def test_negative_surplus_rejected(sharing):

    with pytest.raises(ValueError):

        sharing.constrain_outgoing_by_surplus(
            microgrid=0,
            available_surplus_kw=-1.0,
        )


# ============================================================
# CLEAR AND RESET
# ============================================================

def test_clear_transfer(sharing):

    sharing.set_transfer(
        0,
        1,
        50.0,
    )

    sharing.clear_transfer(
        0,
        1,
    )

    assert (
        sharing.scheduled_power_matrix_kw[
            0,
            1
        ]
        == pytest.approx(
            0.0
        )
    )


def test_reset(sharing):

    sharing.set_transfer(
        0,
        1,
        50.0,
    )

    sharing.set_transfer(
        1,
        2,
        40.0,
    )

    sharing.reset()

    assert np.allclose(
        sharing.scheduled_power_matrix_kw,
        np.zeros(
            (3, 3)
        ),
    )


# ============================================================
# STATE
# ============================================================

def test_get_state(sharing):

    sharing.set_transfer(
        0,
        1,
        100.0,
    )

    state = sharing.get_state()

    assert state[
        "total_scheduled_power_kw"
    ] == pytest.approx(
        100.0
    )

    assert state[
        "total_received_power_kw"
    ] == pytest.approx(
        98.0
    )

    assert state[
        "total_transfer_loss_kw"
    ] == pytest.approx(
        2.0
    )

    assert state[
        "scheduled_power_conserved"
    ]


# ============================================================
# PARAMETER VALIDATION
# ============================================================

def test_invalid_number_of_microgrids():

    parameters = EnergySharingParameters(
        number_of_microgrids=1,
    )

    with pytest.raises(ValueError):

        parameters.validate()


def test_invalid_efficiency():

    parameters = EnergySharingParameters(
        number_of_microgrids=3,
        efficiency=1.2,
    )

    with pytest.raises(ValueError):

        parameters.validate()


def test_invalid_connectivity_values():

    parameters = EnergySharingParameters(
        number_of_microgrids=3,
    )

    connectivity = np.array(
        [
            [0, 2, 1],
            [1, 0, 1],
            [1, 1, 0],
        ]
    )

    capacities = np.zeros(
        (3, 3)
    )

    with pytest.raises(ValueError):

        EnergySharingNetwork(
            parameters,
            connectivity,
            capacities,
        )


def test_self_connectivity_rejected():

    parameters = EnergySharingParameters(
        number_of_microgrids=3,
    )

    connectivity = np.eye(
        3,
        dtype=int,
    )

    capacities = np.zeros(
        (3, 3)
    )

    with pytest.raises(ValueError):

        EnergySharingNetwork(
            parameters,
            connectivity,
            capacities,
        )


def test_capacity_on_disconnected_link_rejected():

    parameters = EnergySharingParameters(
        number_of_microgrids=3,
    )

    connectivity = np.zeros(
        (3, 3),
        dtype=int,
    )

    capacities = np.zeros(
        (3, 3)
    )

    capacities[
        0,
        1
    ] = 100.0

    with pytest.raises(ValueError):

        EnergySharingNetwork(
            parameters,
            connectivity,
            capacities,
        )


# ============================================================
# REPRESENTATION
# ============================================================

def test_repr(sharing):

    text = repr(
        sharing
    )

    assert "Test_Sharing_Network" in text
    assert "microgrids=3" in text
    assert "0.9800" in text