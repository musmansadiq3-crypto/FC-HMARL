import pytest
from environment.constraints import (
    ConstraintParameters,
    clip_pcc_exchange,
    clip_value,
    evaluate_microgrid_constraints,
    is_power_balanced,
    is_transformer_limit_satisfied,
    microgrid_power_balance_residual,
    pcc_exchange_kw,
    positive_part,
    power_balance_violation_kw,
    required_grid_power_for_balance,
    transformer_violation_kw,
)
# ============================================================
# BASIC UTILITIES
# ============================================================

def test_clip_value_inside_range():

    result = clip_value(
        value=5.0,
        minimum=0.0,
        maximum=10.0,
    )

    assert result == pytest.approx(
        5.0
    )
def test_clip_value_above_maximum():

    result = clip_value(
        value=20.0,
        minimum=0.0,
        maximum=10.0,
    )
    assert result == pytest.approx(
        10.0
    )
def test_clip_value_below_minimum():

    result = clip_value(
        value=-5.0,
        minimum=0.0,
        maximum=10.0,
    )
    assert result == pytest.approx(
        0.0
    )


def test_invalid_clip_range_rejected():

    with pytest.raises(ValueError):

        clip_value(
            value=5.0,
            minimum=10.0,
            maximum=0.0,
        )


def test_positive_part_positive():

    assert positive_part(
        5.0
    ) == pytest.approx(
        5.0
    )


def test_positive_part_negative():

    assert positive_part(
        -5.0
    ) == pytest.approx(
        0.0
    )
# ============================================================
# POWER BALANCE RESIDUAL
# ============================================================

def test_exact_power_balance():
    residual = microgrid_power_balance_residual(
        net_local_power_kw=-100.0,
        incoming_sharing_kw=20.0,
        outgoing_sharing_kw=0.0,
        grid_power_kw=80.0,
    )
    assert residual == pytest.approx(
        0.0
    )
def test_power_balance_with_surplus_export():
  
    residual = microgrid_power_balance_residual(
        net_local_power_kw=100.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=60.0,
        grid_power_kw=-40.0,
    )
    assert residual == pytest.approx(
        0.0
    )
def test_nonzero_power_balance_residual():

    residual = microgrid_power_balance_residual(
        net_local_power_kw=-100.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=0.0,
        grid_power_kw=50.0,
    )

    assert residual == pytest.approx(
        -50.0
    )
# ============================================================
# POWER BALANCE BOOLEAN
# ============================================================

def test_is_power_balanced_true():

    assert is_power_balanced(
        net_local_power_kw=-100.0,
        incoming_sharing_kw=20.0,
        outgoing_sharing_kw=0.0,
        grid_power_kw=80.0,
    )


def test_is_power_balanced_false():

    assert not is_power_balanced(
        net_local_power_kw=-100.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=0.0,
        grid_power_kw=50.0,
    )


def test_power_balance_tolerance():

    assert is_power_balanced(
        net_local_power_kw=-100.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=0.0,
        grid_power_kw=100.00001,
        tolerance_kw=0.001,
    )


def test_negative_tolerance_rejected():

    with pytest.raises(ValueError):

        is_power_balanced(
            net_local_power_kw=0.0,
            incoming_sharing_kw=0.0,
            outgoing_sharing_kw=0.0,
            grid_power_kw=0.0,
            tolerance_kw=-1.0,
        )
# ============================================================
# BALANCE VIOLATION MAGNITUDE
# ============================================================
def test_power_balance_violation():

    violation = power_balance_violation_kw(
        net_local_power_kw=-100.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=0.0,
        grid_power_kw=70.0,
    )

    assert violation == pytest.approx(
        30.0
    )
# ============================================================
# REQUIRED GRID POWER
# ============================================================

def test_required_grid_power_for_deficit():
    grid_power = required_grid_power_for_balance(
        net_local_power_kw=-100.0,
        incoming_sharing_kw=20.0,
        outgoing_sharing_kw=0.0,
    )

    assert grid_power == pytest.approx(
        80.0
    )
def test_required_grid_power_for_surplus():
    grid_power = required_grid_power_for_balance(
        net_local_power_kw=100.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=60.0,
    )
    assert grid_power == pytest.approx(
        -40.0
    )
# ============================================================
# TRANSFORMER PARAMETERS
# ============================================================
def test_transformer_active_power_limit():

    parameters = ConstraintParameters(
        transformer_rating_kva=1000.0,
        power_factor=0.90,
    )

    assert (
        parameters.transformer_active_power_limit_kw
        == pytest.approx(900.0)
    )
def test_power_factor_one():

    parameters = ConstraintParameters(
        transformer_rating_kva=1000.0,
        power_factor=1.0,
    )
    assert (
        parameters.transformer_active_power_limit_kw
        == pytest.approx(1000.0)
    )
def test_invalid_transformer_rating():

    parameters = ConstraintParameters(
        transformer_rating_kva=-1000.0,
    )

    with pytest.raises(ValueError):

        parameters.validate()

def test_invalid_power_factor_above_one():

    parameters = ConstraintParameters(
        transformer_rating_kva=1000.0,
        power_factor=1.20,
    )
    with pytest.raises(ValueError):

        parameters.validate()
def test_invalid_power_factor_zero():

    parameters = ConstraintParameters(
        transformer_rating_kva=1000.0,
        power_factor=0.0,
    )

    with pytest.raises(ValueError):

        parameters.validate()


# ============================================================
# PCC EXCHANGE
# ============================================================

def test_pcc_exchange():

    exchange = pcc_exchange_kw(
        grid_power_kw=500.0,
        outgoing_sharing_kw=100.0,
    )

    assert exchange == pytest.approx(
        600.0
    )
def test_pcc_exchange_with_export():

    exchange = pcc_exchange_kw(
        grid_power_kw=-300.0,
        outgoing_sharing_kw=100.0,
    )
    assert exchange == pytest.approx(
        -200.0
    )
# ============================================================
# TRANSFORMER LIMIT
# ============================================================
def test_transformer_limit_satisfied():

    assert is_transformer_limit_satisfied(
        grid_power_kw=500.0,
        outgoing_sharing_kw=100.0,
        active_power_limit_kw=1000.0,
    )
def test_transformer_limit_violated():

    assert not is_transformer_limit_satisfied(
        grid_power_kw=900.0,
        outgoing_sharing_kw=200.0,
        active_power_limit_kw=1000.0,
    )
def test_negative_active_power_limit_rejected():

    with pytest.raises(ValueError):

        is_transformer_limit_satisfied(
            grid_power_kw=0.0,
            outgoing_sharing_kw=0.0,
            active_power_limit_kw=-1.0,
        )
# ============================================================
# TRANSFORMER VIOLATION
# ============================================================

def test_no_transformer_violation():

    violation = transformer_violation_kw(
        grid_power_kw=500.0,
        outgoing_sharing_kw=100.0,
        active_power_limit_kw=1000.0,
    )
    assert violation == pytest.approx(
        0.0
    )
def test_transformer_violation_value():

    violation = transformer_violation_kw(
        grid_power_kw=900.0,
        outgoing_sharing_kw=200.0,
        active_power_limit_kw=1000.0,
    )
    assert violation == pytest.approx(
        100.0
    )
# ============================================================
# PCC CLIPPING
# ============================================================

def test_clip_pcc_exchange_inside_limit():

    grid_power = clip_pcc_exchange(
        grid_power_kw=500.0,
        outgoing_sharing_kw=100.0,
        active_power_limit_kw=1000.0,
    )

    assert grid_power == pytest.approx(
        500.0
    )


def test_clip_pcc_exchange_upper_limit():
    """
    P_out = 200
    limit = 1000

    Maximum allowed grid power:

        1000 - 200 = 800
    """

    grid_power = clip_pcc_exchange(
        grid_power_kw=1000.0,
        outgoing_sharing_kw=200.0,
        active_power_limit_kw=1000.0,
    )

    assert grid_power == pytest.approx(
        800.0
    )
def test_clip_pcc_exchange_lower_limit():
    """
    Constraint:

        -1000 <= P_grid + 200

    therefore:

        P_grid >= -1200
    """

    grid_power = clip_pcc_exchange(
        grid_power_kw=-1500.0,
        outgoing_sharing_kw=200.0,
        active_power_limit_kw=1000.0,
    )

    assert grid_power == pytest.approx(
        -1200.0
    )
# ============================================================
# FULL CONSTRAINT EVALUATION
# ============================================================

def test_full_constraint_evaluation_feasible():

    parameters = ConstraintParameters(
        transformer_rating_kva=1000.0,
        power_factor=1.0,
        balance_tolerance_kw=1e-6,
    )

    result = evaluate_microgrid_constraints(
        net_local_power_kw=-100.0,
        incoming_sharing_kw=20.0,
        outgoing_sharing_kw=0.0,
        grid_power_kw=80.0,
        parameters=parameters,
    )

    assert result[
        "power_balance_satisfied"
    ]

    assert result[
        "transformer_limit_satisfied"
    ]

    assert result[
        "power_balance_residual_kw"
    ] == pytest.approx(
        0.0
    )

    assert result[
        "transformer_violation_kw"
    ] == pytest.approx(
        0.0
    )
def test_full_constraint_evaluation_infeasible_balance():

    parameters = ConstraintParameters(
        transformer_rating_kva=1000.0,
        power_factor=1.0,
    )

    result = evaluate_microgrid_constraints(
        net_local_power_kw=-100.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=0.0,
        grid_power_kw=50.0,
        parameters=parameters,
    )

    assert not result[
        "power_balance_satisfied"
    ]

    assert result[
        "power_balance_violation_kw"
    ] == pytest.approx(
        50.0
    )


def test_full_constraint_evaluation_transformer_violation():

    parameters = ConstraintParameters(
        transformer_rating_kva=1000.0,
        power_factor=1.0,
    )

    result = evaluate_microgrid_constraints(
        net_local_power_kw=-1100.0,
        incoming_sharing_kw=0.0,
        outgoing_sharing_kw=0.0,
        grid_power_kw=1100.0,
        parameters=parameters,
    )

    assert result[
        "power_balance_satisfied"
    ]

    assert not result[
        "transformer_limit_satisfied"
    ]

    assert result[
        "transformer_violation_kw"
    ] == pytest.approx(
        100.0
    )
