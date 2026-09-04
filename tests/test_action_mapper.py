import numpy as np
import pytest

from marl.action_mapper import (
    ActionMapperConfig,
    _clip_normalized_action,
    apply_coordinator_sharing_multiplier,
)


def test_default_config_valid():

    config = ActionMapperConfig()

    config.validate()


def test_negative_reserve_fraction_rejected():

    config = ActionMapperConfig(
        reserve_fraction_of_bess_rating=-0.1
    )

    with pytest.raises(ValueError):
        config.validate()


def test_normalized_zero():

    assert _clip_normalized_action(
        0.0,
        1e-8,
    ) == pytest.approx(0.0)


def test_normalized_upper_limit():

    assert _clip_normalized_action(
        1.0,
        1e-8,
    ) == pytest.approx(1.0)


def test_normalized_lower_limit():

    assert _clip_normalized_action(
        -1.0,
        1e-8,
    ) == pytest.approx(-1.0)


def test_normalized_above_limit_rejected():

    with pytest.raises(ValueError):

        _clip_normalized_action(
            1.1,
            1e-8,
        )


def test_normalized_below_limit_rejected():

    with pytest.raises(ValueError):

        _clip_normalized_action(
            -1.1,
            1e-8,
        )


def test_coordinator_sharing_multiplier_disabled():

    matrix = np.ones(
        (5, 5),
        dtype=np.float64,
    )

    result = (
        apply_coordinator_sharing_multiplier(
            sharing_matrix=matrix,
            coordinator_action=np.array(
                [0.0, 0.0, -1.0]
            ),
            enabled=False,
        )
    )

    assert np.allclose(
        result,
        matrix,
    )


def test_coordinator_sharing_minus_one():

    matrix = np.ones(
        (5, 5),
        dtype=np.float64,
    )

    result = (
        apply_coordinator_sharing_multiplier(
            sharing_matrix=matrix,
            coordinator_action=np.array(
                [0.0, 0.0, -1.0]
            ),
        )
    )

    assert np.allclose(
        result,
        0.0,
    )


def test_coordinator_sharing_zero():

    matrix = np.ones(
        (5, 5),
        dtype=np.float64,
    )

    result = (
        apply_coordinator_sharing_multiplier(
            sharing_matrix=matrix,
            coordinator_action=np.array(
                [0.0, 0.0, 0.0]
            ),
        )
    )

    assert np.allclose(
        result,
        0.5,
    )


def test_coordinator_sharing_plus_one():

    matrix = np.ones(
        (5, 5),
        dtype=np.float64,
    )

    result = (
        apply_coordinator_sharing_multiplier(
            sharing_matrix=matrix,
            coordinator_action=np.array(
                [0.0, 0.0, 1.0]
            ),
        )
    )

    assert np.allclose(
        result,
        matrix,
    )


def test_short_coordinator_action_keeps_sharing():

    matrix = np.ones(
        (5, 5),
        dtype=np.float64,
    )

    result = (
        apply_coordinator_sharing_multiplier(
            sharing_matrix=matrix,
            coordinator_action=np.array(
                [0.0, 0.0]
            ),
        )
    )

    assert np.allclose(
        result,
        matrix,
    )