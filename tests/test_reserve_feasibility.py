from types import SimpleNamespace
import numpy as np
import pytest
from environment.bess import (
    BatteryEnergyStorageSystem,
    BESSParameters,
)
from marl.action_mapper import (
    map_reserve_actions,
)


def make_bess(
    soc=0.60,
    capacity_kwh=1000.0,
    rated_power_kw=250.0,
):
    return BatteryEnergyStorageSystem(
        BESSParameters(
            capacity_kwh=capacity_kwh,
            rated_power_kw=rated_power_kw,
            charging_efficiency=0.95,
            discharging_efficiency=0.95,
            self_discharge_rate=0.001,
            minimum_soc=0.20,
            maximum_soc=0.95,
            time_step_hours=1.0,
        ),
        initial_soc=soc,
    )


def make_env(besses):
    microgrids = [
        SimpleNamespace(bess=b)
        for b in besses
    ]
    return SimpleNamespace(
        num_microgrids=len(microgrids),
        microgrids=microgrids,
    )


def test_full_discharge_uses_all_power_headroom():
    bess = make_bess(soc=0.80)

    reserve = (
        bess.maximum_feasible_upward_reserve_power_kw(
            scheduled_power_kw=250.0,
            reserve_duration_hours=1.0,
        )
    )

    assert reserve == pytest.approx(0.0)


def test_half_discharge_leaves_half_power_headroom():
    bess = make_bess(soc=0.80)

    reserve = (
        bess.maximum_feasible_upward_reserve_power_kw(
            scheduled_power_kw=125.0,
            reserve_duration_hours=1.0,
        )
    )

    assert reserve == pytest.approx(125.0)


def test_minimum_soc_cannot_sell_upward_reserve():
    bess = make_bess(soc=0.20)

    reserve = (
        bess.maximum_feasible_upward_reserve_power_kw(
            scheduled_power_kw=0.0,
            reserve_duration_hours=1.0,
        )
    )

    assert reserve == pytest.approx(0.0)


def test_energy_limit_can_be_tighter_than_power_limit():
    # SOC only slightly above minimum.
    bess = make_bess(
        soc=0.25,
        capacity_kwh=1000.0,
        rated_power_kw=250.0,
    )

    reserve = (
        bess.maximum_feasible_upward_reserve_power_kw(
            scheduled_power_kw=0.0,
            reserve_duration_hours=1.0,
        )
    )

    expected_energy_limit = (
        ((1.0 - 0.001) * 0.25 - 0.20)
        * 1000.0
        * 0.95
    )

    assert reserve == pytest.approx(
        expected_energy_limit
    )
    assert reserve < 250.0


def test_longer_reserve_duration_reduces_energy_limited_power():
    bess = make_bess(
        soc=0.30,
        capacity_kwh=1000.0,
        rated_power_kw=250.0,
    )

    one_hour = (
        bess.maximum_feasible_upward_reserve_power_kw(
            scheduled_power_kw=0.0,
            reserve_duration_hours=1.0,
        )
    )

    two_hour = (
        bess.maximum_feasible_upward_reserve_power_kw(
            scheduled_power_kw=0.0,
            reserve_duration_hours=2.0,
        )
    )

    assert two_hour <= one_hour
    assert two_hour == pytest.approx(
        one_hour / 2.0
    )


def test_mapper_clips_reserve_by_simultaneous_discharge():
    bess = make_bess(soc=0.80)
    env = make_env([bess])

    # coordinator action[1] = +1 -> 100% reserve request.
    reserve = map_reserve_actions(
        coordinator_action=np.array([0.0, 1.0, 0.0]),
        environment=env,
        bess_actions_kw=np.array([200.0]),
        reserve_fraction_of_bess_rating=1.0,
        reserve_duration_hours=1.0,
    )
    assert reserve.shape == (1,)
    assert reserve[0] == pytest.approx(50.0)


def test_mapper_zero_reserve_at_minimum_soc():
    bess = make_bess(soc=0.20)
    env = make_env([bess])

    reserve = map_reserve_actions(
        coordinator_action=np.array([0.0, 1.0, 0.0]),
        environment=env,
        bess_actions_kw=np.array([0.0]),
        reserve_fraction_of_bess_rating=1.0,
        reserve_duration_hours=1.0,
    )
    assert reserve[0] == pytest.approx(0.0)
def test_mapper_respects_participation_request():
    bess = make_bess(soc=0.80)
    env = make_env([bess])

    # action[1] = 0 -> participation = 0.5
    reserve = map_reserve_actions(
        coordinator_action=np.array([0.0, 0.0, 0.0]),
        environment=env,
        bess_actions_kw=np.array([0.0]),
        reserve_fraction_of_bess_rating=1.0,
        reserve_duration_hours=1.0,
    )
    assert reserve[0] == pytest.approx(125.0)
def test_invalid_reserve_duration_rejected():
    bess = make_bess(soc=0.80)
    with pytest.raises(ValueError):
        bess.maximum_feasible_upward_reserve_power_kw(
            scheduled_power_kw=0.0,
            reserve_duration_hours=0.0,
        )
