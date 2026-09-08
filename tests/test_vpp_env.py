import numpy as np
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
from environment.energy_sharing import (
    EnergySharingNetwork,
    EnergySharingParameters,
)
from environment.vpp_env import (
    VPPEnvironment,
)

# ============================================================
# HELPER
# ============================================================

def build_microgrid(
    name,
    pv_capacity_kw,
    bess_capacity_kwh,
    bess_power_kw,
    peak_load_kw,
    transformer_kva,
):
    """
    Build one manuscript-parameterized microgrid.

    The small two-EV fleet is used only for integration testing.
    """

    pv = PhotovoltaicSystem(
        PVParameters(
            rated_capacity_kw=pv_capacity_kw,
            efficiency=1.0,
            critical_irradiance_w_m2=200.0,
            stc_irradiance_w_m2=1000.0,
        ),
        name=f"{name}_PV",
    )

    bess = BatteryEnergyStorageSystem(
        BESSParameters(
            capacity_kwh=bess_capacity_kwh,
            rated_power_kw=bess_power_kw,
            charging_efficiency=0.95,
            discharging_efficiency=0.95,
            self_discharge_rate=0.001,
            minimum_soc=0.20,
            maximum_soc=0.95,
            time_step_hours=1.0,
        ),
        initial_soc=0.60,
        name=f"{name}_BESS",
    )

    ev_fleet = EVFleet(
        EVFleetParameters(
            number_of_evs=2,
            maximum_aggregate_charging_power_kw=20.0,
            time_step_hours=1.0,
        ),
        vehicles=[
            EVRecord(
                ev_id=f"{name}_EV1",
                arrival_time=8.0,
                departure_time=12.0,
                charging_power_kw=7.0,
            ),
            EVRecord(
                ev_id=f"{name}_EV2",
                arrival_time=9.0,
                departure_time=15.0,
                charging_power_kw=11.0,
            ),
        ],
        name=f"{name}_EV_Fleet",
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
        name=f"{name}_Market",
    )

    return Microgrid(
        parameters=MicrogridParameters(
            name=name,
            peak_load_kw=peak_load_kw,
            transformer_rating_kva=transformer_kva,
            power_factor=1.0,
        ),
        pv_system=pv,
        bess=bess,
        ev_fleet=ev_fleet,
        market=market,
    )


# ============================================================
# FIVE-MG FIXTURE
# ============================================================

@pytest.fixture
def vpp():

    microgrids = [
        build_microgrid(
            "MG1",
            pv_capacity_kw=500.0,
            bess_capacity_kwh=1000.0,
            bess_power_kw=250.0,
            peak_load_kw=750.0,
            transformer_kva=1000.0,
        ),

        build_microgrid(
            "MG2",
            pv_capacity_kw=600.0,
            bess_capacity_kwh=1200.0,
            bess_power_kw=300.0,
            peak_load_kw=850.0,
            transformer_kva=1250.0,
        ),

        build_microgrid(
            "MG3",
            pv_capacity_kw=450.0,
            bess_capacity_kwh=900.0,
            bess_power_kw=250.0,
            peak_load_kw=700.0,
            transformer_kva=1000.0,
        ),

        build_microgrid(
            "MG4",
            pv_capacity_kw=550.0,
            bess_capacity_kwh=1100.0,
            bess_power_kw=300.0,
            peak_load_kw=800.0,
            transformer_kva=1250.0,
        ),

        build_microgrid(
            "MG5",
            pv_capacity_kw=700.0,
            bess_capacity_kwh=1400.0,
            bess_power_kw=350.0,
            peak_load_kw=950.0,
            transformer_kva=1500.0,
        ),
    ]

    connectivity = np.ones(
        (5, 5),
        dtype=int,
    )

    np.fill_diagonal(
        connectivity,
        0,
    )

    maximum_power = np.full(
        (5, 5),
        250.0,
        dtype=float,
    )

    np.fill_diagonal(
        maximum_power,
        0.0,
    )

    sharing_network = EnergySharingNetwork(
        parameters=EnergySharingParameters(
            number_of_microgrids=5,
            efficiency=0.98,
        ),
        connectivity_matrix=connectivity,
        maximum_power_matrix_kw=maximum_power,
        name="Five_MG_Sharing",
    )

    return VPPEnvironment(
        microgrids=microgrids,
        energy_sharing_network=sharing_network,
        episode_length_hours=24,
        name="Test_Five_MG_VPP",
    )


# ============================================================
# BASIC ARCHITECTURE
# ============================================================

def test_five_microgrids(vpp):

    assert vpp.num_microgrids == 5


def test_microgrid_names(vpp):

    names = [
        mg.name
        for mg in vpp.microgrids
    ]

    assert names == [
        "MG1",
        "MG2",
        "MG3",
        "MG4",
        "MG5",
    ]


def test_total_manuscript_pv_capacity(vpp):

    total = sum(
        mg.pv_system.rated_capacity_kw
        for mg in vpp.microgrids
    )

    assert total == pytest.approx(
        2800.0
    )


def test_total_manuscript_bess_capacity(vpp):

    total = sum(
        mg.bess.capacity_kwh
        for mg in vpp.microgrids
    )

    assert total == pytest.approx(
        5600.0
    )


def test_total_peak_load(vpp):

    total = sum(
        mg.peak_load_kw
        for mg in vpp.microgrids
    )

    assert total == pytest.approx(
        4050.0
    )


# ============================================================
# ZERO SHARING
# ============================================================

def test_zero_sharing_matrix(vpp):

    sharing = np.zeros(
        (5, 5)
    )

    result = vpp.apply_sharing_actions(
        sharing
    )

    assert np.allclose(
        result["feasible_matrix_kw"],
        sharing,
    )

    assert result[
        "total_scheduled_sharing_kw"
    ] == pytest.approx(
        0.0
    )


# ============================================================
# SHARING EFFICIENCY
# ============================================================

def test_vpp_sharing_efficiency(vpp):

    sharing = np.zeros(
        (5, 5)
    )

    sharing[0, 1] = 100.0

    result = vpp.apply_sharing_actions(
        sharing
    )

    assert result[
        "outgoing_power_kw"
    ][0] == pytest.approx(
        100.0
    )

    assert result[
        "incoming_scheduled_power_kw"
    ][1] == pytest.approx(
        100.0
    )

    assert result[
        "incoming_received_power_kw"
    ][1] == pytest.approx(
        98.0
    )

    assert result[
        "total_sharing_loss_kw"
    ] == pytest.approx(
        2.0
    )


# ============================================================
# SHARING CAPACITY
# ============================================================

def test_sharing_capacity_is_enforced(vpp):

    sharing = np.zeros(
        (5, 5)
    )

    sharing[0, 1] = 500.0

    result = vpp.apply_sharing_actions(
        sharing
    )

    assert result[
        "feasible_matrix_kw"
    ][0, 1] == pytest.approx(
        250.0
    )


# ============================================================
# COMPLETE VPP STEP
# ============================================================

def test_complete_vpp_step(vpp):

    result = vpp.step(
        time=10.0,

        loads_kw=[
            400.0,
            450.0,
            350.0,
            420.0,
            500.0,
        ],

        irradiances_w_m2=[
            500.0,
            500.0,
            500.0,
            500.0,
            500.0,
        ],

        bess_actions_kw=[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],

        sharing_matrix_kw=np.zeros(
            (5, 5)
        ),

        buy_prices_usd_per_kwh=[
            0.20,
            0.20,
            0.20,
            0.20,
            0.20,
        ],

        sell_prices_usd_per_kwh=[
            0.10,
            0.10,
            0.10,
            0.10,
            0.10,
        ],
    )

    assert result[
        "num_microgrids"
    ] == 5

    assert len(
        result["local_results"]
    ) == 5


def test_all_power_balances_close(vpp):

    result = vpp.step(
        time=10.0,

        loads_kw=[
            400.0,
            450.0,
            350.0,
            420.0,
            500.0,
        ],

        irradiances_w_m2=[
            500.0,
            500.0,
            500.0,
            500.0,
            500.0,
        ],

        bess_actions_kw=[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],

        sharing_matrix_kw=np.zeros(
            (5, 5)
        ),

        buy_prices_usd_per_kwh=[
            0.20,
            0.20,
            0.20,
            0.20,
            0.20,
        ],

        sell_prices_usd_per_kwh=[
            0.10,
            0.10,
            0.10,
            0.10,
            0.10,
        ],
    )

    assert result[
        "constraints"
    ][
        "all_power_balanced"
    ]


# ============================================================
# SYSTEM TOTALS
# ============================================================

def test_total_pv_is_sum_of_local_pv(vpp):

    result = vpp.step(
        time=7.0,

        loads_kw=[
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
        ],

        irradiances_w_m2=[
            1000.0,
            1000.0,
            1000.0,
            1000.0,
            1000.0,
        ],

        bess_actions_kw=[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],

        sharing_matrix_kw=np.zeros(
            (5, 5)
        ),

        buy_prices_usd_per_kwh=[
            0.20,
            0.20,
            0.20,
            0.20,
            0.20,
        ],

        sell_prices_usd_per_kwh=[
            0.10,
            0.10,
            0.10,
            0.10,
            0.10,
        ],
    )

    assert result[
        "totals"
    ][
        "pv_power_kw"
    ] == pytest.approx(
        2800.0
    )


def test_total_load_is_sum(vpp):

    result = vpp.step(
        time=7.0,

        loads_kw=[
            100.0,
            200.0,
            300.0,
            400.0,
            500.0,
        ],

        irradiances_w_m2=[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],

        bess_actions_kw=[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],

        sharing_matrix_kw=np.zeros(
            (5, 5)
        ),

        buy_prices_usd_per_kwh=[
            0.20,
            0.20,
            0.20,
            0.20,
            0.20,
        ],

        sell_prices_usd_per_kwh=[
            0.10,
            0.10,
            0.10,
            0.10,
            0.10,
        ],
    )

    assert result[
        "totals"
    ][
        "load_power_kw"
    ] == pytest.approx(
        1500.0
    )


# ============================================================
# SHARING IMPACT
# ============================================================

def test_internal_sharing_is_reflected(vpp):

    sharing = np.zeros(
        (5, 5)
    )

    sharing[
        0,
        1
    ] = 100.0

    result = vpp.step(
        time=10.0,

        loads_kw=[
            200.0,
            600.0,
            350.0,
            420.0,
            500.0,
        ],

        irradiances_w_m2=[
            1000.0,
            200.0,
            500.0,
            500.0,
            500.0,
        ],

        bess_actions_kw=[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],

        sharing_matrix_kw=sharing,

        buy_prices_usd_per_kwh=[
            0.20,
            0.20,
            0.20,
            0.20,
            0.20,
        ],

        sell_prices_usd_per_kwh=[
            0.10,
            0.10,
            0.10,
            0.10,
            0.10,
        ],
    )

    assert result[
        "sharing"
    ][
        "total_scheduled_sharing_kw"
    ] == pytest.approx(
        100.0
    )

    assert result[
        "sharing"
    ][
        "total_received_sharing_kw"
    ] == pytest.approx(
        98.0
    )


# ============================================================
# RESERVE
# ============================================================

def test_reserve_revenue_aggregates(vpp):

    result = vpp.step(
        time=7.0,

        loads_kw=[
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
        ],

        irradiances_w_m2=[
            500.0,
            500.0,
            500.0,
            500.0,
            500.0,
        ],

        bess_actions_kw=[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],

        sharing_matrix_kw=np.zeros(
            (5, 5)
        ),

        buy_prices_usd_per_kwh=[
            0.20,
            0.20,
            0.20,
            0.20,
            0.20,
        ],

        sell_prices_usd_per_kwh=[
            0.10,
            0.10,
            0.10,
            0.10,
            0.10,
        ],

        reserve_actions_kw=[
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
        ],
    )

    # 5 MG * 10 kW * 1 h * $0.05/kWh
    assert result[
        "totals"
    ][
        "reserve_revenue_usd"
    ] == pytest.approx(
        2.5
    )


# ============================================================
# STATE
# ============================================================

def test_local_states_length(vpp):

    states = vpp.get_local_states()

    assert len(states) == 5


def test_global_state(vpp):

    state = vpp.get_global_state()

    assert state[
        "num_microgrids"
    ] == 5

    assert len(
        state["local_states"]
    ) == 5


# ============================================================
# RESET
# ============================================================

def test_vpp_reset(vpp):

    vpp.step(
        time=10.0,

        loads_kw=[
            400.0,
            450.0,
            350.0,
            420.0,
            500.0,
        ],

        irradiances_w_m2=[
            500.0,
            500.0,
            500.0,
            500.0,
            500.0,
        ],

        bess_actions_kw=[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],

        sharing_matrix_kw=np.zeros(
            (5, 5)
        ),

        buy_prices_usd_per_kwh=[
            0.20,
            0.20,
            0.20,
            0.20,
            0.20,
        ],

        sell_prices_usd_per_kwh=[
            0.10,
            0.10,
            0.10,
            0.10,
            0.10,
        ],
    )

    state = vpp.reset(
        bess_initial_socs=[
            0.60,
            0.60,
            0.60,
            0.60,
            0.60,
        ]
    )

    assert state[
        "step"
    ] == 0

    for mg in vpp.microgrids:

        assert mg.bess.soc == pytest.approx(
            0.60
        )


# ============================================================
# EPISODE LENGTH
# ============================================================

def test_episode_done_after_24_steps(vpp):

    result = None

    for hour in range(24):

        result = vpp.step(
            time=float(hour),

            loads_kw=[
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
            ],

            irradiances_w_m2=[
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],

            bess_actions_kw=[
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],

            sharing_matrix_kw=np.zeros(
                (5, 5)
            ),

            buy_prices_usd_per_kwh=[
                0.20,
                0.20,
                0.20,
                0.20,
                0.20,
            ],

            sell_prices_usd_per_kwh=[
                0.10,
                0.10,
                0.10,
                0.10,
                0.10,
            ],
        )

    assert result is not None

    assert result[
        "done"
    ]


# ============================================================
# INPUT VALIDATION
# ============================================================

def test_wrong_load_vector_length(vpp):

    with pytest.raises(ValueError):

        vpp.step(
            time=0.0,

            loads_kw=[
                100.0,
                100.0,
            ],

            irradiances_w_m2=[
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],

            bess_actions_kw=[
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],

            sharing_matrix_kw=np.zeros(
                (5, 5)
            ),

            buy_prices_usd_per_kwh=[
                0.20,
                0.20,
                0.20,
                0.20,
                0.20,
            ],

            sell_prices_usd_per_kwh=[
                0.10,
                0.10,
                0.10,
                0.10,
                0.10,
            ],
        )


def test_negative_reserve_rejected(vpp):

    with pytest.raises(ValueError):

        vpp.step(
            time=0.0,

            loads_kw=[
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
            ],

            irradiances_w_m2=[
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],

            bess_actions_kw=[
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],

            sharing_matrix_kw=np.zeros(
                (5, 5)
            ),

            buy_prices_usd_per_kwh=[
                0.20,
                0.20,
                0.20,
                0.20,
                0.20,
            ],

            sell_prices_usd_per_kwh=[
                0.10,
                0.10,
                0.10,
                0.10,
                0.10,
            ],

            reserve_actions_kw=[
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        )


# ============================================================
# REPRESENTATION
# ============================================================

def test_repr(vpp):

    text = repr(
        vpp
    )

    assert "Test_Five_MG_VPP" in text
    assert "microgrids=5" in text
    assert "episode_length=24" in text
