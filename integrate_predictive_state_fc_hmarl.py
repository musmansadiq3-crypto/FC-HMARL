# ============================================================
# FC-HMARL
# STEP 7J
# REAL CONFIDENCE-AWARE PREDICTIVE STATE INTEGRATION
# ============================================================
#
# PURPOSE
# -------
# Connect the leakage-free, causal 24x4 predictive state:
#
#       S_pred = Phi * Z_hat
#
# generated in Step 7I to the EXISTING FC-HMARL
# HierarchicalStateBuilder.
#
#
# Manuscript local-agent state:
#
#   s_i = [
#       SOC_i,
#       P_i^PV,
#       P_i^load,
#       P_i^EV,
#       P_i^grid,
#       S_pred
#   ]^T
#
#   5 scalar physical states + 96 predictive elements
#
#   dimension = 101
#
#
# Manuscript coordinator state:
#
#   s_c = [
#       P^VPP,
#       E^share,
#       lambda,
#       S_pred
#   ]^T
#
#   3 scalar system states + 96 predictive elements
#
#   dimension = 99
#
#
# Global software state:
#
#   5 x 101 + 99 = 604
#
#
# IMPORTANT
# ---------
# This script is an INTEGRATION VALIDATION step.
#
# The physical scalar values below are neutral placeholders
# used ONLY to verify the real S_pred -> MARL state interface.
#
# We do NOT yet claim that zero physical values represent
# actual VPP operation.
#
# The next stage will connect real/scaled exogenous physical
# profiles to VPPTrainingBridge.
#
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from marl.state_builder import (
    StateBuilderConfig,
    HierarchicalStateBuilder,
    build_local_state,
    build_coordinator_state,
    build_global_state,
    flatten_predictive_state,
    validate_predictive_state,
)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"D:\Molvi paper review\FC_HMARL"
)


INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "forecasting"
    / "predictive_state"
    / "confidence_aware_predictive_state.npz"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "integration"
    / "predictive_state"
)


OUTPUT_ARCHIVE = (
    OUTPUT_DIR
    / "fc_hmarl_real_predictive_state_integration.npz"
)


SUMMARY_FILE = (
    OUTPUT_DIR
    / "fc_hmarl_state_dimensions.csv"
)


FIRST_SAMPLE_FILE = (
    OUTPUT_DIR
    / "fc_hmarl_first_sample_states.csv"
)


# ============================================================
# 2. CONSTANTS
# ============================================================

NUMBER_OF_MICROGRIDS = 5

FORECAST_HORIZON = 24

FORECAST_FEATURES = 4

PREDICTIVE_DIMENSION = (
    FORECAST_HORIZON
    * FORECAST_FEATURES
)

EXPECTED_LOCAL_DIMENSION = (
    5
    + PREDICTIVE_DIMENSION
)

EXPECTED_COORDINATOR_DIMENSION = (
    3
    + PREDICTIVE_DIMENSION
)

EXPECTED_GLOBAL_DIMENSION = (
    NUMBER_OF_MICROGRIDS
    * EXPECTED_LOCAL_DIMENSION
    + EXPECTED_COORDINATOR_DIMENSION
)


# ============================================================
# 3. HELPERS
# ============================================================

def section(title):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# 4. LOAD REAL PREDICTIVE STATE
# ============================================================

section(
    "FC-HMARL STEP 7J - REAL S_pred INTEGRATION"
)


if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Required Step 7I file not found:\n"
        f"{INPUT_FILE}"
    )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


data = np.load(
    INPUT_FILE,
    allow_pickle=True,
)


print(
    "Input archive keys:"
)


for key in data.files:

    print(
        f"  {key}"
    )


predictive_state_matrix = np.asarray(
    data[
        "predictive_state_matrix"
    ],
    dtype=np.float32,
)


predictive_state_flat_saved = np.asarray(
    data[
        "predictive_state_flat"
    ],
    dtype=np.float32,
)


causal_confidence = np.asarray(
    data[
        "causal_confidence_24h"
    ],
    dtype=np.float32,
)


feature_names = np.asarray(
    data[
        "feature_names"
    ]
)


number_of_samples = (
    predictive_state_matrix.shape[0]
)


print()

print(
    f"Predictive-state matrix : "
    f"{predictive_state_matrix.shape}"
)

print(
    f"Saved flattened state   : "
    f"{predictive_state_flat_saved.shape}"
)

print(
    f"Causal confidence       : "
    f"{causal_confidence.shape}"
)


# ============================================================
# 5. CONFIGURE EXISTING STATE BUILDER
# ============================================================

section(
    "CONFIGURING EXISTING HIERARCHICAL STATE BUILDER"
)


state_config = StateBuilderConfig(

    number_of_microgrids=
        NUMBER_OF_MICROGRIDS,

    forecast_horizon=
        FORECAST_HORIZON,

    forecast_features=
        FORECAST_FEATURES,

    flatten_predictive_state=True,

    dtype="float32",
)


state_builder = (
    HierarchicalStateBuilder(
        state_config
    )
)


print(
    f"Number of MGs        : "
    f"{NUMBER_OF_MICROGRIDS}"
)

print(
    f"Forecast horizon     : "
    f"{FORECAST_HORIZON}"
)

print(
    f"Forecast features    : "
    f"{FORECAST_FEATURES}"
)

print(
    f"Predictive dimension : "
    f"{PREDICTIVE_DIMENSION}"
)


# ============================================================
# 6. VALIDATE EVERY REAL PREDICTIVE SAMPLE
# ============================================================

section(
    "VALIDATING ALL REAL PREDICTIVE STATES"
)


maximum_flatten_difference = 0.0


for sample_index in range(
    number_of_samples
):

    predictive_sample = (
        predictive_state_matrix[
            sample_index
        ]
    )


    validated = (
        validate_predictive_state(

            predictive_sample,

            forecast_horizon=
                FORECAST_HORIZON,

            forecast_features=
                FORECAST_FEATURES,
        )
    )


    flattened = (
        flatten_predictive_state(

            validated,

            forecast_horizon=
                FORECAST_HORIZON,

            forecast_features=
                FORECAST_FEATURES,

            dtype="float32",
        )
    )


    if flattened.shape != (
        PREDICTIVE_DIMENSION,
    ):

        raise RuntimeError(

            f"Sample {sample_index} "
            f"did not flatten to 96 dimensions."
        )


    difference = float(

        np.max(

            np.abs(

                flattened
                - predictive_state_flat_saved[
                    sample_index
                ]
            )
        )
    )


    maximum_flatten_difference = max(

        maximum_flatten_difference,

        difference,
    )


print(
    f"Validated samples : "
    f"{number_of_samples}"
)


print(
    f"Maximum difference against "
    f"Step 7I flattened states: "
    f"{maximum_flatten_difference:.12e}"
)


if not np.allclose(
    predictive_state_matrix.reshape(
        number_of_samples,
        -1
    ),
    predictive_state_flat_saved,
    atol=1e-7,
):

    raise RuntimeError(
        "Step 7I and state_builder flattening "
        "are inconsistent."
    )


print()

print(
    "[OK] Existing state_builder accepts all "
    "real Step 7I predictive states."
)


# ============================================================
# 7. NEUTRAL PHYSICAL VALUES FOR INTERFACE VALIDATION
# ============================================================
#
# These values are NOT simulation results.
#
# They are neutral placeholders used only to test the state
# interface before real physical exogenous profiles are wired
# into VPPTrainingBridge.
#
# ============================================================

local_soc = np.zeros(
    NUMBER_OF_MICROGRIDS,
    dtype=np.float32,
)


local_pv = np.zeros(
    NUMBER_OF_MICROGRIDS,
    dtype=np.float32,
)


local_load = np.zeros(
    NUMBER_OF_MICROGRIDS,
    dtype=np.float32,
)


local_ev = np.zeros(
    NUMBER_OF_MICROGRIDS,
    dtype=np.float32,
)


local_grid = np.zeros(
    NUMBER_OF_MICROGRIDS,
    dtype=np.float32,
)


vpp_power = 0.0

energy_sharing = 0.0

market_price = 0.0


# ============================================================
# 8. BUILD FIRST REAL FC-HMARL LOCAL STATES
# ============================================================

section(
    "BUILDING LOCAL AGENT STATES"
)


first_predictive_state = (
    predictive_state_matrix[
        0
    ]
)


local_states = []


for microgrid_index in range(
    NUMBER_OF_MICROGRIDS
):

    local_state = (
        build_local_state(

            soc=
                local_soc[
                    microgrid_index
                ],

            pv_power=
                local_pv[
                    microgrid_index
                ],

            load_power=
                local_load[
                    microgrid_index
                ],

            ev_power=
                local_ev[
                    microgrid_index
                ],

            grid_exchange=
                local_grid[
                    microgrid_index
                ],

            predictive_state=
                first_predictive_state,

            config=
                state_config,
        )
    )


    local_state = np.asarray(
        local_state,
        dtype=np.float32,
    )


    local_states.append(
        local_state
    )


    print(

        f"MG{microgrid_index + 1} "
        f"local state shape: "
        f"{local_state.shape}"
    )


    if local_state.shape != (
        EXPECTED_LOCAL_DIMENSION,
    ):

        raise RuntimeError(

            f"MG{microgrid_index + 1}: "
            f"expected local dimension "
            f"{EXPECTED_LOCAL_DIMENSION}, "
            f"received {local_state.shape}"
        )


print()

print(
    "[OK] All five local-agent states "
    "have 101 dimensions."
)


# ============================================================
# 9. VERIFY LOCAL STATE CONTENT
# ============================================================

section(
    "VERIFYING LOCAL STATE CONTENT"
)


expected_predictive_flat = (
    first_predictive_state.reshape(
        -1
    )
)


for microgrid_index, local_state in enumerate(
    local_states
):

    physical_part = (
        local_state[
            :5
        ]
    )


    predictive_part = (
        local_state[
            5:
        ]
    )


    predictive_difference = float(

        np.max(

            np.abs(

                predictive_part
                - expected_predictive_flat
            )
        )
    )


    print(

        f"MG{microgrid_index + 1}: "

        f"physical={physical_part.tolist()} | "

        f"S_pred difference="
        f"{predictive_difference:.12e}"
    )


    if not np.allclose(
        predictive_part,
        expected_predictive_flat,
        atol=1e-7,
    ):

        raise RuntimeError(

            f"MG{microgrid_index + 1}: "
            f"real S_pred was not inserted "
            f"correctly."
        )


print()

print(
    "[OK] The final 96 components of each "
    "local state are the real causal S_pred."
)


# ============================================================
# 10. BUILD COORDINATOR STATE
# ============================================================

section(
    "BUILDING COORDINATOR STATE"
)


coordinator_state = (
    build_coordinator_state(

        vpp_power=
            vpp_power,

        energy_sharing=
            energy_sharing,

        market_price=
            market_price,

        predictive_state=
            first_predictive_state,

        config=
            state_config,
    )
)


coordinator_state = np.asarray(
    coordinator_state,
    dtype=np.float32,
)


print(
    f"Coordinator state shape: "
    f"{coordinator_state.shape}"
)


if coordinator_state.shape != (
    EXPECTED_COORDINATOR_DIMENSION,
):

    raise RuntimeError(

        "Coordinator dimension mismatch.\n"

        f"Expected: "
        f"{EXPECTED_COORDINATOR_DIMENSION}\n"

        f"Received: "
        f"{coordinator_state.shape}"
    )


coordinator_predictive_part = (
    coordinator_state[
        3:
    ]
)


coordinator_difference = float(

    np.max(

        np.abs(

            coordinator_predictive_part
            - expected_predictive_flat
        )
    )
)


print(
    f"S_pred difference: "
    f"{coordinator_difference:.12e}"
)


if not np.allclose(
    coordinator_predictive_part,
    expected_predictive_flat,
    atol=1e-7,
):

    raise RuntimeError(
        "Real S_pred was not correctly inserted "
        "into coordinator state."
    )


print()

print(
    "[OK] Coordinator state has 99 dimensions."
)

print(
    "[OK] Its final 96 components are "
    "the real causal S_pred."
)


# ============================================================
# 11. BUILD GLOBAL STATE
# ============================================================

section(
    "BUILDING GLOBAL FC-HMARL STATE"
)


global_state = (
    build_global_state(

        local_states=
            local_states,

        coordinator_state=
            coordinator_state,

        config=
            state_config,
    )
)


global_state = np.asarray(
    global_state,
    dtype=np.float32,
)


print(
    f"Global state shape: "
    f"{global_state.shape}"
)


if global_state.shape != (
    EXPECTED_GLOBAL_DIMENSION,
):

    raise RuntimeError(

        f"Expected global dimension "
        f"{EXPECTED_GLOBAL_DIMENSION}, "
        f"received {global_state.shape}"
    )


print()

print(
    "[OK] Global state dimension = 604."
)


# ============================================================
# 12. VERIFY DIMENSION EQUATIONS
# ============================================================

section(
    "VERIFYING FC-HMARL STATE DIMENSIONS"
)


print(
    "Local agent:"
)

print(
    f"  5 physical variables "
    f"+ {PREDICTIVE_DIMENSION} predictive "
    f"= {EXPECTED_LOCAL_DIMENSION}"
)


print()

print(
    "Coordinator:"
)

print(
    f"  3 system variables "
    f"+ {PREDICTIVE_DIMENSION} predictive "
    f"= {EXPECTED_COORDINATOR_DIMENSION}"
)


print()

print(
    "Global software state:"
)

print(
    f"  {NUMBER_OF_MICROGRIDS} "
    f"x {EXPECTED_LOCAL_DIMENSION} "
    f"+ {EXPECTED_COORDINATOR_DIMENSION} "
    f"= {EXPECTED_GLOBAL_DIMENSION}"
)


# ============================================================
# 13. BUILD ALL-SAMPLE PREDICTIVE STATE BATCH
# ============================================================
#
# We do NOT create complete 604-dimensional physical states
# for all 1291 samples here because their physical local
# quantities will come from the actual VPP environment.
#
# We preserve the real S_pred sequence for the bridge.
#
# ============================================================

section(
    "PREPARING PREDICTIVE STATES FOR VPP TRAINING BRIDGE"
)


bridge_predictive_states = (
    predictive_state_matrix.copy()
)


if bridge_predictive_states.shape != (
    number_of_samples,
    FORECAST_HORIZON,
    FORECAST_FEATURES,
):

    raise RuntimeError(
        "Unexpected bridge predictive-state shape."
    )


print(
    f"Bridge predictive-state sequence: "
    f"{bridge_predictive_states.shape}"
)


print(
    f"Each bridge input provides: "
    f"{FORECAST_HORIZON} x "
    f"{FORECAST_FEATURES}"
)


# ============================================================
# 14. SAVE STATE DIMENSION SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    [
        {
            "state":
                "predictive_state",
            "dimension":
                PREDICTIVE_DIMENSION,
            "structure":
                "24 x 4",
        },
        {
            "state":
                "local_agent",
            "dimension":
                EXPECTED_LOCAL_DIMENSION,
            "structure":
                "5 + 96",
        },
        {
            "state":
                "coordinator",
            "dimension":
                EXPECTED_COORDINATOR_DIMENSION,
            "structure":
                "3 + 96",
        },
        {
            "state":
                "global",
            "dimension":
                EXPECTED_GLOBAL_DIMENSION,
            "structure":
                "5 x 101 + 99",
        },
    ]
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# 15. SAVE FIRST-SAMPLE STATE VALUES
# ============================================================

first_sample_rows = []


for microgrid_index, local_state in enumerate(
    local_states
):

    first_sample_rows.append(
        {
            "state_name":
                f"MG{microgrid_index + 1}",
            "dimension":
                len(local_state),
            "physical_dimension":
                5,
            "predictive_dimension":
                96,
            "first_value":
                float(local_state[0]),
            "last_value":
                float(local_state[-1]),
        }
    )


first_sample_rows.append(
    {
        "state_name":
            "Coordinator",
        "dimension":
            len(coordinator_state),
        "physical_dimension":
            3,
        "predictive_dimension":
            96,
        "first_value":
            float(coordinator_state[0]),
        "last_value":
            float(coordinator_state[-1]),
    }
)


pd.DataFrame(
    first_sample_rows
).to_csv(
    FIRST_SAMPLE_FILE,
    index=False,
)


# ============================================================
# 16. SAVE INTEGRATION ARCHIVE
# ============================================================

np.savez_compressed(

    OUTPUT_ARCHIVE,

    predictive_state_matrix=
        bridge_predictive_states,

    predictive_state_flat=
        predictive_state_flat_saved,

    first_local_states=
        np.stack(
            local_states,
            axis=0,
        ),

    first_coordinator_state=
        coordinator_state,

    first_global_state=
        global_state,

    causal_confidence_24h=
        causal_confidence,

    feature_names=
        feature_names,

    number_of_microgrids=
        np.array(
            NUMBER_OF_MICROGRIDS
        ),

    local_state_dimension=
        np.array(
            EXPECTED_LOCAL_DIMENSION
        ),

    coordinator_state_dimension=
        np.array(
            EXPECTED_COORDINATOR_DIMENSION
        ),

    global_state_dimension=
        np.array(
            EXPECTED_GLOBAL_DIMENSION
        ),
)


# ============================================================
# 17. FINAL VALIDATION
# ============================================================

section(
    "STEP 7J FINAL VALIDATION"
)


print(
    f"Real S_pred samples      : "
    f"{number_of_samples}"
)

print(
    f"S_pred matrix/sample     : "
    f"(24, 4)"
)

print(
    f"S_pred flattened         : "
    f"{PREDICTIVE_DIMENSION}"
)

print(
    f"Number of local agents   : "
    f"{NUMBER_OF_MICROGRIDS}"
)

print(
    f"Local state dimension    : "
    f"{EXPECTED_LOCAL_DIMENSION}"
)

print(
    f"Coordinator dimension    : "
    f"{EXPECTED_COORDINATOR_DIMENSION}"
)

print(
    f"Global state dimension   : "
    f"{EXPECTED_GLOBAL_DIMENSION}"
)


print()

print(
    "[OK] Real causal S_pred accepted by "
    "existing state_builder."
)

print(
    "[OK] Local state = 101 dimensions."
)

print(
    "[OK] Coordinator state = 99 dimensions."
)

print(
    "[OK] Global state = 604 dimensions."
)

print(
    "[OK] Existing tested MARL modules were "
    "not modified."
)

print(
    "[OK] Predictive-state sequence is ready "
    "for VPPTrainingBridge."
)


# ============================================================
# 18. OUTPUT LOCATIONS
# ============================================================

section(
    "STEP 7J COMPLETE"
)


print(
    f"Integration archive:\n"
    f"{OUTPUT_ARCHIVE}"
)

print()

print(
    f"State-dimension summary:\n"
    f"{SUMMARY_FILE}"
)

print()

print(
    f"First-sample state summary:\n"
    f"{FIRST_SAMPLE_FILE}"
)

print()

print(
    "STEP 7J COMPLETE."
)