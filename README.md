# FC-HMARL: Forecast-Confidence-Aware Hierarchical Multi-Agent Reinforcement Learning

## Overview

This repository contains the complete reconstructed implementation, validation workflow, experimental records, and publication artifacts for the FC-HMARL framework for coordinated operation of interconnected microgrids within a Virtual Power Plant (VPP).

The implementation combines:

- Multi-variable forecasting
- Forecast uncertainty and confidence modeling
- Five interconnected microgrids
- PV generation
- Battery Energy Storage Systems (BESS)
- Electric Vehicle (EV) fleets
- Grid-market interaction
- Inter-microgrid energy sharing
- Local SAC-based controllers
- Upper-level VPP coordination
- Hierarchical multi-agent reinforcement learning
- Physical feasibility enforcement
- Benchmark and diagnostic ablation evaluation

---

## 1. Repository Structure

The principal project structure is:

    FC_HMARL/
    |
    +-- agents/                 SAC agents and neural networks
    +-- configs/                System, market, forecasting and MARL configuration
    +-- data/                   Raw, processed and publication data
    +-- environment/            Physical VPP and microgrid models
    +-- evaluation/             Forecasting, economic and operational evaluation
    +-- forecasting/            Forecasting, uncertainty and confidence modules
    +-- marl/                   Hierarchical FC-HMARL integration
    +-- outputs/                Checkpoints, results, figures and RL archives
    +-- risk/                   Risk-analysis utilities
    +-- tests/                  Automated software verification
    |
    +-- train_real_fc_hmarl_final_v3.py
    +-- DEVELOPMENT_HISTORY.md
    +-- FINAL_SUBMISSION_MANIFEST.txt
    +-- FINAL_REPRODUCIBILITY_RECORD.txt
    +-- FINAL_TEST_VERIFICATION.txt
    +-- final_project_structure.txt

---

## 2. Forecasting Pipeline

The forecasting pipeline processes four principal variables:

1. PV generation
2. Electrical load
3. EV demand/availability representation
4. Electricity market price

The implemented workflow includes:

- Raw-data processing
- Data cleaning
- Chronological alignment
- Three-sigma filtering
- Min-max normalization
- Chronological train/validation/test splitting
- Sequence construction
- Multi-variable forecasting
- Forecast uncertainty analysis
- Forecast confidence calculation
- Predictive-state generation

The forecasting configuration uses a 168-hour historical input window and a 24-hour prediction horizon.

Forecast information is subsequently integrated into the hierarchical control state.

---

## 3. Data Organization

The project contains data-processing and experimental artifacts for the major FC-HMARL inputs.

The principal data categories include:

- PV generation data
- Electrical load data
- EV charging/behavior data
- Electricity market-price data
- Processed forecasting datasets
- Reinforcement-learning archives
- Validation datasets
- TEST datasets
- Publication-supporting Excel workbooks

Historical and intermediate data artifacts are intentionally retained where useful for development traceability.

---

## 4. Physical VPP Model

The VPP consists of five interconnected microgrids.

The physical environment includes models for:

- PV generation
- Battery Energy Storage Systems
- Electric Vehicle fleets
- Electrical demand
- Grid exchange
- Electricity-market settlement
- Inter-microgrid energy sharing
- Transformer/PCC operation
- Reserve scheduling
- Power-balance constraints
- SOC feasibility
- Charging/discharging feasibility

The final physical implementation includes corrections identified during diagnostic testing for reserve feasibility and transformer/PCC operation.

---

## 5. Five-Microgrid System

The reconstructed VPP contains five microgrids with individual:

- PV capacities
- BESS energy capacities
- BESS power ratings
- EV fleets
- Electrical loads
- Transformer ratings

These microgrids are coordinated through the hierarchical FC-HMARL control framework.

---

## 6. Forecast Confidence Modeling

Forecast uncertainty is incorporated into the control framework through confidence-aware predictive information.

Forecast errors associated with renewable generation, electrical load, EV behavior, and electricity price are used to construct predictive-confidence information.

The confidence information is combined with forecast trajectories before integration into the reinforcement-learning state.

This allows the controller to receive both predicted operating information and information representing forecast reliability.

---

## 7. Predictive-State Construction

The forecasting pipeline produces a 24-hour prediction horizon for four variables:

    24 x 4 = 96 predictive features

The predictive information is incorporated into local and coordinator states.

The implemented state construction provides:

    Predictive state dimension:   96
    Local state dimension:        101
    Coordinator state dimension:  99
    Global state dimension:       604

The global representation combines the five local microgrid states and the coordinator state.

---

## 8. FC-HMARL Architecture

The hierarchical control architecture contains two principal decision layers.

### 8.1 Local Control Layer

Five SAC-based local microgrid agents are implemented.

Each local agent interacts with its corresponding microgrid environment.

The local control layer represents decisions associated with:

- BESS operation
- Local energy management
- Inter-microgrid sharing requests
- Grid interaction through the physical environment

### 8.2 Upper-Level Coordination Layer

A separate SAC-based VPP coordinator is implemented.

The coordinator represents VPP-level decisions associated with:

- Market interaction
- Reserve coordination
- Inter-microgrid coordination
- System-level economic operation

### 8.3 Hierarchical Integration

The local agents and coordinator are integrated through:

- State construction
- Action mapping
- Physical-environment interaction
- Reward assignment
- Replay-buffer storage
- SAC policy updates
- Hierarchical training logic

---

## 9. SAC Implementation

The reinforcement-learning implementation contains:

- Stochastic actor networks
- Twin critic networks
- Target critic networks
- Experience replay
- Soft target updates
- Adam optimization
- Discounted reward learning
- Continuous action processing
- Checkpoint saving and loading

The project maintains separate local agents and a separate upper-level coordinator.

---

## 10. Physical Feasibility Development

Extensive physical-system testing was performed during implementation.

### 10.1 Reserve Feasibility

Reserve scheduling was audited to ensure that reserve commitments remained compatible with available BESS power and energy.

### 10.2 Transformer/PCC Feasibility

Grid exchange and sharing behavior were audited against transformer and Point-of-Common-Coupling limits.

Requested operation is projected to physically feasible operation before market settlement and physical-state construction.

### 10.3 Reward Credit Assignment

Reward inputs were audited so that infeasible requested PCC behavior could be appropriately represented in local-agent reward information while the physical state continued to use feasible realized operation.

These development stages are preserved through diagnostic scripts, tests, corrected implementations, and historical outputs.

---

## 11. Final Training Protocol

The final controller version is:

    FC-HMARL V3

The final training configuration used:

    Number of local agents:       5
    Number of coordinators:       1
    Maximum training episodes:    5000
    Episode horizon:              24 hours
    Discount factor:              0.99
    Learning rate:                1e-4
    Replay-buffer capacity:       1,000,000
    Batch size:                   512
    Soft-update coefficient:      0.005
    Optimizer:                    Adam

The final V3 controller was trained for 5000 episodes.

The TRAIN archive was used for policy learning.

The final training records and checkpoints are retained in the output directories.

---

## 12. TRAIN / VALIDATION / TEST Separation

The final experimental workflow maintains strict separation between:

    TRAIN
    VALIDATION
    TEST

TRAIN data were used for controller learning.

VALIDATION data were used for checkpoint evaluation and final checkpoint selection.

TEST data were reserved for final locked evaluation.

The TEST dataset was not used to select the final checkpoint.

---

## 13. Validation-Based Checkpoint Selection

Saved checkpoints from final V3 training were evaluated using the VALIDATION dataset.

The validation procedure selected:

    Checkpoint 200

as the final controller checkpoint.

Checkpoint 200 was frozen before final TEST evaluation.

---

## 14. Locked Final TEST Evaluation

The frozen checkpoint was evaluated on the final TEST archive.

The final TEST evaluation used:

- Checkpoint 200 only
- Deterministic policy inference
- No policy learning
- No gradient updates
- No replay-buffer updates
- No checkpoint re-ranking
- No TEST-based model selection

The complete TEST archive was evaluated through rolling 24-hour windows.

---

## 15. Final TEST Performance

The locked final TEST evaluation produced the following principal results:

    Mean total return:                 -17573.003806
    Standard deviation:                 3199.954778
    Median total return:               -17100.055266
    Mean net market cost:               8575.627636 USD/day
    Mean grid import:                  47084.772950 kWh/day
    Mean grid export:                    146.364612 kWh/day
    Mean BESS throughput:               2137.710502 kWh/day
    Mean scheduled energy sharing:       127.342998 kWh/day
    Mean power-balance violation:          0
    Transformer violations:                0
    SOC violations:                        0

These results correspond to the frozen validation-selected checkpoint and the final locked TEST protocol.

---

## 16. Benchmark Evaluation

The final FC-HMARL controller was compared with:

1. Passive grid-only operation
2. Rule-based BESS energy-management operation
3. Full FC-HMARL operation

The passive benchmark represents operation without active BESS coordination, energy sharing, or reserve scheduling.

The rule-based benchmark is a transparent reconstructed controller used for comparison and is not presented as an independently optimized reinforcement-learning controller.

The final benchmark results were:

### Passive operation

    Mean return:             -18320.486056
    Mean grid import:         48175.843152 kWh/day
    Mean net market cost:      9155.555160 USD/day
    Peak grid import:          3173.535138 kW

### Rule-based operation

    Mean return:             -17835.675273
    Mean grid import:         46849.009781 kWh/day
    Mean net market cost:      8897.807730 USD/day
    Peak grid import:          3167.593581 kW

### FC-HMARL

    Mean return:             -17573.003806
    Mean grid import:         47084.772950 kWh/day
    Mean net market cost:      8575.627636 USD/day
    Peak grid import:          3155.289877 kW

Relative to passive operation, FC-HMARL reduced net market cost by approximately 6.33% and grid import by approximately 2.26%.

Relative to the reconstructed rule-based benchmark, FC-HMARL reduced net market cost by approximately 3.62%.

---

## 17. Diagnostic Ablation Analysis

Post-training inference-time diagnostic analyses were conducted for:

- Forecast-confidence information
- Inter-microgrid energy sharing
- Upper-level VPP coordination

The recorded mean TEST returns were:

    Full FC-HMARL:       -17573.003806
    No confidence:       -17567.033186
    No sharing:          -17572.128341
    No coordinator:      -18318.391247

These experiments use the frozen trained controller and modify selected inference-time components.

They are therefore reported as post-training inference-time diagnostic ablations and not as independently retrained ablation policies.

The diagnostic results provide strong evidence for the importance of upper-level coordination in the implemented controller.

The confidence and sharing diagnostics are interpreted conservatively because their removal did not degrade the frozen-policy reward under every TEST condition.

---

## 18. Publication Tables

Final publication tables are generated from the validated evaluation results.

These include:

- Main FC-HMARL performance
- Microgrid-level performance
- Coordinator performance
- Economic performance
- Operational performance
- Benchmark comparison
- Benchmark improvement
- Diagnostic ablation analysis

Publication tables are retained under the final results directories.

---

## 19. Publication Figures

The repository includes plotting workflows and figure data for:

- Forecasting performance
- Forecast uncertainty propagation
- Forecast-horizon performance
- Forecast-confidence evolution
- Operational dynamics
- Risk/economic sensitivity
- VPP economic and operational performance
- Case-study comparison
- Training diagnostics
- Final TEST performance
- Benchmark comparison
- Ablation diagnostics

Where historical or manuscript-supporting Excel workbooks are retained, their provenance should be distinguished from results generated directly by the final reconstructed V3 pipeline.

---

## 20. Automated Software Verification

The project contains an extensive automated test suite covering:

- BESS operation
- PV generation
- EV behavior
- Power balance
- Energy sharing
- Market operation
- Microgrid environment
- VPP environment
- Forecast preprocessing
- Forecast datasets
- Forecast models
- Forecast training
- SAC agents
- State construction
- Action mapping
- Hierarchical integration
- Reserve feasibility
- Transformer/PCC feasibility
- Reward routing
- Real VPP training bridge

The final complete software verification result is:

    993 passed
    22 warnings

The warnings are non-failing Pytest collection and PyTorch Transformer warnings.

The exact final verification output is stored in:

    FINAL_TEST_VERIFICATION.txt

---

## 21. Validated Software Environment

The final verified environment is:

    Python:       3.13.9
    PyTorch:      2.13.0+cpu
    NumPy:        2.3.5
    Pandas:       2.3.3
    SciPy:        1.16.3
    Matplotlib:   3.10.6
    Pytest:       8.4.2
    CUDA:         False

The final implementation and test suite were validated using CPU-based PyTorch execution.

The complete environment record is stored in:

    FINAL_REPRODUCIBILITY_RECORD.txt

---

## 22. Development Records

The repository intentionally retains many development artifacts, including:

- Historical scripts
- Intermediate model versions
- Diagnostic scripts
- Corrected implementations
- Validation utilities
- Previous training runs
- Previous evaluation outputs
- Supporting Excel workbooks
- Intermediate figures
- Physical-system audits
- Benchmark-development scripts
- Ablation-development scripts

These files are retained to provide transparent evidence of the implementation and validation process.

They should not all be interpreted as separate final models.

The development sequence is documented in:

    DEVELOPMENT_HISTORY.md

---

## 23. Final Results Directories

Principal final artifacts are organized under:

    outputs/results/
    outputs/checkpoints/
    outputs/rl_data/
    outputs/figures/

The final V3 result branches include records associated with:

- Training
- Validation
- Locked TEST evaluation
- Benchmark evaluation
- Diagnostic ablation evaluation
- Publication tables
- Publication figures

---

## 24. Reproducibility Records

The project root contains the following supporting records:

    DEVELOPMENT_HISTORY.md
    FINAL_SUBMISSION_MANIFEST.txt
    FINAL_REPRODUCIBILITY_RECORD.txt
    FINAL_TEST_VERIFICATION.txt
    final_project_structure.txt

These files document:

- Development history
- Project contents
- Software environment
- Test verification
- Final project structure
- Experimental traceability

---

## 25. Historical Files

Historical files are intentionally retained where they provide evidence of the development process.

Older scripts and outputs may correspond to:

- Earlier forecasting experiments
- Earlier physical-system implementations
- Reserve diagnostics
- PCC diagnostics
- Reward diagnostics
- Earlier training configurations
- Preliminary validation
- Preliminary TEST evaluations

The official final evaluation should be identified through the final V3 workflow and frozen validation-selected checkpoint rather than through historical experimental branches.

---

## 26. Reconstruction and Reproducibility Statement

The original implementation associated with the manuscript was not fully preserved.

The present repository therefore represents a systematic reconstruction based on:

- Manuscript equations
- Manuscript parameter tables
- Manuscript architectural descriptions
- Available research datasets
- Physical-system constraints
- Reconstructed software choices where exact implementation details were unavailable
- Extensive automated verification
- Final TRAIN/VALIDATION/TEST evaluation

Implementation details that were not explicitly preserved in the manuscript should be treated as reconstruction choices rather than claimed as recovered original source-code settings.

This distinction is important for transparent scientific reporting.

---

## 27. Important Final-Model Identification

For reproducibility, the official reconstructed final controller should be identified as:

    FC-HMARL V3

with:

    Final training length:          5000 episodes
    Checkpoint-selection dataset:   VALIDATION
    Frozen final checkpoint:        200
    Final evaluation dataset:       TEST
    Final evaluation mode:          deterministic
    Final software verification:    993 passed

Historical models and scripts remain available for traceability but should not replace this final evaluation chain.

---

## 28. Repository Philosophy

The repository is intentionally comprehensive.

Rather than removing substantive development artifacts, the project preserves the sequence of implementation, debugging, physical validation, model training, validation-based selection, locked TEST evaluation, benchmarking, ablation analysis, and publication preparation.

Automatically generated cache artifacts such as:

    __pycache__/
    .pytest_cache/
    *.pyc

do not represent research work and may be excluded from a final submission archive.

All substantive source code, experiment records, diagnostic scripts, validation outputs, and supporting research artifacts may be retained for traceability.

---

## 29. Final Verification Summary

Final reconstructed framework:

    FC-HMARL V3

System:

    Five interconnected microgrids

Learning architecture:

    Five local SAC agents
    One upper-level SAC coordinator

Training:

    5000 episodes

Checkpoint selection:

    Validation-based

Frozen checkpoint:

    200

Final evaluation:

    Locked TEST evaluation

Software verification:

    993 tests passed

The repository preserves the complete reconstructed implementation and its associated development, verification, evaluation, and publication workflow.
