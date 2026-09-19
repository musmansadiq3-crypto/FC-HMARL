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

    Predictive state dimension:   120
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

## 13. Final Reviewer-Revision Evaluation Protocol

For the final reviewer-revision experiments, independently trained policies were evaluated at the fixed terminal checkpoint:

    Training episode:              5000
    Independent seeds:             11, 22, 33, 44, 55
    TEST samples:                  1291
    Valid rolling TEST windows:    1268
    Evaluation horizon:            24 hours
    Deterministic actions:         True
    Gradient updates:              False
    Replay-buffer writes:          False
    TEST checkpoint selection:     False

Episode 5000 was fixed before TEST evaluation and was not selected or re-ranked using TEST performance.

The earlier validation-selected checkpoint-200 experiment is retained as historical development evidence only. The final Reviewer 3/5 statistical evidence reported below uses the fixed episode-5000 multi-seed protocol.

---

## 14. Final Multi-Seed TEST Evaluation

Across the five independently trained Full FC-HMARL policies, the locked TEST evaluation produced:

    Mean total return:              -17858.9723
    Sample SD:                         142.2834
    95% CI:                    [-18035.6404, -17682.3041]

    Mean net market cost:             8738.2688 USD/day
    Sample SD:                          36.8593
    95% CI:                     [8692.5020, 8784.0357]

    Mean grid import:                46899.3443 kWh/day
    Mean grid export:                  298.2299 kWh/day
    Mean BESS throughput:             6007.4586 kWh/day
    Mean scheduled energy sharing:     265.4387 kWh/day

All five Full FC-HMARL evaluations produced zero transformer-limit violations and zero SOC violations.

The complete seed-level evidence is retained under:

    outputs/comment3_ablation/final_35_test_summaries/
    outputs/comment3_ablation/final_statistical_evidence/

---

## 15. Final Benchmark Evaluation

The final comparison includes:

1. Passive grid-only operation
2. Rule-based BESS energy-management operation
3. Full FC-HMARL

The passive and rule-based controllers are deterministic fixed baselines and are therefore evaluated once rather than assigned artificial seed variability. Full FC-HMARL statistics are calculated across five independently trained policies.

### Passive grid-only

    Mean return:              -18317.0302
    Net market cost:            9155.5552 USD/day
    Grid import:               48175.8432 kWh/day
    Grid export:                  42.9032 kWh/day
    BESS throughput:               0
    Energy sharing:                0

### Rule-based BESS EMS

    Mean return:              -17832.2530
    Net market cost:            8897.8077 USD/day
    Grid import:               46849.0098 kWh/day
    Grid export:                  58.1091 kWh/day
    BESS throughput:            1424.1548 kWh/day
    Energy sharing:                0

### Full FC-HMARL

    Mean return:              -17858.9723 +/- 142.2834
    Net market cost:            8738.2688 +/- 36.8593 USD/day
    Grid import:               46899.3443 +/- 504.7160 kWh/day
    Grid export:                 298.2299 +/- 44.7278 kWh/day
    BESS throughput:            6007.4586 +/- 1878.3620 kWh/day
    Energy sharing:              265.4387 +/- 272.6135 kWh/day

Relative to passive operation, Full FC-HMARL reduced mean net market cost by 4.56%, with a 95% confidence interval of 4.06% to 5.06%.

Relative to the rule-based BESS EMS, Full FC-HMARL reduced mean net market cost by 1.79%, with a 95% confidence interval of 1.28% to 2.31%.

A uniform scalar-return advantage over the rule-based controller was not observed. The hierarchical controller is therefore not presented as uniformly superior according to scalar return. Its additional functionality includes nonzero inter-microgrid energy sharing and coordinated operation.

All benchmark evaluations produced zero transformer-limit and SOC violations.

Final benchmark evidence is retained under:

    outputs/final_benchmark_5000/final_statistical_evidence/

---

## 16. Independently Retrained Ablation Study

Seven controller configurations were independently trained using the same five seeds:

1. Full FC-HMARL
2. No-confidence
3. State-only confidence
4. Reward-only confidence
5. Concatenated forecast-and-confidence
6. Multiplicative weighting
7. No-coordinator

This gives:

    7 configurations x 5 seeds = 35 independent training runs

All configurations were evaluated using the same fixed episode-5000 TEST protocol.

Mean TEST return +/- sample SD:

    Full:               -17858.9718 +/- 142.2920
    No-confidence:      -18266.1551 +/- 1074.0638
    State-only:         -17898.4254 +/- 392.5873
    Reward-only:        -18022.6104 +/- 342.5813
    Concatenated:       -18418.7642 +/- 1480.1728
    Multiplicative:     -18391.9871 +/- 1046.3540
    No-coordinator:     -17579.4071 +/- 4.5765

Seed-matched paired comparisons use:

    Delta R = R_Full - R_Ablation

Results:

    Comparison                 Delta R       p-value
    Full - No-confidence       +407.1833      0.4589
    Full - State-only           +39.4536      0.8535
    Full - Reward-only         +163.6385      0.3518
    Full - Concatenated        +559.7923      0.4552
    Full - Multiplicative      +533.0153      0.2756
    Full - No-coordinator      -279.5647      0.0112

The Full-versus-No-confidence difference is not statistically significant at the conventional 0.05 level. Therefore, the confidence mechanism is not claimed to provide a statistically established scalar-return improvement.

The no-coordinator configuration achieved a significantly higher scalar return in this comparison, but it produced zero inter-microgrid energy sharing and zero reserve coordination. Accordingly, the hierarchy is interpreted as providing coordination functionality rather than guaranteed scalar-return superiority.

All seven configurations produced zero transformer-limit and SOC violations.

Complete ablation statistics are retained under:

    outputs/comment3_ablation/final_statistical_evidence/

---

## 17. Prediction-Interval Coverage Evidence

Prediction intervals were reconstructed using validation-only empirical residual quantiles.

For each forecast target and horizon, the 2.5th and 97.5th percentiles of validation residuals were estimated and then frozen before application to the locked TEST set.

The nominal interval level was:

    95%

Observed locked-TEST prediction-interval coverage probability (PICP):

    Overall:    91.8724%
    PV:         94.6166%
    Load:       89.8980%
    EV:         93.8226%
    Price:      89.1525%

The overall empirical undercoverage relative to the nominal 95% level was 3.1276 percentage points. Therefore, the intervals are not described as achieving exact nominal 95% TEST coverage.

Prediction-interval coverage is reported as a forecasting calibration diagnostic and is distinct from the causal confidence signal used by the controller.

The validation-derived interval parameters were frozen before TEST evaluation. The locked TEST targets matched the archived TEST targets exactly; reconstruction of the validation targets differed from the archived values by at most approximately 2.93e-6 because of numerical precision and inverse scaling.

Complete evidence is retained under:

    outputs/forecasting/prediction_interval_evidence/

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

    969 passed, 37 warnings
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
    Final software verification:    969 passed, 37 warnings

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
