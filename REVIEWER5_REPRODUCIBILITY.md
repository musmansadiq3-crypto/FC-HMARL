# Reviewer 5 Reproducibility Package

## Final Evaluation Protocol

The final reviewer-revision experiments use the fixed terminal checkpoint at episode 5000.

- Seeds: 11, 22, 33, 44, 55
- TEST samples: 1,291
- Valid rolling 24-hour TEST windows: 1,268
- Deterministic policy inference: Yes
- Gradient updates during TEST: No
- Replay-buffer writes during TEST: No
- TEST-based checkpoint selection: No

The earlier checkpoint-200 experiment is retained only as historical development evidence. Final Reviewer 3/5 statistical conclusions use the fixed episode-5000 multi-seed protocol.

## Software Environment

The reproducibility environment was verified using:

- Python 3.13.9
- NumPy 2.3.5
- Pandas 2.3.3
- Matplotlib 3.10.6
- scikit-learn 1.7.2
- SciPy 1.16.3
- PyYAML 6.0.3
- PyTorch 2.13.0+cpu
- pytest 8.4.2

Exact dependencies are pinned in `requirements.txt`.

Clean-environment verification produced:

    969 passed, 37 warnings, 0 failures

## Installation

Create and activate a clean Python environment and run:

    python -m pip install -r requirements.txt
    python -m pip check
    python -m pytest -q

## Final Model Parameters

The final Full FC-HMARL evaluation uses five independently trained policies with seeds:

    11, 22, 33, 44, 55

For every seed, the release retains the episode-5000 parameters for:

- Coordinator
- Local agent 1
- Local agent 2
- Local agent 3
- Local agent 4
- Local agent 5
- Training state

The final Full checkpoints are stored under:

    outputs/comment3_ablation/full/seed_<SEED>/checkpoints/

Replay-buffer archives are not required for deterministic TEST inference and are excluded from the release.

## Controlled Ablation Study

Seven independently retrained configurations were evaluated:

1. Full
2. No-confidence
3. State-only
4. Reward-only
5. Concatenated forecast-and-confidence
6. Multiplicative weighting
7. No-coordinator

Five seeds were used for each configuration, producing 35 independent training runs.

Statistical evidence is stored under:

    outputs/comment3_ablation/final_statistical_evidence/

Seed-matched paired comparisons use Student-t 95% confidence intervals and paired t-tests.

The Full-versus-No-confidence scalar-return difference was not statistically significant (p = 0.4589). Therefore, the confidence mechanism is not claimed to provide a statistically established scalar-return improvement.

## Final Benchmark Evaluation

The final benchmark comparison includes:

- Passive grid-only operation
- Rule-based BESS EMS
- Full FC-HMARL

The passive and rule-based controllers are deterministic fixed baselines and are not assigned artificial seed variance.

Across five independently trained Full FC-HMARL policies:

    Mean net market cost = 8738.2688 USD/day
    SD = 36.8593 USD/day

Mean cost reduction relative to passive:

    4.56%
    95% CI = 4.06% to 5.06%

Mean cost reduction relative to rule-based BESS EMS:

    1.79%
    95% CI = 1.28% to 2.31%

All benchmark evaluations produced zero transformer-limit and SOC violations.

Evidence is stored under:

    outputs/final_benchmark_5000/final_statistical_evidence/

## Prediction-Interval Coverage

Prediction intervals use validation-only empirical residual quantiles. The interval parameters are frozen before application to the locked TEST set.

Nominal coverage:

    95%

Observed TEST PICP:

    Overall = 91.8724%
    PV      = 94.6166%
    Load    = 89.8980%
    EV      = 93.8226%
    Price   = 89.1525%

The intervals therefore show moderate held-out undercoverage and are not claimed to achieve exact nominal 95% TEST coverage.

Prediction-interval coverage is a forecasting diagnostic and is distinct from the causal confidence signal supplied to the controller.

Evidence is stored under:

    outputs/forecasting/prediction_interval_evidence/

## Confidence and Predictive State

The forecasting layer predicts four physical variables:

- PV
- Load
- EV demand
- Electricity price

The confidence-aware predictive state contains these forecast quantities together with the confidence channel. The five-feature state representation must therefore not be interpreted as five independently forecast physical variables.

Confidence calibration is performed using validation information before locked TEST evaluation.

## CVaR Role

Confidence-dependent operational risk information is used within the learning/control formulation.

CVaR reported in the final analysis is a post-processing risk statistic calculated after policy execution. It does not enter the SAC actor or critic gradient updates and is not presented as a CVaR-trained RL objective.

## Main Reproduction Scripts

Ablation training:

    python train_comment3_ablation.py

Ablation evaluation:

    python evaluate_comment3_ablation_test.py

Ablation statistics:

    python generate_comment3_final_statistics.py

Final benchmark evaluation:

    python evaluate_final_benchmarks_5000.py

Benchmark statistics:

    python generate_final_benchmark_statistics.py

Prediction-interval evidence:

    python generate_prediction_interval_evidence.py

## Reproducibility Principle

Final manuscript claims should be interpreted only according to the locked Reviewer 3/5 evidence described above. Historical development experiments are not used as evidence for the final statistical conclusions.
