# FC-HMARL Final Reproducibility Release

This repository contains the final reproducibility package for the
Forecast-Confidence-Aware Hierarchical Multi-Agent Reinforcement Learning
(FC-HMARL) framework.

## Final Experimental Protocol

- Configurations: 7
- Random seeds: 11, 22, 33, 44, 55
- Independent runs: 35
- Training episodes per run: 5000
- Fixed terminal checkpoint: episode 5000
- TEST windows per run: 1268
- Deterministic TEST actions: Yes
- Gradient updates during TEST: No
- Replay-buffer writes during TEST: No
- Checkpoint comparison/reselection on TEST: No

## Configurations

1. full
2. no_confidence
3. state_only
4. reward_only
5. concatenated
6. multiplicative
7. no_coordinator

Each configuration was evaluated using five independent seeds, giving
7 x 5 = 35 final experimental runs.

## Repository Structure

### code/
Contains the training, evaluation, forecasting-confidence, aggregation,
and protocol-verification source code.

### checkpoints/
Contains the final episode-5000 model artifacts associated with the
35 experimental runs.

### results/
Contains the final fixed-policy TEST results for the 35 runs.

### analysis/
Contains the aggregated five-seed results and forecast-confidence
reliability analyses.

### protocol/
Contains the locked experimental protocol, release manifest,
seed configuration, and reproducibility commands.

## Final TEST Protocol

Episode 5000 is the predefined terminal training checkpoint used for
final evaluation.

The TEST split is used only for fixed-policy final evaluation.
No checkpoint is selected, ranked, compared, or re-selected using TEST
performance.

All final TEST evaluations use deterministic actions with no gradient
updates and no replay-buffer writes.

## Verified Release Contents

The final release was checked for:

- 35/35 checkpoint sets
- 245/245 PT artifacts
- 35/35 TEST result sets
- 140/140 TEST artifacts
- 7 configurations
- 5 seeds per configuration
- checkpoint episode 5000 for all final runs
- 1268 TEST windows per run
- no checkpoint comparison on TEST
- no gradient updates during TEST
- no replay writes during TEST

## Reproduction

See:

protocol/RUN_COMMANDS.txt

for the training, evaluation, aggregation, confidence-analysis,
and protocol-verification commands.

Seed and configuration information is provided in:

protocol/seed_configuration.json

## Aggregated Results

Five-seed aggregated results are provided in:

analysis/ablation_5seed_mean_sd.csv

The 35 individual final TEST summaries are provided in:

analysis/all_35_final_test_summaries.csv

## Forecast-Confidence Reliability

Forecast-confidence reliability outputs are provided under:

analysis/confidence_reliability/

## Reproducibility Note

The final reported protocol uses checkpoint episode 5000 and 1268
available TEST windows per run. Earlier intermediate experimental
protocols are not part of this final release.
