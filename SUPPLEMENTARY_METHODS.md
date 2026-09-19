# Supplementary Methods for Final Reviewer Revision

## 1. Forecasting model and predictive state

The forecasting subsystem predicts four physical variables: PV power, load, EV power, and electricity price. A 168-hour historical input window is used to generate a 24-hour-ahead forecast.

The Transformer configuration used in the final forecasting pipeline has hidden dimension 128, four attention heads, two Transformer layers, feed-forward dimension 256, dropout 0.1, GELU activation, and learnable positional encoding.

Forecast-method selection is performed using VALIDATION only. The final hybrid forecast uses:

- PV: daily-seasonal forecast
- Load: Transformer
- EV: Transformer
- Price: daily-seasonal forecast

The TEST split is not used to select these forecasting methods.

For RL, the 24-hour forecasts of four physical variables give 96 forecast values. Confidence information is then appended to the predictive representation, producing the final 120-dimensional predictive state used by the final pipeline. Thus, the fifth channel is confidence information and must not be interpreted as a fifth physical forecasting variable.

## 2. Confidence construction and calibration

Confidence is computed causally from forecasting uncertainty information calibrated without using the locked TEST set for model or checkpoint selection. Confidence values are bounded to [0,1].

The confidence calculation uses the four physical forecast variables with equal weighting in the final confidence configuration. The resulting confidence information is incorporated into the predictive state and confidence-aware control formulation.

The causal confidence variable is distinct from the prediction-interval coverage analysis. Prediction intervals are generated separately for Reviewer 5 using validation residual quantiles and are frozen before application to the locked TEST predictions.

The nominal 95% prediction-interval analysis gives an overall TEST PICP of 91.8724%, indicating moderate held-out undercoverage rather than claiming exact 95% empirical coverage.

## 3. Hierarchical SAC control

The final FC-HMARL architecture contains five local SAC agents and one upper-level coordinator.

Local agents determine local continuous energy-management actions subject to the implemented physical constraints. The coordinator provides the upper-level coordination mechanism associated with inter-MEVPP energy sharing and reserve-related coordination.

The independently retrained no-coordinator ablation removes this coordination functionality; consequently, energy sharing and reserve-related quantities are zero in that configuration.

The final predictive-state dimension used by the control pipeline is 120. The final evaluation uses deterministic policy actions and does not perform gradient updates, replay-buffer writes, or TEST-time checkpoint selection.

## 4. Reward, feasibility, and risk treatment

The control objective combines economic and operational terms implemented in the reward pipeline while enforcing the modeled operating constraints. Transformer and SOC violations are reported separately in the final evaluation evidence.

Confidence-dependent operational risk information is used within the learning/control formulation. CVaR, however, is not used to update the SAC actor or critic networks in the final implementation. CVaR is calculated as a post-execution risk-analysis quantity. The manuscript therefore does not describe CVaR as the SAC training objective.

## 5. Final evaluation protocol

Reviewer 3/5 evidence uses independently evaluated episode-5000 policies for five fixed seeds:

11, 22, 33, 44, and 55.

The locked TEST archive contains 1,291 samples and provides 1,268 valid 24-hour evaluation windows.

During final evaluation:

- deterministic actions are enabled;
- gradient updates are disabled;
- replay-buffer writes are disabled;
- TEST data are not used for checkpoint selection.

The ablation study contains seven configurations evaluated over five seeds, for 35 final runs:

1. Full FC-HMARL
2. No-confidence
3. State-only
4. Reward-only
5. Concatenated forecast-and-confidence
6. Multiplicative weighting
7. No-coordinator

Seed-matched paired statistical comparisons use Full minus Ablation return differences. The Full versus No-confidence comparison is not statistically significant (p = 0.4589), so the final manuscript does not claim statistically established scalar-return superiority from the confidence mechanism.

The no-coordinator configuration achieves a higher scalar return in the paired comparison (Full minus No-coordinator mean difference = -279.564736, p = 0.0112), while eliminating energy-sharing and reserve functionality. Accordingly, the hierarchy is interpreted as enabling coordination functionality rather than as providing universal scalar-return superiority.

## 6. Benchmark interpretation

The final fixed-baseline comparison includes passive grid-only operation, a rule-based BESS EMS, and the five-seed Full FC-HMARL evaluation.

Mean Full FC-HMARL net market cost is 8738.268850, compared with 9155.555160 for passive grid-only operation and 8897.807730 for the rule-based BESS EMS. This corresponds to mean cost reductions of approximately 4.56% and 1.79%, respectively.

These cost results are reported separately from scalar return. The final evidence does not claim uniform scalar-return superiority over the rule-based controller.

## 7. Software reproducibility

The locked software environment is documented in requirements.txt and REVIEWER5_REPRODUCIBILITY.md. Clean-environment verification completed with:

969 passed, 37 warnings, 0 failures.

The final statistical evidence, benchmark evidence, prediction-interval evidence, model checkpoints, evaluation scripts, and reproducibility protocol are retained in the repository.
