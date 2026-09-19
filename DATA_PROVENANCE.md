# Data Provenance and Chronological Split

## 1. Final forecasting data

The FC-HMARL forecasting pipeline uses four hourly variables: PV generation, electrical load, EV charging demand, and electricity price. The unified forecasting dataset contains 8,760 hourly records representing one common 2022 calendar year.

The final data sources represented in the repository are:

- PV generation: NSRDB PV data, 2022.
- Electrical load: Pecan Street data. The processed source represents measurements originating from 2014-2018 and is mapped into the common annual profile used by the forecasting pipeline.
- EV charging demand: ACN-Data, Caltech, 2018-2020, obtained through the ACN-Data REST API. Observed charging-session energy is converted into an hourly EV profile and then into a representative 8,760-hour annual profile.
- Electricity price: the cleaned price series included in the unified forecasting pipeline and aligned to the common 2022 hourly timeline.

The different source periods are therefore not presented as simultaneous measurements from one physical site. They are processed independently and temporally aligned to construct the common representative annual operating dataset used in the experiments.

## 2. Temporal resolution and alignment

All four forecasting variables are represented at an hourly resolution. The unified dataset is checked for exactly 8,760 rows, duplicate timestamps, and missing forecasting targets.

The dataset-building stage performs source cleaning, temporal alignment, and construction of the unified annual dataset. It does not itself perform the machine-learning split.

## 3. Chronological train/validation/test split

After construction of the unified time series, the forecasting configuration applies a chronological split:

- Training: 70%
- Validation: 15%
- Test: 15%

The chronological order is preserved; samples are not randomly shuffled across these three temporal partitions.

Training data are used for model fitting. Validation data are used for forecast-method selection, confidence calibration, and model/checkpoint selection. The TEST partition is reserved for the final locked evaluation and is not used for checkpoint selection.

## 4. RL archive provenance

The RL training, validation, and TEST archives are derived from the corresponding forecasting partitions. The final TEST archive explicitly records:

- source_split = test
- forecast_method_selection_split = validation
- confidence_calibration_split = validation
- test_split_used_for_checkpoint_selection = False

The final Reviewer 3/5 evaluation uses the fixed episode-5000 protocol. The earlier validation-selected checkpoint-200 experiment is retained only as historical development evidence.

## 5. Reproducibility note

The repository contains the preprocessing and archive-construction scripts required to trace the processed data into the forecasting and RL inputs. Source periods are reported explicitly because the unified annual dataset combines processed profiles originating from different datasets and historical periods rather than claiming a single-site contemporaneous measurement campaign.
