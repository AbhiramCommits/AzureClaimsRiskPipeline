# Model Comparison (2023-2024 policy-year holdout)

| Model | MAE | RMSE | Gamma Deviance | Pinball 0.5 | Pinball 0.9 | Pinball 0.99 | Norm. Gini | Calib slope | Calib intercept |
|---|---|---|---|---|---|---|---|---|---|
| global_mean | 11,661.7 | 80,945.3 | 35.9062 | 5,830.8 | 5,816.7 | 5,813.5 | 0.0000 | 0.4975 | 3,534.3 |
| tweedie_glm | 10,294.1 | 80,466.7 | 34.8987 | 5,147.1 | 5,574.9 | 5,671.1 | 0.1236 | 1.3284 | -900.4 |
| lightgbm | 10,698.5 | 80,388.1 | 34.7896 | 3,534.3 | 4,827.4 | 2,985.1 | 0.1181 | 1.0951 | -363.6 |

## LightGBM improvement over baselines (holdout)

| Metric | vs global mean | vs Tweedie GLM |
|---|---|---|
| MAE | 8.26% | -3.93% |
| RMSE | 0.69% | 0.10% |
| Gamma deviance | 3.11% | 0.31% |

Top-decile lift (LightGBM, holdout): **4.0822**
