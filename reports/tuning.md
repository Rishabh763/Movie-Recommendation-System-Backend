# Hyperparameter tuning

Tuned on a **validation split carved out of the training data** (each user's last 10% of
training ratings, by time), so the test set was never used to choose hyperparameters.
Ranking metrics use 5,000 sampled validation users.

## Biased MF (ALS, k = 64, 8 iterations): L2 strength

| reg | Validation RMSE |
|---|---|
| 0.02 | 0.8065 |
| **0.05** | **0.7753** |
| 0.10 | 0.8065 |

## Implicit ALS (k = 128, 12 iterations, reg = 0.05): confidence weighting

Confidence for an observed rating is `c = alpha * w`, where `w = max(rating - 2, 0.5)`
("rating") or `w = 1` ("binary"). Unobserved pairs have confidence 1 and preference 0.

| Weight | alpha | Validation NDCG@10 | Validation Recall@10 |
|---|---|---|---|
| rating | 0.25 | 0.0773 | 0.1081 |
| rating | 0.5 | 0.0827 | 0.1176 |
| **rating** | **1.0** | **0.0856** | **0.1223** |
| rating | 2.0 | 0.0846 | 0.1217 |
| rating | 4.0 | 0.0826 | |
| rating | 10.0 | 0.0774 | |
| binary | 0.25 | 0.0628 | 0.0879 |
| binary | 0.5 | 0.0687 | 0.0968 |
| binary | 1.0 | 0.0742 | 0.1068 |
| binary | 2.0 | 0.0745 | 0.1069 |

The regularization strength barely mattered (alpha = 1: reg 0.05 gives 0.0856, reg 1.0
gives 0.0855). Weighting confidence by the star rating beats treating every rating as an
equal positive by ~15% NDCG: even for "what will they watch next", how much someone liked
past movies carries signal.
