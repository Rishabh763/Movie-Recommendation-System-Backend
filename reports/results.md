Evaluated on MovieLens 32M: 25,458,060 train / 6,267,836 test ratings, 200,945 users, 23,350 movies. Ranking metrics over 20,000 sampled users; relevant = held-out rating >= 4.

| Model | RMSE | Recall@10 | NDCG@10 | Coverage@10 | Fit time |
|---|---|---|---|---|---|
| Popularity (most rated) | – | 0.0744 | 0.0713 | 0.9% | 0s |
| Popularity (damped mean) | – | 0.0260 | 0.0210 | 0.2% | 0s |
| Bias baseline (mu + b_u + b_i) | 0.8666 | 0.0000 | 0.0000 | 0.1% | 9s |
| Item-kNN (centered cosine) | – | 0.1195 | 0.1171 | 7.8% | 52s |
| Biased MF (ALS, k=64) | 0.7883 | 0.0067 | 0.0051 | 22.8% | 178s |
| Implicit ALS (k=128) | – | 0.1262 | 0.1174 | 8.0% | 78s |
