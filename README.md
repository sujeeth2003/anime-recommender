# Anime Recommender (Netflix-style)

Recommends what to watch next from the **Kaggle "Anime Recommendations Database"** (`anime.csv` + `rating.csv`, ~73k users x 12k titles): personalised top-N shelves with "because you rated X highly" explanations, and content-based fallback for titles nobody has rated yet.

> **Data note:** the Kaggle files need a Kaggle login, so they are not bundled and I did not download them. The code reads the exact Kaggle schema (`python run_experiment.py --kaggle path/to/dir`, which also drops the `-1` "watched but unrated" rows for the explicit-rating models). The results below are on a **schema-identical synthetic dataset** (3,000 users x 800 titles, 5% dense, long-tailed popularity, genre-correlated taste), so they show the method and the trade-offs, **not** performance on the real data.

## Models (`recsys/models.py`)
| Model | What it is |
|---|---|
| Popularity | Bayesian-damped mean rating (a title with 3 tens must not outrank one with 3,000 nines) |
| Bias baseline | global mean + item bias + user bias |
| **ALS** (explicit) | alternating least squares with weighted-lambda regularisation on the residual of the bias baseline: predicts *ratings* |
| **Implicit ALS** | Hu-Koren-Volinsky confidence-weighted ALS: learns *who watched what*: optimises the ranking a "watch next" shelf is judged on |
| Content index | TF-IDF over genre + type: works with zero ratings (cold start) |
| Hybrid | ALS score + content-similarity to the user's liked titles |

