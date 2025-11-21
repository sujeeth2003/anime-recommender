# Anime Recommender (Netflix-style)

Recommends what to watch next from the **MyAnimeList 2023 dataset** ([Kaggle: dbdmobile/myanimelist-dataset](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset)): personalised top-N shelves with "because you rated X highly" explanations, and a content-based fallback for titles nobody has rated yet.

## Data
The download has six CSVs. Two are used:
| File | Rows | Used for |
|---|---|---|
| `anime-dataset-2023.csv` | 25k titles | catalogue: name, genres, type |
| `users-score-2023.csv` | 24.3M ratings, 270k users, ratings 1-10 | the ratings |

The other four are not needed: `final_animedataset.csv` and `user-filtered.csv` are pre-joined / pre-filtered copies of the same ratings, `users-details-2023.csv` is profile metadata and `anime-filtered.csv` is an older catalogue. The files are not in this repo (they are several GB): download from Kaggle and point `--mal` at the folder.

**Sampling:** ALS solves one small system per user per iteration, so a full 24M-rating run is a cluster job. The experiment takes a reproducible random sample of **15,000 users who rated 20+ titles** (all of their ratings kept): **2.19M ratings over 12,876 titles, 1.1% dense**, 80/20 split per user. Numbers below are on that sample.

## Models (`recsys/models.py`)
| Model | What it is |
|---|---|
| Popularity | Bayesian-damped mean rating (a title with 3 tens must not outrank one with 3,000 nines) |
| Bias baseline | global mean + item bias + user bias |
| **ALS** (explicit) | alternating least squares with weighted-lambda regularisation on the residual of the bias baseline: predicts *ratings* |
| **Implicit ALS** | Hu-Koren-Volinsky confidence-weighted ALS: learns *who watched what*: optimises the ranking a "watch next" shelf is judged on |
| Content index | TF-IDF over genre + type: works with zero ratings (cold start) |
| Hybrid | ALS score + content-similarity to the user's liked titles |

## Results (synthetic; hold out 20% of each user's ratings; candidates = every unrated title; relevant = held-out rating >= 8)
```
model                          RMSE    P@10    R@10  NDCG@10  coverage
popularity (damped mean)      1.722   0.015   0.068    0.052      2.0%
bias baseline                 1.742   0.014   0.064    0.031      1.9%
ALS ratings (k=24)            1.151   0.013   0.055    0.032     70.6%
implicit ALS (ranking)          n/a   0.029   0.124    0.092     28.1%
hybrid ALS + content            n/a   0.013   0.056    0.033     72.4%
```
What this teaches:
- **Predicting ratings is not ranking.** ALS cuts RMSE by 33% versus popularity but is *no better at top-N* (NDCG 0.032 vs 0.052): it is good at "how would this user score this title" and poor at "which titles will this user actually watch", because people mostly watch popular titles and rating models ignore that.
- **The implicit model fixes ranking**: NDCG@10 0.092, **1.8x popularity**, with 14x the catalogue coverage of popularity (28% of titles recommended to someone vs 2%), so the shelves are personal instead of everyone seeing the same hits.
- The hybrid did not help here because the content signal is weak relative to the collaborative signal on this data; it is there for cold start, which this offline split does not measure (a proper cold-start evaluation needs titles held out entirely).
- "Popularity is a strong baseline, so beat it honestly": the damped popularity baseline is *better* than the naive bias baseline on RMSE here.

## Netflix-style output
```
because you rated highly: Title 43, Title 598, Title 82
  Title 153 [Sports, Shounen, Action]  score 0.64  (similar to Title 43)
  Title 751 [Thriller, Seinen]         score 0.63  (similar to Title 43)
```
The explanation line is the highest content-similarity title among the user's top-rated ones.

## Run
```bash
pip install numpy pandas scipy scikit-learn
python -m unittest discover -s tests       # 5 tests
python run_experiment.py                   # synthetic; add --kaggle DIR for the real data
```
At Kaggle scale the per-user solves in ALS are the cost (73k small ridge systems per half-iteration). They are independent, so they parallelise trivially; a production version would use `implicit`/GPU or vectorised batched solves.
