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

## Results (real data, 15k-user sample; hold out 20% of each user's ratings; candidates = every unrated title; relevant = held-out rating >= 8)
```
model                          RMSE    P@10    R@10  NDCG@10  coverage
popularity (damped mean)      1.515   0.036   0.026    0.043      0.2%
bias baseline                 1.287   0.018   0.014    0.017      0.2%
ALS ratings (k=24)            1.232   0.015   0.011    0.016     11.8%
implicit ALS (ranking)          n/a   0.256   0.224    0.338      6.0%
hybrid ALS + content            n/a   0.020   0.014    0.023     11.5%
```
What this shows:
- **Predicting ratings is not ranking.** ALS cuts RMSE by **19%** versus the popularity baseline (1.515 -> 1.232), but its top-10 is *worse* than popularity (NDCG 0.016 vs 0.043). It answers "how would this user score this title", not "what will this user watch".
- **The implicit model fixes ranking:** NDCG@10 **0.338, 7.9x popularity**, with precision@10 of 0.256 (about 1 in 4 recommended titles is one the user later rated 8+). It is personal: popularity shows everyone the same ~26 titles (0.2% coverage of the catalogue) while implicit ALS spreads across 6%.
- The hybrid does not help top-N here; the genre signal is weak next to 2M real ratings. It exists for cold start, which this offline split does not measure.
- Damped popularity is a strong baseline on RMSE only if you ignore user bias: the simple bias baseline (1.287) beats it, as expected on real data.
- Earlier I ran the same code on a synthetic dataset while I had no access to the real files; those numbers (33% RMSE, 1.8x) do not carry over: the real data shows a smaller rating gain and a much larger ranking gain.

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
