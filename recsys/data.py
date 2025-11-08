"""Data for the Kaggle 'Anime Recommendations Database' (CooperUnion), plus a schema-identical synthetic generator.

Kaggle files (download from kaggle.com/datasets/CooperUnion/anime-recommendations-database; not redistributed here):
  anime.csv   anime_id, name, genre ("Action, Comedy, ..."), type, episodes, rating, members
  rating.csv  user_id, anime_id, rating   (rating = -1 means 'watched, but did not rate')
"""
import numpy as np
import pandas as pd

GENRES = ["Action", "Adventure", "Comedy", "Drama", "Fantasy", "Horror", "Mecha", "Music", "Mystery", "Romance", "Sci-Fi",
          "Slice of Life", "Sports", "Supernatural", "Thriller", "Historical", "Psychological", "School", "Shounen", "Seinen"]


def load_kaggle(directory):
    anime = pd.read_csv(f"{directory}/anime.csv")
    ratings = pd.read_csv(f"{directory}/rating.csv")
    anime["genre"] = anime["genre"].fillna("")
    ratings = ratings[ratings["rating"] >= 1].copy()          # drop the -1 'watched but unrated' rows for explicit-rating models
    return anime, ratings


def synthetic(n_users=3000, n_items=800, avg_ratings=40, seed=0, k_latent=6):
    """Users and titles live in a latent taste space; a title's genres come from its latent position, so genre
    content genuinely carries signal (as in real data) but does not explain everything. Popularity is long-tailed."""
    rng = np.random.default_rng(seed)
    item_f = rng.normal(0, 1, (n_items, k_latent))
    user_f = rng.normal(0, 1, (n_users, k_latent))
    G = rng.normal(0, 1, (k_latent, len(GENRES)))
    genre_score = item_f @ G
    genres = []
    for i in range(n_items):
        top = np.argsort(-genre_score[i])[: rng.integers(2, 5)]
        genres.append(", ".join(GENRES[j] for j in top))
    pop = rng.pareto(1.2, n_items) + 1; pop /= pop.sum()
    quality = rng.normal(0, 0.5, n_items)
    anime = pd.DataFrame({"anime_id": np.arange(1, n_items + 1), "name": [f"Title {i}" for i in range(1, n_items + 1)],
                          "genre": genres, "type": rng.choice(["TV", "Movie", "OVA"], n_items, p=[.6, .2, .2]),
                          "episodes": rng.integers(1, 60, n_items), "members": (pop * 1e6).astype(int)})
    rows = []
    for u in range(n_users):
        n = max(5, int(rng.poisson(avg_ratings)))
        items = rng.choice(n_items, size=min(n, n_items), replace=False, p=pop)
        aff = item_f[items] @ user_f[u] / np.sqrt(k_latent)
        r = np.clip(np.round(6.5 + 1.6 * aff + quality[items] + rng.normal(0, 0.8, len(items))), 1, 10)
        rows += [(u + 1, int(anime.anime_id[i]), int(v)) for i, v in zip(items, r)]
    ratings = pd.DataFrame(rows, columns=["user_id", "anime_id", "rating"])
    anime["rating"] = anime["anime_id"].map(ratings.groupby("anime_id")["rating"].mean())
    return anime, ratings

