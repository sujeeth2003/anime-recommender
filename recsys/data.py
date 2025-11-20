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


def load_mal2023(directory, n_users=15000, min_ratings=20, seed=0):
    """MyAnimeList 2023 dataset (kaggle.com/datasets/dbdmobile/myanimelist-dataset).
    Uses two of its six files:
      anime-dataset-2023.csv   anime_id, Name, Genres, Type, Score, Members, ...   (the catalogue: ~25k titles)
      users-score-2023.csv     user_id, Username, anime_id, Anime Title, rating    (24.3M explicit 1-10 ratings, 270k users)
    The other four are not needed: final_animedataset.csv and user-filtered.csv are pre-joined/pre-filtered copies of the
    same ratings, users-details-2023.csv is profile metadata, and anime-filtered.csv is an older catalogue.
    A full ALS over 24M ratings needs a cluster-style solver, so this takes a reproducible random sample of `n_users`
    users who rated at least `min_ratings` titles, keeping all of each sampled user's ratings."""
    anime = pd.read_csv(f"{directory}/anime-dataset-2023.csv", usecols=["anime_id", "Name", "Genres", "Type", "Score", "Members"])
    anime = anime.rename(columns={"Name": "name", "Genres": "genre", "Type": "type", "Score": "mal_score", "Members": "members"})
    anime["genre"] = anime["genre"].replace("UNKNOWN", "").fillna("")
    ratings = pd.read_csv(f"{directory}/users-score-2023.csv", usecols=["user_id", "anime_id", "rating"], dtype="int32")
    ratings = ratings[ratings.anime_id.isin(anime.anime_id)]
    counts = ratings.groupby("user_id").size()
    eligible = counts[counts >= min_ratings].index.to_numpy()
    keep = np.random.default_rng(seed).choice(eligible, size=min(n_users, len(eligible)), replace=False)
    ratings = ratings[ratings.user_id.isin(keep)].reset_index(drop=True)
    rated = anime[anime.anime_id.isin(ratings.anime_id)].reset_index(drop=True)      # catalogue = titles someone in the sample rated
    return rated, ratings


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


def split_per_user(ratings, test_frac=0.2, seed=0, min_keep=3):
    """Hold out a random fraction of EACH user's ratings (so every test user has training history)."""
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, idx in ratings.groupby("user_id").indices.items():
        if len(idx) > min_keep + 1:
            test_idx += list(rng.choice(idx, size=max(1, int(len(idx) * test_frac)), replace=False))
    mask = np.zeros(len(ratings), bool); mask[test_idx] = True
    return ratings[~mask].reset_index(drop=True), ratings[mask].reset_index(drop=True)
