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


