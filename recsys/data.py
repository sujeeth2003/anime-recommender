"""Data for the Kaggle 'Anime Recommendations Database' (CooperUnion), plus a schema-identical synthetic generator.

Kaggle files (download from kaggle.com/datasets/CooperUnion/anime-recommendations-database; not redistributed here):
  anime.csv   anime_id, name, genre ("Action, Comedy, ..."), type, episodes, rating, members
  rating.csv  user_id, anime_id, rating   (rating = -1 means 'watched, but did not rate')
"""
import numpy as np
import pandas as pd

GENRES = ["Action", "Adventure", "Comedy", "Drama", "Fantasy", "Horror", "Mecha", "Music", "Mystery", "Romance", "Sci-Fi",
          "Slice of Life", "Sports", "Supernatural", "Thriller", "Historical", "Psychological", "School", "Shounen", "Seinen"]


