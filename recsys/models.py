"""Recommenders, from simplest to most capable. All expose fit(train) and score matrix access via user/item index maps."""
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


class Index:
    def __init__(self, ratings, anime):
        self.users = {u: i for i, u in enumerate(sorted(ratings.user_id.unique()))}
        self.items = {a: i for i, a in enumerate(anime.anime_id)}
        self.item_ids = list(anime.anime_id)
        self.nu, self.ni = len(self.users), len(self.items)

    def matrix(self, ratings):
        r = ratings[ratings.user_id.isin(self.users) & ratings.anime_id.isin(self.items)]
        u = r.user_id.map(self.users).to_numpy(); i = r.anime_id.map(self.items).to_numpy()
        return sp.csr_matrix((r.rating.to_numpy(float), (u, i)), shape=(self.nu, self.ni))


