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


class Popularity:
    """Bayesian (damped) mean rating: a title with 3 ratings of 10 must not outrank one with 3000 ratings of 9."""

    def __init__(self, damping=20):
        self.damping = damping

    def fit(self, R):
        cnt = np.asarray((R > 0).sum(0)).ravel(); tot = np.asarray(R.sum(0)).ravel()
        self.mu = tot.sum() / max(cnt.sum(), 1)
        self.item_score = (tot + self.damping * self.mu) / (cnt + self.damping)
        return self

    def predict_all(self, u):
        return self.item_score

