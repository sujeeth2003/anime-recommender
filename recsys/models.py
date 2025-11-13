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

    def predict(self, u, i):
        return self.item_score[i]


class BiasBaseline:
    """global mean + item bias + user bias (the standard strong baseline for rating prediction)."""

    def __init__(self, reg=10.0):
        self.reg = reg

    def fit(self, R):
        coo = R.tocoo(); self.mu = coo.data.mean()
        self.bi = np.zeros(R.shape[1]); self.bu = np.zeros(R.shape[0])
        for _ in range(8):
            res = coo.data - self.mu - self.bu[coo.row]
            self.bi = np.bincount(coo.col, res, R.shape[1]) / (np.bincount(coo.col, minlength=R.shape[1]) + self.reg)
            res = coo.data - self.mu - self.bi[coo.col]
            self.bu = np.bincount(coo.row, res, R.shape[0]) / (np.bincount(coo.row, minlength=R.shape[0]) + self.reg)
        return self

    def predict_all(self, u):
        return self.mu + self.bu[u] + self.bi

    def predict(self, u, i):
        return self.mu + self.bu[u] + self.bi[i]


class ALS:
    """Alternating least squares with weighted-lambda regularisation (ALS-WR, Zhou et al. 2008) on explicit ratings,
    fit to the residual of a bias baseline. Each half-step solves one small k x k ridge system per user (or item)."""

    def __init__(self, k=24, reg=0.08, iters=12, seed=0):
        self.k, self.reg, self.iters, self.rng = k, reg, iters, np.random.default_rng(seed)

    def fit(self, R):
        self.base = BiasBaseline().fit(R)
        coo = R.tocoo()
        res = coo.data - self.base.mu - self.base.bu[coo.row] - self.base.bi[coo.col]
        Rres = sp.csr_matrix((res, (coo.row, coo.col)), shape=R.shape); Rt = Rres.T.tocsr()
        self.U = self.rng.normal(0, 0.1, (R.shape[0], self.k)); self.V = self.rng.normal(0, 0.1, (R.shape[1], self.k))
        for _ in range(self.iters):
            self.U = self._solve(Rres, self.V); self.V = self._solve(Rt, self.U)
        return self

    def _solve(self, M, F):
        out = np.zeros((M.shape[0], self.k)); eye = np.eye(self.k)
        for r in range(M.shape[0]):
            s, e = M.indptr[r], M.indptr[r + 1]
            if s == e: continue
            Fi = F[M.indices[s:e]]
            out[r] = np.linalg.solve(Fi.T @ Fi + self.reg * (e - s) * eye, Fi.T @ M.data[s:e])
        return out

    def predict_all(self, u):
        return self.base.predict_all(u) + self.V @ self.U[u]

    def predict(self, u, i):
        return self.base.predict(u, i) + np.einsum("ij,ij->i", self.U[u], self.V[i])


class ImplicitALS:
    """Implicit-feedback ALS (Hu, Koren, Volinsky 2008): the model learns to reproduce WHO WATCHED WHAT, with confidence
    1 + alpha*rating on watched pairs, and a small weight on every unwatched pair. This optimises the thing a
    'what should I watch next' shelf is judged on (ranking watched titles first), unlike rating-prediction ALS."""

    def __init__(self, k=32, reg=0.1, alpha=4.0, iters=10, seed=0):
        self.k, self.reg, self.alpha, self.iters, self.rng = k, reg, alpha, iters, np.random.default_rng(seed)

    def fit(self, R):
        C = R.copy().tocsr(); C.data = 1.0 + self.alpha * C.data / 10.0        # confidence on observed pairs
        Ct = C.T.tocsr()
        self.U = self.rng.normal(0, 0.1, (R.shape[0], self.k)); self.V = self.rng.normal(0, 0.1, (R.shape[1], self.k))
        for _ in range(self.iters):
            self.U = self._solve(C, self.V); self.V = self._solve(Ct, self.U)
        return self

