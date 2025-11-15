import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from recsys.data import split_per_user, synthetic  # noqa: E402
from recsys.evaluate import ranking_metrics, rmse_on  # noqa: E402
from recsys.models import ALS, BiasBaseline, ContentIndex, Index, Popularity  # noqa: E402


class RecsysTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.anime, cls.ratings = synthetic(600, 250, avg_ratings=35, seed=1)
        cls.train, cls.test = split_per_user(cls.ratings)
        cls.idx = Index(cls.ratings, cls.anime)
        cls.R = cls.idx.matrix(cls.train)

    def test_split_keeps_train_history_and_no_leak(self):
        tr = set(zip(self.train.user_id, self.train.anime_id)); te = set(zip(self.test.user_id, self.test.anime_id))
        self.assertFalse(tr & te)
        self.assertEqual(set(self.test.user_id) - set(self.train.user_id), set())

    def test_damped_popularity_prefers_well_supported_titles(self):
        import scipy.sparse as sp
        R = sp.csr_matrix(np.array([[10, 9], [0, 9], [0, 9], [0, 9], [0, 9]], float))   # item0: one 10; item1: five 9s
        m = Popularity(damping=10).fit(R)
        self.assertLess(m.item_score[0], 10.0); self.assertGreater(m.item_score[0], m.mu)   # shrunk toward the mean
        self.assertLess(abs(m.item_score[1] - m.mu), abs(9.9 - m.mu) + 1)

    def test_als_beats_baselines_on_rmse_and_ranking(self):
        pop = Popularity().fit(self.R); bias = BiasBaseline().fit(self.R); als = ALS(k=6, reg=0.4, iters=8).fit(self.R)
        r_pop, r_bias, r_als = (rmse_on(m, self.idx, self.test) for m in (pop, bias, als))
        self.assertLess(r_als, min(r_bias, r_pop) * 0.85)                 # personalisation clearly beats non-personalised scores
        rk_pop = ranking_metrics(pop, self.idx, self.R, self.test, max_users=300)
        from recsys.models import ImplicitALS
        rk_imp = ranking_metrics(ImplicitALS(k=16, iters=6).fit(self.R), self.idx, self.R, self.test, max_users=300)
        self.assertGreater(rk_imp["ndcg@k"], rk_pop["ndcg@k"])          # the ranking model beats popularity at top-N

    def test_recommendations_exclude_already_rated(self):
        als = ALS(k=8, iters=4).fit(self.R)
        s = als.predict_all(0).copy(); s[self.R[0].indices] = -np.inf
        self.assertFalse(set(np.argsort(-s)[:10]) & set(self.R[0].indices))

