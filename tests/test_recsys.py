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

