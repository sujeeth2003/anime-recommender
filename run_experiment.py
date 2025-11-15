"""Compare popularity, bias baseline, ALS and the hybrid; then show Netflix-style output for one user.

    python run_experiment.py                       # synthetic data with the Kaggle schema
    python run_experiment.py --kaggle path/to/dir  # the real anime.csv + rating.csv
"""
import argparse
import time

from recsys.data import load_kaggle, split_per_user, synthetic
from recsys.evaluate import ranking_metrics, rmse_on
from recsys.models import ALS, BiasBaseline, ContentIndex, Hybrid, ImplicitALS, Index, Popularity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle")
    ap.add_argument("--users", type=int, default=3000)
    ap.add_argument("--items", type=int, default=800)
    a = ap.parse_args()
    anime, ratings = load_kaggle(a.kaggle) if a.kaggle else synthetic(a.users, a.items)
    src = a.kaggle or "synthetic (Kaggle schema)"
    train, test = split_per_user(ratings)
    idx = Index(ratings, anime)
    R = idx.matrix(train)
    print(f"data: {src}: {idx.nu} users x {idx.ni} titles, {len(ratings)} ratings ({len(ratings) / (idx.nu * idx.ni):.1%} dense); "
          f"train {len(train)} / test {len(test)}\n")

