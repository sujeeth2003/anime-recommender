"""Compare popularity, bias baseline, ALS and the hybrid; then show Netflix-style output for one user.

    python run_experiment.py                       # synthetic data with the Kaggle schema
    python run_experiment.py --kaggle path/to/dir  # the real anime.csv + rating.csv
"""
import argparse
import time

from recsys.data import load_kaggle, split_per_user, synthetic
from recsys.evaluate import ranking_metrics, rmse_on
from recsys.models import ALS, BiasBaseline, ContentIndex, Hybrid, ImplicitALS, Index, Popularity


