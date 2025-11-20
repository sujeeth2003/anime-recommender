"""Compare popularity, bias baseline, ALS and the hybrid; then show Netflix-style output for one user.

    python run_experiment.py                       # synthetic data with the Kaggle schema
    python run_experiment.py --kaggle path/to/dir  # the older CooperUnion anime.csv + rating.csv
    python run_experiment.py --mal path/to/dir --users 15000   # MyAnimeList 2023 (6 CSVs)
"""
import argparse
import time

from recsys.data import load_kaggle, load_mal2023, split_per_user, synthetic
from recsys.evaluate import ranking_metrics, rmse_on
from recsys.models import ALS, BiasBaseline, ContentIndex, Hybrid, ImplicitALS, Index, Popularity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle")
    ap.add_argument("--mal", help="folder with the MyAnimeList 2023 files (anime-dataset-2023.csv, users-score-2023.csv)")
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

    t0 = time.time()
    pop = Popularity().fit(R); bias = BiasBaseline().fit(R); als = ALS().fit(R); ials = ImplicitALS().fit(R)
    content = ContentIndex(anime); hyb = Hybrid(als, content, R)
    print(f"fitted in {time.time() - t0:.1f}s\n")
    models = {"popularity (damped mean)": pop, "bias baseline": bias, "ALS ratings (k=24)": als, "implicit ALS (ranking)": ials, "hybrid ALS + content": hyb}
    print(f"{'model':<28}{'RMSE':>7}{'P@10':>8}{'R@10':>8}{'NDCG@10':>9}{'coverage':>10}")
    base = None
    for name, m in models.items():
        rm = rmse_on(m, idx, test) if hasattr(m, "predict") else float("nan")
        rk = ranking_metrics(m, idx, R, test)
        base = base or rm
        print(f"{name:<28}{rm:>7.3f}{rk['precision@k']:>8.3f}{rk['recall@k']:>8.3f}{rk['ndcg@k']:>9.3f}{rk['coverage']:>10.1%}")
    rm_als = rmse_on(als, idx, test)
    print(f"\nALS reduces RMSE by {1 - rm_als / base:.0%} versus the popularity baseline (implicit ALS and the hybrid have no per-pair rating predictor, ranking only).")

    u = 0; uid = sorted(idx.users, key=idx.users.get)[u]
    seen = R[u].indices; liked = sorted(seen, key=lambda i: -R[u, i])[:3]
    print(f"\n--- recommendations for user {uid} ---")
    print("because you rated highly:", ", ".join(anime.name.iloc[i] for i in liked))
    s = ials.predict_all(u).copy(); s[seen] = -1e9
    for i in s.argsort()[::-1][:5]:
        sim_to = max(liked, key=lambda l: float((content.X[i] @ content.X[l].T).toarray()[0, 0]))
        print(f"  {anime.name.iloc[i]:<12} [{anime.genre.iloc[i]}]  score {s[i]:.2f}  (similar to {anime.name.iloc[sim_to]})")


if __name__ == "__main__":
    main()
