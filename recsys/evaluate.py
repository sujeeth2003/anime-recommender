import numpy as np


def rmse_on(model, index, test):
    u = test.user_id.map(index.users).to_numpy(); i = test.anime_id.map(index.items).to_numpy()
    ok = ~(np.isnan(u.astype(float)) | np.isnan(i.astype(float)))
    pred = np.clip(model.predict(u[ok].astype(int), i[ok].astype(int)), 1, 10)
    return float(np.sqrt(np.mean((pred - test.rating.to_numpy()[ok]) ** 2)))


def ranking_metrics(model, index, R_train, test, k=10, like=8, max_users=1500, seed=0):
    """Precision@K, Recall@K, NDCG@K, catalogue coverage. Relevant = held-out titles rated >= `like`.
    Candidates = every title the user has NOT already rated in training (the honest, hard setting)."""
    rng = np.random.default_rng(seed)
    rel = test[test.rating >= like]
    users = [u for u in rel.user_id.unique() if u in index.users]
    users = list(rng.choice(users, min(max_users, len(users)), replace=False))
    by_user = rel.groupby("user_id")["anime_id"].apply(lambda s: {index.items[a] for a in s if a in index.items})
    P, Rc, N, seen_items = [], [], [], set()
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    for uid in users:
        u = index.users[uid]
        s = np.asarray(model.predict_all(u), float).copy()
        s[R_train[u].indices] = -np.inf
        top = np.argpartition(-s, k)[:k]; top = top[np.argsort(-s[top])]
        seen_items.update(top.tolist())
        truth = by_user[uid]
        hits = np.array([t in truth for t in top], float)
        P.append(hits.mean()); Rc.append(hits.sum() / len(truth))
        ideal = disc[: min(len(truth), k)].sum()
        N.append((hits * disc).sum() / ideal)
    return {"precision@k": float(np.mean(P)), "recall@k": float(np.mean(Rc)), "ndcg@k": float(np.mean(N)),
            "coverage": len(seen_items) / index.ni, "users": len(users)}
