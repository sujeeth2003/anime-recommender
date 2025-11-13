import numpy as np


def rmse_on(model, index, test):
    u = test.user_id.map(index.users).to_numpy(); i = test.anime_id.map(index.items).to_numpy()
    ok = ~(np.isnan(u.astype(float)) | np.isnan(i.astype(float)))
    pred = np.clip(model.predict(u[ok].astype(int), i[ok].astype(int)), 1, 10)
    return float(np.sqrt(np.mean((pred - test.rating.to_numpy()[ok]) ** 2)))


