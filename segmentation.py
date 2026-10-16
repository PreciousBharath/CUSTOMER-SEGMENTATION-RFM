"""RFM feature engineering + K-Means clustering pipeline."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def make_transactions(seed=3) -> pd.DataFrame:
    """Synthetic e-commerce orders with 4 hidden customer behaviours."""
    rng = np.random.default_rng(seed)
    personas = [  # (n customers, orders λ, avg spend, recency scale days)
        ("champion", 350, 22, 4200, 12), ("loyal", 700, 9, 2200, 35),
        ("at_risk", 550, 6, 2600, 190), ("new_or_low", 1400, 1.6, 900, 60),
    ]
    rows, cid, end = [], 10000, pd.Timestamp("2025-12-31")
    for _, n, lam, spend, rec in personas:
        for _ in range(n):
            cid += 1
            k = max(1, rng.poisson(lam))
            last = end - pd.Timedelta(days=int(rng.exponential(rec)))
            span = int(rng.integers(30, 720))
            days = np.sort(rng.integers(0, span, k))
            for d in days:
                rows.append((cid, last - pd.Timedelta(days=int(span - d)), max(99, rng.normal(spend, spend * .35))))
    return pd.DataFrame(rows, columns=["customer_id", "order_date", "amount"]).round({"amount": 0})


def rfm_table(tx: pd.DataFrame) -> pd.DataFrame:
    ref = tx.order_date.max() + pd.Timedelta(days=1)
    rfm = tx.groupby("customer_id").agg(
        recency=("order_date", lambda d: (ref - d.max()).days),
        frequency=("order_date", "count"),
        monetary=("amount", "sum"),
    )
    return rfm


def scale(rfm):
    X = np.log1p(rfm[["recency", "frequency", "monetary"]])
    return StandardScaler().fit_transform(X)


def k_search(X, ks=range(2, 9)):
    out = []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=42).fit(X)
        out.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_, sample_size=2000, random_state=42)})
    return pd.DataFrame(out)


SEGMENT_ACTIONS = {
    "Champions": "Reward with VIP perks, early access & referral programme",
    "Loyal Customers": "Upsell premium lines, loyalty points multiplier",
    "At Risk": "Win-back campaign: personalised discount + feedback survey",
    "Promising / New": "Onboarding journey, first-repeat-purchase coupon",
    "Hibernating": "Low-cost reactivation emails; suppress from paid ads",
}


def name_clusters(prof):
    """Champions = best value score; others placed on a frequency x recency quadrant."""
    z = (prof - prof.mean()) / prof.std()
    score = z.frequency + z.monetary - z.recency
    champ = score.idxmax()
    mapping = {champ: "Champions"}
    f_mid, r_mid = prof.frequency.median(), prof.recency.median()
    for c in prof.index.drop(champ):
        hi_f, recent = prof.loc[c, "frequency"] >= f_mid, prof.loc[c, "recency"] <= r_mid
        mapping[c] = ("Loyal Customers" if recent else "At Risk") if hi_f else ("Promising / New" if recent else "Hibernating")
    seen = {}
    for c in sorted(mapping, key=lambda c: -score[c]):  # disambiguate duplicates when k > 4
        n = mapping[c]
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            mapping[c] = f"{n} {seen[n]}"
    return mapping


def cluster(rfm, k=4):
    X = scale(rfm)
    km = KMeans(k, n_init=20, random_state=42).fit(X)
    rfm = rfm.copy()
    rfm["cluster"] = km.labels_
    prof = rfm.groupby("cluster")[["recency", "frequency", "monetary"]].mean()
    mapping = name_clusters(prof)
    rfm["segment"] = rfm.cluster.map(mapping)
    pcs = PCA(2, random_state=42).fit_transform(X)
    rfm["pc1"], rfm["pc2"] = pcs[:, 0], pcs[:, 1]
    sil = silhouette_score(X, km.labels_, sample_size=2000, random_state=42)
    return rfm, sil
