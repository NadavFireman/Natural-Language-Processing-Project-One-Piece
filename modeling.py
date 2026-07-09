"""
modeling.py
One Piece NLP Project — modeling & evaluation layer.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import features as F
import text_features as TF


SEED = 42

STRUCTURED_BOOL = [
    "has_D", "is_world_government", "is_marine", "is_cipher_pol", "is_celestial",
    "is_gorosei", "is_pirate", "is_revolutionary", "is_bounty_hunter",
    "is_yonko", "is_warlord", "is_worst_generation", "is_captain",
    "is_grand_fleet", "is_yonko_crew", "has_devil_fruit", "df_awakened",
    "has_haki", "has_armament_haki", "has_observation_haki",
    "has_conqueror_haki", "is_royalty", "is_deceased", "epithet_present",
]
STRUCTURED_CAT = ["df_type", "race", "gender", "origin_sea"]
STRUCTURED_NUM = ["bounty_n_values", "bounty_changes"]


def make_regressor(seed=SEED):
    try:
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations=600, depth=6, learning_rate=0.05,
            loss_function="RMSE", random_seed=seed, verbose=False), "CatBoost"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_iter=600, max_depth=6, learning_rate=0.05,
            random_state=seed), "HistGradientBoosting"


# ----------------------------------------------------------------------
# Optional sample weights (counter the under-fitting of the high end)
# ----------------------------------------------------------------------
def make_sample_weights(df, yonko=4.0, warlord=2.0,
                        top_quantile=0.9, top_boost=2.0, base=1.0):
    """Optional per-row TRAINING weights, to stop the regression collapsing the
    top of the bounty range toward the mean (it has very few very-high examples).
    Up-weights the most notorious LABELED characters: Yonko and Warlords (from
    the data-derived is_yonko / is_warlord flags) plus the top bounty quantile.

    NOTE: the Gorosei and Imu CANNOT be weighted — they carry no bounty, so they
    are prediction-only rows, never in training. This lever raises predictions
    for profiles that RESEMBLE the weighted groups; it does not by itself fix
    those out-of-distribution archetypes. Pass to evaluate(weight_fn=...) or use
    in the application; the headline comparison stays unweighted (honest)."""
    w = np.full(len(df), float(base))
    if "is_yonko" in df:
        w[df["is_yonko"].fillna(False).to_numpy()] *= yonko
    if "is_warlord" in df:
        w[df["is_warlord"].fillna(False).to_numpy()] *= warlord
    if top_quantile is not None and "bounty" in df:
        b = df["bounty"].astype(float)
        thr = b.quantile(top_quantile)
        w[(b >= thr).to_numpy()] *= top_boost
    return w


def target_frame(df, canon_only=True, wanted_only=False):
    m = df["bounty"].notna()
    if "is_crew" in df:
        m &= ~df["is_crew"].fillna(False)
    if canon_only and "is_non_canon" in df:
        m &= ~df["is_non_canon"].fillna(False)
    # wanted_only=True drops star-rated WG valuations (is_star_bounty: {{B|s}}
    # Marine ratings + {{C|N}} Cross Guild), leaving only plain {{B}} wanted
    # bounties — a single, clean target.
    if wanted_only and "is_star_bounty" in df:
        m &= ~df["is_star_bounty"].fillna(False)
    out = df[m].copy().reset_index(drop=True)
    out["y_log"] = np.log1p(out["bounty"].astype(float))
    return out


def _bins(y_log, n=5):
    q = pd.qcut(y_log, q=min(n, len(np.unique(y_log))), labels=False, duplicates="drop")
    return q.astype(int)


def _structured_matrix(tr, te):
    Xtr = tr[[c for c in STRUCTURED_BOOL if c in tr]].astype(float).reset_index(drop=True)
    Xte = te[[c for c in STRUCTURED_BOOL if c in te]].astype(float).reset_index(drop=True)
    for c in STRUCTURED_NUM:
        if c not in tr:
            continue
        Xtr[c] = pd.to_numeric(tr[c], errors="coerce").fillna(0.0).values
        Xte[c] = pd.to_numeric(te[c], errors="coerce").fillna(0.0).values
    for c in STRUCTURED_CAT:
        if c not in tr:
            continue
        cats = [v for v in tr[c].fillna("NA").unique()]
        for v in cats:
            Xtr[f"{c}={v}"] = (tr[c].fillna("NA").values == v).astype(float)
            Xte[f"{c}={v}"] = (te[c].fillna("NA").values == v).astype(float)
    return Xtr, Xte


def _network_matrix(tr, te):
    cols = [c for c in F.GRAPH_FEATURE_COLS if c in tr]
    return (tr[cols].astype(float).reset_index(drop=True),
            te[cols].astype(float).reset_index(drop=True))


def _text_matrix(tr, te, embedder=None, use_embeddings=True):
    tr = TF.add_clean_tokens(tr); te = TF.add_clean_tokens(te)
    vec = TF.fit_tfidf(tr["text_clean"])
    svd = TF.fit_tfidf_svd(vec, tr["text_clean"])
    Xtr = TF.transform_tfidf(vec, svd, tr["text_clean"]).reset_index(drop=True)
    Xte = TF.transform_tfidf(vec, svd, te["text_clean"]).reset_index(drop=True)
    Xtr["threat_score"] = tr["clean_text"].apply(TF.threat_score).values
    Xte["threat_score"] = te["clean_text"].apply(TF.threat_score).values
    if use_embeddings and embedder is not None:
        etr = TF.embed_texts(embedder, tr["clean_text"].tolist())
        ete = TF.embed_texts(embedder, te["clean_text"].tolist())
        esvd = TF.fit_embedding_svd(etr)
        Etr = TF.transform_embeddings(esvd, etr).reset_index(drop=True)
        Ete = TF.transform_embeddings(esvd, ete).reset_index(drop=True)
        Xtr = pd.concat([Xtr, Etr], axis=1)
        Xte = pd.concat([Xte, Ete], axis=1)
    return Xtr, Xte


def build_fold_matrix(regime, tr, te, embedder=None, use_embeddings=True):
    parts_tr, parts_te = [], []
    if regime in ("text", "combined"):
        a, b = _text_matrix(tr, te, embedder, use_embeddings)
        parts_tr.append(a); parts_te.append(b)
    if regime in ("network", "combined"):
        a, b = _network_matrix(tr, te)
        parts_tr.append(a); parts_te.append(b)
    if regime == "combined":
        a, b = _structured_matrix(tr, te)
        parts_tr.append(a); parts_te.append(b)
    Xtr = pd.concat(parts_tr, axis=1)
    Xte = pd.concat(parts_te, axis=1)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)
    return Xtr.fillna(0.0), Xte.fillna(0.0)


def evaluate(df, regimes=("text", "network", "combined"),
             n_splits=5, seed=SEED, embedder=None, use_embeddings=True,
             canon_only=True, wanted_only=False, weight_fn=None, verbose=True):
    data = target_frame(df, canon_only=canon_only, wanted_only=wanted_only)
    y = data["y_log"].values
    # optional training weights (e.g. weight_fn=make_sample_weights). None = the
    # honest, unweighted baseline. Computed once on the supervised set, then
    # sliced per fold so the SAME weighting applies across all regimes.
    w_all = weight_fn(data) if weight_fn is not None else None
    strat = _bins(data["y_log"])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(data, strat))

    oof = data[["name", "bounty", "y_log"]].copy()
    rows = []

    base_pred = np.zeros(len(data))
    for tr_i, te_i in folds:
        base_pred[te_i] = np.median(y[tr_i])
    oof["pred_log_baseline"] = base_pred
    rows.append(_metric_row("baseline (median)", y, base_pred, data["bounty"].values))

    for regime in regimes:
        pred = np.zeros(len(data))
        for f, (tr_i, te_i) in enumerate(folds):
            tr, te = data.iloc[tr_i], data.iloc[te_i]
            Xtr, Xte = build_fold_matrix(regime, tr, te, embedder, use_embeddings)
            model, engine = make_regressor(seed)
            sw = w_all[tr_i] if w_all is not None else None
            model.fit(Xtr.values, y[tr_i], sample_weight=sw)
            pred[te_i] = model.predict(Xte.values)
            if verbose:
                print(f"  [{regime:<8}] fold {f+1}/{n_splits}  "
                      f"train={len(tr_i)} test={len(te_i)} feats={Xtr.shape[1]}")
        oof[f"pred_log_{regime}"] = pred
        rows.append(_metric_row(regime, y, pred, data["bounty"].values))

    results = pd.DataFrame(rows).set_index("model")
    if verbose:
        print("\n=== CV results (mean over folds) — engine:", make_regressor()[1], "===")
        print(results.round(3).to_string())
    return results, oof


def _metric_row(name, y_log_true, y_log_pred, bounty_true):
    rmse_log = np.sqrt(mean_squared_error(y_log_true, y_log_pred))
    r2 = r2_score(y_log_true, y_log_pred)
    mae_berries = mean_absolute_error(bounty_true, np.expm1(y_log_pred))
    return {"model": name, "R2_log": r2, "RMSE_log": rmse_log,
            "MAE_berries": mae_berries}


def error_analysis(oof, regime="combined", n=10):
    d = oof.copy()
    d["pred_bounty"] = np.expm1(d[f"pred_log_{regime}"])
    d["resid_log"] = d[f"pred_log_{regime}"] - d["y_log"]
    d["abs_resid"] = d["resid_log"].abs()
    cols = ["name", "bounty", "pred_bounty", "resid_log"]
    worst = d.sort_values("abs_resid", ascending=False).head(n)[cols]
    worst = worst.assign(
        bounty=worst["bounty"].astype("int64"),
        pred_bounty=worst["pred_bounty"].round(0).astype("int64"),
        resid_log=worst["resid_log"].round(2),
        direction=np.where(worst["resid_log"] > 0, "over-predicted", "under-predicted"),
    )
    return worst.reset_index(drop=True)