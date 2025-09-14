import time
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE, RandomOverSampler

from .utils import tqdm, month_starts, safe_accuracy
from .models import fit_gbc_smote

def rolling_backtest_month_starts(
    df_adj: pd.DataFrame,
    features: List[str],
    train_years: int,
    min_train_samples: int,
    seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    valid = df_adj.index
    dates_to_eval = month_starts(valid)  # month starts only
    tr_offset = pd.DateOffset(years=train_years)

    recs_g, recs_s = [], []
    print(f"[INFO] Evaluating {len(dates_to_eval):,} month-start dates "
          f"with a {train_years}y window and min {min_train_samples} train samples.")
    t0 = time.time()

    for i, fd in enumerate(dates_to_eval, 1):
        train_start = fd - tr_offset
        mask_train = (valid >= train_start) & (valid < fd)
        n_train = int(mask_train.sum())
        if n_train < min_train_samples:
            if i % 10 == 0:
                elapsed = time.time() - t0
                print(f"[PROGRESS] {i}/{len(dates_to_eval)} folds | elapsed {elapsed:,.1f}s (skipped: not enough train)")
            continue

        X_train = df_adj.loc[mask_train, features]
        y_cat   = df_adj.loc[mask_train, 'target']
        X_test  = df_adj.loc[[fd], features]

        scaler = StandardScaler().fit(X_train)
        Xtr = scaler.transform(X_train)
        Xte = scaler.transform(X_test)

        # --- GBC + SMOTE (robust fallback per fold) ---
        try:
            counts = y_cat.value_counts()
            if counts.min() > 1:
                k = min(5, counts.min() - 1)
                sampler = SMOTE(random_state=seed, k_neighbors=k)
            else:
                sampler = RandomOverSampler(random_state=seed)
            X_res, y_res = sampler.fit_resample(Xtr, y_cat)
        except Exception as e:
            print(f"[WARN] SMOTE failed on {fd.date()} ({type(e).__name__}: {e}). Falling back to RandomOverSampler.")
            sampler = RandomOverSampler(random_state=seed)
            X_res, y_res = sampler.fit_resample(Xtr, y_cat)

        gbc = fit_gbc_smote(Xtr, y_cat, seed=seed) if X_res is None else None
        if gbc is None:
            from sklearn.ensemble import GradientBoostingClassifier
            gbc = GradientBoostingClassifier(random_state=seed)
            gbc.fit(X_res, y_res)

        proba_g = gbc.predict_proba(Xte)[0]
        lbls_g  = list(gbc.classes_)
        mg = {lbl: p for lbl, p in zip(lbls_g, proba_g)}
        pg = np.array([mg.get('negative', 0.0), mg.get('neutral', 0.0), mg.get('positive', 0.0)])
        pred_g = ['negative','neutral','positive'][pg.argmax()]
        recs_g.append({
            'pred_date': fd, 'predicted': pred_g, 'actual': df_adj.at[fd, 'target'],
            'prob_neg': pg[0], 'prob_neutral': pg[1], 'prob_pos': pg[2],
            'dxy_return_2M': df_adj.at[fd, 'dxy_return_2M']
        })

        # --- Softmax (multinomial logistic) ---
        soft = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=seed)
        soft.fit(Xtr, y_cat)
        proba_s = soft.predict_proba(Xte)[0]
        lbls_s  = list(soft.classes_)
        ms = {lbl: p for lbl, p in zip(lbls_s, proba_s)}
        ps = np.array([ms.get('negative', 0.0), ms.get('neutral',  0.0), ms.get('positive', 0.0)])
        pred_s = ['negative','neutral','positive'][ps.argmax()]
        recs_s.append({
            'pred_date': fd, 'predicted': pred_s, 'actual': df_adj.at[fd, 'target'],
            'prob_neg': ps[0], 'prob_neutral': ps[1], 'prob_pos': ps[2],
            'dxy_return_2M': df_adj.at[fd, 'dxy_return_2M']
        })

        if i % 10 == 0:
            elapsed = time.time() - t0
            print(f"[PROGRESS] {i}/{len(dates_to_eval)} folds | elapsed {elapsed:,.1f}s")

    res_g = pd.DataFrame(recs_g).set_index('pred_date').sort_index()
    res_s = pd.DataFrame(recs_s).set_index('pred_date').sort_index()
    safe_accuracy(res_g.get('actual'), res_g.get('predicted'), "GBC (month starts)")
    safe_accuracy(res_s.get('actual'), res_s.get('predicted'), "Softmax (month starts)")
    print(f"[INFO] Produced {len(res_g):,} GBC and {len(res_s):,} Softmax predictions.")
    print(f"[INFO] Total elapsed: {time.time() - t0:,.1f}s")
    return res_g, res_s

def simple_average_ensemble(df_stack: pd.DataFrame, dates: List[pd.Timestamp]) -> pd.DataFrame:
    """Predict by averaging base probabilities on specified dates."""
    use = df_stack[df_stack.index.isin(dates)].copy()
    avg_neg = 0.5*use['prob_neg_gbc'] + 0.5*use['prob_neg_soft']
    avg_neu = 0.5*use['prob_neutral_gbc'] + 0.5*use['prob_neutral_soft']
    avg_pos = 0.5*use['prob_pos_gbc'] + 0.5*use['prob_pos_soft']
    arr = np.vstack([avg_neg.values, avg_neu.values, avg_pos.values]).T
    cls = np.array(['negative','neutral','positive'])
    pred = cls[arr.argmax(axis=1)]
    out = pd.DataFrame({'pred': pred, 'actual': use['actual']}, index=use.index)
    return out

def stacking_backtest_logit(
    df_stack: pd.DataFrame,
    test_dates: List[pd.Timestamp],
    window_years: int = 2,
    C: float = 0.5,
    seed: int = 42
) -> pd.DataFrame:
    """Rolling-window meta-learner using multinomial logistic regression on the 6 base prob features."""
    feats = [
        'prob_neg_gbc','prob_neutral_gbc','prob_pos_gbc',
        'prob_neg_soft','prob_neutral_soft','prob_pos_soft'
    ]
    recs = []
    ro = pd.DateOffset(years=window_years)
    print(f"[INFO] (Logit stacker) Stacking over {len(test_dates):,} monthly dates with {window_years}y window (C={C}).")

    for td in tqdm(test_dates, desc="Stacking (logit)"):
        tr = df_stack[(df_stack.index >= td - ro) & (df_stack.index < td)]
        if tr.shape[0] < 24:
            continue

        Xtr, ytr = tr[feats].values, tr['actual'].values
        Xte, yte = df_stack.loc[[td], feats].values, df_stack.loc[td, 'actual']

        sw = compute_sample_weight("balanced", ytr)
        meta = LogisticRegression(
            multi_class='multinomial', solver='lbfgs',
            C=C, max_iter=1000, random_state=seed
        )
        meta.fit(Xtr, ytr, sample_weight=sw)
        pred = meta.predict(Xte)[0]
        recs.append({'date': td, 'predicted_logit': pred, 'actual': yte})

    result = pd.DataFrame(recs).set_index('date').sort_index()
    safe_accuracy(result.get('actual'), result.get('predicted_logit'), f"Stacked Logit ({window_years}y)")
    print(f"[INFO] (Logit stacker) produced {len(result):,} predictions.")
    return result
