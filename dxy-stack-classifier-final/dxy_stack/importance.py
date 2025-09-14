import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from .models import fit_gbc_smote

def compute_permutation_importance(
    df_adj: pd.DataFrame,
    features: List[str],
    res_gbc: pd.DataFrame,
    seed: int
) -> pd.DataFrame:
    """Compute OOS permutation importance on the last 20% of history before the last backtest date."""
    if len(res_gbc) < 60:
        print("\n[INFO] Skipping permutation importance (insufficient backtest length).")
        return pd.DataFrame(columns=['mean_drop','std'])

    last_date = res_gbc.index.max()
    cut = last_date
    mask_train = df_adj.index < cut
    if mask_train.sum() < 100:
        print("\n[INFO] Skipping permutation importance (insufficient history).")
        return pd.DataFrame(columns=['mean_drop','std'])

    X_all = df_adj.loc[mask_train, features]
    y_all = df_adj.loc[mask_train, 'target']
    n = len(X_all)
    split = int(n * 0.8)
    X_tr = X_all.iloc[:split, :]
    y_tr = y_all.iloc[:split]
    X_va = X_all.iloc[split:, :]
    y_va = y_all.iloc[split:]

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_va_s = pd.DataFrame(scaler.transform(X_va), index=X_va.index, columns=X_va.columns)

    # fit GBC with SMOTE on the training slice
    gbc_imp = fit_gbc_smote(X_tr_s, y_tr, seed=seed)
    pi = permutation_importance(
        estimator=gbc_imp,
        X=X_va_s, y=y_va,
        n_repeats=20, random_state=seed,
        scoring='accuracy', n_jobs=-1
    )
    imp_df = pd.DataFrame({'mean_drop': pi.importances_mean, 'std': pi.importances_std},
                          index=X_va.columns).sort_values('mean_drop', ascending=False)
    return imp_df
