import numpy as np
import pandas as pd

from dxy_stack.config import CFG
from dxy_stack.data import load_and_prepare, build_adjusted_dataset
from dxy_stack.utils import month_starts
from dxy_stack.backtest import (
    rolling_backtest_month_starts,
    simple_average_ensemble,
    stacking_backtest_logit,
)
from dxy_stack.importance import compute_permutation_importance
from dxy_stack.io_utils import save_backtests, save_importances

def main():
    np.random.seed(CFG.SEED)

    # 1) Load & features
    df, features = load_and_prepare(CFG.excel_path, CFG.sheet_name)

    # 2) Build adjusted dataset (strict anti-leakage)
    df_adj = build_adjusted_dataset(df, features, forecast_months=CFG.forecast_months)
    print("[INFO] Class distribution in adjusted data:")
    print(df_adj['target'].value_counts().to_string(), "\n")

    # 3) Rolling backtests (month starts only, SMOTE on for GBC)
    res_gbc, res_soft = rolling_backtest_month_starts(
        df_adj=df_adj,
        features=features,
        train_years=CFG.daily_train_years,
        min_train_samples=CFG.MIN_TRAIN_SAMPLES,
        seed=CFG.SEED
    )

    # 4) Build stacking frame aligned on evaluated dates
    idx = res_gbc.index.intersection(res_soft.index)
    df_stack = pd.DataFrame({
        'prob_neg_gbc':     res_gbc.loc[idx, 'prob_neg'],
        'prob_neutral_gbc': res_gbc.loc[idx, 'prob_neutral'],
        'prob_pos_gbc':     res_gbc.loc[idx, 'prob_pos'],
        'prob_neg_soft':    res_soft.loc[idx, 'prob_neg'],
        'prob_neutral_soft':res_soft.loc[idx, 'prob_neutral'],
        'prob_pos_soft':    res_soft.loc[idx, 'prob_pos'],
        'actual':           res_gbc.loc[idx, 'actual']
    }).dropna()
    print(f"[INFO] Stack input rows: {len(df_stack):,}")

    # 5) Monthly test dates for stacking (after enough history)
    initial_train_end = df_stack.index.min() + pd.DateOffset(years=CFG.stack_train_years)
    test_dates = [d for d in month_starts(df_stack.index) if d and d >= initial_train_end]

    # --- Baselines
    print(f"[Baseline] Always 'neutral' accuracy: {(df_stack['actual']=='neutral').mean():.2%}")
    avg_ens = simple_average_ensemble(df_stack, test_dates)
    if not avg_ens.empty:
        from sklearn.metrics import accuracy_score
        print(f"[Avg Ensemble] Accuracy: {accuracy_score(avg_ens['actual'], avg_ens['pred']):.2%} "
              f"({len(avg_ens)} predictions)")

    # --- Logit stacker (2y, C=0.5); also try 3y for stability
    backtest_stack_logit_2y = stacking_backtest_logit(
        df_stack, test_dates, window_years=CFG.stack_train_years, C=0.5, seed=CFG.SEED
    )
    backtest_stack_logit_3y = stacking_backtest_logit(
        df_stack, test_dates, window_years=3, C=0.5, seed=CFG.SEED
    )

    # 6) Optional: OOS permutation importance
    imp_df = compute_permutation_importance(df_adj, features, res_gbc, seed=CFG.SEED)

    # 7) Save outputs
    save_backtests(
        CFG.output_excel,
        res_gbc, res_soft, df_stack,
        avg_ens, backtest_stack_logit_2y, backtest_stack_logit_3y
    )
    save_importances(CFG.output_importance_excel, imp_df)

    print(f"\n[INFO] Saved backtests to: {CFG.output_excel}")
    if imp_df is not None and not imp_df.empty:
        print(f"[INFO] Saved permutation importances to: {CFG.output_importance_excel}")

if __name__ == "__main__":
    main()
