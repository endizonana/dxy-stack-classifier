import numpy as np
import pandas as pd
from typing import List, Tuple
from .utils import robust_rsi

def load_and_prepare(excel_path: str, sheet_name: str) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif 'Date' in df.columns:
        df['date'] = pd.to_datetime(df['Date'])
    else:
        raise ValueError("No date column found. Expected 'date' or 'Date'.")
    df = df.set_index('date').sort_index()

    # engineered features
    df['dxy_ret_60d'] = df['dxy'] - df['dxy'].shift(60)
    df['dxy_rsi_14']  = robust_rsi(df['dxy'], period=14)
    df['dxy_trend']   = df['dxy'].rolling(window=60, min_periods=30).mean()   # trend proxy
    df['dxy_trend_ret_60d'] = df['dxy_trend'] - df['dxy_trend'].shift(60)

    features = [
        'dxy_1m_rr', 'spread2_usdem', 'spread10_usdem', 'us10', 'citi', 'vix',
        'dxy_ret_60d', 'dxy_rsi_14', 'tips5', 'dxy_trend', 'spread_usyield', 'dxy_trend_ret_60d'
    ]

    df = df.dropna(subset=features + ['dxy']).copy()
    print(f"[INFO] Loaded {len(df):,} rows after feature dropna.")
    return df, features

def build_adjusted_dataset(df: pd.DataFrame, features: List[str], forecast_months: int = 2) -> pd.DataFrame:
    forecast_offset = pd.DateOffset(months=forecast_months)
    records = []

    for pred_date in df.index:
        cutoff_date = pred_date - forecast_offset
        if cutoff_date < df.index[0]:
            continue
        feat_candidates = df.index[df.index <= cutoff_date]
        if len(feat_candidates) == 0:
            continue
        feat_date = feat_candidates[-1]

        dxy_future = df.at[pred_date, 'dxy']
        dxy_past   = df.at[feat_date, 'dxy']
        ret_adj = dxy_future / dxy_past - 1.0

        rec = {'pred_date': pred_date, 'feature_date': feat_date, 'dxy_return_2M': ret_adj}
        for f in features:
            rec[f] = df.at[feat_date, f]
        records.append(rec)

    df_adj = pd.DataFrame(records).set_index('pred_date').dropna(subset=['dxy_return_2M'])

    fixed_tol = df_adj['dxy_return_2M'].abs().mean()
    print(f"[INFO] Adjusted rows: {len(df_adj):,} | Fixed tolerance (|2M ret| mean): {fixed_tol:.6f}")

    def classify(r):
        if r > fixed_tol:   return "positive"
        if r < -fixed_tol:  return "negative"
        return "neutral"

    df_adj['target'] = df_adj['dxy_return_2M'].apply(classify)
    return df_adj
