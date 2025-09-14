import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.metrics import accuracy_score

# Progress bar (falls back to no-op if tqdm missing)
try:
    from tqdm.auto import tqdm  # noqa: F401
except Exception:
    def tqdm(x, **kwargs):  # type: ignore
        return x

def robust_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=period).mean()
    rs = gain / (loss.replace(0, np.nan) + 1e-12)
    return 100 - (100 / (1 + rs))

def safe_accuracy(y_true: pd.Series, y_pred: pd.Series, label: str) -> Optional[float]:
    if y_true is None or y_pred is None or len(y_true) == 0:
        print(f"[{label}] No valid predictions produced.")
        return None
    acc = accuracy_score(y_true, y_pred)
    print(f"[{label}] Accuracy: {acc:.2%}")
    return acc

def month_starts(index: pd.DatetimeIndex) -> List[pd.Timestamp]:
    """Return first available date of each month in the index."""
    out: List[pd.Timestamp] = []
    for per in index.to_series().dt.to_period('M').unique():
        start = pd.Timestamp(per.start_time)
        mask = (index >= start) & (index < start + pd.DateOffset(months=1))
        if mask.any():
            out.append(index[mask][0])
    return out
