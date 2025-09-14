import os
from dataclasses import dataclass
from pathlib import Path

# Resolve repository root from this file's location: dxy_stack/ → repo/
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_EXCEL = _REPO_ROOT / "data" / "DXY_Classification_Data.xlsx"

@dataclass
class Config:
    # IO
    excel_path: str = str(_DEFAULT_EXCEL)   # bundled dataset (change to your own path if desired)
    sheet_name: str = "dxy_classification"
    output_excel: str = str(_REPO_ROOT / "backtest_results.xlsx")
    output_importance_excel: str = str(_REPO_ROOT / "feature_importances_permutation.xlsx")

    # Modeling / CV
    forecast_months: int = 2        # target horizon
    daily_train_years: int = 1      # rolling training window for base models
    stack_train_years: int = 2      # rolling training window for stacker

    # Misc
    MIN_TRAIN_SAMPLES: int = 12     # good for monthly cadence (~1 year)
    SEED: int = 42

CFG = Config()
