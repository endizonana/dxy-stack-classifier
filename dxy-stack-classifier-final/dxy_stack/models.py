import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler

def fit_gbc_smote(X: np.ndarray, y: pd.Series, seed: int = 42) -> GradientBoostingClassifier:
    counts = y.value_counts()
    if counts.min() > 1:
        k = min(5, counts.min() - 1)
        sampler = SMOTE(random_state=seed, k_neighbors=k)
    else:
        sampler = RandomOverSampler(random_state=seed)
    X_res, y_res = sampler.fit_resample(X, y)
    gbc = GradientBoostingClassifier(random_state=seed)
    gbc.fit(X_res, y_res)
    return gbc

def fit_softmax(X: np.ndarray, y: pd.Series, seed: int = 42) -> LogisticRegression:
    lr = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=seed)
    lr.fit(X, y)
    return lr
