# DXY Monthly Classification — Stacking Backtest

Strict **anti‑leakage** backtest to classify the **US Dollar Index (DXY)** direction **2 months ahead**.  
Two base learners (Gradient Boosting + SMOTE, and Multinomial Logistic Regression) feed a **stacked logit** meta‑model and a **simple average ensemble** for stability.

---

## 🚀 Highlights

- **Leak‑free target construction** (features at \(t-2M\) predict class at \(t\)).
- **Monthly evaluation** at first trading day available in your dataset index.
- **Rolling windows**: base models (1y), stacker (2–3y).
- **Class imbalance handled** with SMOTE / RandomOverSampler fallback.
- **Turn‑key**: point the config to your Excel, run one script, get Excel outputs.

---

## 📦 Repository Structure

```
dxy-stack-classifier/
├─ dxy_stack/
│  ├─ __init__.py
│  ├─ backtest.py
│  ├─ config.py
│  ├─ data.py
│  ├─ importance.py
│  ├─ io_utils.py
│  └─ utils.py
├─ scripts/
│  └─ run_backtest.py
├─ tests/
│  └─ test_import.py
├─ .gitignore
├─ LICENSE
├─ pyproject.toml
└─ requirements.txt
```

---

## 📥 Data

> **Bundled sample data**: `data/DXY_Classification_Data.xlsx` is included.
> By default, `dxy_stack/config.py` points to this file. Replace it with your own path anytime.

- Excel **sheet name**: `dxy_classification`
- Required columns (minimum):
  - `date` (or `Date`)
  - `dxy`
  - Features such as: `dxy_1m_rr`, `spread2_usdem`, `spread10_usdem`, `spread_usyield`, `us10`, `tips5`, `citi`, `vix`

Additional engineered features are computed for you:
- `dxy_ret_60d`, `dxy_rsi_14`, `dxy_trend`, `dxy_trend_ret_60d`

> Tip: put your Excel anywhere and update the path in `dxy_stack/config.py`.

---

## 🎯 Target Construction

Let \( M \) denote **months**.

- **Horizon**: \( 2M \)
- **Forward return (anti‑leakage)**:
  \[ r = \frac{DXY_{t+2M}}{DXY_{t}} - 1 \]
  where **features come from \(t-2M\)** (the last available row at or before \(t-2M\)).
- **Classification** (adaptive neutral band):
  \[
  \text{positive if } r > \overline{|r|},\quad
  \text{negative if } r < -\overline{|r|},\quad
  \text{neutral otherwise}
  \]
  where \( \overline{|r|} \) is the in‑sample mean absolute 2‑month return.

Plain‑text fallback:
```
r = (DXY at t+2M) / (DXY at t) - 1
positive if r > mean(|r|)
negative if r < -mean(|r|)
neutral otherwise
```

---

## 🛠️ How to Run

1) **Install**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2) **Configure**
Edit the paths and knobs in `dxy_stack/config.py`:
- Excel path & sheet
- Forecast horizon, training window lengths, seed
- Output Excel paths

3) **Execute**
```bash
python scripts/run_backtest.py
```

4) **Outputs**
- `backtest_results.xlsx`  
  - `gbc_monthstarts`, `softmax_monthstarts` (base models)
  - `stack_inputs` (6 base probabilities + actual)
  - `avg_ensemble`, `stacked_logit_2y`, `stacked_logit_3y`
- `feature_importances_permutation.xlsx`  
  - OOS permutation importance on a validation slice

---

## 📊 Methodology Notes

- **Month starts only** for evaluation to limit sampling drift.
- **Scaling** with `StandardScaler` inside each rolling window.
- **Imbalance**: SMOTE when each minority class has ≥2 samples in the window; otherwise fallback to `RandomOverSampler`.
- **Stacker**: multinomial logistic regression on the 6 base probabilities with **balanced** sample weights.

---

## 🧪 Quick Smoke Test

```bash
pytest -q
```

---

## ❓FAQ

**Q: My sheet uses `Date` instead of `date`. Is that okay?**  
A: Yes. The loader accepts either `date` or `Date` and converts to a datetime index.

**Q: Where do I put the Excel?**  
A: Anywhere. Point `excel_path` in `dxy_stack/config.py` to the file location (absolute path recommended).

**Q: I don’t have some optional features (e.g., `tips5`).**  
A: Ensure the columns listed in `features` exist in your file, or remove them from the list in `data.py`.

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

## 🙌 Credits

Built by Endi Zonana with a focus on **clean anti‑leakage evaluation** and **transparent stacking**.
