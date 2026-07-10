# 📈 Stock Volatility Forecasting
### Testing Five Model Families From Scratch — and Finding That None of Them Beat Guessing "Tomorrow Looks Like Today"

## 🧠 Key Insights (TL;DR)

- A naive persistence model ("tomorrow ≈ today") matched or outperformed all tested models.
- Increasing model complexity (Ridge → ARIMA → GARCH) did not improve predictive performance.
- Residual diagnostics (Ljung-Box p ≈ 0.0000) show remaining autocorrelation → signal still exists.
- Model errors concentrate during volatility regime shifts (e.g., COVID crash), not uniformly.
- GARCH models, while theoretically suited for volatility, did not outperform simpler models even during high-volatility periods.
- Evaluation choice matters: GARCH must be evaluated on variance, not returns.

## 🚀 Overview

This project started as an attempt to take the linear regression and gradient descent math I'd learned in theory and actually implement it myself, rather than just calling `sklearn.linear_model.LinearRegression()`. The goal was never to build the most accurate volatility model possible — it was to make sure I genuinely understood what's happening under the hood: the gradients, the update rule, what regularization actually does to the loss surface, and — it turned out — how easy it is to fool yourself with an evaluation that isn't rigorous enough.

It grew from there. Once I had a working linear model, I kept asking "but is this actually good?" — which led to adding a naive baseline, then walk-forward validation, then testing whether a more standard time-series tool (ARIMA) or a model purpose-built for volatility (GARCH, and later GJR-GARCH) could do better. Along the way I also added residual diagnostics, a fold-by-fold and regime-by-regime breakdown, and a proper GARCH evaluation against the target it's actually designed for, to make sure the "nothing beats naive" conclusion wasn't hiding something more specific underneath. The most useful part of this project ended up being a finding I didn't expect going in: across every model I tried, simple to sophisticated, none of them reliably beat the simplest possible baseline — and the more sophisticated the model got, the worse it tended to do on average, even though a deeper look shows the failures aren't random, they concentrate around volatility regime changes. I think that result, and the process of uncovering it properly, is more interesting than a clean accuracy number would have been, so this README leads with it rather than hiding it.


## 💼 Resume Summary (1–2 lines)

Built a full volatility forecasting pipeline comparing linear models, ARIMA, and GARCH using walk-forward validation; found that naive persistence matches or outperforms complex models, with errors concentrated during regime shifts and residual diagnostics indicating unmodeled structure.

## 🎯 Problem Statement

Predict next-day realized volatility of the S&P 500 using its own recent history. This relies on a known property of markets called volatility clustering — calm and turbulent periods tend to persist for a while — so recent volatility is at least somewhat informative about tomorrow's.

- **Inputs:** the previous 5 days of rolling 21-day realized volatility, plus (in one experiment) lagged volume change and a rolling volume average, plus (for the time-series models) the raw daily return series itself
- **Target:** next day's realized volatility
- **Data:** S&P 500 (`^GSPC`) daily prices via `yfinance`, 2015–2023 (2,014 trading days)

## 🧠 Concepts I Was Practicing

**Linear Regression**
```
y = Xw + b
```

**Batch Gradient Descent**
```
w := w - α · ∇J(w)
b := b - α · ∂J/∂b
```
Written by hand in NumPy. `scikit-learn` is only used afterward, to check my implementation against a known-correct reference — never during training.

**Ridge (L2) and Lasso (L1) Regularization**
```
Ridge:  J(w) = MSE + λΣw²        →  dw += 2λw
Lasso:  J(w) = MSE + λΣ|w|       →  dw += λ·sign(w)
```
Both implemented from scratch, with the bias term deliberately excluded from the penalty in each case.

**Bias-Variance Tradeoff** and **walk-forward validation** — I wanted to actually see these rather than recite the definitions, so I swept regularization strength and used rolling-origin train/test splits instead of trusting a single split.

**ARIMA and GARCH-family models** — used via `statsmodels` and `arch` respectively, since implementing their maximum-likelihood estimation from scratch is a much larger undertaking than the linear optimizer above and wasn't really the point of this stage. These were brought in specifically to test whether standard, purpose-built time-series tools could do what my from-scratch linear model couldn't.

**Residual diagnostics** — ACF/PACF plots and the Ljung-Box test, to check whether my best model's errors were genuinely random or still had exploitable structure left in them, rather than just trusting the aggregate MSE number.

## 🏗️ Project Structure

```
stock-volatility-predictor/
│
├── data/
│   └── raw/                     # Downloaded market data (gitignored)
│
├── src/
│   ├── config.py                # Central place for lag size, λ values, fold count, etc.
│   ├── data_loader.py           # Fetch, clean, cache S&P 500 data
│   ├── features.py              # Log returns, rolling volatility, lag + volume features
│   ├── model.py                  # Gradient descent, Ridge (L2), Lasso (L1) — from scratch
│   ├── evaluate.py              # Metrics, naive baseline, sklearn comparison, plots
│   ├── walk_forward.py          # Rolling-origin walk-forward validation for the linear model
│   ├── arima_baseline.py        # ARIMA, multi-step and rolling one-step-ahead
│   ├── garch_baseline.py        # GARCH / GJR-GARCH, evaluated correctly on squared returns
│   ├── diagnostics.py           # Residual plots, ACF/PACF, Ljung-Box test
│   ├── fold_comparison.py       # Per-fold MSE comparison across all models
│   ├── regime_analysis.py       # High-vol vs low-vol regime performance split
│   ├── plot_returns.py          # Returns/squared-returns plot with fold boundaries marked
│   ├── save_results.py          # Save/load results.json
│   └── predict.py               # Load a trained model, forecast the next trading day
│
├── notebooks/
│   ├── EDA.ipynb                # Data exploration: price, returns, volatility, correlation
│   └── model_analysis.ipynb     # Loads results.json, visualizes the full comparison
│
├── main.py                      # End-to-end pipeline: data → train → tune → validate → save
├── outputs/
│   ├── results.json             # All final metrics, versioned per run
│   ├── trained_model.npz        # Saved weights/bias/normalization stats
│   └── plots/                   # Every saved figure
├── requirements.txt
└── README.md
```

## ⚙️ Pipeline

```
Raw Prices → Log Returns → Rolling Volatility (+ Volume Features)
    → Chronological Train/Test Split → Feature Standardization (train stats only)
    → Ridge & Lasso λ Sweeps → Naive Baseline → Walk-Forward Validation
    → ARIMA (multi-step, then rolling one-step) → GARCH (vs squared returns) → GJR-GARCH
    → Sklearn Comparison → Residual Diagnostics → Fold & Regime Analysis
    → Save Results → Save Model → Predict
```

---

## 🔬 What I Actually Found (Not Just the Clean Version)

### 1. My weights didn't match sklearn's at first, and the reason was multicollinearity

My first gradient descent run produced predictions close to sklearn's `LinearRegression`, but the learned weights differed meaningfully. The EDA notebook makes the reason directly visible: the correlation matrix of the 5 lag features shows every pair correlated at 0.96 or higher. This makes sense once you look at the volatility series itself — it's a slow-moving 21-day rolling average, so a value from 5 days ago is nearly the same number as today's. With features this redundant, the loss surface isn't a clean bowl with one minimum, it's a flat, elongated valley where many different weight combinations give almost identical loss. This is the textbook reason to reach for regularization, which is what led me to implement Ridge next.


### 2. My "same lambda" comparison against sklearn's Ridge was initially invalid

My loss averages squared error over `n` samples; sklearn's Ridge sums it. With ~1,600 training rows, the same nominal λ was roughly 1,600x stronger in my version. Scaling my λ by `n_train` before comparing fixed this — afterward, my weights matched sklearn's Ridge to five decimal places.

### 3. A single train/test split was hiding a much less flattering picture

Once I added a **naive baseline** (simply predicting "tomorrow's volatility = today's volatility") and **walk-forward validation** (5 rolling train/test folds, never training on future data), the picture changed. On a single 80/20 split, my model looked roughly competitive with the naive baseline. Under walk-forward validation the model's average error was clearly worse than just guessing "no change." I hadn't computed the naive baseline until fairly late in the project, and once I did, it was obvious my earlier single-split result had been somewhat flattering.

### 4. Adding volume didn't help, and Lasso independently confirmed why

Adding lagged trading volume change and a rolling volume average as extra features barely changed anything. Lasso — which can drive individual weights all the way to exactly zero — confirmed this cleanly by zeroing out `vol_lag_2` through `vol_lag_4` and both volume features, keeping real weight on only `vol_lag_1` and `vol_lag_5`. Its test MSE landed at 0.000248 — identical to the naive baseline to six decimal places.

### 5. A single multi-step ARIMA forecast was a badly unfair comparison to my daily-updating linear model

My first ARIMA attempt fit once per fold and forecast the entire ~330-day test window in one shot. ARIMA's forecasts converge toward the series' unconditional mean as the horizon grows, so by 20-30 days in it was essentially predicting a flat line regardless of what volatility actually did (MSE 0.0159). Switching to rolling one-step-ahead forecasting brought this down to 0.000776 — still worse than naive, but a fair comparison instead of a broken one.

### 6. My first GARCH evaluation was scored against the wrong target — and fixing it mattered

I initially evaluated GARCH's forecast against my smoothed 21-day rolling volatility, the same target used for the linear model and ARIMA. This is actually a mismatched comparison: GARCH forecasts *conditional variance of returns*, a reactive, single-day quantity, not a backward-looking 21-day average. Scored that way, GARCH looked clearly worse than it should (MSE 0.003166 for GARCH, 0.005714 for GJR-GARCH). I added a second evaluation — GARCH's forecast vs. the *next day's actual squared return*, the target it's actually designed to predict — and removed the mismatched comparison from the main results table once I confirmed the fairer one told a more honest story. Even on its own correct target, though, GARCH's fold-3 error was roughly 15-20x higher than its other folds, so the target mismatch explained some, but not all, of GARCH's weak showing.

### 7. Residual diagnostics revealed the model isn't just "bad" — it's missing something specific

Running a Ljung-Box test on my best model's (Ridge) residuals gave a p-value of essentially 0.0000 — meaning the residuals are **not** random noise; there's still detectable autocorrelation left over. This matters because it's a more precise claim than "the model doesn't work": it means there is *some* structure a linear lag model isn't capturing, even though that structure hasn't (yet) translated into beating naive. It reframes the project's conclusion from "volatility is unpredictable" (a strong claim my evidence doesn't fully support) to "this particular linear approach isn't capturing what's there" (a claim my evidence directly supports).

### 8. The failures are concentrated at regime changes, not spread evenly

Three separate pieces of evidence point at the same thing:
- **Fold-by-fold comparison**: every model — Ridge, ARIMA, GARCH, GJR-GARCH — has its worst fold in the same historical window (fold 3), each roughly 2.4–3x worse than its own average.
- **Regime split** (high vs. low volatility days, split at the median): Ridge's MSE is 1.52x worse on high-volatility days (0.000307) than low-volatility days (0.000202).
- **The EDA plots** independently show the single most volatile day in the whole dataset (April 6, 2020, by the 21-day rolling measure) sits inside fold 3's window — the COVID crash.

One hypothesis I went in with — that GARCH, being purpose-built for volatility, would specifically shine relative to the linear model during this regime shift — turned out **not** to hold up: GARCH's fold-3 degradation was proportionally similar to (if anything slightly worse than) the linear model's. That's a real, slightly counterintuitive finding worth stating plainly rather than the more expected story I'd assumed going in.



---

## ⚠️ A Critical Evaluation Mistake (and Fix)

An early version of this project evaluated all models using MSE on returns.  
This is appropriate for regression models, but **incorrect for GARCH**, which predicts conditional variance.

Fix:
- Evaluated GARCH against **next-day squared returns**
- Added volatility-specific interpretation instead of comparing directly to return models

Takeaway:
Choosing the wrong evaluation metric can make a correct model look incorrect.


## 📊 Full Results

### Ridge Lambda Sweep (5,000 epochs, single 80/20 split)

| λ | Test MSE |
|---|---|
| 0.0 | 0.000266 |
| 0.001 | 0.000260 |
| **0.01** | **0.000255** ✅ |
| 0.1 | 0.000317 |
| 1.0 | 0.000599 |

### Lasso Lambda Sweep

| λ | Test MSE |
|---|---|
| 0.0 | 0.000266 |
| **0.001** | **0.000248** ✅ (matches naive baseline) |
| 0.01 | 0.000249 |
| 0.1 | 0.002009 |
| 1.0 | 0.260348 |

### Sklearn Validation (Ridge, λ=0.01)

| | vol_lag_1 | vol_lag_2 | vol_lag_3 | vol_lag_4 | vol_lag_5 | volume_change | volume_ma |
|---|---|---|---|---|---|---|---|
| Custom Ridge GD | 0.10315 | 0.03505 | 0.00460 | -0.00905 | -0.02241 | 0.00113 | 0.00215 |
| Sklearn Ridge (matched α) | 0.10315 | 0.03505 | 0.00460 | -0.00905 | -0.02241 | 0.00113 | 0.00215 |

Matching to five decimal places confirms the gradient and update rule are implemented correctly.

### The Complete, Honest Final Comparison

| Rank | Method | Test MSE |
|---|---|---|
| 1 | Naive baseline (today = tomorrow) | 0.000248 |
| 1 (tie) | Custom Lasso (single split) | 0.000248 |
| 3 | Custom Ridge (single split) | 0.000255 |
| 4 | Custom Ridge (walk-forward avg, 5 folds) | 0.000398 |
| 5 | ARIMA (rolling one-step, refit weekly) | 0.000776 |
| 6 | GJR-GARCH(1,1,1), Student's t, refit weekly | 0.005714 |
| 7 | ARIMA (single multi-step forecast per fold — unfair comparison, included for completeness) | 0.015889 |

GARCH's own correct evaluation (vs. squared returns, different units, not directly comparable to the table above) averaged **0.00000039** — with fold 3 at 0.00000171, roughly 15-20x its other folds, echoing the same regime-sensitivity seen everywhere else.

**Does any model beat the naive baseline? No.** And the ranking isn't noise — it's roughly monotonic with model complexity.

### Fold-by-Fold Pattern (all models, MSE)

| Fold | Ridge | ARIMA (rolling) | GARCH | GJR-GARCH |
|---|---|---|---|---|
| 1 | 0.000160 | 0.000338 | 0.001338 | 0.001475 |
| 2 | 0.000237 | 0.000418 | 0.001874 | 0.002715 |
| **3** | **0.000960** | **0.002188** | **0.008829** | **0.017292** |
| 4 | 0.000347 | 0.000499 | 0.001967 | 0.003300 |
| 5 | 0.000284 | 0.000434 | 0.001822 | 0.003788 |

Every model's worst fold is the same fold, by a wide margin.

### Regime Split (Ridge, threshold = median volatility)

| Regime | Days | MSE |
|---|---|---|
| Low volatility (below median) | 199 | 0.000202 |
| High volatility (above median) | 199 | 0.000307 |

### Residual Diagnostics (Ridge, best model)

| Statistic | Value |
|---|---|
| Mean residual | 0.000824 (near zero — low systematic bias) |
| Std | 0.015941 |
| Skew | 0.090 (nearly symmetric) |
| Ljung-Box p-value (lag 20) | ≈ 0.0000 (residuals show significant autocorrelation) |



---

## 📈 Visual Insight: Why Naive Works So Well

The volatility series is highly persistent and smooth due to the 21-day rolling window.

→ This makes "today ≈ tomorrow" a very strong baseline.

In contrast:
- More complex models overreact to noise
- Multi-step models drift toward the mean
- Regime shifts break all models simultaneously

This explains why increased model complexity did not translate into better performance.


## 🤔 What I Take Away From This

Beating a naive persistence forecast is a genuinely known-hard problem in volatility forecasting, and I ran into that directly rather than just reading about it. Across five model families, none reliably beat "assume tomorrow looks like today" under walk-forward validation on this dataset, and additional model complexity generally made results worse, not better.

But the deeper diagnostics changed how I'd state that conclusion. The Ljung-Box result shows my best model's residuals still have real structure it isn't capturing — this isn't "volatility is unpredictable," it's "this specific linear, lag-only approach has a ceiling below what's actually there to find." And the fold/regime analysis shows the failures aren't random or evenly spread — they concentrate specifically around volatility regime changes (the COVID crash window most of all), across every model I tried, including the ones purpose-built for volatility. That GARCH didn't show a special advantage there, despite my expecting it to, was itself a useful correction to my assumptions.

A few things I'd want to check before drawing a stronger conclusion: whether a longer/shorter lag window changes anything, whether the naive baseline holds up as well on other tickers or asset classes, whether a model with an explicit regime-switching component (rather than one set of parameters fit across all conditions) handles the fold-3-type periods better, and whether adding genuinely external information (implied volatility, macro data, news) — rather than more transformations of the same past-price data — is what's actually needed to close the gap the Ljung-Box test suggests exists.

## 🧩 Why Didn’t Complex Models Win?

Three key reasons:

1. **Low signal-to-noise ratio** in financial returns  
   → Hard to extract stable predictive patterns

2. **Volatility clustering is real, but slow-moving**  
   → Makes naive persistence surprisingly strong

3. **Regime shifts dominate error**  
   → Models trained on past regimes fail during sudden transitions

Result:
Model sophistication alone is not enough — adaptability matters more.

## ⚠️ Limitations

- Only one asset (S&P 500) and one time period (2015–2023)
- No external features (e.g., VIX, macroeconomic indicators)
- Fixed lag structure (5 days) may not be optimal
- Hyperparameters (λ, ARIMA orders) not fully tuned via walk-forward
- Linear models assume static relationships across regimes

## 🔮 Live Prediction

```bash
python -m main            # trains every model, validates, saves outputs/results.json + trained_model.npz
python -m src.predict     # loads the saved linear model and forecasts the next trading day
```

Example output:
```
Most recent trading data as of: 2026-07-02
Predicted next-day (annualized) volatility: 0.17405
```

Worth noting: this is only ever a next-*trading*-day forecast, since the data skips weekends/holidays. Given the findings above, this forecast should be read with real skepticism rather than confidence — particularly if the market happens to be near a volatility regime change, which is exactly where every model tested here struggled most.

## 📓 Notebooks

- **`notebooks/EDA.ipynb`** — price, returns, and volatility over time; the squared-returns view with fold boundaries marked; the lag-feature correlation heatmap behind the multicollinearity finding; basic volatility statistics
- **`notebooks/model_analysis.ipynb`** — loads `outputs/results.json`, plots the full model comparison, both λ sweeps, and predicted-vs-actual for the best model

## 🛠️ Tech Stack

Python 3.9 · NumPy · Pandas · Matplotlib · yfinance · scikit-learn (validation only) · statsmodels (ARIMA, Ljung-Box, ACF/PACF) · arch (GARCH / GJR-GARCH) · Jupyter

## ▶️ How to Run

```bash
git clone https://github.com/uctiot007/stock-volatility-predictor.git
cd stock-volatility-predictor

python -m venv .venv
source .venv/bin/activate   # Mac/Linux
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt

python -m main
python -m src.predict

jupyter notebook notebooks/EDA.ipynb
jupyter notebook notebooks/model_analysis.ipynb
```

Note: the full `main.py` run trains and validates every model above, including several ARIMA and GARCH refits — expect it to take a few minutes, not seconds. Key parameters (lag size, λ values, fold count, ARIMA/GARCH settings) live in `src/config.py`.

## 🔮 Things I'd Like to Try Next

- **Longer/shorter lag windows** — check whether 5 days is the right amount of history, or whether that choice itself is limiting every model tested
- **A genuinely external feature** — implied volatility (e.g. VIX), macro indicators, or news-based signals, rather than more transformations of the same past price/volume data
- **GARCH's forecast as an input feature** to Ridge/Lasso, rather than only as a standalone model — since GARCH forecasts a different, arguably more relevant quantity (conditional variance) than the lag features do
- **Regime-aware or regime-switching models**, given how consistently every model tested here struggled specifically at regime transitions
- **Testing on other tickers or asset classes** — to see whether "nothing beats naive" is specific to this period/index or a broader pattern
- **A proper walk-forward hyperparameter selection** — right now λ is chosen on the single split, then evaluated separately via walk-forward; ideally λ itself would be selected using walk-forward validation too

## 👤 Author

Akshat — learning quantitative finance and ML by building things, checking my understanding against known-correct implementations, and trying not to let a flattering number — or a comfortable conclusion — stop me from digging one level deeper.

If you spot something I got wrong or could do better, I'd genuinely like to know — feel free to open an issue.