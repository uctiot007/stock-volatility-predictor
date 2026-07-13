# 📈 Stock Volatility Predictor
### Testing Five Model Families From Scratch — and Finding That None of Them Beat Guessing "Tomorrow Looks Like Today"

## 🚀 Overview

This project started as an attempt to take the linear regression and gradient descent math I'd learned in theory and actually implement it myself, rather than just calling `sklearn.linear_model.LinearRegression()`. The goal was never to build the most accurate volatility model possible — it was to make sure I genuinely understood what's happening under the hood: the gradients, the update rule, what regularization actually does to the loss surface, and — it turned out — how easy it is to fool yourself with an evaluation that isn't rigorous enough, even at the level of picking a single hyperparameter.

It grew from there. Once I had a working linear model, I kept asking "but is this actually good?" — which led to adding a naive baseline, then walk-forward validation, then testing whether a more standard time-series tool (ARIMA) or a model purpose-built for volatility (GARCH, and later GJR-GARCH) could do better. Along the way I added residual diagnostics, a fold-by-fold and regime-by-regime breakdown, a proper GARCH evaluation against the target it's actually designed for, and — most recently — a check on whether my own hyperparameter selection process was itself trustworthy. That last check turned up a real methodological issue in my own earlier work, which is probably the single most useful thing this project taught me. The overall finding held up throughout: across every model I tried, simple to sophisticated, none of them reliably beat the simplest possible baseline — and the more sophisticated the model got, the worse it tended to do on average, even though a deeper look shows the failures aren't random, they concentrate around volatility regime changes. I think that result, and the process of uncovering it properly, is more interesting than a clean accuracy number would have been, so this README leads with it rather than hiding it.

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

**Bias-Variance Tradeoff** and **walk-forward validation** — I wanted to actually see these rather than recite the definitions, so I swept regularization strength and used rolling-origin train/test splits instead of trusting a single split — and eventually used walk-forward validation not just to *evaluate* the model, but to *select* the hyperparameter itself, which turned out to matter (see findings below).

**ARIMA and GARCH-family models** — used via `statsmodels` and `arch` respectively, since implementing their maximum-likelihood estimation from scratch is a much larger undertaking than the linear optimizer above and wasn't really the point of this stage.

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
│   ├── walk_forward.py          # Rolling-origin walk-forward validation, used both to
│   │                             # evaluate the model and to select lambda itself
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
    → Walk-Forward Lambda Selection (separate from single-split selection, see findings)
    → Chronological Train/Test Split → Feature Standardization (train stats only)
    → Ridge & Lasso λ Sweeps (single split) → Naive Baseline → Walk-Forward Validation
    → ARIMA (multi-step, then rolling one-step) → GARCH (vs squared returns) → GJR-GARCH
    → Sklearn Comparison → Residual Diagnostics → Fold & Regime Analysis
    → Save Results → Save Model → Predict
```

---

## 🔬 What I Actually Found (Not Just the Clean Version)

### 1. My weights didn't match sklearn's at first, and the reason was multicollinearity

My first gradient descent run produced predictions close to sklearn's `LinearRegression`, but the learned weights differed meaningfully. The EDA notebook makes the reason directly visible: the correlation matrix of the 5 lag features shows every pair correlated at 0.96 or higher — expected, since volatility is a slow-moving 21-day rolling average, so a value from 5 days ago is nearly the same number as today's. With features this redundant, the loss surface isn't a clean bowl with one minimum, it's a flat, elongated valley where many different weight combinations give almost identical loss. This is the textbook reason to reach for regularization.

### 2. My "same lambda" comparison against sklearn's Ridge was initially invalid

My loss averages squared error over `n` samples; sklearn's Ridge sums it. With ~1,600 training rows, the same nominal λ was roughly 1,600x stronger in my version. Scaling my λ by `n_train` before comparing fixed this — afterward, my weights matched sklearn's Ridge to five decimal places.

### 3. A single train/test split was hiding a much less flattering picture

Once I added a **naive baseline** and **walk-forward validation** (5 rolling train/test folds, never training on future data), the picture changed. On a single 80/20 split, my model looked roughly competitive with the naive baseline. Under walk-forward validation the model's average error was clearly worse than just guessing "no change."

### 4. Adding volume didn't help, and Lasso independently confirmed why

Adding lagged trading volume change and a rolling volume average barely changed anything. Lasso zeroed out `vol_lag_2` through `vol_lag_4` and both volume features, keeping real weight on only `vol_lag_1` and `vol_lag_5`. Its test MSE landed at 0.000248 — identical to the naive baseline to six decimal places.

### 5. A single multi-step ARIMA forecast was a badly unfair comparison

My first ARIMA attempt fit once per fold and forecast the entire ~330-day test window in one shot, so it collapsed toward the series' unconditional mean and produced a badly inflated error (MSE 0.0159). Switching to rolling one-step-ahead forecasting brought this down to 0.000776 — still worse than naive, but a fair comparison instead of a broken one.

### 6. My first GARCH evaluation was scored against the wrong target — and fixing it mattered

I initially evaluated GARCH's forecast against my smoothed 21-day rolling volatility, the same target used for the linear model and ARIMA — a mismatched comparison, since GARCH forecasts *conditional variance of returns*, a reactive, single-day quantity. Scored that way, GARCH looked clearly worse than it should. Evaluating GARCH against next-day squared returns (its actual target) instead gives a small, differently-scaled number (average MSE 0.00000039) that isn't directly comparable to the other models' units — but even on its own correct target, GARCH's fold-3 error was roughly 15-20x higher than its other folds, so the target mismatch explained some, but not all, of its weak showing elsewhere.

### 7. Residual diagnostics revealed the model isn't just "bad" — it's missing something specific

A Ljung-Box test on my best model's (Ridge) residuals gave a p-value of essentially 0.0000 — the residuals are **not** random noise; there's still detectable autocorrelation left over. This reframes the project's conclusion from "volatility is unpredictable" (a strong claim my evidence doesn't fully support) to "this particular linear approach isn't capturing what's there" (a claim my evidence directly supports).

### 8. The failures are concentrated at regime changes, not spread evenly

Three separate pieces of evidence point at the same thing: every model's worst fold is the same historical window (fold 3, ~2.4-3x worse than each model's own average); a regime split (high vs. low volatility days) shows Ridge is 1.52x worse on high-volatility days; and the EDA plots independently show the single most volatile day in the dataset (April 6, 2020, the COVID crash) sits inside that same fold. One hypothesis I went in with — that GARCH, being purpose-built for volatility, would specifically shine relative to the linear model during this regime shift — turned out **not** to hold: GARCH's fold-3 degradation was proportionally similar to, if anything slightly worse than, the linear model's.

### 9. My hyperparameter selection method was itself overfitting to one split

After getting the model comparison largely settled, I went back and checked something I'd taken for granted: I'd been choosing Ridge's λ using the single 80/20 split, then separately checking walk-forward performance with that chosen λ. To test whether that selection process was actually sound, I instead selected λ *using* walk-forward validation directly — running the full 5-fold walk-forward for every candidate λ and picking whichever minimized the walk-forward average.

The two methods disagreed:

| Selection method | Chosen λ | Resulting walk-forward MSE |
|---|---|---|
| Single 80/20 split | 0.01 | 0.000398 |
| Walk-forward itself | **0.0** (no regularization) | **0.000357** |

λ=0.01 looked best on the one held-out slice used by the single split, but wasn't actually the best choice once evaluated consistently across multiple historical periods — a smaller-scale version of the same overfitting problem regularization is normally used to prevent, except here it was happening one level up, in how I was tuning the regularization itself. This didn't change the headline conclusion (0.000357 is still worse than naive's 0.000248), but it's a real methodological fix, and a good reminder that "walk-forward validate the final model" and "walk-forward validate the *process that chose* the final model" are different things, and only one of them is actually sufficient.

---

## 📊 Full Results

### Ridge Lambda Sweep — Single Split vs. Walk-Forward Selection

| λ | Single-split Test MSE | Walk-forward avg MSE |
|---|---|---|
| **0.0** | 0.000266 | **0.000357** ✅ (best by walk-forward) |
| 0.001 | 0.000260 | 0.000360 |
| **0.01** | **0.000255** ✅ (best by single split) | 0.000398 |
| 0.1 | 0.000317 | 0.000679 |
| 1.0 | 0.000599 | 0.001369 |

### Lasso Lambda Sweep (single split)

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
| 4 | Custom Ridge (walk-forward avg, λ chosen by walk-forward) | 0.000357 |
| 5 | Custom Ridge (walk-forward avg, λ chosen by single split) | 0.000398 |
| 6 | ARIMA (rolling one-step, refit weekly) | 0.000776 |
| 7 | GJR-GARCH(1,1,1), Student's t, refit weekly | 0.005714 |
| 8 | ARIMA (single multi-step forecast per fold — unfair comparison, included for completeness) | 0.015889 |

GARCH's own correct evaluation (vs. squared returns, different units) averaged **0.00000039**, with fold 3 at 0.00000171 — roughly 15-20x its other folds, echoing the same regime-sensitivity seen everywhere else.

**Does any model beat the naive baseline? No.** But the gap narrows once hyperparameter selection is done properly (0.000357 vs. the earlier 0.000398), and the ranking is roughly monotonic with model complexity.

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

## 🤔 What I Take Away From This

Beating a naive persistence forecast is a genuinely known-hard problem in volatility forecasting, and I ran into that directly rather than just reading about it. Across five model families, none reliably beat "assume tomorrow looks like today" under walk-forward validation, and additional model complexity generally made results worse, not better.

But the deeper diagnostics changed how I'd state that conclusion, and going back to check my own tuning process changed it further. The Ljung-Box result shows my best model's residuals still have real structure it isn't capturing — this isn't "volatility is unpredictable," it's "this specific linear, lag-only approach has a ceiling below what's actually there to find." The fold and regime analysis shows the failures concentrate specifically around volatility regime changes across every model, including the ones purpose-built for volatility. And checking my own hyperparameter selection showed that even a "properly walk-forward validated" model can still be tuned in a way that quietly overfits, if the tuning step itself isn't held to the same standard as the final evaluation.

A few things I'd want to check before drawing a stronger conclusion: whether a longer/shorter lag window changes anything, whether the naive baseline holds up as well on other tickers or asset classes, whether a model with an explicit regime-switching component handles fold-3-type periods better, whether GARCH's forecast is more useful as an *input feature* to the linear model than as a standalone predictor, and whether adding genuinely external information (implied volatility, macro data, news) is what's actually needed to close the gap the Ljung-Box test suggests exists.

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

Note: the full `main.py` run now includes walk-forward-based hyperparameter selection on top of everything else, so it takes noticeably longer than earlier versions — expect several minutes, not seconds. Key parameters (lag size, λ values, fold count, ARIMA/GARCH settings) live in `src/config.py`.

## 🔮 Things I'd Like to Try Next

- **Longer/shorter lag windows** — check whether 5 days is the right amount of history, or whether that choice itself is limiting every model tested
- **A genuinely external feature** — implied volatility (e.g. VIX), macro indicators, or news-based signals, rather than more transformations of the same past price/volume data
- **GARCH's forecast as an input feature** to Ridge/Lasso, rather than only as a standalone model — since GARCH forecasts a different, arguably more relevant quantity (conditional variance) than the lag features do
- **Regime-aware or regime-switching models**, given how consistently every model tested here struggled specifically at regime transitions
- **Testing on other tickers or asset classes** — to see whether "nothing beats naive" is specific to this period/index or a broader pattern
- **Extending walk-forward-based selection to Lasso's λ and to ARIMA/GARCH's settings**, now that I know single-split-based selection can be misleading even after the final model is walk-forward validated

## 👤 Author

Akshat — learning quantitative finance and ML by building things, checking my understanding against known-correct implementations, and trying not to let a flattering number — or a comfortable conclusion, or an unquestioned step in my own process — stop me from digging one level deeper.

If you spot something I got wrong or could do better, I'd genuinely like to know — feel free to open an issue.