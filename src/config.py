"""
Central configuration for the project. Change values here rather than
hunting through main.py, walk_forward.py, arima_baseline.py, etc.
"""

# Data
TICKER = "^GSPC"
START_DATE = "2015-01-01"
END_DATE = "2023-01-01"

# Features
N_LAGS = 5
VOLATILITY_WINDOW = 21  # rolling window (days) for realized volatility
INCLUDE_VOLUME = True

# Train/test split
TRAIN_TEST_SPLIT_RATIO = 0.8

# Regularization sweep
LAMBDAS = [0.0, 0.001, 0.01, 0.1, 1.0]

# Gradient descent
LEARNING_RATE = 0.1
EPOCHS = 5000
WALK_FORWARD_EPOCHS = 2000  # fewer epochs per fold to keep walk-forward runtime reasonable

# Walk-forward validation
N_SPLITS = 5

# ARIMA
ARIMA_ORDER = (1, 0, 1)
ARIMA_REFIT_EVERY = 5

# GARCH / GJR-GARCH
GARCH_REFIT_EVERY = 5
GARCH_PARAMS = {"p": 1, "q": 1, "o": 0, "dist": "normal"}
GJR_GARCH_PARAMS = {"p": 1, "q": 1, "o": 1, "dist": "t"}