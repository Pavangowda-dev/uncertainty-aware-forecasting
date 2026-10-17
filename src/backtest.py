# backtest.py
# Runs the whole forecasting pipeline for many different months

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from features import FEATURES
from metrics import mae, coverage, avg_width
from calibration import get_correction


def train_quantile_models(X, y):
    models = {}
    for q in [0.05, 0.5, 0.95]:
        model = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=q,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
        )
        model.fit(X, y)
        models[q] = model
    return models


def predict_interval(models, X):
    lower = np.maximum(models[0.05].predict(X), 0)
    median = np.maximum(models[0.5].predict(X), 0)
    upper = np.maximum(models[0.95].predict(X), 0)

    lower = np.minimum(lower, median)
    upper = np.maximum(upper, median)
    return lower, median, upper


def run_backtest(df, test_months):
    rows = []        # one row of results per month
    all_preds = []   # all predictions, saved for later analysis

    one_hour = pd.Timedelta(hours=1)

    for month_start in test_months:
        test_start = pd.Timestamp(month_start)
        test_end = test_start + pd.offsets.MonthBegin(1)    # start of the next month
        cal_start = test_start - pd.DateOffset(months=3)

        # three parts, always in time order
        fit = df.loc[:cal_start - one_hour].dropna(subset=FEATURES)
        cal = df.loc[cal_start:test_start - one_hour]
        test = df.loc[test_start:test_end - one_hour]

        # train on the fit part only
        models = train_quantile_models(fit[FEATURES], fit["cnt"])

        # intervals for calibration and test
        lower_cal, median_cal, upper_cal = predict_interval(models, cal[FEATURES])
        lower_t, median_t, upper_t = predict_interval(models, test[FEATURES])

        # calibrate using the calibration part
        correction = get_correction(cal["cnt"], lower_cal, upper_cal, alpha=0.1)
        lower_new = np.maximum(lower_t - correction, 0)
        upper_new = upper_t + correction

        rows.append({
            "month": test_start.strftime("%Y-%m"),
            "fit_rows": len(fit),
            "mae_median": mae(test["cnt"], median_t),
            "coverage_raw": coverage(test["cnt"], lower_t, upper_t),
            "coverage_cal": coverage(test["cnt"], lower_new, upper_new),
            "width_raw": avg_width(lower_t, upper_t),
            "width_cal": avg_width(lower_new, upper_new),
            "correction": correction,
        })

        preds = test[["cnt"]].copy()
        preds["median"] = median_t
        preds["lower_raw"] = lower_t
        preds["upper_raw"] = upper_t
        preds["lower_cal"] = lower_new
        preds["upper_cal"] = upper_new
        all_preds.append(preds)

        print("done:", test_start.strftime("%Y-%m"))

    return pd.DataFrame(rows), pd.concat(all_preds)