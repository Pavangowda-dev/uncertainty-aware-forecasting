# features.py
# This file creates the features (input columns) for our models

import numpy as np

# the columns the models will use as input
FEATURES = [
    "hr", "weekday", "mnth", "workingday", "holiday", "season",
    "weathersit", "temp", "hum", "windspeed",
    "hour_sin", "hour_cos",
    "lag_24", "lag_168", "roll_mean_24", "roll_mean_168",
]

def make_features(df):
    df = df.copy()

    # hour of the day as a circle (so hour 23 and hour 0 are close to each other)
    df["hour_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hr"] / 24)

    # past rentals: same hour yesterday and same hour last week
    df["lag_24"] = df["cnt"].shift(24)
    df["lag_168"] = df["cnt"].shift(168)

    # recent average level (also shifted by 24 hours)
    # this helps the model understand the trend (growth over time)
    df["roll_mean_24"] = df["cnt"].shift(24).rolling(24).mean()
    df["roll_mean_168"] = df["cnt"].shift(24).rolling(168).mean()

    return df