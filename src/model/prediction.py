import numpy as np
import pandas as pd


LAG_WEIGHTS = np.array([0.0, 0.0, 0.0, 0.10, 0.20, 0.30, 0.25, 0.15])

RISK_LABELS = {0: "Green", 1: "Yellow", 2: "Amber", 3: "Red"}


def cold_risk_multiplier(temp_min):
    if temp_min > 6:
        return 0
    elif temp_min > 2:
        return 1
    elif temp_min > -2:
        return 2
    else:
        return 3


def predict_respiratory_risk(forecast_df):
    df = forecast_df.copy()
    df["cold_multiplier"] = df["temp_min"].apply(cold_risk_multiplier)

    multipliers = df["cold_multiplier"].values
    padded = np.concatenate([np.zeros(len(LAG_WEIGHTS) - 1), multipliers])
    predicted = np.convolve(padded, LAG_WEIGHTS, mode="valid")

    df["predicted_risk"] = predicted[:len(df)]
    df["risk_label"] = df["cold_multiplier"].map(RISK_LABELS)
    return df


def find_historical_analogues(forecast_df, historical_df, n=3):
    forecast_temps = forecast_df["temp_min"].values[:7]
    if len(forecast_temps) < 7:
        return []

    hist = historical_df.sort_values("date").reset_index(drop=True)
    hist_temps = hist["temp_min"].values
    hist_dates = hist["date"].values

    best = []
    for i in range(len(hist_temps) - 13):
        window = hist_temps[i:i + 7]
        if np.any(np.isnan(window)):
            continue
        dist = np.sqrt(np.sum((forecast_temps - window) ** 2))
        following = hist_temps[i + 7:i + 14]
        following_dates = hist_dates[i + 7:i + 14]
        best.append({
            "distance": dist,
            "start_date": str(hist_dates[i])[:10],
            "end_date": str(hist_dates[i + 6])[:10],
            "temps": window.tolist(),
            "following_min_temp": float(np.nanmin(following)) if len(following) > 0 else None,
            "following_avg_temp": float(np.nanmean(following)) if len(following) > 0 else None,
        })

    best.sort(key=lambda x: x["distance"])
    return best[:n]
