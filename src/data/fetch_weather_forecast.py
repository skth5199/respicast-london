import os
import pandas as pd
import requests

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
LAT, LON = 51.5, -0.1
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def fetch_7day_forecast():
    os.makedirs(DATA_DIR, exist_ok=True)
    params = {
        "latitude": LAT,
        "longitude": LON,
        "daily": "temperature_2m_min,temperature_2m_max,relative_humidity_2m_mean",
        "timezone": "Europe/London",
    }
    resp = requests.get(FORECAST_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["daily"]
    df = pd.DataFrame({
        "date": pd.to_datetime(data["time"]),
        "temp_min": data["temperature_2m_min"],
        "temp_max": data["temperature_2m_max"],
        "humidity": data["relative_humidity_2m_mean"],
    })
    path = os.path.join(DATA_DIR, "weather_forecast.csv")
    df.to_csv(path, index=False)
    print(f"Saved 7-day forecast: {len(df)} days to {path}")
    return df


def fetch_historical_winters():
    os.makedirs(DATA_DIR, exist_ok=True)
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": "2015-01-01",
        "end_date": "2025-12-31",
        "daily": "temperature_2m_min,temperature_2m_max",
        "timezone": "Europe/London",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()["daily"]
    df = pd.DataFrame({
        "date": pd.to_datetime(data["time"]),
        "temp_min": data["temperature_2m_min"],
        "temp_max": data["temperature_2m_max"],
    })
    df["month"] = df["date"].dt.month
    df = df[df["month"].isin([10, 11, 12, 1, 2, 3])].drop(columns=["month"])
    path = os.path.join(DATA_DIR, "weather_historical_winters.csv")
    df.to_csv(path, index=False)
    print(f"Saved historical winters: {len(df)} days to {path}")
    return df


def fetch_and_save():
    fetch_7day_forecast()
    fetch_historical_winters()


if __name__ == "__main__":
    fetch_and_save()
