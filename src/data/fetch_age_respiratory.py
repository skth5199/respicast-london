import io
import os
import pandas as pd
import requests

BASE_URL = "https://fingertips.phe.org.uk/api/all_data/csv/by_indicator_id"
AREA_TYPE = 402
PARENT_AREA = "E12000007"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

INDICATORS = {
    90810: "asthma_admissions",
    90360: "winter_mortality_all",
}


def _fetch_indicator(indicator_id, label):
    url = f"{BASE_URL}?indicator_ids={indicator_id}&area_type_id={AREA_TYPE}&parent_area_code={PARENT_AREA}"
    print(f"Fetching {label} (indicator {indicator_id})...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    df = df[df["Area Code"].str.startswith("E09", na=False)]
    df = df[df["Sex"] == "Persons"]
    df = df[df["Category Type"].isna() | (df["Category Type"] == "")]
    return df[["Area Code", "Area Name", "Indicator Name", "Age",
               "Time period", "Time period Sortable", "Value"]].copy()


def fetch_and_save():
    os.makedirs(DATA_DIR, exist_ok=True)
    frames = []

    for indicator_id, label in INDICATORS.items():
        df = _fetch_indicator(indicator_id, label)
        df["indicator_label"] = label
        print(f"  {len(df)} rows, ages: {df['Age'].unique()}, boroughs: {df['Area Code'].nunique()}")
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    result.columns = [
        "area_code", "area_name", "indicator_name", "age_group",
        "time_period", "time_period_sortable", "value", "indicator_label",
    ]

    path = os.path.join(DATA_DIR, "respiratory_age_borough.csv")
    result.to_csv(path, index=False)
    print(f"\nSaved {len(result)} rows to {path}")

    latest_year = {}
    for label in INDICATORS.values():
        sub = result[result["indicator_label"] == label]
        if not sub.empty:
            latest_year[label] = sub["time_period_sortable"].max()

    latest = result[
        result.apply(
            lambda r: r["time_period_sortable"] == latest_year.get(r["indicator_label"]),
            axis=1,
        )
    ]
    latest_path = os.path.join(DATA_DIR, "respiratory_age_latest.csv")
    latest.to_csv(latest_path, index=False)
    print(f"Saved {len(latest)} latest-year rows to {latest_path}")

    return result


if __name__ == "__main__":
    fetch_and_save()
