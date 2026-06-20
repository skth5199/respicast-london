import io
import os
import pandas as pd
import requests

BASE_URL = "https://fingertips.phe.org.uk/api/all_data/csv/by_indicator_id"
INDICATOR_COPD = 92302
INDICATOR_ASTHMA = 90810
AREA_TYPE = 402
PARENT_AREA = "E12000007"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def fetch_fingertips_csv(indicator_id):
    url = f"{BASE_URL}?indicator_ids={indicator_id}&area_type_id={AREA_TYPE}&parent_area_code={PARENT_AREA}"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df = df[df["Area Code"].str.startswith("E09", na=False)]
    df = df[df["Sex"] == "Persons"]
    df = df[df["Category Type"].isna() | (df["Category Type"] == "")]
    return df[["Area Code", "Area Name", "Time period", "Value", "Count", "Denominator",
               "Time period Sortable"]].copy()


def fetch_and_save():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Fetching COPD admissions (indicator 92302)...")
    copd = fetch_fingertips_csv(INDICATOR_COPD)
    copd.to_csv(os.path.join(DATA_DIR, "respiratory_copd_borough.csv"), index=False)
    print(f"  {len(copd)} rows, {copd['Area Code'].nunique()} boroughs")

    print("Fetching asthma admissions (indicator 90810)...")
    asthma = fetch_fingertips_csv(INDICATOR_ASTHMA)
    asthma.to_csv(os.path.join(DATA_DIR, "respiratory_asthma_borough.csv"), index=False)
    print(f"  {len(asthma)} rows, {asthma['Area Code'].nunique()} boroughs")

    latest_year = copd["Time period Sortable"].max()
    latest = copd[copd["Time period Sortable"] == latest_year][
        ["Area Code", "Area Name", "Value"]
    ].rename(columns={"Value": "copd_rate"})
    latest.to_csv(os.path.join(DATA_DIR, "respiratory_latest.csv"), index=False)
    print(f"  Latest year: {latest_year}, {len(latest)} boroughs saved to respiratory_latest.csv")


if __name__ == "__main__":
    fetch_and_save()
