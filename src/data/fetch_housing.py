import os
import tempfile
import pandas as pd
import requests

CKAN_URL = "https://data.london.gov.uk/api/3/action/package_show?id=2k55d"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def _get_borough_csv_urls():
    print("Discovering LBSM borough CSV URLs from CKAN...")
    resp = requests.get(CKAN_URL, timeout=30)
    resp.raise_for_status()
    resources = resp.json()["result"]["resources"]
    csv_urls = {}
    prefix = "London Building Stock Model 2 - "
    for r in resources:
        name = r.get("name", "")
        url = r.get("url", "")
        if name.startswith(prefix) and url.endswith(".csv"):
            borough_name = name[len(prefix):]
            csv_urls[borough_name] = url
    print(f"  Found {len(csv_urls)} borough CSVs")
    return csv_urls


def _min_max_normalise(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return series * 0.0
    return (series - mn) / (mx - mn)


def _aggregate_housing(df, group_cols):
    total = df.groupby(group_cols).size().reset_index(name="total_properties")

    epc_poor = df[df["epc_rating"].isin(["E", "F-G"])].groupby(group_cols).size().reset_index(name="n_epc_poor")
    uninsu = df[df["wall_insulation"] == "uninsulated"].groupby(group_cols).size().reset_index(name="n_uninsulated")
    single = df[df["glazing_type"].isin(["single/partial", "single"])].groupby(group_cols).size().reset_index(name="n_single_glazing")
    no_ch = df[df["main_heat_type"].isin(["room/storage heaters", "other", "no heating system"])].groupby(group_cols).size().reset_index(name="n_no_central_heating")

    agg = total
    for extra in [epc_poor, uninsu, single, no_ch]:
        agg = agg.merge(extra, on=group_cols, how="left")

    for col in ["n_epc_poor", "n_uninsulated", "n_single_glazing", "n_no_central_heating"]:
        agg[col] = agg[col].fillna(0)

    agg["pct_epc_poor"] = 100 * agg["n_epc_poor"] / agg["total_properties"]
    agg["pct_uninsulated_walls"] = 100 * agg["n_uninsulated"] / agg["total_properties"]
    agg["pct_single_glazing"] = 100 * agg["n_single_glazing"] / agg["total_properties"]
    agg["pct_no_central_heating"] = 100 * agg["n_no_central_heating"] / agg["total_properties"]

    return agg


def _compute_housing_score(df):
    epc_n = _min_max_normalise(df["pct_epc_poor"])
    wall_n = _min_max_normalise(df["pct_uninsulated_walls"])
    glaz_n = _min_max_normalise(df["pct_single_glazing"])
    heat_n = _min_max_normalise(df["pct_no_central_heating"])
    df["housing_quality_score"] = 0.4 * epc_n + 0.3 * wall_n + 0.15 * glaz_n + 0.15 * heat_n
    return df


def fetch_and_save():
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_urls = _get_borough_csv_urls()

    borough_rows = []
    ward_rows = []
    cols_to_read = [
        "administrative_area", "ward22cd", "ward22nm",
        "epc_rating", "wall_insulation", "glazing_type",
        "main_heat_type", "main_fuel_type",
    ]

    for i, (borough_name, url) in enumerate(csv_urls.items(), 1):
        print(f"  [{i}/{len(csv_urls)}] Processing {borough_name}...")
        try:
            resp = requests.get(url, timeout=300, stream=True)
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                tmp_path = tmp.name
                for data_chunk in resp.iter_content(chunk_size=1024 * 1024):
                    tmp.write(data_chunk)

            chunks = []
            for chunk in pd.read_csv(
                tmp_path,
                usecols=lambda c: c in cols_to_read,
                dtype=str,
                chunksize=50000,
            ):
                chunks.append(chunk)
            os.unlink(tmp_path)
            df = pd.concat(chunks, ignore_index=True)

            borough_agg = _aggregate_housing(df, ["administrative_area"])
            borough_rows.append(borough_agg)

            if "ward22cd" in df.columns and "ward22nm" in df.columns:
                ward_agg = _aggregate_housing(df, ["administrative_area", "ward22cd", "ward22nm"])
                ward_rows.append(ward_agg)

        except Exception as e:
            print(f"    ERROR: {e}")
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            continue

    borough_df = pd.concat(borough_rows, ignore_index=True)
    borough_df = _compute_housing_score(borough_df)
    borough_path = os.path.join(DATA_DIR, "housing_borough.csv")
    borough_df.to_csv(borough_path, index=False)
    print(f"\nSaved borough housing data: {len(borough_df)} boroughs to {borough_path}")

    if ward_rows:
        ward_df = pd.concat(ward_rows, ignore_index=True)
        ward_df = _compute_housing_score(ward_df)
        ward_path = os.path.join(DATA_DIR, "housing_ward.csv")
        ward_df.to_csv(ward_path, index=False)
        print(f"Saved ward housing data: {len(ward_df)} wards to {ward_path}")

    return borough_df


if __name__ == "__main__":
    fetch_and_save()
