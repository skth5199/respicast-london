import math
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


def _bng_to_latlon(easting, northing):
    a, b = 6377563.396, 6356256.909
    F0 = 0.9996012717
    lat0, lon0 = math.radians(49), math.radians(-2)
    N0, E0 = -100000, 400000
    e2 = 1 - (b * b) / (a * a)
    n = (a - b) / (a + b)

    lat = lat0
    for _ in range(10):
        M = b * F0 * (
            (1 + n + 1.25 * n**2 + 1.25 * n**3) * (lat - lat0)
            - (3 * n + 3 * n**2 + 21 / 8 * n**3) * math.sin(lat - lat0) * math.cos(lat + lat0)
            + (15 / 8 * n**2 + 15 / 8 * n**3) * math.sin(2 * (lat - lat0)) * math.cos(2 * (lat + lat0))
            - (35 / 24 * n**3) * math.sin(3 * (lat - lat0)) * math.cos(3 * (lat + lat0))
        )
        lat = (northing - N0 - M) / (a * F0) + lat

    v = a * F0 / math.sqrt(1 - e2 * math.sin(lat)**2)
    rho = a * F0 * (1 - e2) / (1 - e2 * math.sin(lat)**2)**1.5
    eta2 = v / rho - 1
    tan_lat = math.tan(lat)
    sec_lat = 1 / math.cos(lat)
    VII = tan_lat / (2 * rho * v)
    VIII = tan_lat / (24 * rho * v**3) * (5 + 3 * tan_lat**2 + eta2 - 9 * tan_lat**2 * eta2)
    IX = tan_lat / (720 * rho * v**5) * (61 + 90 * tan_lat**2 + 45 * tan_lat**4)
    X = sec_lat / v
    XI = sec_lat / (6 * v**3) * (v / rho + 2 * tan_lat**2)
    XII = sec_lat / (120 * v**5) * (5 + 28 * tan_lat**2 + 24 * tan_lat**4)

    dE = easting - E0
    out_lat = lat - VII * dE**2 + VIII * dE**4 - IX * dE**6
    out_lon = lon0 + X * dE - XI * dE**3 + XII * dE**5

    return math.degrees(out_lat), math.degrees(out_lon)


def _postcode_sector(postcode):
    parts = str(postcode).strip().split()
    if len(parts) == 2:
        return parts[0] + " " + parts[1][0]
    return postcode


def _aggregate_postcode(df):
    df["postcode_sector"] = df["postcode_locator"].apply(_postcode_sector)
    df["easting"] = pd.to_numeric(df["easting"], errors="coerce")
    df["northing"] = pd.to_numeric(df["northing"], errors="coerce")

    group_cols = ["administrative_area", "ward22cd", "ward22nm", "postcode_sector"]
    agg = _aggregate_housing(df, group_cols)

    coords = df.groupby(group_cols).agg(
        mean_easting=("easting", "mean"),
        mean_northing=("northing", "mean"),
    ).reset_index()
    agg = agg.merge(coords, on=group_cols, how="left")
    return agg


def _compute_housing_score(df, fuel_poverty_pct=None):
    epc_n = _min_max_normalise(df["pct_epc_poor"])
    wall_n = _min_max_normalise(df["pct_uninsulated_walls"])
    glaz_n = _min_max_normalise(df["pct_single_glazing"])
    heat_n = _min_max_normalise(df["pct_no_central_heating"])
    df["building_fabric_score"] = 0.4 * epc_n + 0.3 * wall_n + 0.15 * glaz_n + 0.15 * heat_n
    if fuel_poverty_pct is not None:
        fp_n = _min_max_normalise(fuel_poverty_pct)
        df["housing_quality_score"] = (
            0.25 * epc_n + 0.20 * wall_n + 0.10 * glaz_n
            + 0.10 * heat_n + 0.35 * fp_n
        )
    else:
        df["housing_quality_score"] = df["building_fabric_score"]
    return df


def fetch_and_save():
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_urls = _get_borough_csv_urls()

    borough_rows = []
    ward_rows = []
    postcode_rows = []
    cols_to_read = [
        "administrative_area", "ward22cd", "ward22nm",
        "postcode_locator", "easting", "northing",
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

                if "postcode_locator" in df.columns:
                    pc_agg = _aggregate_postcode(df)
                    postcode_rows.append(pc_agg)

        except Exception as e:
            print(f"    ERROR: {e}")
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            continue

    borough_df = pd.concat(borough_rows, ignore_index=True)
    vuln_path = os.path.join(DATA_DIR, "vulnerability_inputs.csv")
    fp_series = None
    if os.path.exists(vuln_path):
        vuln = pd.read_csv(vuln_path)
        borough_df = borough_df.merge(
            vuln[["area_name", "fuel_poverty_pct"]],
            left_on="administrative_area", right_on="area_name", how="left",
        ).drop(columns=["area_name"])
        borough_df["fuel_poverty_pct"] = pd.to_numeric(borough_df["fuel_poverty_pct"], errors="coerce")
        fp_series = borough_df["fuel_poverty_pct"]
    borough_df = _compute_housing_score(borough_df, fuel_poverty_pct=fp_series)
    borough_path = os.path.join(DATA_DIR, "housing_borough.csv")
    borough_df.to_csv(borough_path, index=False)
    print(f"\nSaved borough housing data: {len(borough_df)} boroughs to {borough_path}")

    if ward_rows:
        ward_df = pd.concat(ward_rows, ignore_index=True)
        ward_df = _compute_housing_score(ward_df)
        ward_path = os.path.join(DATA_DIR, "housing_ward.csv")
        ward_df.to_csv(ward_path, index=False)
        print(f"Saved ward housing data: {len(ward_df)} wards to {ward_path}")

    if postcode_rows:
        pc_df = pd.concat(postcode_rows, ignore_index=True)
        pc_df = _compute_housing_score(pc_df)
        valid = pc_df["mean_easting"].notna() & pc_df["mean_northing"].notna()
        latlons = pc_df.loc[valid].apply(
            lambda r: _bng_to_latlon(r["mean_easting"], r["mean_northing"]), axis=1,
        )
        pc_df.loc[valid, "lat"] = [ll[0] for ll in latlons]
        pc_df.loc[valid, "lon"] = [ll[1] for ll in latlons]
        pc_path = os.path.join(DATA_DIR, "housing_postcode.csv")
        pc_df.to_csv(pc_path, index=False)
        print(f"Saved postcode housing data: {len(pc_df)} postcodes to {pc_path}")

    return borough_df


if __name__ == "__main__":
    fetch_and_save()
