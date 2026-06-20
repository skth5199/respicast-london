import json
import os
import requests

ONS_URL = "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/WD_MAY_2023_UK_BGC/FeatureServer/0/query"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
BATCH_SIZE = 500


def fetch_london_ward_geojson():
    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, "london_wards.geojson")

    all_features = []
    offset = 0

    print("Downloading London ward boundaries from ONS...")
    while True:
        params = {
            "where": "LAD23CD LIKE 'E09%'",
            "outFields": "WD23CD,WD23NM,LAD23CD,LAD23NM",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
            "resultRecordCount": BATCH_SIZE,
            "resultOffset": offset,
        }
        resp = requests.get(ONS_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        features = data.get("features", [])
        if not features:
            break
        all_features.extend(features)
        print(f"  Fetched {len(all_features)} wards...")
        offset += BATCH_SIZE
        if len(features) < BATCH_SIZE:
            break

    london_geojson = {
        "type": "FeatureCollection",
        "features": all_features,
    }

    with open(output_path, "w") as f:
        json.dump(london_geojson, f)

    print(f"Saved {len(all_features)} London ward boundaries to {output_path}")
    return london_geojson


if __name__ == "__main__":
    fetch_london_ward_geojson()
