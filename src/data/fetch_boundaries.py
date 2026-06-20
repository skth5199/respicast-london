import json
import os
import requests
import topojson as tp

TOPO_URL = "https://martinjc.github.io/UK-GeoJSON/json/eng/topo_lad.json"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def fetch_london_geojson():
    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, "london_boroughs.geojson")

    print("Downloading UK LAD boundaries (TopoJSON)...")
    resp = requests.get(TOPO_URL, timeout=60)
    resp.raise_for_status()
    topo_data = resp.json()

    object_name = list(topo_data.get("objects", {}).keys())[0]
    print(f"  TopoJSON object: '{object_name}'")

    topology = tp.Topology(topo_data, object_name=object_name)
    geojson_str = topology.to_geojson()
    geojson = json.loads(geojson_str) if isinstance(geojson_str, str) else geojson_str

    london_features = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        code = props.get("LAD13CD", "")
        if code.startswith("E09"):
            london_features.append(feature)

    london_geojson = {
        "type": "FeatureCollection",
        "features": london_features,
    }

    with open(output_path, "w") as f:
        json.dump(london_geojson, f)

    print(f"Saved {len(london_features)} London borough boundaries to {output_path}")
    return london_geojson


if __name__ == "__main__":
    fetch_london_geojson()
