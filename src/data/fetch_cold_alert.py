import requests

COLD_URL = "https://ukhsa-dashboard.data.gov.uk/api/proxy/alerts/v1/cold"
LONDON_GEO_CODE = "E12000007"

SEVERITY_MAP = {"Green": 0, "Yellow": 1, "Amber": 2, "Red": 3}


def get_london_cold_alert():
    try:
        resp = requests.get(COLD_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        for record in data:
            if record.get("geography_code") == LONDON_GEO_CODE:
                return {
                    "status": record.get("status", "Unknown"),
                    "refresh_date": record.get("refresh_date", ""),
                    "geography_name": record.get("geography_name", "London"),
                }
    except Exception:
        pass
    return {"status": "Unknown", "refresh_date": "", "geography_name": "London"}


def simulate_cold_alert(temp_min, sim_date_str):
    if temp_min > 6:
        status = "Green"
    elif temp_min > 2:
        status = "Yellow"
    elif temp_min > -2:
        status = "Amber"
    else:
        status = "Red"
    return {"status": status, "refresh_date": sim_date_str, "geography_name": "London (simulated)"}


def alert_severity(status):
    return SEVERITY_MAP.get(status, 0)


if __name__ == "__main__":
    alert = get_london_cold_alert()
    print(f"London Cold Alert: {alert['status']} (severity {alert_severity(alert['status'])})")
    print(f"Refreshed: {alert['refresh_date']}")
