# RespiCast London

Borough-level respiratory risk warnings for NHS GPs, pharmacists, and planners.

Every winter, cold and damp homes quietly fuel one of the largest seasonal pressures on the NHS. RespiCast London turns regional cold weather alerts into borough-level respiratory illness warnings — combining live UKHSA cold alerts, deprivation data, respiratory admissions, housing quality, and weather forecasts into a single actionable dashboard.

Built for the **Health in Climate London 2026 Hackathon**.

## Features

- **Live UKHSA Cold Weather Alert** — real-time alert status for London
- **7-Day Respiratory Risk Forecast** — weather-driven predictions with epidemiological lag-effect modelling
- **Historical Analogue Matching** — finds similar past winter periods to contextualise forecasts
- **Borough Vulnerability Map** — choropleth map combining deprivation, fuel poverty, COPD admissions, and housing quality
- **Housing Quality Analysis** — London Building Stock Model v2 data with ward-level drill-down
- **Borough Deep Dive** — COPD trend charts, vulnerability component breakdowns, and key metrics

## Data Sources

| Source | Data | API |
|--------|------|-----|
| UKHSA | Cold weather alerts | Live REST API |
| OHID Fingertips | COPD & asthma admissions, fuel poverty | REST API |
| MHCLG | Indices of Deprivation 2025 | Excel download |
| Open-Meteo | 7-day forecast + ERA5 historical reanalysis | REST API (no key) |
| GLA | London Building Stock Model v2 | CSV downloads |
| ONS | Borough & ward boundaries | ArcGIS REST API |

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Data is fetched automatically on first run. The housing data fetch takes several minutes (33 borough CSVs, 5–97 MB each).

To pre-fetch data:
```bash
python -m src.data.fetch_respiratory
python -m src.data.fetch_vulnerability
python -m src.data.fetch_boundaries
python -m src.data.fetch_weather_forecast
python -m src.data.fetch_ward_boundaries
python -m src.data.fetch_housing
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect the repo, set main file to `app.py`
4. Deploy

## Vulnerability Model

Composite score per borough (0–1 scale):

| Component | Weight | Source |
|-----------|--------|--------|
| IMD Deprivation | 0.30 | IoD2025 |
| Fuel Poverty | 0.25 | Fingertips |
| COPD Admissions | 0.25 | Fingertips |
| Housing Quality | 0.20 | LBSM v2 |

Risk level = vulnerability score × cold alert severity (0–3).

## Prediction Model

- **Cold risk multiplier**: UKHSA-aligned temperature thresholds (>6°C Green, 2–6°C Yellow, -2–2°C Amber, <-2°C Red)
- **Lag-effect convolution**: cold exposure at day T → respiratory admissions peak at T+3 to T+7
- **Historical analogues**: Euclidean distance matching against 10 years of winter data
