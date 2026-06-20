import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.data.fetch_cold_alert import get_london_cold_alert, alert_severity
from src.model.vulnerability import compute_vulnerability_scores, build_risk_table, compute_age_risk_scores

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

ALERT_COLOURS = {
    "Green": "#2e7d32",
    "Yellow": "#f9a825",
    "Amber": "#ef6c00",
    "Red": "#c62828",
    "Unknown": "#757575",
}

RISK_COLOURS = {"Green": "#2e7d32", "Yellow": "#f9a825", "Amber": "#ef6c00", "Red": "#c62828"}

FORMULA_VULN = (
    "Vulnerability = 0.30 × Deprivation + 0.25 × Fuel Poverty "
    "+ 0.25 × COPD Rate + 0.20 × Housing Quality. "
    "Each component is min-max normalised across London boroughs (0 = lowest, 1 = highest)."
)
FORMULA_VULN_NO_HOUSING = (
    "Vulnerability = 0.40 × Deprivation + 0.30 × Fuel Poverty "
    "+ 0.30 × COPD Rate. "
    "Each component is min-max normalised across London boroughs (0 = lowest, 1 = highest)."
)
FORMULA_HOUSING = (
    "Housing Score = 0.40 × Poor EPC (E/F/G) + 0.30 × Uninsulated Walls "
    "+ 0.15 × Single Glazing + 0.15 × No Central Heating. Higher = worse housing stock."
)
FORMULA_CHILD_RISK = (
    "Childhood Risk = min-max normalised asthma admission rate (ages 0–9) "
    "across London boroughs. 0 = lowest borough, 1 = highest."
)
FORMULA_ELDERLY_RISK = (
    "Elderly Risk = 0.50 × COPD rate (normalised) + 0.50 × Winter Mortality 85+ (normalised). "
    "Combines chronic respiratory disease burden with cold-weather mortality for the elderly."
)
FORMULA_RISK_LEVEL = (
    "Risk Level = Vulnerability Score × Cold Alert Severity "
    "(Green=0, Yellow=1, Amber=2, Red=3). "
    "Result: ≤0 Low, ≤1 Moderate, ≤2 High, >2 Very High."
)
FORMULA_COLD_MULT = (
    "Daily cold risk: >6°C = Green (no risk), 2–6°C = Yellow (low), "
    "-2–2°C = Amber (moderate), <-2°C = Red (high). Based on UKHSA thresholds."
)
FORMULA_PREDICTED_RISK = (
    "Predicted risk uses lag-weighted cold multipliers over 8 days "
    "(weights: 0, 0, 0, 0.10, 0.20, 0.30, 0.25, 0.15), "
    "reflecting the 3–7 day delay between cold exposure and respiratory admissions."
)

st.set_page_config(page_title="RespiCast London", layout="wide", page_icon="🫁")


def _score_label(score, labels=("Low", "Moderate", "High", "Very High")):
    if pd.isna(score):
        return "N/A"
    if score < 0.25:
        return labels[0]
    if score < 0.50:
        return labels[1]
    if score < 0.75:
        return labels[2]
    return labels[3]


def _borough_rank(value, series):
    rank = int((series.dropna() >= value).sum())
    total = int(series.dropna().count())
    return f"Rank {rank} of {total}"


def ensure_data():
    core_files = {
        "respiratory": os.path.join(DATA_DIR, "respiratory_latest.csv"),
        "copd": os.path.join(DATA_DIR, "respiratory_copd_borough.csv"),
        "vulnerability": os.path.join(DATA_DIR, "vulnerability_inputs.csv"),
        "boundaries": os.path.join(DATA_DIR, "london_boroughs.geojson"),
    }
    extra_files = {
        "housing": os.path.join(DATA_DIR, "housing_borough.csv"),
        "ward_boundaries": os.path.join(DATA_DIR, "london_wards.geojson"),
        "age_respiratory": os.path.join(DATA_DIR, "respiratory_age_borough.csv"),
    }

    missing_core = {k: v for k, v in core_files.items() if not os.path.exists(v)}
    missing_extra = {k: v for k, v in extra_files.items() if not os.path.exists(v)}

    if not missing_core and not missing_extra:
        return

    with st.spinner("Fetching data for the first time (this may take several minutes)..."):
        if "respiratory" in missing_core or "copd" in missing_core:
            from src.data.fetch_respiratory import fetch_and_save as fetch_resp
            fetch_resp()
        if "vulnerability" in missing_core:
            from src.data.fetch_vulnerability import fetch_and_save as fetch_vuln
            fetch_vuln()
        if "boundaries" in missing_core:
            from src.data.fetch_boundaries import fetch_london_geojson
            fetch_london_geojson()
        if "ward_boundaries" in missing_extra:
            from src.data.fetch_ward_boundaries import fetch_london_ward_geojson
            fetch_london_ward_geojson()
        if "age_respiratory" in missing_extra:
            from src.data.fetch_age_respiratory import fetch_and_save as fetch_age
            fetch_age()
        if "housing" in missing_extra:
            from src.data.fetch_housing import fetch_and_save as fetch_housing
            fetch_housing()


@st.cache_data(ttl=3600)
def load_data():
    vuln = pd.read_csv(os.path.join(DATA_DIR, "vulnerability_inputs.csv"))
    resp = pd.read_csv(os.path.join(DATA_DIR, "respiratory_latest.csv"))
    copd_ts = pd.read_csv(os.path.join(DATA_DIR, "respiratory_copd_borough.csv"))
    with open(os.path.join(DATA_DIR, "london_boroughs.geojson")) as f:
        geojson = json.load(f)

    housing = None
    housing_path = os.path.join(DATA_DIR, "housing_borough.csv")
    if os.path.exists(housing_path):
        housing = pd.read_csv(housing_path)

    housing_ward = None
    ward_path = os.path.join(DATA_DIR, "housing_ward.csv")
    if os.path.exists(ward_path):
        housing_ward = pd.read_csv(ward_path)

    ward_geojson = None
    ward_geo_path = os.path.join(DATA_DIR, "london_wards.geojson")
    if os.path.exists(ward_geo_path):
        with open(ward_geo_path) as f:
            ward_geojson = json.load(f)

    housing_postcode = None
    pc_path = os.path.join(DATA_DIR, "housing_postcode.csv")
    if os.path.exists(pc_path):
        housing_postcode = pd.read_csv(pc_path)

    age_data = None
    age_path = os.path.join(DATA_DIR, "respiratory_age_borough.csv")
    if os.path.exists(age_path):
        age_data = pd.read_csv(age_path)

    age_latest = None
    age_latest_path = os.path.join(DATA_DIR, "respiratory_age_latest.csv")
    if os.path.exists(age_latest_path):
        age_latest = pd.read_csv(age_latest_path)

    return vuln, resp, copd_ts, geojson, housing, housing_ward, ward_geojson, housing_postcode, age_data, age_latest


@st.cache_data(ttl=1800)
def load_forecast():
    from src.data.fetch_weather_forecast import fetch_7day_forecast, fetch_historical_winters
    from src.model.prediction import predict_respiratory_risk, find_historical_analogues, enrich_analogues_with_outcomes

    forecast_path = os.path.join(DATA_DIR, "weather_forecast.csv")
    hist_path = os.path.join(DATA_DIR, "weather_historical_winters.csv")

    if not os.path.exists(forecast_path):
        fetch_7day_forecast()
    if not os.path.exists(hist_path):
        fetch_historical_winters()

    forecast_df = pd.read_csv(forecast_path, parse_dates=["date"])
    hist_df = pd.read_csv(hist_path, parse_dates=["date"])

    prediction_df = predict_respiratory_risk(forecast_df)
    analogues = find_historical_analogues(forecast_df, hist_df, n=3)

    wm_path = os.path.join(DATA_DIR, "respiratory_age_borough.csv")
    if os.path.exists(wm_path):
        wm_df = pd.read_csv(wm_path)
        wm_london = wm_df[wm_df["indicator_label"].isin(["winter_mortality_all", "winter_mortality_85plus"])]
        analogues = enrich_analogues_with_outcomes(analogues, wm_london)

    return prediction_df, analogues


def render_alert_banner(alert):
    status = alert["status"]
    colour = ALERT_COLOURS.get(status, "#757575")
    text_colour = "#fff" if status in ("Green", "Amber", "Red", "Unknown") else "#000"
    refresh = alert["refresh_date"][:10] if alert["refresh_date"] else "N/A"

    st.markdown(f"""
    <div style="background:{colour}; color:{text_colour}; padding:16px 24px;
                border-radius:8px; margin-bottom:20px; text-align:center;">
        <h2 style="margin:0; color:{text_colour};">
            UKHSA Cold Weather Alert: {status}
        </h2>
        <p style="margin:4px 0 0; opacity:0.9; color:{text_colour};">
            Region: London &nbsp;|&nbsp; Last refreshed: {refresh}
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_forecast_section(prediction_df, analogues):
    st.subheader("7-Day Respiratory Risk Forecast")

    cols_badge = st.columns(len(prediction_df))
    for i, row in prediction_df.iterrows():
        day_label = row["date"].strftime("%a %d")
        colour = RISK_COLOURS.get(row["risk_label"], "#757575")
        with cols_badge[i]:
            st.markdown(
                f'<div style="background:{colour}; color:white; padding:8px; '
                f'border-radius:6px; text-align:center; font-size:0.85em;">'
                f'<b>{day_label}</b><br>{row["risk_label"]}<br>'
                f'{row["temp_min"]:.0f}°C min</div>',
                unsafe_allow_html=True,
            )

    st.caption(FORMULA_COLD_MULT)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=prediction_df["date"], y=prediction_df["predicted_risk"],
            name="Predicted Risk", mode="lines+markers",
            line=dict(color="#c62828", width=3),
            fill="tozeroy", fillcolor="rgba(198,40,40,0.1)",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=prediction_df["date"], y=prediction_df["temp_min"],
            name="Min Temperature (°C)", mode="lines+markers",
            line=dict(color="#1565c0", width=2, dash="dot"),
        ),
        secondary_y=True,
    )

    for level, y_val, colour in [("Yellow", 1, "#f9a825"), ("Amber", 2, "#ef6c00"), ("Red", 3, "#c62828")]:
        fig.add_hline(y=y_val, line_dash="dash", line_color=colour,
                      annotation_text=level, annotation_position="top left",
                      secondary_y=False)

    fig.update_layout(
        height=350, margin={"t": 30, "b": 30, "l": 0, "r": 0},
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Risk Level (0–3)", range=[0, 3.5], secondary_y=False)
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True)

    st.plotly_chart(fig, config={"responsive": True})
    st.caption(FORMULA_PREDICTED_RISK)

    if analogues:
        with st.expander("Historical Analogues — similar past weather patterns & hospital outcomes"):
            for j, a in enumerate(analogues, 1):
                st.markdown(
                    f"**Match {j}** ({a['start_date']} to {a['end_date']}) — "
                    f"Distance: {a['distance']:.1f} | "
                    f"Following week: min {a['following_min_temp']:.1f}°C, "
                    f"avg {a['following_avg_temp']:.1f}°C"
                )

                wm_all = a.get("winter_mortality_all")
                wm_85 = a.get("winter_mortality_85plus")
                if wm_all is not None or wm_85 is not None:
                    wy = a.get("winter_year", "Unknown")
                    parts = []
                    if wm_all is not None:
                        avg_all = a.get("winter_mortality_all_avg")
                        pct = a.get("winter_mortality_pct_above")
                        severity = "above" if pct and pct > 0 else "below"
                        parts.append(f"Winter mortality index: **{wm_all}** (London avg: {avg_all})")
                        if pct is not None:
                            parts.append(f"**{abs(pct):.1f}% {severity}** average")
                    if wm_85 is not None:
                        avg_85 = a.get("winter_mortality_85plus_avg")
                        parts.append(f"85+ mortality index: **{wm_85}** (avg: {avg_85})")
                    st.markdown(
                        f"&nbsp;&nbsp;&nbsp;&nbsp;📊 *Winter {wy}*: " + " | ".join(parts)
                    )
                else:
                    st.markdown(
                        f"&nbsp;&nbsp;&nbsp;&nbsp;*No winter mortality data for {a.get('winter_year', 'this period')}*"
                    )


def render_map(risk_df, geojson, color_col="vulnerability_score", title="Vulnerability"):
    hover = {
        "vulnerability_score": ":.2f",
        "vuln_rating": True,
        "risk_level": True,
        "area_code": False,
        color_col: ":.2f",
    }
    if "vuln_rating" not in risk_df.columns:
        hover.pop("vuln_rating", None)

    fig = px.choropleth_map(
        risk_df,
        geojson=geojson,
        locations="area_code",
        featureidkey="properties.LAD13CD",
        color=color_col,
        color_continuous_scale="YlOrRd",
        range_color=[0, risk_df[color_col].max() if risk_df[color_col].max() > 0 else 1],
        hover_name="area_name",
        hover_data=hover,
        map_style="carto-positron",
        center={"lat": 51.5, "lon": -0.1},
        zoom=9,
        opacity=0.7,
    )
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=550,
        coloraxis_colorbar_title=title,
    )
    return fig


def render_age_risk_section(selected, risk_df, age_data, age_latest):
    st.markdown(f"**Age Risk Profile — {selected}**")

    row = risk_df[risk_df["area_name"] == selected].iloc[0]
    area_code = row["area_code"]

    col_child, col_elderly = st.columns(2)

    with col_child:
        st.markdown("##### Children (0–9 yrs)")
        asthma_val = row.get("asthma_0_9_rate")
        child_risk = row.get("childhood_risk")
        if pd.notna(asthma_val):
            st.metric("Asthma Admissions (per 100k)", f"{asthma_val:.1f}")
            st.metric(
                "Childhood Risk Score",
                f"{child_risk:.3f} ({_score_label(child_risk)})",
                delta=_borough_rank(child_risk, risk_df["childhood_risk"]),
                delta_color="off",
                help=FORMULA_CHILD_RISK,
            )
        else:
            st.info("No childhood asthma data available")

    with col_elderly:
        st.markdown("##### Elderly (85+ yrs)")
        wm_85 = row.get("winter_mort_85_index")
        elderly_risk = row.get("elderly_risk")
        if pd.notna(wm_85):
            st.metric("Winter Mortality Index (85+)", f"{wm_85:.1f}")
            st.metric(
                "Elderly Risk Score",
                f"{elderly_risk:.3f} ({_score_label(elderly_risk)})",
                delta=_borough_rank(elderly_risk, risk_df["elderly_risk"]),
                delta_color="off",
                help=FORMULA_ELDERLY_RISK,
            )
        else:
            st.info("No elderly mortality data available")

    both_high = (
        pd.notna(child_risk) and child_risk >= 0.75
        and pd.notna(elderly_risk) and elderly_risk >= 0.75
    )
    if both_high:
        st.error("⚠️ **NHS Priority Borough** — both childhood and elderly risk scores are in the top quartile")
    elif pd.notna(child_risk) and child_risk >= 0.75:
        st.warning("High childhood respiratory risk — prioritise paediatric preparedness")
    elif pd.notna(elderly_risk) and elderly_risk >= 0.75:
        st.warning("High elderly mortality risk — prioritise geriatric and care home preparedness")

    if age_data is not None and not age_data.empty:
        borough_age = age_data[age_data["area_code"] == area_code]

        asthma_ts = borough_age[
            (borough_age["indicator_label"] == "asthma_admissions")
            & (borough_age["age_group"] == "0-9 yrs")
        ].sort_values("time_period_sortable")

        wm_ts = borough_age[
            borough_age["indicator_label"] == "winter_mortality_85plus"
        ].sort_values("time_period_sortable")

        if not asthma_ts.empty or not wm_ts.empty:
            col_asthma_trend, col_wm_trend = st.columns(2)

            with col_asthma_trend:
                if not asthma_ts.empty:
                    st.markdown("**Childhood Asthma Trend (0–9)**")
                    chart = asthma_ts[["time_period", "value"]].rename(
                        columns={"time_period": "Year", "value": "Rate per 100k"}
                    ).dropna(subset=["Rate per 100k"])
                    if not chart.empty:
                        st.line_chart(chart.set_index("Year"))

            with col_wm_trend:
                if not wm_ts.empty:
                    st.markdown("**Winter Mortality Trend (85+)**")
                    chart = wm_ts[["time_period", "value"]].rename(
                        columns={"time_period": "Winter", "value": "Mortality Index"}
                    ).dropna(subset=["Mortality Index"])
                    if not chart.empty:
                        st.line_chart(chart.set_index("Winter"))


def render_housing_section(housing_df, housing_ward_df, ward_geojson, housing_postcode_df):
    st.subheader("Housing Quality & Cold Vulnerability")

    if housing_df is None or housing_df.empty:
        st.warning("Housing data not yet available. Run `python -m src.data.fetch_housing` to fetch LBSM data.")
        return

    st.caption(FORMULA_HOUSING)

    col_chart, col_stats = st.columns([3, 2])

    with col_chart:
        sorted_housing = housing_df.sort_values("housing_quality_score", ascending=True)
        fig = px.bar(
            sorted_housing,
            x="housing_quality_score",
            y="administrative_area",
            orientation="h",
            color="housing_quality_score",
            color_continuous_scale="YlOrRd",
            labels={"housing_quality_score": "Housing Quality Score", "administrative_area": "Borough"},
        )
        fig.update_layout(
            height=700, margin={"t": 10, "l": 0, "r": 0, "b": 0},
            coloraxis_colorbar_title="Score",
            yaxis=dict(dtick=1),
        )
        st.plotly_chart(fig, config={"responsive": True})

    with col_stats:
        st.markdown("**London-wide Housing Statistics**")
        avg = housing_df[["pct_epc_poor", "pct_uninsulated_walls", "pct_single_glazing", "pct_no_central_heating"]].mean()
        st.metric("Avg % Poor EPC (E/F-G)", f"{avg['pct_epc_poor']:.1f}%")
        st.metric("Avg % Uninsulated Walls", f"{avg['pct_uninsulated_walls']:.1f}%")
        st.metric("Avg % Single Glazing", f"{avg['pct_single_glazing']:.1f}%")
        st.metric("Avg % No Central Heating", f"{avg['pct_no_central_heating']:.1f}%")

        worst = housing_df.nlargest(5, "housing_quality_score")[["administrative_area", "housing_quality_score"]]
        worst.columns = ["Borough", "Score"]
        worst["Score"] = worst["Score"].round(3)
        worst["Rating"] = worst["Score"].apply(lambda s: _score_label(s, ("Good", "Fair", "Poor", "Very Poor")))
        st.markdown("**Most vulnerable housing stock**")
        st.dataframe(worst, hide_index=True)

    if housing_ward_df is not None and ward_geojson is not None and not housing_ward_df.empty:
        st.markdown("---")
        st.markdown("**Ward-Level Drill-Down**")
        ward_boroughs = sorted(housing_ward_df["administrative_area"].dropna().unique())
        selected_borough = st.selectbox("Select borough for ward breakdown", ward_boroughs, key="ward_borough")

        if selected_borough:
            ward_data = housing_ward_df[housing_ward_df["administrative_area"] == selected_borough].copy()
            col_ward_map, col_ward_table = st.columns([3, 2])

            with col_ward_map:
                ward_code_key = None
                if ward_geojson and ward_geojson.get("features"):
                    sample_props = ward_geojson["features"][0].get("properties", {})
                    for key in sample_props:
                        if key.upper().startswith("WD") and key.upper().endswith("CD"):
                            ward_code_key = key
                            break

                if ward_code_key and "ward22nm" in ward_data.columns:
                    geo_ward_name_key = ward_code_key.replace("CD", "NM")
                    lad_code_key = ward_code_key.replace("WD", "LAD")

                    geo_name_to_code = {}
                    for feat in ward_geojson["features"]:
                        p = feat.get("properties", {})
                        lad_nm = p.get(lad_code_key.replace("CD", "NM"), "")
                        if lad_nm == selected_borough:
                            geo_name_to_code[p.get(geo_ward_name_key, "")] = p.get(ward_code_key, "")

                    ward_data_map = ward_data.copy()
                    ward_data_map["geo_ward_code"] = ward_data_map["ward22nm"].map(geo_name_to_code)
                    ward_data_map = ward_data_map.dropna(subset=["geo_ward_code"])

                    if not ward_data_map.empty:
                        fig_ward = px.choropleth_map(
                            ward_data_map,
                            geojson=ward_geojson,
                            locations="geo_ward_code",
                            featureidkey=f"properties.{ward_code_key}",
                            color="housing_quality_score",
                            color_continuous_scale="YlOrRd",
                            hover_name="ward22nm",
                            hover_data={"housing_quality_score": ":.3f", "geo_ward_code": False},
                            map_style="carto-positron",
                            center={"lat": 51.5, "lon": -0.1},
                            zoom=11,
                            opacity=0.7,
                        )
                        fig_ward.update_layout(
                            margin={"r": 0, "t": 0, "l": 0, "b": 0},
                            height=400,
                            coloraxis_colorbar_title="Score",
                        )
                        st.plotly_chart(fig_ward, config={"responsive": True})
                    else:
                        st.info("Could not match ward names to boundaries.")
                else:
                    st.info("Ward boundary data not available for map display.")

            with col_ward_table:
                display_ward = ward_data[["ward22nm", "housing_quality_score",
                                          "pct_epc_poor", "pct_uninsulated_walls"]].copy()
                display_ward.columns = ["Ward", "Score", "% Poor EPC", "% Uninsulated"]
                display_ward = display_ward.sort_values("Score", ascending=False)
                display_ward["Score"] = display_ward["Score"].round(3)
                display_ward["% Poor EPC"] = display_ward["% Poor EPC"].round(1)
                display_ward["% Uninsulated"] = display_ward["% Uninsulated"].round(1)
                st.dataframe(display_ward, hide_index=True, height=400)

            if housing_postcode_df is not None and not housing_postcode_df.empty:
                st.markdown("**Postcode-Sector Drill-Down**")
                pc_data = housing_postcode_df[
                    housing_postcode_df["administrative_area"] == selected_borough
                ].copy()
                pc_data = pc_data.dropna(subset=["lat", "lon"])

                if not pc_data.empty:
                    fig_pc = px.scatter_map(
                        pc_data,
                        lat="lat",
                        lon="lon",
                        color="housing_quality_score",
                        size="total_properties",
                        color_continuous_scale="YlOrRd",
                        hover_name="postcode_sector",
                        hover_data={
                            "housing_quality_score": ":.3f",
                            "total_properties": True,
                            "pct_epc_poor": ":.1f",
                            "pct_uninsulated_walls": ":.1f",
                            "lat": False,
                            "lon": False,
                        },
                        map_style="carto-positron",
                        center={"lat": pc_data["lat"].mean(), "lon": pc_data["lon"].mean()},
                        zoom=12,
                        opacity=0.8,
                        size_max=20,
                    )
                    fig_pc.update_layout(
                        margin={"r": 0, "t": 0, "l": 0, "b": 0},
                        height=400,
                        coloraxis_colorbar_title="Housing Score",
                    )
                    st.plotly_chart(fig_pc, config={"responsive": True})
                    st.caption(
                        f"{len(pc_data)} postcode sectors in {selected_borough} — "
                        f"bubble size = number of properties, colour = housing vulnerability score"
                    )


def main():
    ensure_data()

    st.title("RespiCast London")
    st.caption("Borough-level respiratory risk warnings for NHS planners")

    alert = get_london_cold_alert()
    render_alert_banner(alert)
    severity = alert_severity(alert["status"])

    # --- Forecast section ---
    try:
        prediction_df, analogues = load_forecast()
        render_forecast_section(prediction_df, analogues)
        peak_multiplier = prediction_df["cold_multiplier"].max()
    except Exception as e:
        st.warning(f"Forecast data unavailable: {e}")
        prediction_df, analogues = None, None
        peak_multiplier = 0

    st.divider()

    # --- Vulnerability map + risk table ---
    (vuln, resp, copd_ts, geojson, housing, housing_ward,
     ward_geojson, housing_postcode, age_data, age_latest) = load_data()
    scored = compute_vulnerability_scores(vuln, resp, housing_df=housing)

    if age_latest is not None and not age_latest.empty:
        scored = compute_age_risk_scores(scored, age_latest)

    risk_df = build_risk_table(scored, severity)

    if prediction_df is not None and peak_multiplier > 0:
        risk_df["predicted_risk_score"] = risk_df["vulnerability_score"] * peak_multiplier

    has_housing = housing is not None and "housing_quality_score" in risk_df.columns and risk_df["housing_quality_score"].notna().any()
    has_age = "childhood_risk" in risk_df.columns and risk_df["childhood_risk"].notna().any()

    risk_df["vuln_rating"] = risk_df["vulnerability_score"].apply(_score_label)

    # --- Methodology expander ---
    with st.expander("How scores are computed"):
        vuln_formula = FORMULA_VULN if has_housing else FORMULA_VULN_NO_HOUSING
        st.markdown(f"""
**Vulnerability Score** (0–1 scale)
{vuln_formula}
Scale: Low (< 0.25) · Moderate (0.25–0.50) · High (0.50–0.75) · Very High (> 0.75)

**Risk Level**
{FORMULA_RISK_LEVEL}

**Housing Quality Score** (0–1 scale)
{FORMULA_HOUSING}
Scale: Good (< 0.25) · Fair (0.25–0.50) · Poor (0.50–0.75) · Very Poor (> 0.75)

**Childhood Risk** (0–1 scale)
{FORMULA_CHILD_RISK}

**Elderly Risk** (0–1 scale)
{FORMULA_ELDERLY_RISK}

**Daily Cold Risk**
{FORMULA_COLD_MULT}

**Predicted Respiratory Risk** (0–3 scale)
{FORMULA_PREDICTED_RISK}
        """)

    if severity == 0 and peak_multiplier == 0:
        st.info(
            "Cold alert is currently **Green** and forecast shows no cold risk. "
            "The map shows baseline vulnerability. During cold periods, risk scores "
            "will reflect the combined weather + vulnerability signal."
        )

    map_mode = "Current Vulnerability"
    if prediction_df is not None and peak_multiplier > 0:
        map_mode = st.radio(
            "Map view",
            ["Current Vulnerability", "Predicted Risk (peak day)"],
            horizontal=True,
        )

    col_map, col_table = st.columns([3, 2])

    with col_map:
        st.subheader("Borough Vulnerability Map")
        if map_mode == "Predicted Risk (peak day)" and "predicted_risk_score" in risk_df.columns:
            fig = render_map(risk_df, geojson, color_col="predicted_risk_score", title="Predicted Risk")
        else:
            fig = render_map(risk_df, geojson)
        st.plotly_chart(fig, config={"responsive": True})

    with col_table:
        st.subheader("Risk Rankings")
        st.caption(FORMULA_VULN if has_housing else FORMULA_VULN_NO_HOUSING)

        display_cols = ["area_name", "vulnerability_score", "risk_level",
                        "imd_score", "fuel_poverty_pct", "copd_rate"]
        col_names = ["Borough", "Vulnerability", "Risk Level",
                     "IMD Score", "Fuel Poverty %", "COPD Rate"]

        if has_housing:
            display_cols.append("housing_quality_score")
            col_names.append("Housing Score")

        if has_age:
            display_cols.extend(["childhood_risk", "elderly_risk"])
            col_names.extend(["Child Risk", "Elderly Risk"])

        display_df = risk_df[display_cols].copy()
        display_df.columns = col_names
        display_df["Vulnerability"] = display_df["Vulnerability"].round(3)
        display_df.insert(2, "Rating", display_df["Vulnerability"].apply(_score_label))
        display_df.insert(3, "Rank", range(1, len(display_df) + 1))
        if has_age:
            display_df["Child Risk"] = display_df["Child Risk"].round(3)
            display_df["Elderly Risk"] = display_df["Elderly Risk"].round(3)
        st.dataframe(display_df, width="stretch", height=500, hide_index=True)

    st.divider()

    # --- Borough deep dive ---
    st.subheader("Borough Deep Dive")
    boroughs = sorted(risk_df["area_name"].tolist())
    selected = st.selectbox("Select a borough", boroughs)

    if selected:
        row = risk_df[risk_df["area_name"] == selected].iloc[0]
        borough_ts = copd_ts[copd_ts["Area Name"] == selected].sort_values("Time period Sortable")

        col_trend, col_breakdown = st.columns(2)

        with col_trend:
            st.markdown(f"**COPD Emergency Admissions Trend — {selected}**")
            if not borough_ts.empty:
                chart_data = borough_ts[["Time period", "Value"]].copy()
                chart_data = chart_data.rename(columns={"Time period": "Year", "Value": "Rate per 100k"})
                chart_data = chart_data.dropna(subset=["Rate per 100k"])
                st.line_chart(chart_data.set_index("Year"))
            else:
                st.write("No trend data available")

        with col_breakdown:
            st.markdown(f"**Vulnerability Components — {selected}**")
            st.caption("Each component normalised 0–1 across boroughs, then weighted. Bar height = score × weight.")
            components = [
                {"Component": "IMD Deprivation", "Score": row["imd_norm"], "Weight": 0.30 if has_housing else 0.40},
                {"Component": "Fuel Poverty", "Score": row["fp_norm"], "Weight": 0.25 if has_housing else 0.30},
                {"Component": "COPD Admissions", "Score": row["copd_norm"], "Weight": 0.25 if has_housing else 0.30},
            ]
            if has_housing and pd.notna(row.get("housing_norm")):
                components.append({"Component": "Housing Quality", "Score": row["housing_norm"], "Weight": 0.20})

            comp_df = pd.DataFrame(components)
            comp_df["Weighted"] = comp_df["Score"] * comp_df["Weight"]

            colours = ["#ef5350", "#ff9800", "#42a5f5", "#66bb6a"][:len(comp_df)]
            fig_bar = px.bar(
                comp_df, x="Component", y="Weighted", color="Component",
                color_discrete_sequence=colours, text="Weighted",
            )
            fig_bar.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_bar.update_layout(
                showlegend=False, yaxis_title="Weighted Score",
                yaxis_range=[0, 0.5], height=350, margin={"t": 10},
            )
            st.plotly_chart(fig_bar, config={"responsive": True})

        vuln_score = row["vulnerability_score"]
        n_metrics = 5 if has_housing else 4
        if has_age:
            n_metrics += 2
        metric_cols = st.columns(n_metrics)
        metric_cols[0].metric(
            "Vulnerability Score",
            f"{vuln_score:.3f} ({_score_label(vuln_score)})",
            delta=_borough_rank(vuln_score, risk_df["vulnerability_score"]),
            delta_color="off",
            help=FORMULA_VULN if has_housing else FORMULA_VULN_NO_HOUSING,
        )
        metric_cols[1].metric("IMD Score", f"{row['imd_score']:.1f}",
                              help="Index of Multiple Deprivation — higher = more deprived. Source: MHCLG IoD2025.")
        metric_cols[2].metric("Fuel Poverty", f"{row['fuel_poverty_pct']:.1f}%",
                              help="% of households in fuel poverty. Source: OHID Fingertips.")
        copd_val = f"{row['copd_rate']:.1f}" if pd.notna(row['copd_rate']) else "N/A"
        metric_cols[3].metric("COPD Rate (per 100k)", copd_val,
                              help="COPD emergency admissions per 100,000 population. Source: OHID Fingertips.")
        idx = 4
        if has_housing and pd.notna(row.get("housing_quality_score")):
            h_score = row["housing_quality_score"]
            metric_cols[idx].metric(
                "Housing Quality",
                f"{h_score:.3f} ({_score_label(h_score, ('Good', 'Fair', 'Poor', 'Very Poor'))})",
                delta=_borough_rank(h_score, risk_df["housing_quality_score"]),
                delta_color="off",
                help=FORMULA_HOUSING,
            )
            idx += 1
        if has_age:
            child_risk = row.get("childhood_risk")
            elderly_risk = row.get("elderly_risk")
            if pd.notna(child_risk):
                metric_cols[idx].metric(
                    "Child Risk",
                    f"{child_risk:.3f} ({_score_label(child_risk)})",
                    delta=_borough_rank(child_risk, risk_df["childhood_risk"]),
                    delta_color="off",
                    help=FORMULA_CHILD_RISK,
                )
            else:
                metric_cols[idx].metric("Child Risk", "N/A")
            if pd.notna(elderly_risk):
                metric_cols[idx + 1].metric(
                    "Elderly Risk",
                    f"{elderly_risk:.3f} ({_score_label(elderly_risk)})",
                    delta=_borough_rank(elderly_risk, risk_df["elderly_risk"]),
                    delta_color="off",
                    help=FORMULA_ELDERLY_RISK,
                )
            else:
                metric_cols[idx + 1].metric("Elderly Risk", "N/A")

        if has_age:
            st.markdown("---")
            render_age_risk_section(selected, risk_df, age_data, age_latest)

    st.divider()

    # --- Housing section ---
    render_housing_section(housing, housing_ward, ward_geojson, housing_postcode)

    st.divider()
    st.caption(
        "Data sources: OHID Fingertips (respiratory admissions, winter mortality), MHCLG IoD2025 (deprivation), "
        "Fingertips (fuel poverty), UKHSA (cold alerts), Open-Meteo (weather forecast & reanalysis), "
        "GLA London Building Stock Model v2 (housing). "
        "Built for the Health in Climate London 2026 Hackathon."
    )


if __name__ == "__main__":
    main()
