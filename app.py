import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.data.fetch_cold_alert import get_london_cold_alert, alert_severity
from src.model.vulnerability import compute_vulnerability_scores, build_risk_table

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

ALERT_COLOURS = {
    "Green": "#2e7d32",
    "Yellow": "#f9a825",
    "Amber": "#ef6c00",
    "Red": "#c62828",
    "Unknown": "#757575",
}

RISK_COLOURS = {"Green": "#2e7d32", "Yellow": "#f9a825", "Amber": "#ef6c00", "Red": "#c62828"}

st.set_page_config(page_title="RespiCast London", layout="wide", page_icon="🫁")


def ensure_data():
    needed = [
        os.path.join(DATA_DIR, "respiratory_latest.csv"),
        os.path.join(DATA_DIR, "respiratory_copd_borough.csv"),
        os.path.join(DATA_DIR, "vulnerability_inputs.csv"),
        os.path.join(DATA_DIR, "london_boroughs.geojson"),
    ]
    if all(os.path.exists(f) for f in needed):
        return

    with st.spinner("Fetching data for the first time..."):
        from src.data.fetch_respiratory import fetch_and_save as fetch_resp
        from src.data.fetch_vulnerability import fetch_and_save as fetch_vuln
        from src.data.fetch_boundaries import fetch_london_geojson

        if not os.path.exists(needed[0]):
            fetch_resp()
        if not os.path.exists(needed[2]):
            fetch_vuln()
        if not os.path.exists(needed[3]):
            fetch_london_geojson()


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

    return vuln, resp, copd_ts, geojson, housing, housing_ward, ward_geojson


@st.cache_data(ttl=1800)
def load_forecast():
    from src.data.fetch_weather_forecast import fetch_7day_forecast, fetch_historical_winters
    from src.model.prediction import predict_respiratory_risk, find_historical_analogues

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

    if analogues:
        with st.expander("Historical Analogues — similar past weather patterns"):
            for j, a in enumerate(analogues, 1):
                st.markdown(
                    f"**Match {j}** ({a['start_date']} to {a['end_date']}) — "
                    f"Distance: {a['distance']:.1f} | "
                    f"Following week: min {a['following_min_temp']:.1f}°C, "
                    f"avg {a['following_avg_temp']:.1f}°C"
                )


def render_map(risk_df, geojson, color_col="vulnerability_score", title="Vulnerability"):
    fig = px.choropleth_map(
        risk_df,
        geojson=geojson,
        locations="area_code",
        featureidkey="properties.LAD13CD",
        color=color_col,
        color_continuous_scale="YlOrRd",
        range_color=[0, risk_df[color_col].max() if risk_df[color_col].max() > 0 else 1],
        hover_name="area_name",
        hover_data={
            "vulnerability_score": ":.2f",
            "risk_level": True,
            "area_code": False,
            color_col: ":.2f",
        },
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


def render_housing_section(housing_df, housing_ward_df, ward_geojson):
    st.subheader("Housing Quality & Cold Vulnerability")

    if housing_df is None or housing_df.empty:
        st.warning("Housing data not yet available. Run `python -m src.data.fetch_housing` to fetch LBSM data.")
        return

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
                    borough_code = None
                    for feat in ward_geojson["features"]:
                        p = feat.get("properties", {})
                        if p.get(lad_code_key, "").startswith("E09"):
                            borough_code = p.get(lad_code_key)
                            lad_name_key = lad_code_key.replace("CD", "NM")
                            if p.get(lad_name_key) == selected_borough:
                                break
                            borough_code = None

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
    vuln, resp, copd_ts, geojson, housing, housing_ward, ward_geojson = load_data()
    scored = compute_vulnerability_scores(vuln, resp, housing_df=housing)
    risk_df = build_risk_table(scored, severity)

    if prediction_df is not None and peak_multiplier > 0:
        risk_df["predicted_risk_score"] = risk_df["vulnerability_score"] * peak_multiplier

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
        display_cols = ["area_name", "vulnerability_score", "risk_level",
                        "imd_score", "fuel_poverty_pct", "copd_rate"]
        col_names = ["Borough", "Vulnerability", "Risk Level",
                     "IMD Score", "Fuel Poverty %", "COPD Rate"]

        has_housing = housing is not None and "housing_quality_score" in risk_df.columns
        if has_housing and risk_df["housing_quality_score"].notna().any():
            display_cols.append("housing_quality_score")
            col_names.append("Housing Score")

        display_df = risk_df[display_cols].copy()
        display_df.columns = col_names
        display_df["Vulnerability"] = display_df["Vulnerability"].round(3)
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

        n_metrics = 5 if has_housing else 4
        metric_cols = st.columns(n_metrics)
        metric_cols[0].metric("Vulnerability Score", f"{row['vulnerability_score']:.3f}")
        metric_cols[1].metric("IMD Score", f"{row['imd_score']:.1f}")
        metric_cols[2].metric("Fuel Poverty", f"{row['fuel_poverty_pct']:.1f}%")
        copd_val = f"{row['copd_rate']:.1f}" if pd.notna(row['copd_rate']) else "N/A"
        metric_cols[3].metric("COPD Rate (per 100k)", copd_val)
        if has_housing and pd.notna(row.get("housing_quality_score")):
            metric_cols[4].metric("Housing Quality", f"{row['housing_quality_score']:.3f}")

    st.divider()

    # --- Housing section ---
    render_housing_section(housing, housing_ward, ward_geojson)

    st.divider()
    st.caption(
        "Data sources: OHID Fingertips (respiratory admissions), MHCLG IoD2025 (deprivation), "
        "Fingertips (fuel poverty), UKHSA (cold alerts), Open-Meteo (weather forecast & reanalysis), "
        "GLA London Building Stock Model v2 (housing). "
        "Built for the Health in Climate London 2026 Hackathon."
    )


if __name__ == "__main__":
    main()
