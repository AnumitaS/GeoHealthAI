import json
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pydeck as pdk
import streamlit as st

from modules.facility_recommender import predict_facility_locations

st.set_page_config(
    page_title="GeoHealth-AI Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏥 GeoHealth-AI: Dynamic Decision Support System")
st.markdown(
    "Spatial boundary, highway/railway networks, and population density analytics "
    "for dynamic health facility location gap recommendations."
)

PROJECT_ROOT = Path(__file__).resolve().parent
GENERATED_DATA_DIR = PROJECT_ROOT / "generated_data"
CONSTANT_DATA_DIR = PROJECT_ROOT / "constant_data"


def get_processed_states():
    if not GENERATED_DATA_DIR.exists():
        return []
    dirs = [d.name for d in GENERATED_DATA_DIR.iterdir() if d.is_dir()]
    return sorted([d.replace("_", " ").title() for d in dirs])


available_states = get_processed_states()

if not available_states:
    st.error("⚠️ No state data found inside `generated_data/`! Run `python main.py` first.")
    st.stop()


# ---------------------------------------------------------
# CACHED LOADERS
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_official_soi_border():
    soi_path = CONSTANT_DATA_DIR / "india-soi.geojson"
    if soi_path.exists():
        try:
            return gpd.read_file(soi_path).to_crs("EPSG:4326")
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def load_state_population(state_name: str) -> int:
    clean_name = state_name.strip().lower().replace(" ", "_")
    pop_file = GENERATED_DATA_DIR / clean_name / "state_populations.json"
    if pop_file.exists():
        with open(pop_file, "r") as f:
            data = json.load(f)
            return data.get(state_name, 0)
    return 0


@st.cache_data(show_spinner="Loading state layers...")
def load_state_layers(state_name: str):
    clean_name = state_name.strip().lower().replace(" ", "_")
    state_dir = GENERATED_DATA_DIR / clean_name

    layers = {}
    layer_names = ["state_border", "districts", "highways", "railways", "villages", "health_facilities"]

    for name in layer_names:
        file_path = state_dir / f"{name}.parquet"
        if file_path.exists():
            layers[name] = gpd.read_parquet(file_path)
        else:
            layers[name] = None

    return layers


soi_border_gdf = load_official_soi_border()

# Sidebar Options
st.sidebar.header("📍 Region Selection")
default_idx = available_states.index("West Bengal") if "West Bengal" in available_states else 0
selected_state = st.sidebar.selectbox("Select Target State:", available_states, index=default_idx)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Dynamic AI Gap Parameters")
coverage_radius_km = st.sidebar.slider("Service Coverage Radius (km)", 5.0, 30.0, 12.0, 1.0)
cluster_eps_km = st.sidebar.slider("Clustering Epsilon (km)", 5.0, 30.0, 15.0, 1.0)
min_villages = st.sidebar.slider("Min Villages Per Cluster", 2, 20, 5, 1)

st.sidebar.markdown("---")
st.sidebar.header("🗺️ Map Views & Controls")
show_png_map = st.sidebar.checkbox("Show Static Map PNG Image", value=True)
show_soi_outer = st.sidebar.checkbox("Official India SOI Border", value=True)
show_state_border = st.sidebar.checkbox("State Boundary", value=True)
show_districts = st.sidebar.checkbox("District Boundaries", value=True)
show_pop_heatmap = st.sidebar.checkbox("Population Density Heatmap", value=True)
show_highways = st.sidebar.checkbox("National Highways", value=True)
show_railways = st.sidebar.checkbox("Railway Tracks", value=True)
show_villages = st.sidebar.checkbox("Census Village Centroids", value=False)
show_existing_health = st.sidebar.checkbox("Active Health Facilities (Excl. SCE)", value=True)
show_buffers = st.sidebar.checkbox("Service Coverage Buffers", value=True)
show_ai_sites = st.sidebar.checkbox("Dynamic AI Proposed Facilities", value=True)

clean_state_name = selected_state.strip().lower().replace(" ", "_")
state_dir = GENERATED_DATA_DIR / clean_state_name
png_map_path = state_dir / f"{clean_state_name}_map.png"

state_layers = load_state_layers(selected_state)
total_pop_2011 = load_state_population(selected_state)
projected_pop_2026 = total_pop_2011 * 1.17

if state_layers.get("state_border") is None:
    st.warning(f"⚠️ Layer state_border.parquet is missing in `/generated_data/{clean_state_name}/`.")
    st.stop()

health_gdf = state_layers.get("health_facilities")

ai_recs_gdf = predict_facility_locations(
    villages_gdf=state_layers.get("villages"),
    health_gdf=health_gdf,
    state_border_gdf=state_layers.get("state_border"),
    coverage_radius_km=coverage_radius_km,
    cluster_eps_km=cluster_eps_km,
    min_villages_per_cluster=min_villages,
)

if show_png_map:
    if png_map_path.exists():
        st.markdown(f"### 🖼️ High-Resolution Static Map (`{clean_state_name}_map.png`)")
        st.image(str(png_map_path), caption=f"Healthcare Coverage Overview - {selected_state}", use_container_width=True)

st.markdown("### 🗺️ Dynamic 3D Spatial Visualization")
target_state_gdf = state_layers.get("state_border")
bounds = target_state_gdf.total_bounds
center_lat = (bounds[1] + bounds[3]) / 2.0
center_lon = (bounds[0] + bounds[2]) / 2.0

deck_layers = []

# Master Survey of India National Outer Border Layer
if show_soi_outer and soi_border_gdf is not None:
    deck_layers.append(
        pdk.Layer(
            "GeoJsonLayer",
            data=json.loads(soi_border_gdf.to_json()),
            stroked=True,
            filled=False,
            get_line_color="[225, 29, 72, 255]",
            get_line_width=2500,
        )
    )

if show_state_border and target_state_gdf is not None:
    deck_layers.append(
        pdk.Layer(
            "GeoJsonLayer",
            data=json.loads(target_state_gdf.to_json()),
            stroked=True,
            filled=False,
            get_line_color="[15, 23, 42, 255]",
            get_line_width=1500,
        )
    )

if show_districts and state_layers.get("districts") is not None:
    deck_layers.append(
        pdk.Layer(
            "GeoJsonLayer",
            data=json.loads(state_layers["districts"].to_json()),
            stroked=True,
            filled=False,
            get_line_color="[120, 113, 108, 180]",
            get_line_width=800,
        )
    )

if show_pop_heatmap and state_layers.get("villages") is not None:
    vil_pop_df = state_layers["villages"].copy()
    vil_pop_df["Longitude"] = vil_pop_df.geometry.x
    vil_pop_df["Latitude"] = vil_pop_df.geometry.y
    if "population" not in vil_pop_df.columns:
        vil_pop_df["population"] = 500

    deck_layers.append(
        pdk.Layer(
            "HeatmapLayer",
            data=vil_pop_df,
            get_position=["Longitude", "Latitude"],
            get_weight="population",
            radius_pixels=60,
            intensity=1.5,
            threshold=0.05,
            opacity=0.6,
        )
    )

if show_highways and state_layers.get("highways") is not None:
    deck_layers.append(
        pdk.Layer(
            "GeoJsonLayer",
            data=json.loads(state_layers["highways"].to_json()),
            stroked=True,
            filled=False,
            get_line_color="[185, 28, 28, 220]",
            get_line_width=400,
        )
    )

if show_railways and state_layers.get("railways") is not None:
    deck_layers.append(
        pdk.Layer(
            "GeoJsonLayer",
            data=json.loads(state_layers["railways"].to_json()),
            stroked=True,
            filled=False,
            get_line_color="[51, 65, 85, 200]",
            get_line_width=300,
        )
    )

if show_villages and state_layers.get("villages") is not None:
    vil_gdf = state_layers["villages"].copy()
    vil_gdf["Longitude"] = vil_gdf.geometry.x
    vil_gdf["Latitude"] = vil_gdf.geometry.y
    deck_layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=vil_gdf,
            get_position=["Longitude", "Latitude"],
            get_fill_color="[2, 132, 199, 120]",
            get_radius=300,
            pickable=True,
        )
    )

health_df = pd.DataFrame()
if health_gdf is not None and not health_gdf.empty:
    health_df = pd.DataFrame({"Longitude": health_gdf.geometry.x, "Latitude": health_gdf.geometry.y})

if show_existing_health and not health_df.empty:
    deck_layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=health_df,
            get_position=["Longitude", "Latitude"],
            get_fill_color="[21, 128, 61, 220]",
            get_radius=800,
            pickable=True,
        )
    )

if show_buffers and not health_df.empty:
    deck_layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=health_df,
            get_position=["Longitude", "Latitude"],
            get_fill_color="[234, 179, 8, 30]",
            get_line_color="[21, 128, 61, 150]",
            get_line_width=20,
            get_radius=coverage_radius_km * 1000,
            stroked=True,
        )
    )

ai_sites_df = pd.DataFrame()
if ai_recs_gdf is not None and not ai_recs_gdf.empty:
    ai_sites_df = pd.DataFrame({
        "ID": ai_recs_gdf.get("recommendation_id", [f"AI_Site_{i+1}" for i in range(len(ai_recs_gdf))]),
        "Longitude": ai_recs_gdf.geometry.x,
        "Latitude": ai_recs_gdf.geometry.y,
    })

if show_ai_sites and not ai_sites_df.empty:
    deck_layers.append(
        pdk.Layer(
            "ColumnLayer",
            data=ai_sites_df,
            get_position=["Longitude", "Latitude"],
            get_elevation=18000,
            elevation_scale=1,
            radius=2500,
            get_fill_color="[217, 119, 6, 240]",
            pickable=True,
        )
    )

st.pydeck_chart(
    pdk.Deck(
        layers=deck_layers,
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=6.8, pitch=35),
        map_style=pdk.map_styles.CARTO_LIGHT,
    )
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("State Selected", selected_state)
with col2:
    st.metric("Projected Population", f"{projected_pop_2026/1e6:.2f} M")
with col3:
    st.metric("Active Patient Facilities", len(health_df))
with col4:
    st.metric("AI Proposed Sites", len(ai_sites_df))

if not ai_sites_df.empty:
    st.markdown(f"### 🎯 Recommended Locations ({len(ai_sites_df)} Proposed Sites)")
    st.dataframe(ai_sites_df[["ID", "Latitude", "Longitude"]], use_container_width=True)