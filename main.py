import json
import os
import re
from pathlib import Path
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

from modules.facility_recommender import predict_facility_locations

PROJECT_ROOT = Path(__file__).resolve().parent
CONSTANT_DATA_DIR = PROJECT_ROOT / "constant_data"
GENERATED_DATA_DIR = PROJECT_ROOT / "generated_data"
GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Dataset paths
OFFICIAL_SOI_BORDER = os.path.join(CONSTANT_DATA_DIR, "india-soi.geojson")
STATE_BORDER_FILE = os.path.join(CONSTANT_DATA_DIR, "StateBoundary.parquet")

INDIA_DISTRICT_BORDER = os.path.join(CONSTANT_DATA_DIR, "2011_Dist.parquet")
VILLEGE_DATA = os.path.join(CONSTANT_DATA_DIR, "Census_villages.parquet")
RAILWAY_DATA = os.path.join(CONSTANT_DATA_DIR, "GatiShakti_Railway_tracks.parquet")
HIGHWAY_DATA = os.path.join(CONSTANT_DATA_DIR, "GatiShakti_MORTH_National_Highways.parquet")
INDIA_HEALTH_FACILITY_DATA = os.path.join(CONSTANT_DATA_DIR, "INDIA_HEALTH_FACILITIES_NIC.geojson")
POPULATION_DATA = os.path.join(CONSTANT_DATA_DIR, "2011-IndiaStateDistSbDistVill-0000.xlsx")


def load_dataset(filepath: str):
    """Loads geospatial and tabular files safely."""
    if not os.path.exists(filepath):
        print(f"⚠️ File not found: {filepath}")
        return None

    print(f"📦 Loading {Path(filepath).name}...")
    try:
        if filepath.endswith((".geojson", ".json")):
            return gpd.read_file(filepath)
        elif filepath.endswith(".parquet"):
            return gpd.read_parquet(filepath)
        elif filepath.endswith((".xlsx", ".xls")):
            df = pd.read_excel(filepath)
            df.columns = [str(c).strip().lower() for c in df.columns]
            return df
    except Exception as e:
        print(f"❌ Failed to load {filepath}: {e}")
        return None


def get_official_state_boundaries():
    """
    Loads state boundaries from StateBoundary.parquet and masks them using
    india-soi.geojson to enforce official Survey of India external borders.
    """
    states_gdf = load_dataset(STATE_BORDER_FILE)
    if states_gdf is None:
        raise FileNotFoundError("`StateBoundary.parquet` is missing from constant_data/.")

    soi_gdf = load_dataset(OFFICIAL_SOI_BORDER)
    states_gdf = states_gdf.to_crs("EPSG:4326")

    if soi_gdf is not None and not soi_gdf.empty:
        print("🇮🇳 Applying Survey of India mask (india-soi.geojson) to state boundaries...")
        soi_gdf = soi_gdf.to_crs("EPSG:4326")
        
        # Buffer 0 repairs self-intersecting geometries
        soi_gdf["geometry"] = soi_gdf["geometry"].buffer(0)
        states_gdf["geometry"] = states_gdf["geometry"].buffer(0)
        
        try:
            masked_states = gpd.overlay(states_gdf, soi_gdf, how="intersection", keep_geom_type=True)
            if not masked_states.empty:
                return masked_states
        except Exception as e:
            print(f"⚠️ Spatial overlay failed ({e}). Falling back to clip...")
            clipped = gpd.clip(states_gdf, soi_gdf)
            if not clipped.empty:
                return clipped

    print("⚠️ `india-soi.geojson` fallback: using StateBoundary.parquet directly...")
    return states_gdf


def find_state_column(gdf: gpd.GeoDataFrame) -> str:
    """Identifies the state name column in StateBoundary.parquet."""
    keywords = ["st_nm", "stname", "st_name", "state", "state_name", "statename", "name"]
    
    for col in gdf.columns:
        clean_col = re.sub(r'[^a-zA-Z0-9]', '', col.lower())
        for kw in keywords:
            if kw in clean_col:
                return col

    for col in gdf.columns:
        if gdf[col].dtype == "object":
            sample_vals = [str(v).lower() for v in gdf[col].dropna().head(20)]
            if any("bengal" in v or "maharashtra" in v or "delhi" in v for v in sample_vals):
                return col

    raise KeyError(f"Could not identify state name column in dataset. Columns: {list(gdf.columns)}")


def generate_state_map_image(state_name: str, state_dir: Path, loaded_layers: dict):
    """Generates and saves a high-resolution PNG image including AI proposed health facilities."""
    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f1f5f9")

    # Layer 1: State Border
    if "state_border" in loaded_layers and loaded_layers["state_border"] is not None:
        loaded_layers["state_border"].plot(ax=ax, color="none", edgecolor="#0f172a", linewidth=2.0, zorder=5)

    # Layer 2: District Boundaries
    if "districts" in loaded_layers and loaded_layers["districts"] is not None:
        loaded_layers["districts"].plot(ax=ax, color="none", edgecolor="#94a3b8", linewidth=0.8, linestyle="--", zorder=4)

    # Layer 3: Transport Lines
    if "railways" in loaded_layers and loaded_layers["railways"] is not None:
        loaded_layers["railways"].plot(ax=ax, color="#475569", linewidth=0.7, label="Railways", zorder=2)

    if "highways" in loaded_layers and loaded_layers["highways"] is not None:
        loaded_layers["highways"].plot(ax=ax, color="#dc2626", linewidth=1.2, label="Highways", zorder=3)

    # Layer 4: Active Health Facilities
    if "health_facilities" in loaded_layers and loaded_layers["health_facilities"] is not None:
        loaded_layers["health_facilities"].plot(
            ax=ax, color="#16a34a", markersize=18, alpha=0.8, label="Active Health Facilities", zorder=6
        )

    # Layer 5: AI Proposed Health Facilities
    if "ai_proposed" in loaded_layers and loaded_layers["ai_proposed"] is not None and not loaded_layers["ai_proposed"].empty:
        loaded_layers["ai_proposed"].plot(
            ax=ax, color="#d97706", marker="^", markersize=60, label="AI Proposed Facilities", zorder=7
        )

    ax.set_title(f"Healthcare Gap & Facility Recommendation Map - {state_name}", fontsize=15, fontweight="bold", pad=15)
    ax.set_axis_off()
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=10)

    clean_name = state_name.strip().lower().replace(" ", "_")
    output_image_path = state_dir / f"{clean_name}_map.png"
    plt.tight_layout()
    plt.savefig(output_image_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  🖼️  Saved Map PNG with AI Recommendations: {output_image_path.name}")


def process_and_generate_resources():
    print("🚀 Initializing GeoHealth-AI Resource Generator...")

    # Load state boundaries masked with official Survey of India border
    states_gdf = get_official_state_boundaries()
    state_col = find_state_column(states_gdf)
    print(f"✅ Identified State Column: '{state_col}'")

    available_states = sorted(list(states_gdf[state_col].dropna().unique()))

    print("\n--------------------------------------------------")
    print("Available States:")
    for idx, name in enumerate(available_states, 1):
        print(f" [{idx:2d}] {name}")
    print(" [ 0] PROCESS ALL STATES")
    print("--------------------------------------------------")

    user_choice = input("\nEnter State Index to Process (or 0 for ALL): ").strip()
    selected_targets = []

    if user_choice == "0" or not user_choice:
        selected_targets = available_states
    else:
        try:
            choice_idx = int(user_choice) - 1
            if 0 <= choice_idx < len(available_states):
                selected_targets = [available_states[choice_idx]]
            else:
                selected_targets = available_states
        except ValueError:
            selected_targets = available_states

    districts_gdf = load_dataset(INDIA_DISTRICT_BORDER)
    highways_gdf = load_dataset(HIGHWAY_DATA)
    railways_gdf = load_dataset(RAILWAY_DATA)
    villages_gdf = load_dataset(VILLEGE_DATA)
    health_gdf = load_dataset(INDIA_HEALTH_FACILITY_DATA)
    pop_df = load_dataset(POPULATION_DATA)

    for state_name in selected_targets:
        clean_state_name = str(state_name).strip().lower().replace(" ", "_")
        state_dir = GENERATED_DATA_DIR / clean_state_name
        state_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n✂️ Processing official boundaries for: {state_name}...")

        target_state = states_gdf[states_gdf[state_col] == state_name].to_crs("EPSG:4326")
        target_state.to_parquet(state_dir / "state_border.parquet", index=False)

        if pop_df is not None:
            state_pop_col = [c for c in pop_df.columns if "state" in c or "st_name" in c][0]
            val_pop_col = [c for c in pop_df.columns if "tot_p" in c or "population" in c or "persons" in c][0]
            
            matched_pop = pop_df[pop_df[state_pop_col].astype(str).str.strip().str.lower() == str(state_name).strip().lower()]
            total_val = int(matched_pop[val_pop_col].sum()) if not matched_pop.empty else 0
            
            with open(state_dir / "state_populations.json", "w") as f:
                json.dump({str(state_name): total_val}, f, indent=2)

        raw_layers = {
            "districts": districts_gdf,
            "highways": highways_gdf,
            "railways": railways_gdf,
            "villages": villages_gdf,
            "health_facilities": health_gdf,
        }

        saved_layers = {"state_border": target_state}

        for layer_key, layer_gdf in raw_layers.items():
            if layer_gdf is not None and not layer_gdf.empty:
                try:
                    clipped = gpd.clip(layer_gdf.to_crs("EPSG:4326"), target_state)

                    if layer_key == "health_facilities" and not clipped.empty:
                        type_cols = [c for c in clipped.columns if any(k in c.lower() for k in ["type", "cat", "facility", "level"])]
                        if type_cols:
                            t_col = type_cols[0]
                            non_patient_codes = ["sce", "sub center", "sub-center", "sub centre", "sub-centre", "sc"]
                            pattern = r'|'.join([rf'\b{re.escape(code)}\b' for code in non_patient_codes])
                            clipped = clipped[~clipped[t_col].astype(str).str.strip().str.lower().str.contains(pattern, regex=True)]

                    if layer_key == "villages" and not clipped.empty:
                        pop_cols = [c for c in clipped.columns if any(k in c.lower() for k in ["tot_p", "pop", "persons"])]
                        val_col = pop_cols[0] if pop_cols else None
                        centroids = clipped.to_crs(epsg=3857).geometry.centroid.to_crs(epsg=4326)
                        
                        clipped = gpd.GeoDataFrame(
                            {"geometry": centroids, "population": pd.to_numeric(clipped[val_col], errors="coerce").fillna(500) if val_col else 500},
                            crs="EPSG:4326",
                        )

                    if not clipped.empty:
                        clipped.to_parquet(state_dir / f"{layer_key}.parquet", index=False)
                        saved_layers[layer_key] = clipped
                except Exception as e:
                    print(f"  └─ Skipped {layer_key}: {e}")

        # Compute AI recommendations for PNG map rendering
        ai_proposed_gdf = predict_facility_locations(
            villages_gdf=saved_layers.get("villages"),
            health_gdf=saved_layers.get("health_facilities"),
            state_border_gdf=target_state,
        )
        saved_layers["ai_proposed"] = ai_proposed_gdf

        generate_state_map_image(str(state_name), state_dir, saved_layers)

    print("\n🎉 Pre-processing complete! PNG maps now include AI recommended facility markers.")


if __name__ == "__main__":
    process_and_generate_resources()
