import re
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONSTANT_DATA_DIR = PROJECT_ROOT / "constant_data"
OFFICIAL_SOI_BORDER = CONSTANT_DATA_DIR / "india-soi.geojson"

# Population capacity mapping for healthcare facility types
POPULATION_CAP_MAP = {
    "MCH": 500000,
    "MC": 500000,
    "DH": 300000,
    "SDH": 150000,
    "RH": 100000,
    "CHC": 100000,
    "BPHC": 100000,
    "PHC": 30000,
    "UPHC": 30000,
    "SC": 5000,
}

# Facility tiers ordered from highest capacity to primary care
FACILITY_HIERARCHY = ["MCH", "MC", "DH", "SDH", "CHC", "BPHC", "PHC", "UPHC", "SC"]


def predict_facility_type(cluster_population: float, is_urban: bool = False) -> dict:
    """Predicts the recommended facility tier and required units based on cluster population."""
    if cluster_population <= 0:
        return {
            "recommended_facility": "SC",
            "units_needed": 1,
            "unit_capacity": POPULATION_CAP_MAP["SC"],
        }

    candidate_tiers = (
        ["MC", "DH", "SDH", "UPHC"]
        if is_urban
        else ["DH", "SDH", "CHC", "BPHC", "PHC", "SC"]
    )

    selected_facility = "SC"
    for facility in candidate_tiers:
        cap = POPULATION_CAP_MAP[facility]
        if cluster_population >= cap:
            selected_facility = facility
            break

    unit_capacity = POPULATION_CAP_MAP[selected_facility]
    units_needed = int(np.ceil(cluster_population / unit_capacity))

    return {
        "recommended_facility": selected_facility,
        "units_needed": units_needed,
        "unit_capacity": unit_capacity,
    }


def filter_effective_facilities(health_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Excludes non-patient Sub-Centers (SCE / SC) from healthcare gap calculations."""
    if health_gdf is None or health_gdf.empty:
        return health_gdf

    type_cols = [
        c
        for c in health_gdf.columns
        if any(
            k in c.lower()
            for k in ["type", "cat", "facility", "level", "nin", "nin_facility"]
        )
    ]
    if not type_cols:
        return health_gdf

    type_col = type_cols[0]
    non_patient_codes = [
        "sce",
        "sub center",
        "sub-center",
        "sub centre",
        "sub-centre",
        "sc",
    ]
    pattern = r"|".join([rf"\b{re.escape(code)}\b" for code in non_patient_codes])

    return health_gdf[
        ~health_gdf[type_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.contains(pattern, regex=True)
    ]


def predict_facility_locations(
    villages_gdf: gpd.GeoDataFrame,
    health_gdf: gpd.GeoDataFrame,
    state_border_gdf: gpd.GeoDataFrame,
    coverage_radius_km: float = 12.0,
    cluster_eps_km: float = 15.0,
    min_villages_per_cluster: int = 5,
) -> gpd.GeoDataFrame:
    """Identifies unserved village clusters, calculates population-weighted centroids,

    and predicts facility types/units strictly within official SOI borders.
    """
    empty_cols = [
        "recommendation_id",
        "geometry",
        "unserved_population",
        "village_count",
        "recommended_facility",
        "units_needed",
        "unit_capacity",
    ]

    if villages_gdf is None or villages_gdf.empty:
        return gpd.GeoDataFrame(columns=empty_cols, crs="EPSG:4326")

    clean_health = filter_effective_facilities(health_gdf)

    if clean_health is not None and not clean_health.empty:
        health_projected = clean_health.to_crs(epsg=3857)
        villages_projected = villages_gdf.to_crs(epsg=3857)

        buffers = health_projected.geometry.buffer(coverage_radius_km * 1000)
        combined_buffers = buffers.unary_union

        unserved_mask = ~villages_projected.geometry.within(combined_buffers)
        unserved_villages = villages_gdf[unserved_mask].copy()
    else:
        unserved_villages = villages_gdf.copy()

    if unserved_villages.empty:
        return gpd.GeoDataFrame(columns=empty_cols, crs="EPSG:4326")

    coords_rad = np.radians(
        unserved_villages[["geometry"]]
        .apply(lambda p: [p.geometry.y, p.geometry.x], axis=1)
        .tolist()
    )
    kms_per_radian = 6371.0088
    epsilon = cluster_eps_km / kms_per_radian

    db = DBSCAN(eps=epsilon, min_samples=min_villages_per_cluster, metric="haversine")
    unserved_villages["cluster"] = db.fit_predict(coords_rad)

    clustered = unserved_villages[unserved_villages["cluster"] != -1].copy()
    if clustered.empty:
        return gpd.GeoDataFrame(columns=empty_cols, crs="EPSG:4326")

    # Detect population column name dynamically
    pop_col = None
    possible_pop_cols = [
        "POPULATION",
        "TOT_P",
        "TOTAL_POP",
        "pop",
        "tot_p",
        "population",
    ]
    for col in possible_pop_cols:
        if col in clustered.columns:
            pop_col = col
            break

    # Load Survey of India boundary for territorial validation
    soi_gdf = None
    if OFFICIAL_SOI_BORDER.exists():
        try:
            soi_gdf = gpd.read_file(OFFICIAL_SOI_BORDER).to_crs("EPSG:4326")
            soi_gdf["geometry"] = soi_gdf["geometry"].buffer(0)
        except Exception as e:
            print(f"⚠️ Failed to load SOI boundary: {e}")

    valid_boundary = state_border_gdf
    if valid_boundary is not None and soi_gdf is not None:
        try:
            valid_boundary = gpd.clip(valid_boundary.to_crs("EPSG:4326"), soi_gdf)
        except Exception:
            pass

    boundary_union = (
        valid_boundary.unary_union
        if (valid_boundary is not None and not valid_boundary.empty)
        else None
    )

    records = []

    for cluster_id, group in clustered.groupby("cluster"):
        total_pop = 0
        if pop_col is not None:
            # Convert non-numeric values gracefully
            pop_series = pd.to_numeric(group[pop_col], errors="coerce").fillna(0)
            total_pop = float(pop_series.sum())

        # Population-Weighted Centroid calculation
        if total_pop > 0:
            center_lon = (group.geometry.x * pop_series).sum() / total_pop
            center_lat = (group.geometry.y * pop_series).sum() / total_pop
        else:
            center_lon = group.geometry.x.mean()
            center_lat = group.geometry.y.mean()

        point_gdf = gpd.GeoDataFrame(
            {"geometry": gpd.points_from_xy([center_lon], [center_lat])},
            crs="EPSG:4326",
        )

        # Ensure site falls within boundary limits
        if boundary_union is not None and not point_gdf.within(boundary_union).iloc[0]:
            continue

        # Determine Urban/Rural context if attribute exists
        is_urban = False
        if "IS_URBAN" in group.columns:
            is_urban = group["IS_URBAN"].astype(bool).any()

        # Predict facility details based on cluster population
        facility_info = predict_facility_type(total_pop, is_urban=is_urban)

        records.append(
            {
                "recommendation_id": f"AI_Site_{cluster_id + 1}",
                "geometry": point_gdf.geometry.iloc[0],
                "unserved_population": int(total_pop),
                "village_count": len(group),
                "recommended_facility": facility_info["recommended_facility"],
                "units_needed": facility_info["units_needed"],
                "unit_capacity": facility_info["unit_capacity"],
            }
        )

    if not records:
        return gpd.GeoDataFrame(columns=empty_cols, crs="EPSG:4326")

    return gpd.GeoDataFrame(records, crs="EPSG:4326")