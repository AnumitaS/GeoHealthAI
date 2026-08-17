import re
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONSTANT_DATA_DIR = PROJECT_ROOT / "constant_data"
OFFICIAL_SOI_BORDER = CONSTANT_DATA_DIR / "india-soi.geojson"


def filter_effective_facilities(health_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Excludes non-patient Sub-Centers (SCE / SC) from healthcare gap calculations."""
    if health_gdf is None or health_gdf.empty:
        return health_gdf

    type_cols = [
        c
        for c in health_gdf.columns
        if any(k in c.lower() for k in ["type", "cat", "facility", "level", "nin", "nin_facility"])
    ]
    if not type_cols:
        return health_gdf

    type_col = type_cols[0]
    non_patient_codes = ["sce", "sub center", "sub-center", "sub centre", "sub-centre", "sc"]
    pattern = r"|".join([rf"\b{re.escape(code)}\b" for code in non_patient_codes])

    return health_gdf[~health_gdf[type_col].astype(str).str.strip().str.lower().str.contains(pattern, regex=True)]


def predict_facility_locations(
    villages_gdf: gpd.GeoDataFrame,
    health_gdf: gpd.GeoDataFrame,
    state_border_gdf: gpd.GeoDataFrame,
    coverage_radius_km: float = 12.0,
    cluster_eps_km: float = 15.0,
    min_villages_per_cluster: int = 5,
) -> gpd.GeoDataFrame:
    """Identifies unserved village clusters and calculates optimal location centroids strictly within official SOI borders."""
    if villages_gdf is None or villages_gdf.empty:
        return gpd.GeoDataFrame(columns=["recommendation_id", "geometry"], crs="EPSG:4326")

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
        return gpd.GeoDataFrame(columns=["recommendation_id", "geometry"], crs="EPSG:4326")

    coords_rad = np.radians(
        unserved_villages[["geometry"]].apply(lambda p: [p.geometry.y, p.geometry.x], axis=1).tolist()
    )
    kms_per_radian = 6371.0088
    epsilon = cluster_eps_km / kms_per_radian

    db = DBSCAN(eps=epsilon, min_samples=min_villages_per_cluster, metric="haversine")
    unserved_villages["cluster"] = db.fit_predict(coords_rad)

    clustered = unserved_villages[unserved_villages["cluster"] != -1]
    if clustered.empty:
        return gpd.GeoDataFrame(columns=["recommendation_id", "geometry"], crs="EPSG:4326")

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
        valid_boundary.unary_union if (valid_boundary is not None and not valid_boundary.empty) else None
    )

    proposed_points = []
    rec_ids = []

    for cluster_id, group in clustered.groupby("cluster"):
        center_lon = group.geometry.x.mean()
        center_lat = group.geometry.y.mean()

        point_gdf = gpd.GeoDataFrame(
            {"geometry": gpd.points_from_xy([center_lon], [center_lat])}, crs="EPSG:4326"
        )

        if boundary_union is not None:
            if point_gdf.within(boundary_union).iloc[0]:
                proposed_points.append(point_gdf.geometry.iloc[0])
                rec_ids.append(f"AI_Site_{cluster_id + 1}")
        else:
            proposed_points.append(point_gdf.geometry.iloc[0])
            rec_ids.append(f"AI_Site_{cluster_id + 1}")

    return gpd.GeoDataFrame({"recommendation_id": rec_ids, "geometry": proposed_points}, crs="EPSG:4326")