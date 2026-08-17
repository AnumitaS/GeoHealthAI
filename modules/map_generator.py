from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def render_state_map(state_name: str, layers: dict, pop_info: dict, save_path: Path):
    """
    Renders Population Density Heatmap integrated with Districts, Railways, 
    Highways, Census Villages, Health Facilities, and AI Site markers.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    target_crs = "EPSG:4326"

    fig, ax = plt.subplots(figsize=(14, 16), dpi=300)
    ax.set_facecolor("#fcfcfc")

    districts = layers.get("districts")
    pop_map = pop_info.get("district_pop_map", {})

    # Layer 1: Population Density Heatmap (Choropleth by District)
    if districts is not None and not districts.empty:
        dist_gdf = districts.to_crs(target_crs).copy()
        d_name_col = [c for c in dist_gdf.columns if "dist" in c.lower() or "name" in c.lower()][0]
        
        # Map population values to districts
        dist_gdf["proj_pop"] = dist_gdf[d_name_col].apply(
            lambda name: pop_map.get(str(name).lower().replace(" ", ""), 0)
        )

        if dist_gdf["proj_pop"].sum() > 0:
            dist_gdf.plot(
                column="proj_pop",
                ax=ax,
                cmap="YlOrRd",
                legend=True,
                legend_kwds={"label": "Projected District Population", "orientation": "horizontal", "shrink": 0.6, "pad": 0.02},
                edgecolor="#78716c",
                linewidth=0.5,
                alpha=0.65
            )
        else:
            dist_gdf.plot(ax=ax, facecolor="#f5f5f4", edgecolor="#78716c", linewidth=0.5)

    # Layer 2: Census Villages
    villages = layers.get("villages")
    if villages is not None and not villages.empty:
        villages = villages.to_crs(target_crs)
        if "Point" in villages.geometry.iloc[0].geom_type:
            villages.plot(ax=ax, color="#0284c7", markersize=0.2, alpha=0.4)
        else:
            villages.plot(ax=ax, edgecolor="#38bdf8", facecolor="none", linewidth=0.1, alpha=0.3)

    # Layer 3: Railways
    railways = layers.get("railways")
    if railways is not None and not railways.empty:
        railways.to_crs(target_crs).plot(ax=ax, color="#334155", linewidth=0.7, linestyle=":", alpha=0.8)

    # Layer 4: Highways
    highways = layers.get("highways")
    if highways is not None and not highways.empty:
        highways.to_crs(target_crs).plot(ax=ax, color="#b91c1c", linewidth=1.2, alpha=0.9)

    # Layer 5: Outer State Border
    state_border = layers.get("state_border")
    if state_border is not None and not state_border.empty:
        state_border.to_crs(target_crs).plot(ax=ax, edgecolor="#0f172a", facecolor="none", linewidth=2.0)

    # Layer 6: Existing Health Facilities
    health = layers.get("health_facilities")
    if health is not None and not health.empty:
        health.to_crs(target_crs).plot(ax=ax, color="#15803d", marker="+", markersize=25, linewidth=1.2, alpha=0.9)

    # Layer 7: AI Proposed Candidate Sites
    ai_recs = layers.get("ai_recommendations")
    if ai_recs is not None and not ai_recs.empty:
        ai_recs.to_crs(target_crs).plot(
            ax=ax, color="#d97706", marker="*", markersize=150, edgecolor="#451a03", linewidth=0.8, alpha=1.0
        )

    pop_str = pop_info.get("formatted_str", "Population Data N/A")
    ax.set_title(f"{state_name} - Population Density & Integrated Infrastructure GIS Map\n[{pop_str}]", fontsize=14, fontweight="bold", pad=20)
    ax.set_axis_off()

    legend_elements = [
        Line2D([0], [0], color="#0f172a", lw=2.0, label="State Boundary"),
        Line2D([0], [0], color="#78716c", lw=0.6, label="District Boundary"),
        Line2D([0], [0], color="#b91c1c", lw=1.5, label="National Highways (MORTH)"),
        Line2D([0], [0], color="#334155", lw=1.0, linestyle=":", label="Railway Network"),
        Line2D([0], [0], marker="o", color="w", label="Census Villages", markerfacecolor="#0284c7", markersize=4),
        Line2D([0], [0], marker="+", color="#15803d", label="Existing Health Facilities", markersize=8, markeredgewidth=1.5, linestyle="None"),
        Line2D([0], [0], marker="*", color="#d97706", label="AI Proposed Facility Sites", markersize=10, markeredgecolor="#451a03", linestyle="None")
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, facecolor="white", edgecolor="#cbd5e1")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"✅ Full Infrastructure & Population Heatmap saved to:\n   {save_path}")
    plt.close()


def render_ai_recommendation_map(state_name: str, layers: dict, save_path: Path):
    """
    Renders dedicated PNG map focused on proposed facility locations and 12km service buffers.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    target_crs = "EPSG:3857"

    fig, ax = plt.subplots(figsize=(14, 16), dpi=300)
    ax.set_facecolor("#fafaf9")

    state_border = layers.get("state_border")
    if state_border is not None and not state_border.empty:
        state_border.to_crs(target_crs).plot(ax=ax, edgecolor="#1e293b", facecolor="#f1f5f9", linewidth=1.8)

    districts = layers.get("districts")
    if districts is not None and not districts.empty:
        districts.to_crs(target_crs).plot(ax=ax, edgecolor="#cbd5e1", facecolor="none", linewidth=0.6, linestyle="--")

    health = layers.get("health_facilities")
    if health is not None and not health.empty:
        health.to_crs(target_crs).plot(ax=ax, color="#94a3b8", marker="o", markersize=12, alpha=0.5)

    ai_recs = layers.get("ai_recommendations")
    num_recs = 0
    if ai_recs is not None and not ai_recs.empty:
        ai_recs_proj = ai_recs.to_crs(target_crs)
        num_recs = len(ai_recs_proj)

        buffers = ai_recs_proj.geometry.buffer(12000)
        buffers.plot(ax=ax, facecolor="#fef08a", edgecolor="#eab308", linewidth=1.2, alpha=0.35)

        ai_recs_proj.plot(ax=ax, color="#d97706", marker="*", markersize=200, edgecolor="#451a03", linewidth=1.0, zorder=5)

        for idx, row in ai_recs_proj.iterrows():
            ax.annotate(
                text=str(row.get("recommendation_id", f"Site_{idx+1}")),
                xy=(row.geometry.x, row.geometry.y),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color="#78350f"
            )

    ax.set_title(
        f"{state_name} - AI Recommended Health Facility Placement Plan\n"
        f"[Target Gap Locations: {num_recs} | Service Radius Buffer: 12 km]",
        fontsize=14, fontweight="bold", pad=20, color="#1e293b"
    )
    ax.set_axis_off()

    legend_elements = [
        Line2D([0], [0], color="#1e293b", lw=1.8, label="State Boundary"),
        Line2D([0], [0], color="#cbd5e1", lw=0.6, linestyle="--", label="District Boundaries"),
        Line2D([0], [0], marker="o", color="w", label="Existing Health Network", markerfacecolor="#94a3b8", markersize=6),
        Line2D([0], [0], marker="*", color="#d97706", label="AI Proposed Facility Site", markersize=12, markeredgecolor="#451a03", linestyle="None"),
        Line2D([0], [0], color="#eab308", lw=8, alpha=0.35, label="12km Service Radius Buffer")
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, facecolor="white", edgecolor="#e2e8f0")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"✅ AI Recommendation Plan Map PNG saved to:\n   {save_path}")
    plt.close()