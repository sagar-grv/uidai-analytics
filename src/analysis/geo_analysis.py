"""
UIDAI Aadhaar - Deep Geo-Spatial Analytics
===========================================
Phase 11: State & District Level Deep Dive with "Proof of Inequality".
Inputs: 'uidai_gold_master.csv'
Outputs: 
1. Pareto Chart (Concentration Risk)
2. State Performance Matrix (Scatter)
3. District Anomaly Report (3-Sigma Deviations)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.figsize'] = (14, 8)

base_path = Path(r"c:\Users\sagar\Downloads\uidai dataset")
processed_path = base_path / "processed"
output_path = base_path / "geo_outputs"
output_path.mkdir(exist_ok=True)

print("=" * 80)
print("UIDAI GEO-SPATIAL DEEP DIVE")
print("=" * 80)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1/3] Loading Cleaned Data...")
df = pd.read_csv(processed_path / "uidai_gold_master.csv")
print(f"  Records: {len(df):,}")

# =============================================================================
# 2. STATE LEVEL PROOF: PERFORMANCE MATRIX
# =============================================================================
print("\n[2/3] Generating State Performance Matrix...")

# Aggregate metrics
state_metrics = df.groupby("state_clean").agg(
    Total_Volume=("total_count", "sum"),
    Biometric_Vol=("total_count", lambda x: x[df["type"]=="Biometric"].sum()),
    Demographic_Vol=("total_count", lambda x: x[df["type"]=="Demographic"].sum()),
    Districts=("district", "nunique")
).reset_index()

# Efficiency Metric: Biometric Updates per Demographic Update (>1 means tech adoption)
state_metrics["Tech_Efficiency_Ratio"] = state_metrics["Biometric_Vol"] / (state_metrics["Demographic_Vol"] + 1)
state_metrics["Avg_Volume_Per_District"] = state_metrics["Total_Volume"] / state_metrics["Districts"]

# Filter for plotting (remove tiny states/UTs for clarity if needed, or keep all)
# We highlight top 10 by volume
top_states = state_metrics.nlargest(10, "Total_Volume")["state_clean"].tolist()

# Plot: Bubbles (Size = Total Volume)
plt.figure(figsize=(14, 8))
sns.scatterplot(
    data=state_metrics, 
    x="Districts", 
    y="Tech_Efficiency_Ratio", 
    size="Total_Volume", 
    sizes=(100, 3000), 
    alpha=0.6, 
    hue="Total_Volume",
    palette="viridis",
    legend=False
)

# Annotate Top States
for i, row in state_metrics.iterrows():
    if row["state_clean"] in top_states or row["Tech_Efficiency_Ratio"] > 2.5:
        plt.text(
            row["Districts"], 
            row["Tech_Efficiency_Ratio"], 
            row["state_clean"], 
            fontsize=9, 
            weight='bold'
        )

plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label="1:1 Ratio (Parity)")
plt.title("State Performance Matrix: Operational Scale vs. Tech Efficiency\n(Size = Total Volume)", fontsize=16)
plt.xlabel("Number of Districts (Operational Centers)", fontsize=12)
plt.ylabel("Tech Efficiency (Biometric / Demographic Ratio)", fontsize=12)
plt.tight_layout()
plt.savefig(output_path / "state_performance_matrix.png")
print(f"  Saved: state_performance_matrix.png")

# =============================================================================
# 3. DISTRICT LEVEL PROOF: CONCENTRATION & ANOMALIES
# =============================================================================
print("\n[3/3] Analyzing District Concentration & Anomalies...")

district_agg = df.groupby(["state_clean", "district"])["total_count"].sum().reset_index()
district_agg = district_agg.sort_values("total_count", ascending=False)

# Pareto Calculation
district_agg["Cumulative_Pct"] = district_agg["total_count"].cumsum() / district_agg["total_count"].sum()
top_20_pct_count = int(len(district_agg) * 0.2)
top_20_load = district_agg.iloc[top_20_pct_count]["Cumulative_Pct"]

print(f"  Pareto Proof: Top 20% of districts handle {top_20_load:.1%} of Total Load.")

# Plot Pareto
plt.figure(figsize=(14, 6))
# Top 50 Districts Bar Chart
top_50 = district_agg.head(50)
sns.barplot(x="district", y="total_count", data=top_50, palette="magma")
plt.xticks(rotation=90, fontsize=8)
plt.title(f"Concentration Risk: Top 50 Districts (out of {len(district_agg)})", fontsize=16)
plt.ylabel("Total Transactions", fontsize=12)
plt.tight_layout()
plt.savefig(output_path / "district_pareto_top50.png")
print(f"  Saved: district_pareto_top50.png")

# HYPER-LOCAL ANOMALIES (Z-Score by State)
print("  Identifying Hyper-Local Anomalies (District > 3-Sigma of State Mean)...")
anomalies = []

for state in district_agg["state_clean"].unique():
    state_data = district_agg[district_agg["state_clean"] == state]
    if len(state_data) < 3: continue # Skip tiny states
    
    mean = state_data["total_count"].mean()
    std = state_data["total_count"].std()
    
    # Z-Score
    state_data["z_score"] = (state_data["total_count"] - mean) / (std + 1e-5)
    
    outliers = state_data[state_data["z_score"] > 3.0]
    for _, row in outliers.iterrows():
        anomalies.append({
            "State": state,
            "District": row["district"],
            "Volume": row["total_count"],
            "State_Avg": mean,
            "Z_Score": row["z_score"]
        })

anomaly_df = pd.DataFrame(anomalies).sort_values("Z_Score", ascending=False)
anomaly_path = output_path / "district_anomalies_3sigma.csv"
anomaly_df.to_csv(anomaly_path, index=False)
print(f"  Detected {len(anomaly_df)} Hyper-Local Anomalies. Saved to {anomaly_path}")

# Generate Text Report
with open(output_path / "geo_insights.txt", "w") as f:
    f.write("UIDAI GEO-SPATIAL DEEP DIVE REPORT\n")
    f.write("==================================\n\n")
    f.write(f"1. CONCENTRATION PROOF (Pareto Principle)\n")
    f.write(f"   - FACT: Top 20% of districts handle {top_20_load:.1%} of ALL traffic.\n")
    f.write(f"   - FACT: Top 10 Districts alone handle {(district_agg.head(10)['total_count'].sum()/district_agg['total_count'].sum()):.1%} of national volume.\n")
    f.write(f"   - INSIGHT: Infrastructure failure in these top 10 nodes cripples the national network.\n\n")
    
    f.write(f"2. HYPER-LOCAL ANOMALIES (Z > 3.0)\n")
    f.write(f"   The following districts are statistical outliers consuming disproportionate resources within their states:\n")
    for i, row in anomaly_df.head(10).iterrows():
        f.write(f"   - {row['District']} ({row['State']}): {row['Volume']:,} (Z-Score: {row['Z_Score']:.1f})\n")
    
print("\nGEO ANALYSIS COMPLETE.")
