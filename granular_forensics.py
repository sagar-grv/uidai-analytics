"""
UIDAI Aadhaar - Hyper-Granular Forensic Scan
=============================================
Phase 12.5: Deep Dive into Age-Bands and Pincode Clusters.
Goal: Detect "Ghost Beneficiaries" (Child Enrollment Fraud) and Sub-District Bottlenecks.

Inputs: Raw 'api_data_aadhar_enrolment' (for age breakdowns), 'uidai_gold_master.csv'
Outputs: 
1. Age-Band Anomaly Report (Districts with suspicious 0-5 yr spikes)
2. Pincode Cluster Heatmap Data
3. Weekly Operational Efficiency Cycle
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

base_path = Path(r"c:\Users\sagar\Downloads\uidai dataset")
enrollment_path = base_path / "api_data_aadhar_enrolment" / "api_data_aadhar_enrolment"
processed_path = base_path / "processed"
output_path = base_path / "granular_outputs"
output_path.mkdir(exist_ok=True)

print("=" * 80)
print("UIDAI HYPER-GRANULAR FORENSIC SCAN")
print("=" * 80)

# =============================================================================
# 1. AGE-BAND ANOMALY DETECTION ("GHOST CHILDREN" CHECK)
# =============================================================================
print("\n[1/3] Scanning for Child Enrollment Anomalies (0-5 Years)...")

# Load Raw Enrollment Data to get Age Columns
files = list(enrollment_path.glob("*.csv"))
df_enroll = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Clean District Names (basic clean needed for grouping)
df_enroll["district_clean"] = df_enroll["district"].astype(str).str.strip().str.title()
df_enroll["state_clean"] = df_enroll["state"].astype(str).str.strip().str.title()

# Aggregate by District
district_age = df_enroll.groupby(["state_clean", "district_clean"]).agg(
    Age_0_5=("age_0_5", "sum"),
    Age_5_17=("age_5_17", "sum"),
    Age_18_Plus=("age_18_greater", "sum")
).reset_index()

district_age["Total"] = district_age["Age_0_5"] + district_age["Age_5_17"] + district_age["Age_18_Plus"]
district_age = district_age[district_age["Total"] > 1000] # Filter distincts with meaningful volume

# Calculate Child Ratio
district_age["Child_Ratio"] = district_age["Age_0_5"] / district_age["Total"]
national_avg_child = district_age["Child_Ratio"].mean()
std_child = district_age["Child_Ratio"].std()

print(f"  National Avg Child Enrollment Ratio (0-5 yrs): {national_avg_child:.2%}")

# Z-Score for Child Ratio
district_age["Z_Child"] = (district_age["Child_Ratio"] - national_avg_child) / std_child

# Identify Suspicious Spike (> 3 Sigma)
suspicious_districts = district_age[district_age["Z_Child"] > 3.0].sort_values("Z_Child", ascending=False)

print(f"  Found {len(suspicious_districts)} districts with suspicious Child Enrollment Spikes (Potential Fraud/Data Push).")
print("  Top 5 Anomalies:")
for i, row in suspicious_districts.head(5).iterrows():
    print(f"    - {row['district_clean']} ({row['state_clean']}): {row['Child_Ratio']:.1%} Children (Z: {row['Z_Child']:.1f})")

# Save Anomaly List
suspicious_districts.to_csv(output_path / "child_enrollment_anomalies.csv", index=False)

# =============================================================================
# 2. PINCODE MICRO-CLUSTER ANALYSIS
# =============================================================================
print("\n[2/3] Analyzing Pincode Micro-Clusters (First 3 Digits)...")
# Load Gold Master
df_gold = pd.read_csv(processed_path / "uidai_gold_master.csv")

# Extract Prefix
df_gold["pin_prefix"] = df_gold["pincode"].astype(str).str[:3]

# Aggregate
pin_stats = df_gold.groupby("pin_prefix")["total_count"].sum().reset_index()
pin_stats = pin_stats.sort_values("total_count", ascending=False)

# Concentration in Prefixes
top_10_prefixes = pin_stats.head(10)
print(f"  Top Pincode Zone: {top_10_prefixes.iloc[0]['pin_prefix']} (Vol: {top_10_prefixes.iloc[0]['total_count']:,})")

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x="pin_prefix", y="total_count", data=top_10_prefixes, palette="cool")
plt.title("Top 10 Pincode Zones (Micro-Markets)", fontsize=14)
plt.xlabel("Pincode Prefix (3-Digit)", fontsize=12)
plt.ylabel("Total Volume", fontsize=12)
plt.savefig(output_path / "pincode_micro_clusters.png")
print("  Saved: pincode_micro_clusters.png")

# =============================================================================
# 3. WEEKLY OPERATIONAL CYCLE (The "Sunday Slump")
# =============================================================================
print("\n[3/3] Measuring Weekend Efficiency Lag...")
df_gold["date"] = pd.to_datetime(df_gold["date"])
df_gold["day_name"] = df_gold["date"].dt.day_name()

day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekly_vol = df_gold.groupby("day_name")["total_count"].mean().reindex(day_order)

# Calculate "Slump"
avg_weekday = weekly_vol[:5].mean()
sunday_vol = weekly_vol["Sunday"]
slump_pct = (avg_weekday - sunday_vol) / avg_weekday

print(f"  Sunday Operational Drop: {slump_pct:.1%} vs Weekday Avg")

# Plot
plt.figure(figsize=(10, 5))
sns.lineplot(x=weekly_vol.index, y=weekly_vol.values, marker="o", linewidth=2.5, color="#8e44ad")
plt.title("Daily Operational Efficiency Cycle (The 'Heartbeat')", fontsize=14)
plt.ylabel("Avg Daily Transactions", fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig(output_path / "weekly_efficiency_cycle.png")
print("  Saved: weekly_efficiency_cycle.png")

# Generate Insight Text
with open(output_path / "granular_insights.txt", "w") as f:
    f.write("UIDAI HYPER-GRANULAR FORENSIC INSIGHTS\n")
    f.write("======================================\n\n")
    f.write("1. THE 'GHOST CHILD' ANOMALY (0-5 Year Skew)\n")
    f.write(f"   - National Avg: {national_avg_child:.1%}\n")
    f.write(f"   - Suspicious Districts (>3 Sigma):\n")
    for i, row in suspicious_districts.head(5).iterrows():
        f.write(f"     * {row['district_clean']} ({row['state_clean']}): {row['Child_Ratio']:.1%} (Z: {row['Z_Child']:.1f})\n")
    f.write("\n")
    f.write("2. PINCODE MICRO-ZONES\n")
    f.write(f"   - Most active zone: {top_10_prefixes.iloc[0]['pin_prefix']} (Likely major metro hub)\n")
    f.write("\n")
    f.write("3. OPERATIONAL HEARTBEAT\n")
    f.write(f"   - Sunday Slump: {slump_pct:.1%} drop in efficiency.\n")
    f.write("   - Recommendation: Use Sundays for system maintenance to zero impact.\n")

print("\nGRANULAR SCAN COMPLETE.")
