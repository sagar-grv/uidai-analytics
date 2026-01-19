"""
UIDAI Aadhaar Data - Benford's Law Visualization
=================================================
Generates a specialized forensic chart comparing actual data distribution
vs. Benford's Law theoretical distribution to prove data anomalies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set aesthetic style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'sans-serif']

base_path = Path(r"c:\Users\sagar\Downloads\uidai dataset")
output_path = base_path / "visualizations"
output_path.mkdir(exist_ok=True)

print("Loading datasets for Benford Analysis...")
enrollment_path = base_path / "api_data_aadhar_enrolment" / "api_data_aadhar_enrolment"
enrollment_df = pd.concat([pd.read_csv(f) for f in enrollment_path.glob("*.csv")], ignore_index=True)

biometric_path = base_path / "api_data_aadhar_biometric" / "api_data_aadhar_biometric"
biometric_df = pd.concat([pd.read_csv(f) for f in biometric_path.glob("*.csv")], ignore_index=True)

demographic_path = base_path / "api_data_aadhar_demographic" / "api_data_aadhar_demographic"
demographic_df = pd.concat([pd.read_csv(f) for f in demographic_path.glob("*.csv")], ignore_index=True)

# Parse dates and calculate totals
for df in [enrollment_df, biometric_df, demographic_df]:
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")

enrollment_df["total"] = enrollment_df["age_0_5"] + enrollment_df["age_5_17"] + enrollment_df["age_18_greater"]
biometric_df["total"] = biometric_df["bio_age_5_17"] + biometric_df["bio_age_17_"]
demographic_df["total"] = demographic_df["demo_age_5_17"] + demographic_df["demo_age_17_"]

def get_leading_digit_dist(df):
    daily_totals = df.groupby("date")["total"].sum()
    daily_totals = daily_totals[daily_totals > 0]
    digits = daily_totals.astype(str).str[0].astype(int)
    counts = digits.value_counts(normalize=True).sort_index()
    # Ensure all digits 1-9
    aligned = pd.Series(0.0, index=np.arange(1, 10))
    aligned.update(counts)
    return aligned

# Calculate distributions
benford_dist = np.log10(1 + 1 / np.arange(1, 10))
enroll_dist = get_leading_digit_dist(enrollment_df)
bio_dist = get_leading_digit_dist(biometric_df)
demo_dist = get_leading_digit_dist(demographic_df)

# Prepare dataframe for plotting
plot_data = []
for digit in range(1, 10):
    plot_data.append({"Digit": digit, "Frequency": benford_dist[digit-1], "Type": "Benford's Law (Expected)"})
    plot_data.append({"Digit": digit, "Frequency": enroll_dist.get(digit, 0), "Type": "Enrollment (Actual)"})
    plot_data.append({"Digit": digit, "Frequency": bio_dist.get(digit, 0), "Type": "Biometric (Actual)"})
    plot_data.append({"Digit": digit, "Frequency": demo_dist.get(digit, 0), "Type": "Demographic (Actual)"})

vis_df = pd.DataFrame(plot_data)

# =============================================================================
# PLOT 1: COMPREHENSIVE COMPARISON
# =============================================================================
plt.figure(figsize=(14, 8))

# Define colors: Benford (Black dashed), Others (Distinct colors)
colors = {
    "Benford's Law (Expected)": "#333333",
    "Enrollment (Actual)": "#2ecc71",
    "Biometric (Actual)": "#e74c3c",
    "Demographic (Actual)": "#3498db"
}

# Create bar chart for actuals
bar_data = vis_df[vis_df["Type"] != "Benford's Law (Expected)"]
benford_line = vis_df[vis_df["Type"] == "Benford's Law (Expected)"]

ax = sns.barplot(x="Digit", y="Frequency", hue="Type", data=bar_data, palette=colors, alpha=0.7)

# Overlay Benford's Law as a line
plt.plot(np.arange(0, 9), benford_line["Frequency"], color='#333333', marker='o', 
         linestyle='--', linewidth=2.5, label="Benford's Law (Theoretical)", zorder=10)

# Aesthetics
plt.title("FORENSIC AUDIT: Benford's Law Deviation Analysis\n(Daily Transaction Volumes)", fontsize=16, fontweight='bold', pad=20)
plt.ylabel("Frequency (Probability)", fontsize=12)
plt.xlabel("Leading Digit (1-9)", fontsize=12)
plt.legend(title="Dataset Source", title_fontsize=12, fontsize=10, loc='upper right')

# Add annotation box
textstr = (
    "INTERPRETATION:\n"
    "• Benford's Law (Black Line) represents natural, organic data.\n"
    "• Large deviations indicate unnatural constraints, synthetic generation,\n"
    "  or batch processing (e.g., quotas or caps).\n"
    "• Biometric (Red) shows extreme deviation at digit '3' and '4'."
)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
plt.text(0.5, 0.75, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

# Save
save_path = output_path / "benford_forensic_analysis.png"
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved visualization to: {save_path}")

# =============================================================================
# PLOT 2: SMALL MULTIPLES (SEPARATE VIEW)
# =============================================================================
# Create a cleaner separate view for the report
g = sns.FacetGrid(vis_df[vis_df["Type"] != "Benford's Law (Expected)"], col="Type", height=5, aspect=1)
g.map_dataframe(sns.barplot, x="Digit", y="Frequency", color="#666666", alpha=0.6)

# Add Benford line to each subplot
def plot_benford(**kwargs):
    plt.plot(np.arange(0, 9), benford_dist, color='red', marker='x', linestyle='--', linewidth=2, label="Benford")

g.map(plot_benford)
g.set_axis_labels("Leading Digit", "Frequency")
g.fig.suptitle("Forensic Audit by Dataset (Red Line = Natural Expected Pattern)", fontsize=16, y=1.05)
g.add_legend()

save_path_2 = output_path / "benford_breakdown.png"
plt.savefig(save_path_2, dpi=300, bbox_inches='tight')
print(f"Saved breakdown to: {save_path_2}")

print("Benford visualization complete.")
