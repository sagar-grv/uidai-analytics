"""
UIDAI Aadhaar Data Analysis - Visualization Script
===================================================
Generates comprehensive visualizations for hackathon submission
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Create output directory
output_dir = Path(r"c:\Users\sagar\Downloads\uidai dataset\visualizations")
output_dir.mkdir(exist_ok=True)

# Load datasets
print("Loading datasets...")
base_path = Path(r"c:\Users\sagar\Downloads\uidai dataset")

enrollment_path = base_path / "api_data_aadhar_enrolment" / "api_data_aadhar_enrolment"
enrollment_df = pd.concat([pd.read_csv(f) for f in enrollment_path.glob("*.csv")], ignore_index=True)

biometric_path = base_path / "api_data_aadhar_biometric" / "api_data_aadhar_biometric"
biometric_df = pd.concat([pd.read_csv(f) for f in biometric_path.glob("*.csv")], ignore_index=True)

demographic_path = base_path / "api_data_aadhar_demographic" / "api_data_aadhar_demographic"
demographic_df = pd.concat([pd.read_csv(f) for f in demographic_path.glob("*.csv")], ignore_index=True)

# Parse dates
enrollment_df["date"] = pd.to_datetime(enrollment_df["date"], format="%d-%m-%Y", errors="coerce")
biometric_df["date"] = pd.to_datetime(biometric_df["date"], format="%d-%m-%Y", errors="coerce")
demographic_df["date"] = pd.to_datetime(demographic_df["date"], format="%d-%m-%Y", errors="coerce")

# Calculate totals
enrollment_df["total"] = enrollment_df["age_0_5"] + enrollment_df["age_5_17"] + enrollment_df["age_18_greater"]
biometric_df["total"] = biometric_df["bio_age_5_17"] + biometric_df["bio_age_17_"]
demographic_df["total"] = demographic_df["demo_age_5_17"] + demographic_df["demo_age_17_"]

# Standardize state names
def standardize_state(state):
    state = str(state).strip().title()
    # Common corrections
    corrections = {
        'Andhra Prades': 'Andhra Pradesh',
        'West Bengli': 'West Bengal', 
        'Westbengal': 'West Bengal',
        'West Bengal': 'West Bengal',
        'Pondicherry': 'Puducherry',
        'Orissa': 'Odisha',
        'Chattisgarh': 'Chhattisgarh',
        'Chhatisgarh': 'Chhattisgarh',
        'Uttrakhand': 'Uttarakhand',
        'Uttaranchal': 'Uttarakhand',
        'Dadra And Nagar Haveli': 'Dadra And Nagar Haveli And Daman And Diu',
        'Daman And Diu': 'Dadra And Nagar Haveli And Daman And Diu',
        'Jammu And Kashmir': 'Jammu And Kashmir',
        'Jammu & Kashmir': 'Jammu And Kashmir',
    }
    return corrections.get(state, state)

enrollment_df["state_clean"] = enrollment_df["state"].apply(standardize_state)
biometric_df["state_clean"] = biometric_df["state"].apply(standardize_state)
demographic_df["state_clean"] = demographic_df["state"].apply(standardize_state)

print("Creating visualizations...")

# =============================================================================
# VISUALIZATION 1: Age Distribution - Enrollment
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Enrollment Age Distribution
age_data = [enrollment_df["age_0_5"].sum(), enrollment_df["age_5_17"].sum(), enrollment_df["age_18_greater"].sum()]
age_labels = ['Age 0-5', 'Age 5-17', 'Age 18+']
colors = ['#3498db', '#e74c3c', '#2ecc71']

axes[0].pie(age_data, labels=age_labels, autopct='%1.1f%%', colors=colors, explode=[0.02, 0.02, 0.02])
axes[0].set_title('Enrollment by Age Group\n(Total: {:,})'.format(sum(age_data)))

# Biometric Age Distribution
bio_data = [biometric_df["bio_age_5_17"].sum(), biometric_df["bio_age_17_"].sum()]
bio_labels = ['Age 5-17', 'Age 17+']
axes[1].pie(bio_data, labels=bio_labels, autopct='%1.1f%%', colors=['#e74c3c', '#2ecc71'], explode=[0.02, 0.02])
axes[1].set_title('Biometric Updates by Age Group\n(Total: {:,})'.format(sum(bio_data)))

# Demographic Age Distribution
demo_data = [demographic_df["demo_age_5_17"].sum(), demographic_df["demo_age_17_"].sum()]
demo_labels = ['Age 5-17', 'Age 17+']
axes[2].pie(demo_data, labels=demo_labels, autopct='%1.1f%%', colors=['#e74c3c', '#2ecc71'], explode=[0.02, 0.02])
axes[2].set_title('Demographic Updates by Age Group\n(Total: {:,})'.format(sum(demo_data)))

plt.tight_layout()
plt.savefig(output_dir / '01_age_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 01_age_distribution.png")

# =============================================================================
# VISUALIZATION 2: Top 15 States - Enrollment
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Enrollment by State
state_enroll = enrollment_df.groupby("state_clean")["total"].sum().sort_values(ascending=True).tail(15)
axes[0].barh(state_enroll.index, state_enroll.values, color='#3498db')
axes[0].set_xlabel('Total Enrollments')
axes[0].set_title('Top 15 States by Enrollment')
for i, v in enumerate(state_enroll.values):
    axes[0].text(v + 10000, i, f'{v:,.0f}', va='center', fontsize=9)

# Biometric by State
state_bio = biometric_df.groupby("state_clean")["total"].sum().sort_values(ascending=True).tail(15)
axes[1].barh(state_bio.index, state_bio.values, color='#e74c3c')
axes[1].set_xlabel('Total Biometric Updates')
axes[1].set_title('Top 15 States by Biometric Updates')
for i, v in enumerate(state_bio.values):
    axes[1].text(v + 50000, i, f'{v:,.0f}', va='center', fontsize=9)

# Demographic by State
state_demo = demographic_df.groupby("state_clean")["total"].sum().sort_values(ascending=True).tail(15)
axes[2].barh(state_demo.index, state_demo.values, color='#2ecc71')
axes[2].set_xlabel('Total Demographic Updates')
axes[2].set_title('Top 15 States by Demographic Updates')
for i, v in enumerate(state_demo.values):
    axes[2].text(v + 50000, i, f'{v:,.0f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / '02_top_states.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 02_top_states.png")

# =============================================================================
# VISUALIZATION 3: Time Series Analysis
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Daily enrollment trend
daily_enroll = enrollment_df.groupby("date")["total"].sum()
axes[0].plot(daily_enroll.index, daily_enroll.values, color='#3498db', linewidth=1.5)
axes[0].fill_between(daily_enroll.index, daily_enroll.values, alpha=0.3)
axes[0].set_title('Daily Enrollment Trend')
axes[0].set_ylabel('Enrollments')
axes[0].axhline(daily_enroll.mean(), color='red', linestyle='--', label=f'Mean: {daily_enroll.mean():,.0f}')
axes[0].legend()

# Daily biometric trend
daily_bio = biometric_df.groupby("date")["total"].sum()
axes[1].plot(daily_bio.index, daily_bio.values, color='#e74c3c', linewidth=1.5)
axes[1].fill_between(daily_bio.index, daily_bio.values, alpha=0.3, color='#e74c3c')
axes[1].set_title('Daily Biometric Update Trend')
axes[1].set_ylabel('Updates')
axes[1].axhline(daily_bio.mean(), color='blue', linestyle='--', label=f'Mean: {daily_bio.mean():,.0f}')
axes[1].legend()

# Daily demographic trend
daily_demo = demographic_df.groupby("date")["total"].sum()
axes[2].plot(daily_demo.index, daily_demo.values, color='#2ecc71', linewidth=1.5)
axes[2].fill_between(daily_demo.index, daily_demo.values, alpha=0.3, color='#2ecc71')
axes[2].set_title('Daily Demographic Update Trend')
axes[2].set_ylabel('Updates')
axes[2].set_xlabel('Date')
axes[2].axhline(daily_demo.mean(), color='blue', linestyle='--', label=f'Mean: {daily_demo.mean():,.0f}')
axes[2].legend()

plt.tight_layout()
plt.savefig(output_dir / '03_time_series.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 03_time_series.png")

# =============================================================================
# VISUALIZATION 4: Monthly Trends
# =============================================================================
enrollment_df["month"] = enrollment_df["date"].dt.month
biometric_df["month"] = biometric_df["date"].dt.month
demographic_df["month"] = demographic_df["date"].dt.month

fig, ax = plt.subplots(figsize=(12, 6))

monthly_enroll = enrollment_df.groupby("month")["total"].sum()
monthly_bio = biometric_df.groupby("month")["total"].sum()
monthly_demo = demographic_df.groupby("month")["total"].sum()

x = np.arange(len(monthly_enroll))
width = 0.25

bars1 = ax.bar(x - width, monthly_enroll.values, width, label='Enrollments', color='#3498db')
bars2 = ax.bar(x, monthly_bio.values, width, label='Biometric Updates', color='#e74c3c')
bars3 = ax.bar(x + width, monthly_demo.values, width, label='Demographic Updates', color='#2ecc71')

ax.set_xlabel('Month')
ax.set_ylabel('Count')
ax.set_title('Monthly Comparison: Enrollments vs Updates')
ax.set_xticks(x)
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticklabels([month_names[m-1] for m in monthly_enroll.index])
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '04_monthly_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 04_monthly_comparison.png")

# =============================================================================
# VISUALIZATION 5: Day of Week Analysis
# =============================================================================
enrollment_df["day_of_week"] = enrollment_df["date"].dt.day_name()
biometric_df["day_of_week"] = biometric_df["date"].dt.day_name()
demographic_df["day_of_week"] = demographic_df["date"].dt.day_name()

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, (df, title, color) in enumerate([
    (enrollment_df, 'Enrollments', '#3498db'),
    (biometric_df, 'Biometric Updates', '#e74c3c'),
    (demographic_df, 'Demographic Updates', '#2ecc71')
]):
    dow_data = df.groupby("day_of_week")["total"].sum().reindex(day_order)
    axes[i].bar(range(7), dow_data.values, color=color)
    axes[i].set_xticks(range(7))
    axes[i].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    axes[i].set_title(f'{title} by Day of Week')
    axes[i].set_ylabel('Total Count')

plt.tight_layout()
plt.savefig(output_dir / '05_day_of_week.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 05_day_of_week.png")

# =============================================================================
# VISUALIZATION 6: Heatmap - State vs Month
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 12))

# Top 20 states for enrollment
top_states = enrollment_df.groupby("state_clean")["total"].sum().sort_values(ascending=False).head(20).index
state_month = enrollment_df[enrollment_df["state_clean"].isin(top_states)].pivot_table(
    values="total", index="state_clean", columns="month", aggfunc="sum"
)

sns.heatmap(state_month, cmap='YlOrRd', annot=False, fmt='.0f', ax=ax)
ax.set_title('Enrollment Heatmap: Top 20 States vs Month')
ax.set_xlabel('Month')
ax.set_ylabel('State')
# Dynamic month labels based on actual columns
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticklabels([month_names[int(m)-1] for m in state_month.columns])

plt.tight_layout()
plt.savefig(output_dir / '06_heatmap_state_month.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 06_heatmap_state_month.png")

# =============================================================================
# VISUALIZATION 7: Update Ratio Analysis
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Calculate bio vs demo ratio by state
state_bio_sum = biometric_df.groupby("state_clean")["total"].sum()
state_demo_sum = demographic_df.groupby("state_clean")["total"].sum()

# Merge and get ratio
combined = pd.DataFrame({
    'Biometric': state_bio_sum,
    'Demographic': state_demo_sum
}).dropna()
combined['Ratio'] = combined['Biometric'] / combined['Demographic']
combined = combined.sort_values('Ratio', ascending=True).tail(20)

combined[['Biometric', 'Demographic']].plot(kind='barh', ax=ax, color=['#e74c3c', '#2ecc71'])
ax.set_xlabel('Total Updates')
ax.set_title('Top 20 States: Biometric vs Demographic Updates')
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / '07_bio_vs_demo.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 07_bio_vs_demo.png")

# =============================================================================
# VISUALIZATION 8: Anomaly Detection - Z-Score
# =============================================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Daily enrollment with z-score highlighting
daily_enroll = enrollment_df.groupby("date")["total"].sum()
z_scores = (daily_enroll - daily_enroll.mean()) / daily_enroll.std()
anomalies = abs(z_scores) > 2

axes[0].plot(daily_enroll.index, daily_enroll.values, color='#3498db', linewidth=1)
axes[0].scatter(daily_enroll.index[anomalies], daily_enroll.values[anomalies], 
                color='red', s=50, zorder=5, label='Anomalies (|z| > 2)')
axes[0].axhline(daily_enroll.mean() + 2*daily_enroll.std(), color='orange', linestyle='--', label='Upper bound')
axes[0].axhline(daily_enroll.mean() - 2*daily_enroll.std(), color='orange', linestyle='--', label='Lower bound')
axes[0].set_title('Enrollment Anomaly Detection (Z-Score Method)')
axes[0].set_ylabel('Daily Enrollments')
axes[0].legend()

# Biometric anomalies
daily_bio = biometric_df.groupby("date")["total"].sum()
z_scores_bio = (daily_bio - daily_bio.mean()) / daily_bio.std()
anomalies_bio = abs(z_scores_bio) > 2

axes[1].plot(daily_bio.index, daily_bio.values, color='#e74c3c', linewidth=1)
axes[1].scatter(daily_bio.index[anomalies_bio], daily_bio.values[anomalies_bio], 
                color='blue', s=50, zorder=5, label='Anomalies (|z| > 2)')
axes[1].axhline(daily_bio.mean() + 2*daily_bio.std(), color='orange', linestyle='--', label='Upper bound')
axes[1].axhline(daily_bio.mean() - 2*daily_bio.std(), color='orange', linestyle='--', label='Lower bound')
axes[1].set_title('Biometric Update Anomaly Detection (Z-Score Method)')
axes[1].set_ylabel('Daily Updates')
axes[1].set_xlabel('Date')
axes[1].legend()

plt.tight_layout()
plt.savefig(output_dir / '08_anomaly_detection.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 08_anomaly_detection.png")

# =============================================================================
# VISUALIZATION 9: District-Level Concentration
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Top 30 districts by enrollment
district_enroll = enrollment_df.groupby(["state_clean", "district"])["total"].sum().sort_values(ascending=False).head(30)
district_labels = [f"{d} ({s[:3]})" for s, d in district_enroll.index]

ax.barh(range(30), district_enroll.values, color='#9b59b6')
ax.set_yticks(range(30))
ax.set_yticklabels(district_labels, fontsize=8)
ax.set_xlabel('Total Enrollments')
ax.set_title('Top 30 Districts by Enrollment')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(output_dir / '09_top_districts.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 09_top_districts.png")

# =============================================================================
# VISUALIZATION 10: Summary Dashboard
# =============================================================================
fig = plt.figure(figsize=(16, 12))

# Create grid
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Summary stats
ax1 = fig.add_subplot(gs[0, 0])
stats = ['Enrollments', 'Biometric', 'Demographic']
values = [enrollment_df["total"].sum(), biometric_df["total"].sum(), demographic_df["total"].sum()]
colors = ['#3498db', '#e74c3c', '#2ecc71']
bars = ax1.bar(stats, values, color=colors)
ax1.set_title('Total Volume by Category')
ax1.set_ylabel('Count')
for bar, val in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500000, 
             f'{val/1e6:.1f}M', ha='center', fontsize=10)

# Age distribution for enrollment
ax2 = fig.add_subplot(gs[0, 1])
age_data = [enrollment_df["age_0_5"].sum(), enrollment_df["age_5_17"].sum(), enrollment_df["age_18_greater"].sum()]
ax2.pie(age_data, labels=['0-5', '5-17', '18+'], autopct='%1.1f%%', colors=['#3498db', '#e74c3c', '#2ecc71'])
ax2.set_title('Enrollment Age Distribution')

# Records per dataset
ax3 = fig.add_subplot(gs[0, 2])
records = [len(enrollment_df), len(biometric_df), len(demographic_df)]
ax3.bar(['Enrollment', 'Biometric', 'Demographic'], records, color=['#3498db', '#e74c3c', '#2ecc71'])
ax3.set_title('Records per Dataset')
ax3.set_ylabel('Row Count')

# Monthly trend
ax4 = fig.add_subplot(gs[1, :])
monthly_enroll = enrollment_df.groupby("month")["total"].sum()
monthly_bio = biometric_df.groupby("month")["total"].sum()
monthly_demo = demographic_df.groupby("month")["total"].sum()
ax4.plot(monthly_enroll.index, monthly_enroll.values, 'o-', label='Enrollments', color='#3498db')
ax4.plot(monthly_bio.index, monthly_bio.values, 's-', label='Biometric', color='#e74c3c')
ax4.plot(monthly_demo.index, monthly_demo.values, '^-', label='Demographic', color='#2ecc71')
ax4.set_title('Monthly Trends Comparison')
ax4.set_xlabel('Month')
ax4.set_ylabel('Count')
ax4.legend()
ax4.set_xticks(range(3, 13))
ax4.set_xticklabels(['Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax4.grid(True, alpha=0.3)

# Top 10 states
ax5 = fig.add_subplot(gs[2, :2])
top_states = enrollment_df.groupby("state_clean")["total"].sum().sort_values(ascending=True).tail(10)
ax5.barh(top_states.index, top_states.values, color='#3498db')
ax5.set_title('Top 10 States by Enrollment')
ax5.set_xlabel('Total Enrollments')

# Key metrics
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')
metrics_text = f"""
KEY METRICS
-----------
Data Period: Mar - Dec 2025

Total Records: {len(enrollment_df) + len(biometric_df) + len(demographic_df):,}

States/UTs: {enrollment_df['state_clean'].nunique()}
Districts: {enrollment_df['district'].nunique()}
Pincodes: {enrollment_df['pincode'].nunique():,}

Peak Enrollment Day: Jul 1
Peak Biometric Day: Jul 1

Duplicates Found:
  - Enrollment: 22,957
  - Biometric: 94,896
  - Demographic: 473,601
"""
ax6.text(0.1, 0.9, metrics_text, transform=ax6.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('UIDAI Aadhaar Data Analysis - Summary Dashboard', fontsize=16, fontweight='bold', y=0.98)
plt.savefig(output_dir / '10_summary_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 10_summary_dashboard.png")

print("\n" + "="*60)
print(f"All visualizations saved to: {output_dir}")
print("="*60)
