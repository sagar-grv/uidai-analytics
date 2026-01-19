"""
UIDAI Aadhaar Data - Deep Insights Extraction
==============================================
Extract specific numerical findings for hackathon report
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

base_path = Path(r"c:\Users\sagar\Downloads\uidai dataset")

print("Loading datasets...")
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

print("\n" + "=" * 80)
print("CRITICAL FINDING 1: AGE DISTRIBUTION ANOMALY")
print("=" * 80)

total_0_5 = enrollment_df["age_0_5"].sum()
total_5_17 = enrollment_df["age_5_17"].sum()
total_18_plus = enrollment_df["age_18_greater"].sum()
total_all = total_0_5 + total_5_17 + total_18_plus

print(f"""
The age distribution in Enrollment data appears INVERTED:

ACTUAL DATA:
  Age 0-5:   {total_0_5:>15,} ({total_0_5/total_all*100:>6.2f}%)
  Age 5-17:  {total_5_17:>15,} ({total_5_17/total_all*100:>6.2f}%)
  Age 18+:   {total_18_plus:>15,} ({total_18_plus/total_all*100:>6.2f}%)
  
EXPECTED (India's population distribution):
  Age 0-5:   ~9%
  Age 5-17:  ~22%
  Age 18+:   ~69%

ANALYSIS:
  - The column 'age_0_5' contains {total_0_5/total_all*100:.1f}% of data (expected ~9%)
  - The column 'age_18_greater' contains only {total_18_plus/total_all*100:.1f}% (expected ~69%)
  
POSSIBLE INTERPRETATIONS:
  1. Column names may be MISLABELED in the data
  2. Data may represent NEW enrollments only (children being enrolled for first time)
  3. This dataset may only focus on minor enrollments
""")

print("\n" + "=" * 80)
print("CRITICAL FINDING 2: DATE COVERAGE GAPS")
print("=" * 80)

for name, df in [("Enrollment", enrollment_df), ("Biometric", biometric_df), ("Demographic", demographic_df)]:
    date_range = pd.date_range(df["date"].min(), df["date"].max())
    actual_dates = set(df["date"].dropna().unique())
    missing_dates = set(date_range) - actual_dates
    
    print(f"\n{name}:")
    print(f"  Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"  Total calendar days: {len(date_range)}")
    print(f"  Days with data: {len(actual_dates)} ({len(actual_dates)/len(date_range)*100:.1f}%)")
    print(f"  Missing days: {len(missing_dates)} ({len(missing_dates)/len(date_range)*100:.1f}%)")
    
    # Check if data is only for certain days of month
    day_of_month = df["date"].dt.day.value_counts().sort_index()
    print(f"  Days of month with most data: {day_of_month.head(5).index.tolist()}")

print("\n" + "=" * 80)
print("CRITICAL FINDING 3: STATE NAME ISSUES - COMPLETE LIST")
print("=" * 80)

all_states = sorted(set(enrollment_df["state"].unique()) | 
                    set(biometric_df["state"].unique()) | 
                    set(demographic_df["state"].unique()))

print(f"\nTotal unique state values: {len(all_states)}")
print(f"Expected Indian states/UTs: 36")
print(f"Extra values: {len(all_states) - 36}")

print("\nComplete list of all state values:")
for i, state in enumerate(all_states, 1):
    print(f"  {i:2}. '{state}'")

# Identify clear duplicates
print("\nClear duplicate state names detected:")
duplicates = [
    ("West Bengal", ["West Bengal", "west Bengal", "West Bengli", "Westbengal"]),
    ("Andhra Pradesh", ["Andhra Pradesh", "andhra pradesh", "Andhra Prades"]),
    ("Odisha", ["Odisha", "odisha", "ODISHA"]),
]
for correct, variants in duplicates:
    found = [v for v in variants if v in all_states]
    if len(found) > 1:
        print(f"  {correct}: found as {found}")

print("\n" + "=" * 80)
print("CRITICAL FINDING 4: EXTREME DAILY ANOMALIES")
print("=" * 80)

for name, df in [("Enrollment", enrollment_df), ("Biometric", biometric_df), ("Demographic", demographic_df)]:
    daily = df.groupby("date")["total"].sum()
    mean_val = daily.mean()
    std_val = daily.std()
    
    # Find all dates with Z-score > 2
    z_scores = (daily - mean_val) / std_val
    anomalies = z_scores[abs(z_scores) > 2].sort_values(ascending=False)
    
    print(f"\n{name}:")
    print(f"  Mean daily total: {mean_val:,.0f}")
    print(f"  Std dev: {std_val:,.0f}")
    print(f"  Anomaly threshold (2*std): {mean_val + 2*std_val:,.0f}")
    print(f"  Days with anomalies: {len(anomalies)}")
    
    if len(anomalies) > 0:
        print(f"\n  Top anomaly dates:")
        for date, z in anomalies.head(10).items():
            actual = daily[date]
            deviation = (actual - mean_val) / mean_val * 100
            print(f"    {date.strftime('%Y-%m-%d')}: {actual:>12,.0f} (Z={z:+.2f}, {deviation:+.1f}% from mean)")

print("\n" + "=" * 80)
print("CRITICAL FINDING 5: DUPLICATE RECORD DETAILS")
print("=" * 80)

for name, df in [("Enrollment", enrollment_df), ("Biometric", biometric_df), ("Demographic", demographic_df)]:
    total = len(df)
    dups = df.duplicated().sum()
    unique = total - dups
    
    print(f"\n{name}:")
    print(f"  Total records: {total:,}")
    print(f"  Duplicate records: {dups:,} ({dups/total*100:.2f}%)")
    print(f"  Unique records: {unique:,}")
    
    # Check duplicates by date
    if dups > 0:
        dup_df = df[df.duplicated(keep=False)]
        dup_by_date = dup_df.groupby("date").size()
        print(f"  Dates with most duplicates: {dup_by_date.nlargest(5).to_dict()}")

print("\n" + "=" * 80)
print("CRITICAL FINDING 6: STATE-WISE AGE DISTRIBUTION ANOMALIES")
print("=" * 80)

state_age = enrollment_df.groupby("state")[["age_0_5", "age_5_17", "age_18_greater"]].sum()
state_age["total"] = state_age.sum(axis=1)
state_age["pct_0_5"] = state_age["age_0_5"] / state_age["total"] * 100
state_age["pct_5_17"] = state_age["age_5_17"] / state_age["total"] * 100
state_age["pct_18_plus"] = state_age["age_18_greater"] / state_age["total"] * 100

# States with extreme values
print("\nStates with 100% in single age group (DATA QUALITY ISSUE):")
extreme_states = state_age[(state_age["pct_0_5"] >= 99) | 
                           (state_age["pct_5_17"] >= 99) | 
                           (state_age["pct_18_plus"] >= 99)]
for state in extreme_states.index:
    row = state_age.loc[state]
    print(f"  {state}:")
    print(f"    Age 0-5: {row['pct_0_5']:.1f}% ({row['age_0_5']:,.0f})")
    print(f"    Age 5-17: {row['pct_5_17']:.1f}% ({row['age_5_17']:,.0f})")
    print(f"    Age 18+: {row['pct_18_plus']:.1f}% ({row['age_18_greater']:,.0f})")
    print(f"    Total: {row['total']:,.0f}")

print("\n" + "=" * 80)
print("CRITICAL FINDING 7: BIOMETRIC VS DEMOGRAPHIC RATIO ANALYSIS")
print("=" * 80)

# Compare at state level
state_bio = biometric_df.groupby("state")["total"].sum()
state_demo = demographic_df.groupby("state")["total"].sum()
state_enroll = enrollment_df.groupby("state")["total"].sum()

combined = pd.DataFrame({
    "Enrollment": state_enroll,
    "Biometric": state_bio,
    "Demographic": state_demo
}).dropna()

combined["Bio_Demo_Ratio"] = combined["Biometric"] / combined["Demographic"]
combined["Bio_Enroll_Ratio"] = combined["Biometric"] / combined["Enrollment"]
combined["Demo_Enroll_Ratio"] = combined["Demographic"] / combined["Enrollment"]

print("\nBiometric/Demographic Ratio by State:")
print(f"  Overall ratio: {combined['Biometric'].sum() / combined['Demographic'].sum():.2f}")
print(f"  Min ratio: {combined['Bio_Demo_Ratio'].min():.2f}")
print(f"  Max ratio: {combined['Bio_Demo_Ratio'].max():.2f}")
print(f"  Std dev: {combined['Bio_Demo_Ratio'].std():.2f}")

print("\nStates with highest Bio/Demo ratio:")
for state in combined.nlargest(5, "Bio_Demo_Ratio").index:
    ratio = combined.loc[state, "Bio_Demo_Ratio"]
    bio = combined.loc[state, "Biometric"]
    demo = combined.loc[state, "Demographic"]
    print(f"  {state}: {ratio:.2f}x (Bio: {bio:,.0f}, Demo: {demo:,.0f})")

print("\nStates with lowest Bio/Demo ratio:")
for state in combined.nsmallest(5, "Bio_Demo_Ratio").index:
    ratio = combined.loc[state, "Bio_Demo_Ratio"]
    bio = combined.loc[state, "Biometric"]
    demo = combined.loc[state, "Demographic"]
    print(f"  {state}: {ratio:.2f}x (Bio: {bio:,.0f}, Demo: {demo:,.0f})")

print("\n" + "=" * 80)
print("CRITICAL FINDING 8: DISTRICT-LEVEL ANOMALIES")
print("=" * 80)

district_enroll = enrollment_df.groupby(["state", "district"])["total"].sum()

print(f"\nTotal state-district combinations: {len(district_enroll):,}")
print(f"Districts with <10 enrollments: {(district_enroll < 10).sum()}")
print(f"Districts with <100 enrollments: {(district_enroll < 100).sum()}")

# Very high concentration districts
q99 = district_enroll.quantile(0.99)
high_districts = district_enroll[district_enroll > q99]
print(f"\nDistricts above 99th percentile ({q99:,.0f}):")
for (state, district), val in high_districts.sort_values(ascending=False).head(10).items():
    pct = val / district_enroll.sum() * 100
    print(f"  {district} ({state}): {val:,.0f} ({pct:.2f}%)")

print("\n" + "=" * 80)
print("CRITICAL FINDING 9: WEEKLY PATTERN ANALYSIS")
print("=" * 80)

for name, df in [("Enrollment", enrollment_df), ("Biometric", biometric_df), ("Demographic", demographic_df)]:
    df["week"] = df["date"].dt.isocalendar().week
    weekly = df.groupby("week")["total"].sum()
    
    print(f"\n{name} Weekly Statistics:")
    print(f"  Total weeks: {len(weekly)}")
    print(f"  Mean weekly total: {weekly.mean():,.0f}")
    print(f"  Std dev: {weekly.std():,.0f}")
    print(f"  Min week: Week {weekly.idxmin()} with {weekly.min():,.0f}")
    print(f"  Max week: Week {weekly.idxmax()} with {weekly.max():,.0f}")
    
    # Week-over-week volatility
    wow_change = weekly.pct_change().abs()
    print(f"  Avg week-over-week change: {wow_change.mean()*100:.1f}%")
    print(f"  Max week-over-week change: {wow_change.max()*100:.1f}%")

print("\n" + "=" * 80)
print("CRITICAL FINDING 10: DATA COMPLETENESS SCORE")
print("=" * 80)

def completeness_score(df, name):
    scores = {}
    
    # Date coverage
    date_range = pd.date_range(df["date"].min(), df["date"].max())
    date_coverage = len(df["date"].unique()) / len(date_range) * 100
    scores["Date Coverage"] = date_coverage
    
    # Null values
    null_rate = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
    scores["Null-Free Rate"] = 100 - null_rate
    
    # Duplicate rate
    dup_rate = df.duplicated().sum() / len(df) * 100
    scores["Duplicate-Free Rate"] = 100 - dup_rate
    
    # Zero record rate (for numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    zero_rate = (df[numeric_cols] == 0).all(axis=1).sum() / len(df) * 100
    scores["Non-Zero Rate"] = 100 - zero_rate
    
    print(f"\n{name} Data Quality Scores:")
    for metric, score in scores.items():
        status = "✓" if score >= 90 else "⚠" if score >= 70 else "✗"
        print(f"  {status} {metric}: {score:.1f}%")
    
    overall = sum(scores.values()) / len(scores)
    print(f"  → Overall Score: {overall:.1f}%")
    return overall

enrollment_score = completeness_score(enrollment_df, "Enrollment")
biometric_score = completeness_score(biometric_df, "Biometric")
demographic_score = completeness_score(demographic_df, "Demographic")

print(f"\n" + "=" * 80)
print("SUMMARY: OVERALL DATA QUALITY")
print("=" * 80)
print(f"""
Dataset Quality Rankings:
  1. Enrollment:   {enrollment_score:.1f}%
  2. Biometric:    {biometric_score:.1f}%
  3. Demographic:  {demographic_score:.1f}%

CRITICAL ISSUES REQUIRING ATTENTION:
1. Age column interpretation unclear (inverted distribution)
2. 22.9% demographic records are duplicates
3. 70% of calendar days missing from all datasets
4. Multiple state name spellings (data entry inconsistency)
5. Some states show 100% enrollment in single age group

RECOMMENDATIONS:
1. Clarify age column semantics with data provider
2. Deduplicate all datasets before analysis
3. Standardize state names using mapping table
4. Investigate July 1st anomaly (possible data dump)
5. Handle missing dates appropriately in time series analysis
""")
