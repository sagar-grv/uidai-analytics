"""
UIDAI Aadhaar Data - Forensic & Behavioral Analysis
====================================================
Advanced statistical tests for "hidden" insights: Benford's Law, Entropy,
Gini Coefficients, and Lead-Lag Correlation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chisquare, entropy
from scipy.signal import correlate
import matplotlib.pyplot as plt
import math
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
print("1. BENFORD'S LAW ANALYSIS (FORENSIC AUDIT)")
print("=" * 80)
print("Testing if the daily transaction volumes follow natural numeric distribution patterns.")
print("Benford's Law states that leading digits (1-9) appear with specific probabilities in natural data.")
print("Deviation suggests: Artificial data generation, arbitrary thresholds, or operational constraints.")

def benford_test(df, name):
    # Extract leading digit from 'total' column
    # We use daily totals to see if DAILY VOLUME follows Benford
    daily_totals = df.groupby("date")["total"].sum()
    leading_digits = daily_totals.astype(str).str[0].astype(int)
    
    # Remove zeros (Benford doesn't apply)
    leading_digits = leading_digits[leading_digits != 0]
    
    observed_counts = leading_digits.value_counts().sort_index()
    total_count = observed_counts.sum()
    
    # Expected proportions according to Benford's Law
    expected_probs = np.log10(1 + 1 / np.arange(1, 10))
    expected_counts = expected_probs * total_count
    
    # Ensure all digits 1-9 are present in observed, fill 0 if missing
    observed_aligned = pd.Series(0, index=np.arange(1, 10))
    observed_aligned.update(observed_counts)
    
    # Chi-square test
    chi2_stat, p_val = chisquare(observed_aligned, expected_counts)
    
    print(f"\n{name} Dataset Analysis:")
    print(f"  Observed Digits Distribution (1-9): {observed_aligned.values}")
    print(f"  Chi-Square Statistic: {chi2_stat:.2f}")
    print(f"  P-value: {p_val:.4f}")
    
    status = "✅ PASS (Natural Distribution)" if p_val > 0.05 else "⚠️ FAIL (Potential Anomaly/Artificial)"
    print(f"  Result: {status}")
    
    # Calculate Mean Absolute Deviation (MAD) - robust metric
    mad = np.mean(np.abs(observed_aligned/total_count - expected_probs))
    print(f"  MAD (Mean Absolute Deviation): {mad:.4f}")
    if mad > 0.015:
        print("    -> High deviation suggests data might be constrained or synthetic.")

benford_test(enrollment_df, "Enrollment")
benford_test(biometric_df, "Biometric")
benford_test(demographic_df, "Demographic")

print("\n" + "=" * 80)
print("2. GEOSPATIAL CLUSTERING (PINCODE DECODING)")
print("=" * 80)
print("Using first 3 digits of Pincode to identify 'Regional Zones' independent of State names.")
print("This solves the 'West Bengal' spelling issue and finds sub-state hotspots.")

def pincode_clustering(df, name):
    # Create Pincode Zone (First 3 digits)
    # Filter valid pincodes first (6 digits)
    valid_pincodes = df[df["pincode"].astype(str).str.match(r'^\d{6}$')]
    valid_pincodes["pincode_zone"] = valid_pincodes["pincode"].astype(str).str[:3]
    
    zone_counts = valid_pincodes.groupby("pincode_zone")["total"].sum().sort_values(ascending=False)
    
    print(f"\n{name} - Top 5 High-Activity Zones:")
    for zone, count in zone_counts.head(5).items():
        # Get Sample state for context
        sample_state = valid_pincodes[valid_pincodes["pincode_zone"] == zone]["state"].iloc[0]
        pct = count / zone_counts.sum() * 100
        print(f"  Zone {zone}xxx ({sample_state}): {count:,} ({pct:.2f}%)")
        
    # Concentration Analysis
    top_10_pct = zone_counts.head(10).sum() / zone_counts.sum() * 100
    print(f"  Top 10 Zones share: {top_10_pct:.2f}% of total volume")

pincode_clustering(enrollment_df, "Enrollment")
pincode_clustering(biometric_df, "Biometric")

print("\n" + "=" * 80)
print("3. OPERATIONAL ENTROPY (SYSTEM RANDOMNESS)")
print("=" * 80)
print("Measuring the entropy of daily transaction volumes.")
print("Low Entropy = Highly predictable/regular (potential bot activity/batch processing)")
print("High Entropy = Chaotic/Unpredictable (ad-hoc user driven)")

def calculate_entropy(df, name):
    daily = df.groupby("date")["total"].sum()
    # Normalize to probabilities
    probs = daily / daily.sum()
    ent = entropy(probs)
    
    # Max possible entropy (uniform distribution)
    max_ent = np.log(len(daily))
    
    normalized_ent = ent / max_ent
    
    print(f"\n{name}:")
    print(f"  Shannon Entropy: {ent:.4f}")
    print(f"  Normalized Entropy (0-1): {normalized_ent:.4f}")
    
    if normalized_ent < 0.8:
        print("  -> LOW ENTROPY: Operations are unusually regular/spikey.")
    else:
        print("  -> HIGH ENTROPY: Operations are well-distributed/organic.")

calculate_entropy(enrollment_df, "Enrollment")
calculate_entropy(biometric_df, "Biometric")
calculate_entropy(demographic_df, "Demographic")

print("\n" + "=" * 80)
print("4. WORKLOAD INEQUALITY (LORENZ & GINI)")
print("=" * 80)
print("Exact calculation of infrastructure load imbalance across Districts.")
print("Gini Coefficient: 0 = Perfect Equality, 1 = Perfect Inequality")

def calculate_gini(df, name):
    # District level aggregation
    dist_counts = df.groupby(["state", "district"])["total"].sum().sort_values()
    values = dist_counts.values
    
    # Lorenz curve
    cumulative_values = np.cumsum(values)
    cumulative_values_norm = cumulative_values / cumulative_values[-1]
    
    # Gini coefficient calculation
    n = len(values)
    gini = (2 * np.sum(np.arange(1, n + 1) * values)) / (n * np.sum(values)) - (n + 1) / n
    
    print(f"\n{name} District Inequality:")
    print(f"  Gini Coefficient: {gini:.4f}")
    
    if gini > 0.6:
        print("  -> HIGH INEQUALITY: A few districts handle most of the load.")
    else:
        print("  -> LOW INEQUALITY: Load is evenly distributed.")
        
    # What % of districts handle 80% of volume?
    total_vol = values.sum()
    target = 0.8 * total_vol
    running = 0
    count = 0
    # Reverse iterate
    for v in values[::-1]:
        running += v
        count += 1
        if running >= target:
            break
            
    pct_districts = count / n * 100
    print(f"  Pareto Check: {pct_districts:.1f}% of districts handle 80% of volume.")

calculate_gini(enrollment_df, "Enrollment")
calculate_gini(biometric_df, "Biometric")

print("\n" + "=" * 80)
print("5. LEAD-LAG ANALYSIS (CORRECTION PATTERNS)")
print("=" * 80)
print("Does a spike in Enrollment lead to a spike in Demographic Updates later?")
print("Testing time-lag correlation (Cross-Correlation).")

# Align dates
daily_enroll = enrollment_df.groupby("date")["total"].sum()
daily_demo = demographic_df.groupby("date")["total"].sum()

# Reindex to full range to handle missing dates (fill with 0)
full_range = pd.date_range(start=min(daily_enroll.index.min(), daily_demo.index.min()),
                           end=max(daily_enroll.index.max(), daily_demo.index.max()))
daily_enroll = daily_enroll.reindex(full_range, fill_value=0)
daily_demo = daily_demo.reindex(full_range, fill_value=0)

# Normalize
norm_enroll = (daily_enroll - daily_enroll.mean()) / daily_enroll.std()
norm_demo = (daily_demo - daily_demo.mean()) / daily_demo.std()

# Cross correlation
lags = np.arange(-30, 31) # +/- 30 days
corr = [norm_enroll.corr(norm_demo.shift(lag)) for lag in lags]
max_corr = max(corr)
max_lag = lags[np.argmax(corr)]

print("\nEnrollment vs. Demographic Update Correlation:")
print(f"  Max Correlation: {max_corr:.4f}")
print(f"  At Lag: {max_lag} days")

if max_lag > 0:
    print(f"  -> Interpretation: Updates tend to follow Enrollments by {max_lag} days.")
elif max_lag < 0:
    print(f"  -> Interpretation: Enrollments tend to follow Updates by {abs(max_lag)} days (Unlikely).")
else:
    print("  -> Interpretation: Events happen simultaneously (Synchronized camps?).")

print("\n" + "=" * 80)
print("ADVANCED FORENSIC ANALYSIS COMPLETE")
print("=" * 80)
