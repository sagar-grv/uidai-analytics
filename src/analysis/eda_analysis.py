"""
UIDAI Aadhaar Data Analysis - Exploratory Data Analysis
========================================================
Hackathon: Unlocking Societal Trends in Aadhaar Enrolment and Updates

This script performs comprehensive EDA on:
1. Aadhaar Enrollment Data
2. Biometric Update Data  
3. Demographic Update Data
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
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# =============================================================================
# DATA LOADING
# =============================================================================
print("=" * 60)
print("LOADING DATASETS")
print("=" * 60)

base_path = Path(r"c:\Users\sagar\Downloads\uidai dataset")

# Load Enrollment Data
enrollment_path = base_path / "api_data_aadhar_enrolment" / "api_data_aadhar_enrolment"
enrollment_files = list(enrollment_path.glob("*.csv"))
print(f"\nEnrollment files found: {len(enrollment_files)}")
enrollment_df = pd.concat([pd.read_csv(f) for f in enrollment_files], ignore_index=True)
print(f"Enrollment records loaded: {len(enrollment_df):,}")

# Load Biometric Update Data
biometric_path = base_path / "api_data_aadhar_biometric" / "api_data_aadhar_biometric"
biometric_files = list(biometric_path.glob("*.csv"))
print(f"\nBiometric files found: {len(biometric_files)}")
biometric_df = pd.concat([pd.read_csv(f) for f in biometric_files], ignore_index=True)
print(f"Biometric records loaded: {len(biometric_df):,}")

# Load Demographic Update Data
demographic_path = base_path / "api_data_aadhar_demographic" / "api_data_aadhar_demographic"
demographic_files = list(demographic_path.glob("*.csv"))
print(f"\nDemographic files found: {len(demographic_files)}")
demographic_df = pd.concat([pd.read_csv(f) for f in demographic_files], ignore_index=True)
print(f"Demographic records loaded: {len(demographic_df):,}")

# =============================================================================
# BASIC DATA EXPLORATION
# =============================================================================
print("\n" + "=" * 60)
print("BASIC DATA EXPLORATION")
print("=" * 60)

def explore_dataset(df, name):
    """Perform basic exploration of a dataset"""
    print(f"\n{'='*40}")
    print(f"DATASET: {name}")
    print(f"{'='*40}")
    
    print(f"\n📊 Shape: {df.shape}")
    print(f"\n📋 Columns: {list(df.columns)}")
    print(f"\n📈 Data Types:")
    print(df.dtypes)
    
    print(f"\n🔍 First 5 rows:")
    print(df.head())
    
    print(f"\n📉 Statistical Summary:")
    print(df.describe())
    
    print(f"\n❓ Missing Values:")
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
    print(missing_df[missing_df['Missing Count'] > 0] if missing.sum() > 0 else "No missing values!")
    
    print(f"\n🔢 Unique Values per Column:")
    for col in df.columns:
        print(f"  {col}: {df[col].nunique():,}")
    
    return missing

# Explore each dataset
print("\n" + "=" * 60)
print("ENROLLMENT DATA EXPLORATION")
print("=" * 60)
enrollment_missing = explore_dataset(enrollment_df, "Enrollment")

print("\n" + "=" * 60)
print("BIOMETRIC UPDATE DATA EXPLORATION")
print("=" * 60)
biometric_missing = explore_dataset(biometric_df, "Biometric Update")

print("\n" + "=" * 60)
print("DEMOGRAPHIC UPDATE DATA EXPLORATION")
print("=" * 60)
demographic_missing = explore_dataset(demographic_df, "Demographic Update")

# =============================================================================
# DATA QUALITY ISSUES IDENTIFICATION
# =============================================================================
print("\n" + "=" * 60)
print("DATA QUALITY ISSUES IDENTIFICATION")
print("=" * 60)

issues = []

# Check 1: Date format and validity
print("\n🗓️ DATE ANALYSIS:")
for name, df in [("Enrollment", enrollment_df), ("Biometric", biometric_df), ("Demographic", demographic_df)]:
    print(f"\n{name} - Date column sample values:")
    print(df['date'].head(10).tolist())
    
    # Try to parse dates
    try:
        df['date_parsed'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
        invalid_dates = df['date_parsed'].isnull().sum()
        if invalid_dates > 0:
            issues.append(f"{name}: {invalid_dates:,} invalid dates found ({invalid_dates/len(df)*100:.2f}%)")
            print(f"  ⚠️ Invalid dates: {invalid_dates:,}")
        else:
            print(f"  ✅ All dates valid")
        
        # Date range
        print(f"  📅 Date range: {df['date_parsed'].min()} to {df['date_parsed'].max()}")
    except Exception as e:
        issues.append(f"{name}: Error parsing dates - {e}")
        print(f"  ❌ Error parsing dates: {e}")

# Check 2: State name consistency
print("\n🗺️ STATE NAME ANALYSIS:")
for name, df in [("Enrollment", enrollment_df), ("Biometric", biometric_df), ("Demographic", demographic_df)]:
    states = df['state'].unique()
    print(f"\n{name} - Unique states: {len(states)}")
    print(f"  States: {sorted(states)[:10]}...")  # First 10

# Check 3: Numeric columns for negative/zero values
print("\n🔢 NUMERIC VALUE ANALYSIS:")

# Enrollment numeric columns
enrollment_numeric_cols = ['age_0_5', 'age_5_17', 'age_18_greater']
print("\nEnrollment numeric columns:")
for col in enrollment_numeric_cols:
    if col in enrollment_df.columns:
        neg_count = (enrollment_df[col] < 0).sum()
        zero_count = (enrollment_df[col] == 0).sum()
        print(f"  {col}: Negative={neg_count:,}, Zero={zero_count:,}, Max={enrollment_df[col].max():,}, Min={enrollment_df[col].min()}")
        if neg_count > 0:
            issues.append(f"Enrollment {col}: {neg_count:,} negative values")

# Biometric numeric columns
biometric_numeric_cols = ['bio_age_5_17', 'bio_age_17_']
print("\nBiometric numeric columns:")
for col in biometric_numeric_cols:
    if col in biometric_df.columns:
        neg_count = (biometric_df[col] < 0).sum()
        zero_count = (biometric_df[col] == 0).sum()
        print(f"  {col}: Negative={neg_count:,}, Zero={zero_count:,}, Max={biometric_df[col].max():,}, Min={biometric_df[col].min()}")
        if neg_count > 0:
            issues.append(f"Biometric {col}: {neg_count:,} negative values")

# Demographic numeric columns
demographic_numeric_cols = ['demo_age_5_17', 'demo_age_17_']
print("\nDemographic numeric columns:")
for col in demographic_numeric_cols:
    if col in demographic_df.columns:
        neg_count = (demographic_df[col] < 0).sum()
        zero_count = (demographic_df[col] == 0).sum()
        print(f"  {col}: Negative={neg_count:,}, Zero={zero_count:,}, Max={demographic_df[col].max():,}, Min={demographic_df[col].min()}")
        if neg_count > 0:
            issues.append(f"Demographic {col}: {neg_count:,} negative values")

# Check 4: Pincode validity
print("\n📮 PINCODE ANALYSIS:")
for name, df in [("Enrollment", enrollment_df), ("Biometric", biometric_df), ("Demographic", demographic_df)]:
    # Convert to string for analysis
    pincodes = df['pincode'].astype(str)
    
    # Check length (Indian pincodes are 6 digits)
    invalid_length = pincodes.str.len() != 6
    invalid_count = invalid_length.sum()
    
    # Check for non-numeric
    non_numeric = ~pincodes.str.match(r'^\d{6}$')
    non_numeric_count = non_numeric.sum()
    
    print(f"\n{name}:")
    print(f"  Unique pincodes: {df['pincode'].nunique():,}")
    print(f"  Invalid length (not 6 digits): {invalid_count:,}")
    print(f"  Invalid format: {non_numeric_count:,}")
    
    if invalid_count > 0:
        issues.append(f"{name}: {invalid_count:,} pincodes with invalid length")

# Check 5: Duplicate records
print("\n🔄 DUPLICATE ANALYSIS:")
for name, df in [("Enrollment", enrollment_df), ("Biometric", biometric_df), ("Demographic", demographic_df)]:
    dup_count = df.duplicated().sum()
    print(f"{name}: {dup_count:,} duplicate rows ({dup_count/len(df)*100:.2f}%)")
    if dup_count > 0:
        issues.append(f"{name}: {dup_count:,} duplicate rows")

# Check 6: Outliers using IQR
print("\n📊 OUTLIER ANALYSIS (IQR Method):")

def detect_outliers_iqr(series, name):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((series < lower) | (series > upper)).sum()
    print(f"  {name}: {outliers:,} outliers (IQR bounds: [{lower:.0f}, {upper:.0f}])")
    return outliers

print("\nEnrollment outliers:")
for col in enrollment_numeric_cols:
    if col in enrollment_df.columns:
        detect_outliers_iqr(enrollment_df[col], col)

print("\nBiometric outliers:")
for col in biometric_numeric_cols:
    if col in biometric_df.columns:
        detect_outliers_iqr(biometric_df[col], col)

print("\nDemographic outliers:")
for col in demographic_numeric_cols:
    if col in demographic_df.columns:
        detect_outliers_iqr(demographic_df[col], col)

# =============================================================================
# SUMMARY OF DATA QUALITY ISSUES
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY OF DATA QUALITY ISSUES")
print("=" * 60)

if issues:
    print("\n⚠️ Issues Found:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("\n✅ No major data quality issues found!")

# =============================================================================
# KEY STATISTICS SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("KEY STATISTICS SUMMARY")
print("=" * 60)

# Enrollment totals
print("\n📈 ENROLLMENT TOTALS:")
total_0_5 = enrollment_df['age_0_5'].sum()
total_5_17 = enrollment_df['age_5_17'].sum()
total_18_plus = enrollment_df['age_18_greater'].sum()
total_enrollment = total_0_5 + total_5_17 + total_18_plus
print(f"  Age 0-5: {total_0_5:,} ({total_0_5/total_enrollment*100:.1f}%)")
print(f"  Age 5-17: {total_5_17:,} ({total_5_17/total_enrollment*100:.1f}%)")
print(f"  Age 18+: {total_18_plus:,} ({total_18_plus/total_enrollment*100:.1f}%)")
print(f"  TOTAL: {total_enrollment:,}")

# Biometric update totals
print("\n🔐 BIOMETRIC UPDATE TOTALS:")
total_bio_5_17 = biometric_df['bio_age_5_17'].sum()
total_bio_17_plus = biometric_df['bio_age_17_'].sum()
total_biometric = total_bio_5_17 + total_bio_17_plus
print(f"  Age 5-17: {total_bio_5_17:,} ({total_bio_5_17/total_biometric*100:.1f}%)")
print(f"  Age 17+: {total_bio_17_plus:,} ({total_bio_17_plus/total_biometric*100:.1f}%)")
print(f"  TOTAL: {total_biometric:,}")

# Demographic update totals
print("\n📝 DEMOGRAPHIC UPDATE TOTALS:")
total_demo_5_17 = demographic_df['demo_age_5_17'].sum()
total_demo_17_plus = demographic_df['demo_age_17_'].sum()
total_demographic = total_demo_5_17 + total_demo_17_plus
print(f"  Age 5-17: {total_demo_5_17:,} ({total_demo_5_17/total_demographic*100:.1f}%)")
print(f"  Age 17+: {total_demo_17_plus:,} ({total_demo_17_plus/total_demographic*100:.1f}%)")
print(f"  TOTAL: {total_demographic:,}")

# State-wise summary
print("\n🗺️ TOP 10 STATES BY ENROLLMENT:")
enrollment_df['total_enrollment'] = enrollment_df['age_0_5'] + enrollment_df['age_5_17'] + enrollment_df['age_18_greater']
state_enrollment = enrollment_df.groupby('state')['total_enrollment'].sum().sort_values(ascending=False)
print(state_enrollment.head(10).to_string())

print("\n🗺️ TOP 10 STATES BY BIOMETRIC UPDATES:")
biometric_df['total_biometric'] = biometric_df['bio_age_5_17'] + biometric_df['bio_age_17_']
state_biometric = biometric_df.groupby('state')['total_biometric'].sum().sort_values(ascending=False)
print(state_biometric.head(10).to_string())

print("\n🗺️ TOP 10 STATES BY DEMOGRAPHIC UPDATES:")
demographic_df['total_demographic'] = demographic_df['demo_age_5_17'] + demographic_df['demo_age_17_']
state_demographic = demographic_df.groupby('state')['total_demographic'].sum().sort_values(ascending=False)
print(state_demographic.head(10).to_string())

# =============================================================================
# DATA LIMITATIONS AND OBSERVATIONS
# =============================================================================
print("\n" + "=" * 60)
print("DATA LIMITATIONS AND OBSERVATIONS")
print("=" * 60)

print("""
🔍 KEY OBSERVATIONS:

1. NO GENDER DATA AVAILABLE
   - The datasets do not include gender information
   - This limits gender-gap analysis mentioned in the strategic framework
   
2. LIMITED AGE GRANULARITY
   - Enrollment: 0-5, 5-17, 18+
   - Updates: 5-17, 17+
   - Cannot do detailed age-cohort analysis (e.g., senior citizens 60+)

3. NO UPDATE TYPE BREAKDOWN
   - Biometric updates don't specify type (fingerprint vs iris)
   - Demographic updates don't specify type (address vs name vs DoB)

4. AGGREGATED DATA
   - Data is aggregated at pincode level per day
   - Cannot track individual enrollment-to-update lifecycle

5. PINCODE AS ONLY LOCATION GRANULARITY
   - No explicit urban/rural classification
   - Would need external mapping for rural/urban proxy

6. SHORT DATE RANGE
   - Data appears to be from March 2025
   - Limited historical trend analysis possible

POTENTIAL ANALYSIS THEMES (Given Data Limitations):
✅ Theme 2 (System Health) - Anomaly Detection, Geographic Hotspots
✅ Temporal pattern analysis - Daily/Weekly trends
✅ State/District level comparisons
✅ Age group distribution analysis
⚠️ Theme 1 (Digital Divide) - Limited without gender/urban-rural data
⚠️ Theme 3 (Forecasting) - Limited historical data
""")

print("\n" + "=" * 60)
print("EDA COMPLETE - Ready for Deep Dive Analysis")
print("=" * 60)
