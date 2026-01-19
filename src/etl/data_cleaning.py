"""
UIDAI Aadhaar Data - Remediation & Cleaning Pipeline
=====================================================
Prerequisite for Theme 2 (Health) & Theme 3 (Forecasting).
Produces 'uidai_gold_master.csv' by fixing:
1. State Name Inconsistencies (68 -> 37 unique)
2. Exact Duplicate Records (Optimization)
3. Date Gaps (Zero-filling for Time Series)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

base_path = Path(r"c:\Users\sagar\Downloads\uidai dataset")
output_path = base_path / "processed"
output_path.mkdir(exist_ok=True)

print("=" * 80)
print("UIDAI DATA REMEDIATION PIPELINE")
print("=" * 80)

# =============================================================================
# 1. STATE MAPPING DICTIONARY
# =============================================================================
state_map = {
    'andhra pradesh': 'Andhra Pradesh', 'andhra prades': 'Andhra Pradesh',
    'arunachal pradesh': 'Arunachal Pradesh',
    'assam': 'Assam',
    'bihar': 'Bihar',
    'chhattisgarh': 'Chhattisgarh', 'chhatisgarh': 'Chhattisgarh', 'chattisgarh': 'Chhattisgarh',
    'chandigarh': 'Chandigarh',
    'dadra and nagar haven': 'Dadra and Nagar Haveli', 'dadra & nagar haveli': 'Dadra and Nagar Haveli',
    'daman and diu': 'Daman and Diu', 'daman & diu': 'Daman and Diu',
    'delhi': 'Delhi', 'new delhi': 'Delhi', 'nct of delhi': 'Delhi',
    'goa': 'Goa',
    'gujarat': 'Gujarat',
    'haryana': 'Haryana',
    'himachal pradesh': 'Himachal Pradesh',
    'jammu and kashmir': 'Jammu and Kashmir', 'jammu & kashmir': 'Jammu and Kashmir',
    'jharkhand': 'Jharkhand',
    'karnataka': 'Karnataka',
    'kerala': 'Kerala',
    'lakshadweep': 'Lakshadweep',
    'madhya pradesh': 'Madhya Pradesh',
    'maharashtra': 'Maharashtra',
    'manipur': 'Manipur',
    'meghalaya': 'Meghalaya',
    'mizoram': 'Mizoram',
    'nagaland': 'Nagaland',
    'odisha': 'Odisha', 'orissa': 'Odisha',
    'puducherry': 'Puducherry', 'pondicherry': 'Puducherry',
    'punjab': 'Punjab',
    'rajasthan': 'Rajasthan',
    'sikkim': 'Sikkim',
    'tamil nadu': 'Tamil Nadu',
    'telangana': 'Telangana',
    'tripura': 'Tripura',
    'uttar pradesh': 'Uttar Pradesh', 'uttar prades': 'Uttar Pradesh',
    'uttarakhand': 'Uttarakhand', 'uttaranchal': 'Uttarakhand',
    'west bengal': 'West Bengal', 'westbengal': 'West Bengal', 'west bengli': 'West Bengal', 
    'west bangal': 'West Bengal', 'west  bengal': 'West Bengal'
}

def clean_state_name(name):
    if pd.isna(name): return "Unknown"
    name = str(name).strip().title()
    name = " ".join(name.split()) # Remove multiple spaces
    
    if name.lower() in state_map:
        return state_map[name.lower()]
    
    # Fuzzy / generic mapping lookup
    lower_name = name.lower()
    for k, v in state_map.items():
        if k in lower_name:
            return v
    
    return name

# =============================================================================
# 2. DATA LOADING & PROCESSING
# =============================================================================
def load_and_clean(loader_path, file_pattern, type_label, age_cols):
    print(f"\nProcessing {type_label}...")
    files = list(loader_path.glob(file_pattern))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    
    initial_len = len(df)
    print(f"  Loaded {initial_len:,} rows")
    
    # Date parsing
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    
    # State Standardization
    df["state_clean"] = df["state"].apply(clean_state_name)
    
    # Deduplication
    df_dedup = df.drop_duplicates()
    dups = initial_len - len(df_dedup)
    print(f"  Removed {dups:,} duplicates ({(dups/initial_len*100):.1f}%)")
    
    # Standardize Column Names
    # We want: date, state, district, pincode, total_count, type
    # Sum age columns to get total
    df_dedup["total_count"] = df_dedup[age_cols].sum(axis=1)
    
    # keep only relevant columns
    cols_to_keep = ["date", "state_clean", "district", "pincode", "total_count"]
    clean_df = df_dedup[cols_to_keep].copy()
    clean_df["type"] = type_label
    
    return clean_df

# Define paths and columns
# Enrollment: age_0_5, age_5_17, age_18_greater
enroll_path = base_path / "api_data_aadhar_enrolment" / "api_data_aadhar_enrolment"
enroll_cols = ["age_0_5", "age_5_17", "age_18_greater"]

# Biometric: bio_age_5_17, bio_age_17_
bio_path = base_path / "api_data_aadhar_biometric" / "api_data_aadhar_biometric"
bio_cols = ["bio_age_5_17", "bio_age_17_"]

# Demographic: demo_age_5_17, demo_age_17_
demo_path = base_path / "api_data_aadhar_demographic" / "api_data_aadhar_demographic"
demo_cols = ["demo_age_5_17", "demo_age_17_"]

# processing
df_enroll = load_and_clean(enroll_path, "*.csv", "Enrollment", enroll_cols)
df_bio = load_and_clean(bio_path, "*.csv", "Biometric", bio_cols)
df_demo = load_and_clean(demo_path, "*.csv", "Demographic", demo_cols)

# =============================================================================
# 3. CONSOLIDATION & GOLD MASTER CREATION
# =============================================================================
print("\nCreating Gold Master Dataset...")
master_df = pd.concat([df_enroll, df_bio, df_demo], ignore_index=True)

# Drop any null dates or zero counts
master_df = master_df.dropna(subset=["date"])
master_df = master_df[master_df["total_count"] > 0]

print(f"Total Gold Master Records: {len(master_df):,}")
print(f"Unique States: {master_df['state_clean'].nunique()}")

# Save
gold_path = output_path / "uidai_gold_master.csv"
master_df.to_csv(gold_path, index=False)
print(f"Saved: {gold_path}")

# =============================================================================
# 4. TIME SERIES PREPARATION (IMPUTATION)
# =============================================================================
print("\nPreparing Time Series Aggregates (Daily)...")
# Aggregate to Country Level Daily
daily_ts = master_df.pivot_table(index="date", columns="type", values="total_count", aggfunc="sum")

# Fill missing dates with 0
full_date_range = pd.date_range(start=daily_ts.index.min(), end=daily_ts.index.max())
daily_ts = daily_ts.reindex(full_date_range, fill_value=0)
daily_ts.index.name = "date"

ts_path = output_path / "uidai_daily_timeseries.csv"
daily_ts.to_csv(ts_path)
print(f"Saved Time Series: {ts_path}")
print(f"Date Range: {daily_ts.index.min().date()} to {daily_ts.index.max().date()}")
print(f"Total Days: {len(daily_ts)}")

print("\nREMEDIATION COMPLETE.")
