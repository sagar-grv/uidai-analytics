"""Quick EDA Summary Script"""
import pandas as pd
import numpy as np
from pathlib import Path

base_path = Path(r"c:\Users\sagar\Downloads\uidai dataset")

# Load all datasets
print("Loading datasets...")
enrollment_path = base_path / "api_data_aadhar_enrolment" / "api_data_aadhar_enrolment"
enrollment_df = pd.concat([pd.read_csv(f) for f in enrollment_path.glob("*.csv")], ignore_index=True)

biometric_path = base_path / "api_data_aadhar_biometric" / "api_data_aadhar_biometric"
biometric_df = pd.concat([pd.read_csv(f) for f in biometric_path.glob("*.csv")], ignore_index=True)

demographic_path = base_path / "api_data_aadhar_demographic" / "api_data_aadhar_demographic"
demographic_df = pd.concat([pd.read_csv(f) for f in demographic_path.glob("*.csv")], ignore_index=True)

print("\n" + "="*60)
print("DATASET OVERVIEW")
print("="*60)
print(f"Enrollment records: {len(enrollment_df):,}")
print(f"Biometric records: {len(biometric_df):,}")
print(f"Demographic records: {len(demographic_df):,}")

print("\nCOLUMNS:")
print(f"Enrollment: {list(enrollment_df.columns)}")
print(f"Biometric: {list(biometric_df.columns)}")
print(f"Demographic: {list(demographic_df.columns)}")

# Parse dates
enrollment_df["date"] = pd.to_datetime(enrollment_df["date"], format="%d-%m-%Y", errors="coerce")
biometric_df["date"] = pd.to_datetime(biometric_df["date"], format="%d-%m-%Y", errors="coerce")
demographic_df["date"] = pd.to_datetime(demographic_df["date"], format="%d-%m-%Y", errors="coerce")

print("\n" + "="*60)
print("DATE RANGES")
print("="*60)
print(f"Enrollment: {enrollment_df['date'].min()} to {enrollment_df['date'].max()}")
print(f"Biometric: {biometric_df['date'].min()} to {biometric_df['date'].max()}")
print(f"Demographic: {demographic_df['date'].min()} to {demographic_df['date'].max()}")

print("\n" + "="*60)
print("UNIQUE VALUES")
print("="*60)
print(f"Enrollment - States: {enrollment_df['state'].nunique()}, Districts: {enrollment_df['district'].nunique()}, Pincodes: {enrollment_df['pincode'].nunique()}")
print(f"Biometric - States: {biometric_df['state'].nunique()}, Districts: {biometric_df['district'].nunique()}, Pincodes: {biometric_df['pincode'].nunique()}")
print(f"Demographic - States: {demographic_df['state'].nunique()}, Districts: {demographic_df['district'].nunique()}, Pincodes: {demographic_df['pincode'].nunique()}")

print("\n" + "="*60)
print("MISSING VALUES")
print("="*60)
print("Enrollment:", enrollment_df.isnull().sum().to_dict())
print("Biometric:", biometric_df.isnull().sum().to_dict())
print("Demographic:", demographic_df.isnull().sum().to_dict())

print("\n" + "="*60)
print("TOTAL COUNTS")
print("="*60)
total_enroll = enrollment_df[["age_0_5", "age_5_17", "age_18_greater"]].sum().sum()
total_bio = biometric_df[["bio_age_5_17", "bio_age_17_"]].sum().sum()
total_demo = demographic_df[["demo_age_5_17", "demo_age_17_"]].sum().sum()
print(f"Total Enrollments: {total_enroll:,}")
print(f"Total Biometric Updates: {total_bio:,}")
print(f"Total Demographic Updates: {total_demo:,}")

print("\n" + "="*60)
print("AGE DISTRIBUTION")
print("="*60)
print("\nEnrollment by Age Group:")
print(f"  Age 0-5: {enrollment_df['age_0_5'].sum():,} ({enrollment_df['age_0_5'].sum()/total_enroll*100:.1f}%)")
print(f"  Age 5-17: {enrollment_df['age_5_17'].sum():,} ({enrollment_df['age_5_17'].sum()/total_enroll*100:.1f}%)")
print(f"  Age 18+: {enrollment_df['age_18_greater'].sum():,} ({enrollment_df['age_18_greater'].sum()/total_enroll*100:.1f}%)")

print("\nBiometric Updates by Age Group:")
print(f"  Age 5-17: {biometric_df['bio_age_5_17'].sum():,} ({biometric_df['bio_age_5_17'].sum()/total_bio*100:.1f}%)")
print(f"  Age 17+: {biometric_df['bio_age_17_'].sum():,} ({biometric_df['bio_age_17_'].sum()/total_bio*100:.1f}%)")

print("\nDemographic Updates by Age Group:")
print(f"  Age 5-17: {demographic_df['demo_age_5_17'].sum():,} ({demographic_df['demo_age_5_17'].sum()/total_demo*100:.1f}%)")
print(f"  Age 17+: {demographic_df['demo_age_17_'].sum():,} ({demographic_df['demo_age_17_'].sum()/total_demo*100:.1f}%)")

print("\n" + "="*60)
print("TOP 10 STATES")
print("="*60)
enrollment_df["total"] = enrollment_df["age_0_5"] + enrollment_df["age_5_17"] + enrollment_df["age_18_greater"]
state_enroll = enrollment_df.groupby("state")["total"].sum().sort_values(ascending=False)
print("\nEnrollment:")
for i, (state, count) in enumerate(state_enroll.head(10).items(), 1):
    print(f"  {i}. {state}: {count:,}")

biometric_df["total"] = biometric_df["bio_age_5_17"] + biometric_df["bio_age_17_"]
state_bio = biometric_df.groupby("state")["total"].sum().sort_values(ascending=False)
print("\nBiometric Updates:")
for i, (state, count) in enumerate(state_bio.head(10).items(), 1):
    print(f"  {i}. {state}: {count:,}")

demographic_df["total"] = demographic_df["demo_age_5_17"] + demographic_df["demo_age_17_"]
state_demo = demographic_df.groupby("state")["total"].sum().sort_values(ascending=False)
print("\nDemographic Updates:")
for i, (state, count) in enumerate(state_demo.head(10).items(), 1):
    print(f"  {i}. {state}: {count:,}")

print("\n" + "="*60)
print("ALL STATES")
print("="*60)
all_states = sorted(set(enrollment_df["state"].unique()) | set(biometric_df["state"].unique()) | set(demographic_df["state"].unique()))
print(f"Total unique states/UTs: {len(all_states)}")
print(all_states)

print("\n" + "="*60)
print("DATA QUALITY CHECKS")
print("="*60)

# Check for duplicates
print("\nDuplicates:")
print(f"  Enrollment: {enrollment_df.duplicated().sum():,}")
print(f"  Biometric: {biometric_df.duplicated().sum():,}")
print(f"  Demographic: {demographic_df.duplicated().sum():,}")

# Check for zeros
print("\nRecords with all zeros:")
enroll_zeros = ((enrollment_df["age_0_5"] == 0) & (enrollment_df["age_5_17"] == 0) & (enrollment_df["age_18_greater"] == 0)).sum()
bio_zeros = ((biometric_df["bio_age_5_17"] == 0) & (biometric_df["bio_age_17_"] == 0)).sum()
demo_zeros = ((demographic_df["demo_age_5_17"] == 0) & (demographic_df["demo_age_17_"] == 0)).sum()
print(f"  Enrollment: {enroll_zeros:,}")
print(f"  Biometric: {bio_zeros:,}")
print(f"  Demographic: {demo_zeros:,}")

# Check for negative values
print("\nNegative values:")
print(f"  Enrollment age_0_5: {(enrollment_df['age_0_5'] < 0).sum()}")
print(f"  Enrollment age_5_17: {(enrollment_df['age_5_17'] < 0).sum()}")
print(f"  Enrollment age_18+: {(enrollment_df['age_18_greater'] < 0).sum()}")

print("\n" + "="*60)
print("TEMPORAL ANALYSIS")
print("="*60)
print("\nUnique dates:")
print(f"  Enrollment: {enrollment_df['date'].nunique()}")
print(f"  Biometric: {biometric_df['date'].nunique()}")
print(f"  Demographic: {demographic_df['date'].nunique()}")

print("\nDate with most enrollments:")
date_enroll = enrollment_df.groupby("date")["total"].sum().sort_values(ascending=False)
print(f"  {date_enroll.index[0].strftime('%Y-%m-%d')}: {date_enroll.iloc[0]:,}")

print("\nDate with most biometric updates:")
date_bio = biometric_df.groupby("date")["total"].sum().sort_values(ascending=False)
print(f"  {date_bio.index[0].strftime('%Y-%m-%d')}: {date_bio.iloc[0]:,}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
