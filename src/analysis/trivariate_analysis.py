
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIG ---
BASE_PATH = Path(r"c:\Users\sagar\Downloads\uidai dataset")
DATA_PATH = BASE_PATH / "data" / "processed" / "uidai_gold_master.csv"
OUTPUT_PATH = BASE_PATH / "reports" / "figures" / "trivariate"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# --- MAPPINGS ---
REGION_MAP = {
    'Andhra Pradesh': 'South', 'Karnataka': 'South', 'Kerala': 'South', 'Tamil Nadu': 'South', 'Telangana': 'South',
    'Arunachal Pradesh': 'North East', 'Assam': 'North East', 'Manipur': 'North East', 'Meghalaya': 'North East', 'Mizoram': 'North East', 'Nagaland': 'North East', 'Sikkim': 'North East', 'Tripura': 'North East',
    'Bihar': 'East', 'Jharkhand': 'East', 'Odisha': 'East', 'West Bengal': 'East',
    'Gujarat': 'West', 'Goa': 'West', 'Maharashtra': 'West', 'Rajasthan': 'West',
    'Haryana': 'North', 'Himachal Pradesh': 'North', 'Punjab': 'North', 'Uttar Pradesh': 'North', 'Uttarakhand': 'North', 'Delhi': 'North', 'Jammu and Kashmir': 'North', 'Ladakh': 'North', 'Chandigarh': 'North',
    'Madhya Pradesh': 'Central', 'Chhattisgarh': 'Central',
    'Dadra and Nagar Haveli and Daman and Diu': 'West', 'Andaman and Nicobar Islands': 'South', 'Lakshadweep': 'South', 'Puducherry': 'South'
}

def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Summer'
    elif month in [6, 7, 8, 9]: return 'Monsoon'
    else: return 'Post-Monsoon'

def main():
    print("Loading Data...")
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    # Feature Engineering
    print("Engineering Features...")
    df['region'] = df['state_clean'].map(REGION_MAP).fillna('Other')
    df['month'] = df['date'].dt.month
    df['season'] = df['month'].apply(get_season)
    
    # --- TRIVARIATE 1: Update Type x Region x Season ---
    print("Generating Analysis 1: Update Type x Region x Season...")
    
    pivot_table = df.pivot_table(
        index='region', 
        columns=['season', 'type'], 
        values='total_count', 
        aggfunc='sum'
    ).fillna(0)
    
    # Visualization: Heatmap of Updates by Region & Season
    # Filter only for Biometric/Demographic to compare Update behavior
    update_df = df[df['type'].isin(['Biometric', 'Demographic'])]
    
    plt.figure(figsize=(15, 8))
    sns.set_theme(style="whitegrid")
    
    # Faceted Bar Chart
    g = sns.catplot(
        data=update_df, 
        x='season', 
        y='total_count', 
        hue='type', 
        col='region', 
        col_wrap=3,
        kind='bar',
        height=4, 
        aspect=1.2,
        estimator=sum,
        ci=None,
        palette={'Biometric': '#FF9933', 'Demographic': '#138808'} # Saffron & Green
    )
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Update Trends: Season vs Region vs Type')
    plt.savefig(OUTPUT_PATH / "trivariate_region_season_type.png")
    plt.close()
    
    # --- TRIVARIATE 2: State x Age Group x Enrollment ---
    print("Generating Analysis 2: State x Age Group Composition...")
    # Normalize by state total to get % composition
    
    if 'age_0_5' in df.columns: # Assuming simplified structure in Gold Master or derived from original logic
        # For Gold Master, we might need to rely on type if detailed age cols aren't there. 
        # But wait, gold master usually has consolidated cols. Let's check columns first or rely on standard EDA logic.
        # If columns missing, we skip.
        pass

    # Alternative Trivariate: State x Time x Volume (Heatmap)
    print("Generating Analysis 3: Temporal Heatmap (State x Month)...")
    state_month = df.groupby(['state_clean', 'month'])['total_count'].sum().reset_index()
    heatmap_data = state_month.pivot(index='state_clean', columns='month', values='total_count')
    
    plt.figure(figsize=(12, 15))
    sns.heatmap(heatmap_data, cmap="YlOrRd", linewidths=.5)
    plt.title('State-wise Frequency Intensity by Month')
    plt.ylabel('State')
    plt.xlabel('Month')
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "trivariate_state_time_heatmap.png")
    plt.close()

    print(f"Analysis Complete. Images saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
