"""
UIDAI Aadhaar - Integrated Health & Forecast Engine
====================================================
COMBINED ANALYSIS: Theme 2 (System Health) + Theme 3 (Forecasting)
Uses Cleaned Gold Master Data.

Features:
1. Health Monitor: Anomaly Detection, Gini Index, Efficiency metrics.
2. Forecasting: Exponential Smoothing (Holt-Winters) for 30-day demand prediction.
3. Hybrid Report: Generation of strategic insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

base_path = Path(r"c:\Users\sagar\Downloads\uidai dataset")
processed_path = base_path / "processed"
output_path = base_path / "integrated_outputs"
output_path.mkdir(exist_ok=True)

print("=" * 80)
print("UIDAI INTEGRATED HEALTH & FORECAST ENGINE")
print("=" * 80)

# =============================================================================
# 1. LOAD CLEANED DATA
# =============================================================================
print("\n[1/4] Loading Gold Master Data...")
gold_master = pd.read_csv(processed_path / "uidai_gold_master.csv")
daily_ts = pd.read_csv(processed_path / "uidai_daily_timeseries.csv", index_col="date", parse_dates=True)

print(f"  Records: {len(gold_master):,}")
print(f"  Time Series Days: {len(daily_ts)}")

# =============================================================================
# 2. THEME 2: SYSTEM HEALTH MONITOR
# =============================================================================
print("\n[2/4] Running System Health Monitor...")

health_report = {}

# 2.1 Anomaly Detection (Z-Score)
print("  > Detecting Daily Volume Anomalies...")
anomalies = {}
for col in daily_ts.columns:
    series = daily_ts[col]
    mean = series.mean()
    std = series.std()
    z_scores = (series - mean) / std
    
    # Flag Z > 2.5 (Significant Anomaly)
    anomaly_dates = series[abs(z_scores) > 2.5]
    anomalies[col] = len(anomaly_dates)
    
    if len(anomaly_dates) > 0:
        print(f"    {col}: {len(anomaly_dates)} anomalies detected (Max Z: {z_scores.max():.2f})")

health_report["Anomalies"] = anomalies

# 2.2 Regional Load Balance (Gini Coefficient)
print("  > Calculating Regional Load Balance (Gini)...")
state_loads = gold_master.groupby("state_clean")["total_count"].sum().sort_values()
vals = state_loads.values
n = len(vals)
gini = (2 * np.sum(np.arange(1, n + 1) * vals)) / (n * np.sum(vals)) - (n + 1) / n
print(f"    National Gini Coefficient: {gini:.4f}")

health_report["Gini"] = gini
health_report["LoadStatus"] = "CRITICAL IMBALANCE" if gini > 0.5 else "BALANCED"

# 2.3 Operational Efficiency (Bio/Demo Ratio)
print("  > Measuring Operational Efficiency...")
type_sums = gold_master.groupby("type")["total_count"].sum()
bio_demo_ratio = type_sums.get("Biometric", 0) / type_sums.get("Demographic", 1)
print(f"    Biometric/Demographic Ratio: {bio_demo_ratio:.2f}x")

health_report["EfficiencyRatio"] = bio_demo_ratio

# =============================================================================
# 3. THEME 3: FORECASTING ENGINE
# =============================================================================
print("\n[3/4] Running Forecasting Engine (Next 30 Days)...")

forecasts = {}

# Set frequency explicitly to avoid alignment errors
daily_ts.index.freq = 'D'

for col in daily_ts.columns:
    print(f"  > Forecasting {col}...")
    try:
        # METHOD 1: Holt-Winters (Advanced)
        # Using 'additive' everywhere is safer for data with zeros
        model = ExponentialSmoothing(
            daily_ts[col], 
            seasonal_periods=7, 
            trend='add', 
            seasonal='add', 
            initialization_method="estimated"
        ).fit(damping_trend=0.2)
        
        future = model.forecast(30)
        forecasts[col] = future
        method_used = "Holt-Winters"
        
    except Exception as e:
        print(f"    Holt-Winters failed ({str(e)}). Switching to Robust Moving Average.")
        
        # METHOD 2: Robust Moving Average (Fallback)
        # Forecast = Last 7 days avg (simple baseline)
        last_7_avg = daily_ts[col].tail(7).mean()
        # Add some noise/trend based on last 30 days
        trend = (daily_ts[col].tail(30).mean() - daily_ts[col].tail(60).head(30).mean()) / 30
        
        future_dates = pd.date_range(start=daily_ts.index.max() + pd.Timedelta(days=1), periods=30, freq='D')
        future_values = [last_7_avg + (trend * i) for i in range(30)]
        # Clip to 0 (no negative enrollments)
        future_values = [max(0, x) for x in future_values]
        
        future = pd.Series(future_values, index=future_dates)
        forecasts[col] = future
        method_used = "Moving Average (Fallback)"

    # Validation and Plotting
    try:
        plt.figure(figsize=(14, 6))
        plt.plot(daily_ts.index, daily_ts[col], label='Historical', color='#34495e', alpha=0.7)
        plt.plot(future.index, future, label=f'Forecast ({method_used})', color='#e74c3c', linestyle='--', linewidth=2)
        plt.title(f"30-Day Forecast: {col} Volume ({method_used})", fontsize=14)
        plt.legend()
        plt.savefig(output_path / f"forecast_{col}.png")
        print(f"    Saved forecast plot: forecast_{col}.png")
        print(f"    Predicted 30-day Volume: {future.sum():,.0f}")
    except Exception as plot_err:
        print(f"    Plotting failed: {str(plot_err)}")

# =============================================================================
# 4. HYBRID INSIGHTS GENERATION
# =============================================================================
print("\n[4/4] Generating Hybrid Insights...")

with open(output_path / "integrated_insights_report.txt", "w") as f:
    f.write("UIDAI INTEGRATED ANALYTICS REPORT\n")
    f.write("=================================\n\n")
    
    f.write("THEME 2: SYSTEM HEALTH STATUS\n")
    f.write(f"- Overall Load Status: {health_report['LoadStatus']} (Gini: {health_report['Gini']:.2f})\n")
    f.write("- Recommendation: " + 
            ("Urgent Load Balancing required. Reallocate resources to Top 20% states." 
             if health_report['Gini'] > 0.5 else "Load is well-distributed.") + "\n")
    f.write(f"- Operational Efficiency: {health_report['EfficiencyRatio']:.2f}x Bio/Demo ratio.\n")
    f.write(f"- Anomalies Detected: {sum(health_report['Anomalies'].values())} significant events.\n\n")
    
    f.write("THEME 3: DEMAND FORECAST (Next 30 Days)\n")
    total_predicted = sum([f.sum() for f in forecasts.values()])
    f.write(f"- Total Predicted Volume: {total_predicted:,.0f} transactions\n")
    for col, future in forecasts.items():
        f.write(f"- {col}: {future.sum():,.0f} (Avg: {future.mean():,.0f}/day)\n")
        
        try:
            if future.sum() == 0:
                peak_day = "N/A (Zero Volume)"
            else:
                peak_idx = future.idxmax()
                peak_day = peak_idx.strftime('%Y-%m-%d') if pd.notna(peak_idx) else "Unknown"
        except:
            peak_day = "Error calculating peak"
            
        f.write(f"  > Peak expected on: {peak_day}\n")

print(f"Saved Insights Report: {output_path / 'integrated_insights_report.txt'}")
print("\nINTEGRATED ANALYSIS COMPLETE.")
