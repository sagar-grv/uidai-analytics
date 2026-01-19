
# 🇮🇳 UIDAI Aadhaar Advanced Analytics & Forensic Dashboard

![Made in India](https://img.shields.io/badge/Made%20with-%E2%9D%A4%EF%B8%8F-red) ![Python](https://img.shields.io/badge/Python-3.10-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)

## 📋 Executive Summary
This project is an advanced forensic analysis and interactive dashboard for the UIDAI Aadhaar dataset (2026). It goes beyond standard EDA to uncover deep behavioral patterns, systemic anomalies, and infrastructure bottlenecks using statistical laws (Benford's Law) and Machine Learning.

## 🚀 Key Features
*   **Forensic Audit**: Benford's Law validation proving synthetic data characteristics.
*   **Infrastructure Health**: Gini Coefficient analysis revealing that 33% of districts handle 80% of the load.
*   **"Camp Mode" Detection**: Zero-lag correlation proving synchronized weekend enrollment drives.
*   **Interactive Dashboard**: A PowerBI-style Streamlit app with:
    *   **Tricolor Theme**: Saffron/White/Green national identity palette.
    *   **ML Simulator**: Random Forest model to forecast demand.
    *   **Forensic Lab**: Tabs dedicated to anomaly detection.
    *   **ChatBot**: "Aadhaar-Bot" for natural language SQL-like queries.

## 📂 Project Structure
```
.
├── app/
│   └── dashboard.py          # Main Streamlit Application (The Command Center)
├── src/
│   ├── analysis/             # Forensic, Geo, and Statistical Analysis Scripts
│   ├── etl/                  # Data Pipeline Scripts
│   └── visualization/        # Plot Generation Scripts
├── data/
│   ├── processed/            # Gold Master Datasets
│   └── raw/                  # Source Data
├── reports/
│   ├── generated/            # PDF and Markdown Reports
│   └── figures/              # High-Res Plots and Outputs
├── requirements.txt          # Dependencies
└── README.md                 # Project Documentation
```

*(Note: Data files are excluded from the repo for privacy and size constraints)*

## 🛠️ Installation & Setup

1.  **Clone the Repository**:
    ```bash
    git clone <your-repo-url>
    cd uidai-analytics
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Dashboard**:
    ```bash
    streamlit run app/dashboard.py
    ```

## 📊 Dashboard Preview
The dashboard features a **"Made in India"** theme with animated KPI cards, geospatial heatmaps, and a dedicated 'Forensic Lab' tab.

---
**Made with ❤️ in India** | Powered by UIDAI Open Data
