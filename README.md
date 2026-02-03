# AdTech Lead Conversion Forecaster (Daily Snapshot Model)

## 📊 Business Problem
AdTech campaigns move fast. Waiting until the end of the month (EOM) to see if a campaign hit its target conversion rate leads to inefficient budget allocation. This model provides **real-time EOM forecasts** based on intra-month performance.

## 🛠️ Solution
I developed a **Daily Snapshot Pipeline** that uses Gradient Boosting to project final conversion rates. The pipeline includes:
- **PySpark ETL:** Handles Hundreds of thousands of lead records with efficient window functions.
- **Bayesian Prior Logic:** Mitigates "cold-start" issues for new campaigns using state-level conversion profiles.
- **Baseline Comparison:** Directly compares ML performance against the legacy "Media Team Baseline" (2-month rolling average).

## 📈 Results
The LightGBM model achieved a **~15% error reduction (MAE)** compared to the legacy rolling average baseline. This allows media buyers to scale winning campaigns and cut losers up to 20 days earlier than the previous manual process.



## 🏗️ Technical Architecture
1. **Ingestion:** Securely pulls from CRM replicas using Databricks Secrets.
2. **Feature Engineering:** Calculates cumulative daily run-rates and historical lags.
3. **Training:** Implements volume-weighted training to prioritize high-spend campaigns.
4. **Inference:** Generates automated forecasts for the current active month.
