import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql import SparkSession

# --- 1. CONFIGURATION ---
def get_db_password():
    """
    Retrieves password from Databricks Secrets or Env Vars fallback.
    """
    try:
        from pyspark.dbutils import DBUtils
        spark = SparkSession.builder.getOrCreate()
        dbutils = DBUtils(spark)
        return dbutils.secrets.get(scope="crm_secrets", key="crm_replica_password")
    except Exception:
        return os.getenv("DB_PASSWORD")
      
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "crm-replica.analytics.internal"),
    "database": os.getenv("DB_NAME", "production_crm"),
    "user": os.getenv("DB_USER", "analyst_user")
}

def get_spark_session():
    return SparkSession.builder.appName("LeadConversionModel").getOrCreate()

# --- 2. DATA INGESTION ---
def load_leads_data(spark):
    password = get_db_password()
    if not password:
        raise ValueError("Database password could not be retrieved. Check your Secret Scope or Environment Variables.")
    two_years_ago = (datetime.today() - relativedelta(years=2)).strftime('%Y-%m-%d %H:%M:%S')
    
    # Redacted table name for public Git
    query = f"(SELECT id, register_date, sub1, state, disposition_note FROM lead_registrations WHERE register_date >= '{two_years_ago}') as filtered"
    
    return (spark.read.format("mysql")
            .option("dbtable", query)
            .option("host", DB_CONFIG["host"])
            .option("user", DB_CONFIG["user"])
            .option("password", password)
            .load())

# --- 3. FEATURE ENGINEERING ---
def preprocess_and_feature_engineer(raw_df):
    # 3a. Normalization
    df = raw_df.withColumn("state", 
        F.when(F.col("state").isin("California", "CA"), "CA")
         .when(F.col("state").isin("Texas", "TX"), "TX")
         .when(F.col("state").isin("Florida", "FL"), "FL")
         .when(F.col("state").isin("New York", "NY"), "NY")
         .otherwise("UNKNOWN"))

    processed_df = (df.withColumn("sale", F.when(F.col("disposition_note").contains("Sold"), 1).otherwise(0))
                      .withColumn("installed", F.when(F.col("disposition_note").contains("Installed"), 1).otherwise(0))
                      .withColumn("date", F.to_date("register_date"))
                      .withColumn("month_start", F.trunc("date", "month")))

    # 3b. Daily Snapshot (Cumulative) Logic
    w_cum = Window.partitionBy("sub1", "month_start").orderBy("date")
    processed_df = (processed_df
        .withColumn("current_registrations", F.count("id").over(w_cum))
        .withColumn("current_sales", F.sum("sale").over(w_cum))
        .withColumn("current_installed", F.sum("installed").over(w_cum))
        .withColumn("day_of_month", F.dayofmonth("date"))
        .withColumn("current_run_rate", F.when(F.col("current_sales") > 0, F.col("current_installed")/F.col("current_sales")).otherwise(0.0)))

    # 3c. Bayesian Smoothing & Lags (The "Truth" Table)
    w_hist = Window.partitionBy("sub1").orderBy("month_start")
    w_roll_2m = w_hist.rowsBetween(-2, -1)

    df_monthly_truth = (processed_df.groupBy("sub1", "month_start")
        .agg(F.sum("sale").alias("final_total_sales"), F.sum("installed").alias("final_total_installed"))
        .withColumn("snapshot_target_pct", F.when(F.col("final_total_sales") > 0, F.col("final_total_installed")/F.col("final_total_sales")).otherwise(0.0))
        .withColumn("pct_lag_1", F.lag("snapshot_target_pct", 1).over(w_hist))
        # --- MEDIA TEAM BASELINE (Rolling 2 Month Average) ---
        .withColumn("media_team_baseline", 
            F.when(F.sum("final_total_sales").over(w_roll_2m) > 0, 
                   F.sum("final_total_installed").over(w_roll_2m) / F.sum("final_total_sales").over(w_roll_2m))
            .otherwise(F.col("pct_lag_1")))
        .na.fill(0))

    # 3d. Assembly
    first_day_curr = F.trunc(F.current_date(), "month")
    targets = df_monthly_truth.select("sub1", "month_start", "snapshot_target_pct", "final_total_sales").filter(F.col("month_start") < first_day_curr)
    features = df_monthly_truth.select("sub1", "month_start", "pct_lag_1", "media_team_baseline")

    return processed_df.join(targets, on=["sub1", "month_start"], how="left") \
                        .join(features, on=["sub1", "month_start"], how="left")

# --- 4. MODELING ---
def run_model_pipeline(enriched_df, split_date):
    pdf = enriched_df.filter(F.col("snapshot_target_pct").isNotNull()).toPandas()
    
    # Adding daily snapshot features to the training list
    features = ['pct_lag_1', 'media_team_baseline', 'current_run_rate', 'day_of_month']
    
    train_mask = pd.to_datetime(pdf['month_start']) < pd.to_datetime(split_date)
    val_mask = pd.to_datetime(pdf['month_start']) >= pd.to_datetime(split_date)
    
    dtrain = lgb.Dataset(pdf.loc[train_mask, features], label=pdf.loc[train_mask, 'snapshot_target_pct'], weight=pdf.loc[train_mask, 'final_total_sales'])
    dval = lgb.Dataset(pdf.loc[val_mask, features], label=pdf.loc[val_mask, 'snapshot_target_pct'], weight=pdf.loc[val_mask, 'final_total_sales'], reference=dtrain)

    model = lgb.train({'objective': 'regression', 'metric': 'rmse', 'verbose': -1}, dtrain, valid_sets=[dval], callbacks=[lgb.early_stopping(50)])
    
    # Calculate MAE for Model vs Baseline
    pdf['preds'] = model.predict(pdf[features])
    model_mae = np.mean(np.abs(pdf.loc[val_mask, 'snapshot_target_pct'] - pdf.loc[val_mask, 'preds']))
    base_mae = np.mean(np.abs(pdf.loc[val_mask, 'snapshot_target_pct'] - pdf.loc[val_mask, 'media_team_baseline']))
    
    print(f"\nVALIDATION RESULTS:\nModel MAE: {model_mae:.4f}\nMedia Team Baseline MAE: {base_mae:.4f}")
    print(f"Improvement: {((base_mae - model_mae) / base_mae)*100:.2f}%")
    
    return model, features

if __name__ == "__main__":
    spark = get_spark_session()
    raw = load_leads_data(spark)
    enriched = preprocess_and_feature_engineer(raw)
    
    split_date = (datetime.today() - relativedelta(months=3)).strftime('%Y-%m-01')
    model, feature_list = run_model_pipeline(enriched, split_date)
    
    # Predict Current Month
    current_data = enriched.filter(F.col("snapshot_target_pct").isNull()).toPandas()
    if not current_data.empty:
        current_data['predicted_eom_pct'] = np.clip(model.predict(current_data[feature_list]), 0, 1)
        print("\n--- LATEST PREDICTIONS ---")
        print(current_data[['sub1', 'predicted_eom_pct', 'media_team_baseline']].head())
