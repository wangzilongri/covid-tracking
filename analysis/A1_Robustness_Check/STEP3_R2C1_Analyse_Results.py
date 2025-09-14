# Databricks notebook source
import os
import pyspark

from pyspark.sql.types import *
from pyspark.sql import Row

import pandas as pd
import numpy as np
from pyspark.sql import functions as F

# COMMAND ----------

base_data_path = "/mnt/users/zilongwang/TLGRF_linear"
results_spark_df = spark.read.parquet(os.path.join(base_data_path, "R2C1_predictions.parquet")).filter(F.col("t") > 30).orderBy(["t","c"])

display(results_spark_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot Predicted Growth Rate vs Actual Growth Rate for each county

# COMMAND ----------

if False:

    df_plot = results_spark_df.select("c", "t", "r_tc", "pred_d_log_I", "pred_d_I").toPandas()
    import matplotlib.pyplot as plt
    import math

    # Get sorted list of counties
    counties = sorted(df_plot["c"].unique())
    n = len(counties)
    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, c_id in enumerate(counties):
        ax = axes[idx]
        sub_df = df_plot[df_plot["c"] == c_id].sort_values("t")
        
        ax.plot(sub_df["t"], sub_df["r_tc"], label="$r_{tc}$", color="black", linewidth=1.8)
        ax.plot(sub_df["t"], sub_df["pred_d_log_I"], label="$\widehat{d_{\log I}}$", linestyle="--", color="C1")
        ax.plot(sub_df["t"], sub_df["pred_d_I"], label="$\widehat{d_I}$", linestyle=":", color="C2")
        
        ax.set_title(f"County {c_id}")
        ax.set_xlabel("t")
        ax.set_ylabel("Growth / Diff")
        ax.grid(True)

    # Remove extra axes if county count doesn't fill the grid
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    # Add global legend once
    from matplotlib.lines import Line2D
    custom_legend = [
        Line2D([0], [0], color="black", lw=1.8, label="$r_{tc}$"),
        Line2D([0], [0], color="C1", linestyle="--", label="$\widehat{d_{\log I}}$"),
        Line2D([0], [0], color="C2", linestyle=":", label="$\widehat{d_I}$")
    ]

    fig.legend(handles=custom_legend, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    plt.show()


# COMMAND ----------

df_debug = results_spark_df.select("c", "t", "r_tc", "pred_d_log_I", "pred_d_I").filter("t > 30").toPandas()

display(df_debug)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compute Results

# COMMAND ----------

from pyspark.sql import functions as F

# Add squared error and absolute error columns
results_spark_df = results_spark_df \
    .withColumn("se_log", (F.col("pred_d_log_I") - F.col("r_tc"))**2) \
    .withColumn("se_lin", (F.col("pred_d_I") - F.col("r_tc"))**2) \
    .withColumn("ae_log", F.abs(F.col("pred_d_log_I") - F.col("r_tc"))) \
    .withColumn("ae_lin", F.abs(F.col("pred_d_I") - F.col("r_tc"))) \
    .withColumn("se_exp_log", (F.exp(F.col("pred_d_log_I")) - F.col("r_tc"))**2) \
    .withColumn("ae_exp_log", F.abs(F.exp(F.col("pred_d_log_I")) - F.col("r_tc")))

# Compute mean RMSE and MAE for each day
daily_metrics = results_spark_df.groupBy("t").agg(
    F.sqrt(F.avg("se_log")).alias("mean_rmse_log"),
    F.sqrt(F.avg("se_lin")).alias("mean_rmse_lin"),
    F.sqrt(F.avg("se_exp_log")).alias("mean_rmse_exp_log"),
    F.avg("ae_log").alias("mean_mae_log"),
    F.avg("ae_lin").alias("mean_mae_lin"),
    F.avg("ae_exp_log").alias("mean_mae_exp_log"),
).orderBy("t")


# COMMAND ----------

display(results_spark_df)

# COMMAND ----------

daily_pd = daily_metrics.toPandas()


# COMMAND ----------

display(daily_pd)

# COMMAND ----------

import matplotlib.pyplot as plt

# Determine y-axis max for consistent scaling
y_max_rmse = max(daily_pd["mean_rmse_log"].max(), daily_pd["mean_rmse_lin"].max())
y_max_mae = max(daily_pd["mean_mae_log"].max(), daily_pd["mean_mae_lin"].max())

# Round up to nearest 0.25 for axis limits
y_max_rmse = np.ceil(y_max_rmse * 4) / 4
y_max_mae = np.ceil(y_max_mae * 4) / 4

y_max = max(y_max_rmse, y_max_mae) + 0.5

# RMSE plot
plt.figure(figsize=(10, 4))
plt.plot(daily_pd["t"], daily_pd["mean_rmse_lin"], label="Linear TLRF", color="blue")
plt.plot(daily_pd["t"], daily_pd["mean_rmse_log"], label="Exponential TLRF", color="orange")
#plt.plot(daily_pd["t"], daily_pd["mean_rmse_exp_log"], label="exp log model")

plt.xlabel("Day (t)")
plt.ylabel("Mean RMSE across counties")
plt.title("Daily RMSE of Growth Rate Estimates")
#plt.yticks(np.arange(0, y_max + 0.01, 1.00))
plt.ylim(0, y_max)
plt.legend()
plt.tight_layout()
plt.show()

# MAE plot
plt.figure(figsize=(10, 4))
plt.plot(daily_pd["t"], daily_pd["mean_mae_lin"], label="Linear TLRF", color="blue")
plt.plot(daily_pd["t"], daily_pd["mean_mae_log"], label="Exponential TLRF", color="orange")
#plt.plot(daily_pd["t"], daily_pd["mean_mae_exp_log"], label="exp log model")


plt.xlabel("Day (t)")
plt.ylabel("Mean MAE across counties")
plt.title("Daily MAE of Growth Rate Estimates")
#plt.yticks(np.arange(0, y_max + 0.01, 1.00))
plt.ylim(0, y_max)
plt.legend()
plt.tight_layout()
plt.show()


# COMMAND ----------

overall_summary = {
    "avg_daily_mean_rmse_log": daily_pd["mean_rmse_log"].mean(),
    "avg_daily_mean_rmse_lin": daily_pd["mean_rmse_lin"].mean(),
    "avg_daily_mean_mae_log": daily_pd["mean_mae_log"].mean(),
    "avg_daily_mean_mae_lin": daily_pd["mean_mae_lin"].mean()
}

display(overall_summary)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 7 Days Ahead Prediction Error

# COMMAND ----------

from pyspark.sql import Window
import pyspark.sql.functions as F

# Base DataFrame: assume it's called df
df = results_spark_df.withColumn("pred_I_7ahead_lin", F.col("I_tc") + 7 * F.col("pred_d_I"))
df = df.withColumn("pred_I_7ahead_log", F.col("I_tc") * F.exp(7 * F.col("pred_d_log_I")))
#df = df.withColumn("pred_I_7ahead_log", F.col("I_tc") + 7 * F.col("pred_d_log_I"))

# Create ground truth I_{t+7, c} by self-joining
df_lagged = df.select(
    F.col("c").alias("c_lag"),
    F.col("t").alias("t_lag"),
    F.col("I_tc").alias("I_tc_7days_later")
).withColumn("t_lag", F.col("t_lag") - 7)

# Join with original df on (c, t)
df_joined = df.join(
    df_lagged,
    (df["c"] == df_lagged["c_lag"]) & (df["t"] == df_lagged["t_lag"]),
    how="inner"
).drop("c_lag", "t_lag")

# Final schema includes:
# - pred_I_7ahead_lin
# - pred_I_7ahead_log
# - I_tc_7days_later (true value)


# COMMAND ----------

display(df_joined)

# COMMAND ----------

from pyspark.sql.functions import log1p
from pyspark.sql.functions import col
# To avoid log(0), use log1p (i.e., log(1 + x)) or filter out zeros
df_joined_log = df_joined.filter(
    (col("I_tc_7days_later") > 0) &
    (col("pred_I_7ahead_lin") > 0) &
    (col("pred_I_7ahead_log") > 0)
).withColumns({
    "log_true": F.log(col("I_tc_7days_later")),
    "log_pred_lin": F.log(col("pred_I_7ahead_lin")),
    "log_pred_log": F.log(col("pred_I_7ahead_log")),
})

df_joined_log = df_joined_log.withColumns({
    "log_sq_err_lin": F.pow(col("log_pred_lin") - col("log_true"), 2),
    "log_sq_err_log": F.pow(col("log_pred_log") - col("log_true"), 2),
    "log_abs_err_lin": F.abs(col("log_pred_lin") - col("log_true")),
    "log_abs_err_log": F.abs(col("log_pred_log") - col("log_true")),
})
daily_log_metrics = df_joined_log.groupBy("t").agg(
    F.sqrt(F.mean("log_sq_err_lin")).alias("log_rmse_lin"),
    F.sqrt(F.mean("log_sq_err_log")).alias("log_rmse_log"),
    F.mean("log_abs_err_lin").alias("log_mae_lin"),
    F.mean("log_abs_err_log").alias("log_mae_log")
).orderBy("t")

daily_log_metrics_pd = daily_log_metrics.toPandas()




# COMMAND ----------

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(daily_log_metrics_pd["t"], daily_log_metrics_pd["log_rmse_lin"], label="Linear TLRF", color="blue")
plt.plot(daily_log_metrics_pd["t"], daily_log_metrics_pd["log_rmse_log"], label="Exponential TLRF", color="orange")
plt.ylabel("Log RMSE")
plt.xlabel("t")
plt.title("Root Mean Squared Error (RMSE) in One-Week Ahead Case Predictions")
plt.legend()
plt.grid(True)
plt.yticks(ticks=plt.yticks()[0])
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(daily_log_metrics_pd["t"], daily_log_metrics_pd["log_mae_lin"], label="Linear TLRF", color="blue")
plt.plot(daily_log_metrics_pd["t"], daily_log_metrics_pd["log_mae_log"], label="Exponential TLRF", color="orange")
plt.ylabel("Log MAE")
plt.xlabel("t")
plt.title("Mean Absolute Error (MAE) in One-Week Ahead Case Prediction")
plt.legend()
plt.grid(True)
plt.yticks(ticks=plt.yticks()[0])
plt.show()


# COMMAND ----------

daily_log_metrics_pd = daily_log_metrics.toPandas()
avg_log_rmse_lin = daily_log_metrics_pd["log_rmse_lin"].median()
avg_log_rmse_log = daily_log_metrics_pd["log_rmse_log"].median()
avg_log_mae_lin = daily_log_metrics_pd["log_mae_lin"].median()
avg_log_mae_log = daily_log_metrics_pd["log_mae_log"].median()
import pandas as pd

summary_table = pd.DataFrame({
    "Method": ["Linear Model", "Exponential Model"],
    "MAE": [avg_log_mae_lin, avg_log_mae_log],
    "RMSE": [avg_log_rmse_lin, avg_log_rmse_log]
})

summary_table = summary_table.round(3)
from IPython.display import display
display(summary_table)
print("\\\\\n".join([
    f"{row.Method} & {row['MAE']:.3f} & {row['RMSE']:.3f}"
    for _, row in summary_table.iterrows()
]))


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Plot Predictions

# COMMAND ----------

import math
import matplotlib.pyplot as plt
import numpy as np
import pyspark.sql.functions as F
from matplotlib.lines import Line2D

if False:

    # Ensure we have log-transformed columns
    df_plot = df_joined_log.select(
        "c", "t",
        F.log(col("pred_I_7ahead_lin")).alias("log_pred_lin"),
        F.log(col("pred_I_7ahead_log")).alias("log_pred_log"),
        F.log(col("I_tc_7days_later")).alias("log_true")
    ).toPandas()

    # List of unique counties
    counties = sorted(df_plot["c"].unique())
    n = len(counties)
    cols = 3  # e.g. 3 plots per row
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, c_id in enumerate(counties):
        ax = axes[idx]
        sub = df_plot[df_plot["c"] == c_id].sort_values("t")
        t = sub["t"]
        ax.plot(t, sub["log_true"], label="Log True", color="black", linewidth=2)
        ax.plot(t, sub["log_pred_lin"], label="Log Pred — Linear", ls="--", color="C1")
        ax.plot(t, sub["log_pred_log"], label="Log Pred — Exponential", ls=":", color="C2")
        ax.set_title(f"County {c_id}")
        ax.set_xlabel("t")
        ax.set_ylabel("log(cases)")
        ax.grid(True)

    # Remove unused subplots if any
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    custom_lines = [
        Line2D([0], [0], color='black', lw=2, label='Log True'),
        Line2D([0], [0], color='C1', linestyle='--', label='Linear Model'),
        Line2D([0], [0], color='C2', linestyle=':', label='Exponential Model'),
    ]

    fig.legend(handles=custom_lines, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    plt.show()


# COMMAND ----------

