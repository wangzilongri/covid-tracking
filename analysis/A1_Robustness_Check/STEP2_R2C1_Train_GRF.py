# Databricks notebook source
import os
import pyspark

from pyspark.sql.types import *
from pyspark.sql import Row
from pyspark.sql import functions as F

# COMMAND ----------

import pandas as pd
import numpy as np

# COMMAND ----------

from econml.grf import RegressionForest


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Load Datasets

# COMMAND ----------

base_data_path = "/mnt/users/zilongwang/TLGRF"
df_path = os.path.join(base_data_path, "TLGRF_R2C1_data")
fd_odd_path = os.path.join(base_data_path, "TLGRF_R2C1_data_fd_odd")
fd_even_path = os.path.join(base_data_path, "TLGRF_R2C1_data_fd_even")

# COMMAND ----------

df_fd_even = spark.read.parquet(fd_even_path)

display(df_fd_even)

# COMMAND ----------

df_fd_odd = spark.read.parquet(fd_odd_path)
display(df_fd_odd)

# COMMAND ----------

t_max_odd = df_fd_odd.agg(F.max("t")).collect()[0][0]
t_max_even = df_fd_even.agg(F.max("t")).collect()[0][0]
t_max = max(t_max_odd, t_max_even)
print(t_max)


# COMMAND ----------

from sklearn.model_selection import GridSearchCV


# COMMAND ----------

import multiprocessing as mp

# Predefine param grid
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_leaf": [2, 5, 10],
    "max_features": ["auto", "sqrt", 0.5]
}

# Step 1: Pre-materialize all needed data
def collect_pd(t):
    df_source = df_fd_even if t % 2 == 0 else df_fd_odd
    return (t, df_source.filter(F.col("t") <= t).toPandas())

t_index_range = list(range(21, t_max + 1))
#t_index_range = list(range(21, 50 + 1))

materialized_data = [collect_pd(t) for t in t_index_range]

# Step 2: Parallelizable function
def process_t_index(t_and_df):
    t_index, df_pd = t_and_df

    # Extract features and targets
    X = df_pd[[f"X_{i}" for i in range(1, 7)]].values
    y_log = df_pd["d_log_I"].values
    y_lin = df_pd["d_I"].values

    # Fit mis-specified model
    grid_log = GridSearchCV(
        estimator=RegressionForest(),
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_log.fit(X, y_log)
    best_forest_log = grid_log.best_estimator_

    # Fit correctly specified model
    grid_lin = GridSearchCV(
        estimator=RegressionForest(),
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_lin.fit(X, y_lin)
    best_forest_lin = grid_lin.best_estimator_

    # Predict for t = t_index
    df_pred = df_pd[df_pd["t"] == t_index]
    X_pred = df_pred[[f"X_{i}" for i in range(1, 7)]].values

    y_pred_log = best_forest_log.predict(X_pred)
    y_pred_lin = best_forest_lin.predict(X_pred)

    # Get current I_t value
    I_tc = df_pred["I_tc"].values

    # Compute 7-day ahead forecasts

    # This is assuming we knew the true form
    pred_I_7ahead_lin = I_tc + 7 * y_pred_lin
    pred_I_7ahead_log = I_tc * np.exp(7 * y_pred_log)

    lengths = {
        "c": len(df_pred["c"]),
        "t": len(df_pred),
        "I_tc": len(I_tc),
        "pred_d_log_I": len(y_pred_log),
        "pred_d_I": len(y_pred_lin),
        "r_tc": len(df_pred["r_tc"]),
    }
    print(f"Lengths at t={t_index}: {lengths}")

    pred_df = pd.DataFrame({
        "c": df_pred["c"].values.ravel(),
        "t": [t_index] * len(df_pred),
        "I_tc": I_tc.ravel(),
        "pred_d_log_I": y_pred_log.ravel(),
        "pred_d_I": y_pred_lin.ravel(),
        "r_tc": df_pred["r_tc"].values.ravel(),
    })

    return pred_df, best_forest_log, best_forest_lin



# Step 3: Parallel execution
with mp.Pool(processes=mp.cpu_count()) as pool:
    results = pool.map(process_t_index, materialized_data)

# Step 4: Unpack results
all_preds = [res[0] for res in results]
best_estimators_log = {t: res[1] for (t, _), res in zip(materialized_data, results)}
best_estimators_lin = {t: res[2] for (t, _), res in zip(materialized_data, results)}

# Step 5: Combine predictions
results_df = pd.concat(all_preds, ignore_index=True)

# Done
display(results_df)


# COMMAND ----------

if False:
    # Prepare data
    all_preds = []
    best_estimators_log = {}
    best_estimators_lin = {}
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_leaf": [2, 5, 10],
        "max_features": ["auto", "sqrt", 0.5]
    }
    for t_index in range(21, 101):

        # Step 1: Filter PySpark DataFrame based on t_index
        if t_index % 2 == 0:
            df_pd = df_fd_even.filter(F.col("t") <= t_index).toPandas()
        else:
            df_pd = df_fd_odd.filter(F.col("t") <= t_index).toPandas()

        # Step 2: Extract features and targets
        X = df_pd[[f"X_{i}" for i in range(1, 7)]].values
        y_log = df_pd["d_log_I"].values  # Mis-specified (log-linear)
        y_lin = df_pd["d_I"].values      # Correct (linear)

        # Step 3: Train Regression Forests
        grid_log = GridSearchCV(
            estimator=RegressionForest(),
            param_grid=param_grid,
            cv=3,  # Cross-validation splits
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_log.fit(X, y_log)
        best_forest_log = grid_log.best_estimator_

        grid_lin = GridSearchCV(
            estimator=RegressionForest(),
            param_grid=param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_lin.fit(X, y_lin)
        best_forest_lin = grid_lin.best_estimator_


        # Step 4: Predict for t == current t_index
        df_pred = df_pd[df_pd["t"] == t_index]
        X_pred = df_pred[[f"X_{i}" for i in range(1, 7)]].values

        y_pred_log = best_forest_log.predict(X_pred)
        y_pred_lin = best_forest_lin.predict(X_pred)

        # Step 5: Output predictions
        pred_df = pd.DataFrame({
            "c": df_pred["c"].values.ravel(),
            "t": [t_index] * len(df_pred),
            "pred_d_log_I": y_pred_log.ravel(),
            "pred_d_I": y_pred_lin.ravel(),
            "r_tc": df_pred["r_tc"].values.ravel()  # ground truth growth rate
        })
        all_preds.append(pred_df)

    # Concatenate all predictions
    results_df = pd.concat(all_preds, ignore_index=True)

    display(results_df)


# COMMAND ----------

# Convert Pandas DataFrame to PySpark DataFrame
results_spark_df = spark.createDataFrame(results_df)

# Save as Parquet
results_spark_df.write.mode("overwrite").parquet(os.path.join(base_data_path, "R2C1_predictions.parquet"))

if False:
    # You can also persist the best models using joblib
    import joblib
    for t in best_estimators_log:
        joblib.dump(best_estimators_log[t], os.path.join(base_data_path, f"best_forest_log_t{t}.pkl"))
        joblib.dump(best_estimators_lin[t], os.path.join(base_data_path,f"best_forest_lin_t{t}.pkl"))

# COMMAND ----------

results_spark_df = results_spark_df.orderBy(["t","c"])
display(results_spark_df)

# COMMAND ----------

