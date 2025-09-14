# Databricks notebook source
import os
from pyspark.sql.types import *
from pyspark.sql import Row
from pyspark.sql import functions as F

# COMMAND ----------

base_data_path = "/mnt/users/zilongwang/TLGRF_linear"

# COMMAND ----------

import numpy as np
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType

# Set random seed
np.random.seed(42)

# Parameters
T = 365  # time periods
C = 1000  # counties
K = 6    # features

# Storage
records = []
C_history = {}  # (c, t) -> cumulative cases C_tc

for c in range(1, C + 1):
    C_tc = 100.0  # initialize cumulative
    for t in range(1, T + 1):
        X = np.random.uniform(low=0.0, high=1.0, size=K)
        rate = 10*float(np.sum(X[:2]))
        intercept = float(np.sum(X[:]))

        C_tc += rate  # strictly non-negative accumulation
        C_history[(c, t)] = C_tc

        # Define I_tc := C_tc - C_{t-22,c}
        #if t > 22:
        #    C_lag22 = C_history[(c, t - 22)]
        #    I_tc = C_tc - C_lag22
        #else:
        I_tc = C_tc

        log_C_tc = float(np.log(C_tc + 0.001))
        log_I_tc = float(np.log(I_tc + 0.001))

        row_data = {
            "t": t,
            "c": c,
            "C_tc": float(C_tc),
            "log_C_tc": log_C_tc,
            "I_tc": float(I_tc),
            "log_I_tc": log_I_tc,
            "alpha_tc": intercept,
            "r_tc": rate
        }

        for k in range(K):
            row_data[f"X_{k+1}"] = float(X[k])

        records.append(Row(**row_data))

# Define schema
schema = StructType([
    StructField("t", IntegerType(), False),
    StructField("c", IntegerType(), False),
    StructField("C_tc", DoubleType(), False),
    StructField("log_C_tc", DoubleType(), False),
    StructField("I_tc", DoubleType(), False),
    StructField("log_I_tc", DoubleType(), False),
    StructField("alpha_tc", DoubleType(), False),
    StructField("r_tc", DoubleType(), False),
] + [StructField(f"X_{k+1}", DoubleType(), False) for k in range(K)])

# Create Spark DataFrame
df_spark = spark.createDataFrame(records, schema=schema)


# COMMAND ----------

display(df_spark)

#df.to_csv(df_path, index=False)
#C_logistic_model.write().overwrite().save("/mnt/users/zilongwang/sku_rank/sku_rank_holdout_C_logistic_model")


# COMMAND ----------

df_path = os.path.join(base_data_path, "TLGRF_R2C1_data")
df_spark.write.mode("overwrite").parquet(df_path)


# COMMAND ----------

import matplotlib.pyplot as plt

# Convert to pandas for plotting
df_pandas = df_spark.select("t", "c", "I_tc").filter(F.col("t") >= 23).toPandas()

# Create figure and axis
plt.figure(figsize=(12, 6))

# Plot I_tc over time for each county using pure matplotlib
for c in sorted(df_pandas["c"].unique())[:4]:
    county_data = df_pandas[df_pandas["c"] == c]
    plt.plot(
        county_data["t"],
        county_data["I_tc"],
        label=f"County {c}"
    )

# Set labels and title
plt.xlabel("Time (t)")
plt.ylabel(r"$I_{t,c}$ (Observed Outcome)")
plt.title(r"Observed Outcome Incident $I_{t,c}$ Over Time by County")

# Add legend
plt.legend(title="County", bbox_to_anchor=(1.05, 1), loc="upper left")

# Improve layout
plt.tight_layout()

# Show plot
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Odd and Even

# COMMAND ----------

from pyspark.sql import Window
import pyspark.sql.functions as F



# COMMAND ----------

# Step 1: Filter for t >= 20
#df_filtered = df_spark.filter("t >= 20")

# Step 2: Create lagged columns by county
window_spec = Window.partitionBy("c").orderBy("t")
df_with_lags = df_spark.withColumn("I_prev", F.lag("I_tc").over(window_spec)) \
                          .withColumn("log_I_prev", F.lag("log_I_tc").over(window_spec))

# Step 3: Filter rows where lag is not null and time >= 20
df_model = df_with_lags.filter(F.col("I_prev").isNotNull())

# Step 4: Add dependent variables
df_model = df_model.withColumn("d_log_I", F.col("log_I_tc") - F.col("log_I_prev")) \
                   .withColumn("d_I", F.col("I_tc") - F.col("I_prev"))


# COMMAND ----------

# Even
df_model_even = df_model.filter("t % 2 == 0")

# Step 5: Collect to pandas
df_fd_even = df_model_even.select(
    "t", "c", "I_tc", "log_I_tc", "r_tc", "d_log_I", "d_I",
    "X_1", "X_2", "X_3", "X_4", "X_5", "X_6"
)

display(df_fd_even)

df_fd_even_path = os.path.join(base_data_path, "TLGRF_R2C1_data_fd_even")
df_fd_even.write.mode("overwrite").parquet(df_fd_even_path)


# COMMAND ----------

# Even
df_model_odd = df_model.filter("t % 2 == 1")

# Step 5: Collect to pandas
df_fd_odd = df_model_odd.select(
    "t", "c", "I_tc", "log_I_tc", "r_tc", "d_log_I", "d_I",
    "X_1", "X_2", "X_3", "X_4", "X_5", "X_6"
)


display(df_fd_odd)


df_fd_odd_path = os.path.join(base_data_path, "TLGRF_R2C1_data_fd_odd")
df_fd_odd.write.mode("overwrite").parquet(df_fd_odd_path)

# COMMAND ----------

