# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v5"
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Lab 03 — Scoring the fleet in batch
# MAGIC
# MAGIC By the end of this notebook you'll have scored the `@prod` model from Unity Catalog two ways:
# MAGIC from a real-time serving endpoint for one engine on demand, and as a batch over the whole
# MAGIC fleet. Both start from the same featurized telemetry.
# MAGIC
# MAGIC Real-time suits a caller that needs a single engine scored on the spot. Batch suits maintenance
# MAGIC planning, where you score the fleet on a cadence to decide what to service next rather than on
# MAGIC every sensor tick.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure the workshop
# MAGIC
# MAGIC Set up the environment and redefine the functions this step needs: read the deploy-written
# MAGIC config, point MLflow at Unity Catalog so `models:/…@prod` resolves to the registered model, and
# MAGIC redefine the shared `featurize` transform.

# COMMAND ----------

import json

import mlflow
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType
from pyspark.sql.window import Window

# Unity Catalog the workshop uses. Notebooks run on Databricks and can't read the
# repo's local .env, so set this to the catalog you deployed into (the `catalog`
# value in .env, or the infra default). It must be an existing catalog.
CATALOG = "pdm_ws_catalog"

CONFIG_PATH = f"/Volumes/{CATALOG}/config/outputs/deploy.json"
with open(CONFIG_PATH) as f:
    cfg = json.load(f)
SCHEMA = cfg["data_schema"]

spark.sql(f"USE {CATALOG}.{SCHEMA}")
mlflow.set_registry_uri("databricks-uc")

MODEL_NAME = f"{CATALOG}.{SCHEMA}.engine_health_forecast"
MODEL_URI = f"models:/{MODEL_NAME}@prod"

# COMMAND ----------

def featurize(cycles, norm_stats):
    """Turn raw engine cycles into the model's features, using the training statistics."""
    EPSILON = 1e-8
    WINDOW = 15       # smoothing window in cycles

    # gd_norm_stats carries only the sensors we kept, so recover that set from its columns.
    active = [c[:-5] for c in norm_stats.columns if c.endswith("_mean")]
    noise = ["sensor_7", "sensor_10", "sensor_12", "sensor_16"]
    redundant = ["sensor_4", "sensor_13", "sensor_14"]
    retained = [c for c in active if c not in noise + redundant]

    df = cycles.withColumn(
        "op_condition",
        F.concat_ws(
            "-",
            F.format_string("%03.0f", F.round("setting_1", 0)),
            F.format_string("%05.2f", F.round("setting_2", 2)),
            F.format_string("%03.0f", F.round("setting_3", 0)),
        ),
    ).join(F.broadcast(norm_stats), on="op_condition", how="left")

    for c in active:
        safe_std = F.coalesce(F.col(f"{c}_std"), F.lit(0.0))
        df = df.withColumn(f"{c}_norm", (F.col(c) - F.col(f"{c}_mean")) / (safe_std + F.lit(EPSILON)))

    w = Window.partitionBy("engine_family", "unit_number").orderBy("time_cycles").rowsBetween(-WINDOW + 1, 0)
    for c in retained:
        df = df.withColumn(f"{c}_norm_ma_{WINDOW}", F.mean(f"{c}_norm").over(w))
        df = df.withColumn(f"{c}_norm_std_{WINDOW}", F.coalesce(F.stddev(f"{c}_norm").over(w), F.lit(0.0)))

    features = [f"{c}_norm_ma_{WINDOW}" for c in retained] + [f"{c}_norm_std_{WINDOW}" for c in retained]
    return df.select("engine_family", "source_flag", "unit_number", "time_cycles", "op_condition", *features)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Featurize the incoming telemetry
# MAGIC
# MAGIC In production the telemetry lands raw, and a featurization pipeline turns it into the model's
# MAGIC features before anything scores it. That pipeline isn't part of this lab, so we run `featurize`
# MAGIC inline over the fleet to stand in for it, and both inference paths below start from its output.
# MAGIC
# MAGIC The still-running fleet here is the test split of `sl_engine_cycles`, the held-out engines we
# MAGIC reuse from training as a stand-in for units currently in service.

# COMMAND ----------

source_df = spark.table("sl_engine_cycles").where("source_flag = 'test'")
source_features_df = featurize(source_df, spark.table("gd_norm_stats"))

display(source_features_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Real-time inference
# MAGIC
# MAGIC The registered model can sit behind a low-latency endpoint for a caller that needs a single
# MAGIC engine scored on demand. Model Serving stands one up straight from the Unity Catalog model, with
# MAGIC no retraining or repackaging. Serving pins a concrete version, so we resolve what `@prod` points
# MAGIC at and deploy that; the endpoint scales to zero when idle, so it costs nothing between calls.
# MAGIC
# MAGIC Deploying an endpoint and calling it are live actions, so whether to run the next two cells is
# MAGIC the presenter's call. The deploy cell waits for the endpoint to come up, which takes a few
# MAGIC minutes on a first deploy.

# COMMAND ----------

import time

from mlflow import MlflowClient
from mlflow.deployments import get_deploy_client

ENDPOINT = "engine-health-forecast"
prod_version = MlflowClient(registry_uri="databricks-uc").get_model_version_by_alias(MODEL_NAME, "prod").version

client = get_deploy_client("databricks")
client.create_endpoint(
    name=ENDPOINT,
    config={
        "served_entities": [{
            "entity_name": MODEL_NAME,
            "entity_version": prod_version,
            "workload_size": "Small",
            "scale_to_zero_enabled": True,
        }],
    },
)

# create_endpoint returns before provisioning finishes, so wait until the endpoint is ready.
while client.get_endpoint(ENDPOINT)["state"]["ready"] != "READY":
    time.sleep(30)

# COMMAND ----------

# MAGIC %md
# MAGIC A caller scores one engine by sending its recent featurized cycles to the endpoint in the
# MAGIC `dataframe_records` shape the model's signature enforces: the feature columns from above,
# MAGIC covering at least the 30-cycle window the model reads. The endpoint returns that engine's
# MAGIC predicted RUL in cycles.
# MAGIC
# MAGIC A production pipeline calls it the same way. It featurizes the raw telemetry upstream with this
# MAGIC same `featurize`, keeps a rolling buffer of each engine's latest featurized cycles, and its
# MAGIC streaming or app layer posts that buffer to the endpoint whenever it needs a fresh estimate. The
# MAGIC cell below pulls one engine's featurized cycles and makes exactly that call.

# COMMAND ----------

one_engine = source_features_df.where("engine_family = 'FD001' AND unit_number = 1").toPandas()

client.predict(
    endpoint=ENDPOINT,
    inputs={"dataframe_records": one_engine.to_dict(orient="records")},
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch inference
# MAGIC
# MAGIC The standard way to score a registered model across a fleet is `mlflow.pyfunc.spark_udf`, which
# MAGIC wraps the model as a Spark UDF and rebuilds its logged environment on every executor. That is
# MAGIC what normally lets a model run on the workers even when the base image doesn't
# MAGIC carry its dependencies.
# MAGIC
# MAGIC It doesn't fit this model. `spark_udf` scores one row at a time: each input row maps to one
# MAGIC prediction. Our LSTM needs a full engine trajectory — the window of cycles ending at the point
# MAGIC we're scoring, grouped per engine — so a row-at-a-time UDF has no way to hand it that sequence.
# MAGIC The cell below shows the shape of the attempt; it can't score this model correctly, so it's
# MAGIC left commented out.

# COMMAND ----------

# Commented out on purpose: spark_udf scores row by row and can't pass a per-engine trajectory
# to the model, so this does not work here. Shown for contrast only.
# predict_udf = mlflow.pyfunc.spark_udf(spark, MODEL_URI, env_manager="virtualenv")
# predictions = source_features_df.withColumn(
#     "predicted_rul", predict_udf(*[source_features_df[c] for c in source_features_df.columns])
# )

# COMMAND ----------

# MAGIC %md
# MAGIC That leaves two ways to score the fleet with this model:
# MAGIC
# MAGIC 1. **Deploy, infer, destroy.** Stand up the serving endpoint (the deploy cell above), push the
# MAGIC    fleet through it in batches, and tear it down when the run finishes. The endpoint carries the
# MAGIC    model's environment, so `torch` is present and the work runs off the driver; you pay only for
# MAGIC    the minutes the run takes.
# MAGIC 2. **Score on the driver.** Collect the featurized fleet to the driver and score it with pandas.
# MAGIC    It's the simplest path and fine for a fleet this size, but it pulls every row onto one node,
# MAGIC    so it won't hold at true fleet scale — that's where option 1 earns its keep.
# MAGIC
# MAGIC We run option 2 here.

# COMMAND ----------

model = mlflow.pyfunc.load_model(MODEL_URI)

prediction_schema = StructType([
    StructField("engine_family", StringType()),
    StructField("unit_number", IntegerType()),
    StructField("predicted_rul", DoubleType()),
])

features_pdf = source_features_df.toPandas()
predictions_pdf = features_pdf.groupby(["engine_family", "unit_number"], group_keys=False).apply(
    lambda pdf: model.predict(pdf)
)
predictions = spark.createDataFrame(predictions_pdf, schema=prediction_schema)

display(predictions.limit(5))
