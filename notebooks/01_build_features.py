# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ml_v5"
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Lab 01 — Feature Engineering
# MAGIC
# MAGIC By the end of this notebook you'll have a full medallion pipeline for the NASA
# MAGIC C-MAPSS jet-engine data, built on Databricks: **bronze** tables holding the raw
# MAGIC settings, sensor readings, and failure labels as they land; **silver** tables that
# MAGIC reunite them into one row per engine operating cycle plus one RUL value per engine;
# MAGIC and **gold** feature tables ready to train a model for **Remaining Useful Life
# MAGIC (RUL)**, how many operating cycles an engine has left.
# MAGIC
# MAGIC Everything runs on **serverless** compute, and every table is governed by **Unity
# MAGIC Catalog**, which tracks bronze -> silver -> gold lineage for you automatically.

# COMMAND ----------

# MAGIC %md
# MAGIC ## The dataset
# MAGIC
# MAGIC This workshop uses NASA's C-MAPSS turbofan dataset: a collection of simulated
# MAGIC aircraft engines, each run from healthy operation all the way to failure.
# MAGIC
# MAGIC The data covers four engine fleets, FD001 through FD004, and is split into a training
# MAGIC set and a test set. Each row is one operating cycle of one engine, and it holds the
# MAGIC cycle number, the engine's unit number within its fleet, the operational settings that
# MAGIC define the operating condition, and the sensor readings.
# MAGIC
# MAGIC The original dataset ships as flat text files. For this workshop we reshape it into two
# MAGIC feeds that arrive separately, the way telemetry often does in practice: the three
# MAGIC operational settings land as tab-separated files, and the 21 sensor readings arrive as
# MAGIC streaming JSON messages. Reuniting them into one row per cycle is the silver layer's job.
# MAGIC
# MAGIC The training set records each engine all the way to failure, so its last cycle is the
# MAGIC failure itself, and from that we can work out how much life remained at every earlier
# MAGIC cycle. The test set records each engine only up to some point before it fails; a
# MAGIC separate file gives the true remaining life as of each engine's final recorded cycle.
# MAGIC
# MAGIC Each fleet runs under its own range of conditions, set by the combination of the three
# MAGIC operational settings. Some fleets are held at a single steady condition for their whole
# MAGIC life, while others are cycled through six different flight conditions.
# MAGIC
# MAGIC More detail is in Saxena, Goebel, Simon, and Eklund, "Damage Propagation Modeling for
# MAGIC Aircraft Engine Run-to-Failure Simulation" (PHM 2008).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure the workshop
# MAGIC
# MAGIC Everything environment-specific comes from one deploy-written config file, so the
# MAGIC same notebook points at your own catalog with no edits. The one value we can't
# MAGIC derive is the path to that file.

# COMMAND ----------

import json

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Unity Catalog the workshop uses. Notebooks run on Databricks and can't read the
# repo's local .env, so set this to the catalog you deployed into (the `catalog`
# value in .env, or the infra default). It must be an existing catalog.
CATALOG = "pdm_ws_catalog"

# The deploy configuration file, written to the catalog's config volume by deploy.sh.
CONFIG_PATH = f"/Volumes/{CATALOG}/config/outputs/deploy.json"

with open(CONFIG_PATH) as f:
    cfg = json.load(f)

SCHEMA = cfg["data_schema"]
REGION = cfg["aws_region"]

# The published AWS flows. `engine_data/` and `engine_labels/` are the producer's
# static prefixes — the data-layout constants we keep visible here, derived from the
# one bucket name in config.
S3_SETTINGS_PATH = f"s3://{cfg['data_bucket']}/engine_data/"
S3_LABELS_PATH = f"s3://{cfg['data_bucket']}/engine_labels/"
KINESIS_STREAM = cfg["kinesis_stream_name"]

# Unity Catalog service credential that grants read access to the Kinesis stream.
# UC vends short-lived AWS credentials at read time — no keys live in this notebook.
KINESIS_SERVICE_CREDENTIAL = cfg["kinesis_service_credential_name"]

spark.sql(f"USE {CATALOG}.{SCHEMA}")
CHECKPOINT = f"/Volumes/{CATALOG}/{SCHEMA}/ops/checkpoints"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze — land the raw sources
# MAGIC
# MAGIC Bronze holds the raw data landed as-is, one table per source, so we always have a
# MAGIC faithful copy to replay from. Two of our three sources are files on S3 and one is a
# MAGIC Kinesis stream, so the layer splits naturally into a file path and a stream path.

# COMMAND ----------

# MAGIC %md
# MAGIC ### File ingests
# MAGIC
# MAGIC The producer drops tab-separated files under `engine_data/`. **Auto Loader**
# MAGIC (`cloudFiles`) ingests them incrementally and exactly once; it remembers which
# MAGIC files it has already seen in its checkpoint, so re-running only picks up new files.
# MAGIC
# MAGIC We give it an explicit schema for the seven columns we expect and turn on
# MAGIC `rescuedDataColumn`. Partway through the run the producer starts emitting an extra
# MAGIC column; because it isn't in our schema, Auto Loader parks it in `_rescued_data`
# MAGIC instead of failing the stream. You'll see that column fill in for the later files.

# COMMAND ----------

settings_schema = (
    "engine_family STRING, source_flag STRING, unit_number INT, time_cycles INT, "
    "setting_1 DOUBLE, setting_2 DOUBLE, setting_3 DOUBLE"
)

settings_stream = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("sep", "\t")
    .option("header", "true")
    .option("cloudFiles.schemaLocation", f"{CHECKPOINT}/br_engine_settings/schema")
    .option("rescuedDataColumn", "_rescued_data")
    .schema(settings_schema)
    .load(S3_SETTINGS_PATH)
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn("_source_file", F.col("_metadata.file_path"))
)

settings_query = (
    settings_stream.writeStream.option(
        "checkpointLocation", f"{CHECKPOINT}/br_engine_settings/commit"
    )
    .trigger(availableNow=True)
    .toTable(f"{CATALOG}.{SCHEMA}.br_engine_settings")
)
settings_query.awaitTermination()

# What this run actually read: numFilesProcessed counts only the files Auto Loader
# hadn't seen, so re-running with nothing new reports zero. The per-file breakdown
# grows by new files and never re-counts old ones — that's the ingest being incremental.
new_files = sum(
    int(s["metrics"].get("numFilesProcessed", 0))
    for p in settings_query.recentProgress
    for s in p["sources"]
)
total = spark.table("br_engine_settings").count()
print(f"This run ingested {new_files} new file(s) from S3; the table now holds {total}.")

display(
    spark.table("br_engine_settings")
    .groupBy("_source_file")
    .agg(F.count("*").alias("rows"), F.min("_ingested_at").alias("first_ingested"))
    .orderBy(F.col("first_ingested").desc())
    .limit(10)
)

# COMMAND ----------

# MAGIC %md
# MAGIC A quick look at where the schema drift landed. The extra column shows up in
# MAGIC `_rescued_data` only for the later files, exactly as Auto Loader promises.

# COMMAND ----------

display(
    spark.table("br_engine_settings")
    .where("_rescued_data IS NOT NULL")
    .select("_source_file", "_rescued_data")
    .limit(5)
)

# COMMAND ----------

# MAGIC %md
# MAGIC The failure labels for the truncated test engines arrive the same way, as small
# MAGIC tab-separated files under `engine_labels/`, one terminal RUL per engine instance.
# MAGIC The producer republishes them on every run, at fresh paths, so the same engine's
# MAGIC label can land more than once; a plain **batch read** of the whole prefix is the
# MAGIC simplest thing that works (not every source needs streaming), and we clean up the
# MAGIC duplicates in silver. We keep the two `_ingested_at` / `_source_file` audit columns
# MAGIC so bronze stays a faithful record of what landed.

# COMMAND ----------

rul_labels = (
    spark.read.format("csv")
    .option("sep", "\t")
    .option("header", "true")
    .schema("engine_family STRING, source_flag STRING, unit_number INT, terminal_rul INT")
    .load(S3_LABELS_PATH)
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn("_source_file", F.col("_metadata.file_path"))
)
rul_labels.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.br_rul_labels")

print(f"br_rul_labels now holds {spark.table('br_rul_labels').count()} raw label row(s).")
display(spark.table("br_rul_labels").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Stream ingest
# MAGIC
# MAGIC The sensor readings arrive on Kinesis as **batched JSON messages**. Each Kinesis
# MAGIC record is an array of independent, fully-keyed sensor rows (the engine keys plus a
# MAGIC wide row of all 21 sensors for one cycle). The producer shuffles the fleet before
# MAGIC batching, so a run's records spread across shards rather than clustering by engine
# MAGIC instance. Databricks reads Kinesis natively; we keep the raw JSON payload untouched
# MAGIC at bronze so we always have the source of record to replay from.
# MAGIC
# MAGIC > Note. Access is granted through the Unity Catalog **service credential** named in
# MAGIC > the config cell (the `serviceCredential` option); on classic compute you could
# MAGIC > use an instance profile instead, but on serverless the service credential is the
# MAGIC > way.

# COMMAND ----------

sensor_stream = (
    spark.readStream.format("kinesis")
    .option("streamName", KINESIS_STREAM)
    .option("region", REGION)
    .option("serviceCredential", KINESIS_SERVICE_CREDENTIAL)
    .option("initialPosition", "TRIM_HORIZON")
    # availableNow is best-effort on Kinesis; a longer fetch window drains more of
    # the shards per run so fewer re-runs are needed to pick up every record.
    .option("maxFetchDuration", "30s")
    .load()
    .select(
        F.col("partitionKey").alias("partition_key"),
        F.col("data").cast("string").alias("data"),  # raw JSON message, kept as-is
        F.col("shardId").alias("shard_id"),
        F.col("sequenceNumber").alias("sequence_number"),
        F.col("approximateArrivalTimestamp").alias("approximate_arrival_ts"),
    )
    .withColumn("_ingested_at", F.current_timestamp())
)

sensor_query = (
    sensor_stream.writeStream.option(
        "checkpointLocation", f"{CHECKPOINT}/br_sensor_readings/commit"
    )
    .trigger(availableNow=True)
    .toTable(f"{CATALOG}.{SCHEMA}.br_sensor_readings")
)
sensor_query.awaitTermination()

# What this run actually read: Kinesis commits the last sequence number to the
# checkpoint, so re-running with nothing new reports zero. Records read together in
# a run share an _ingested_at, so the breakdown gains a group per run, never
# re-reading records already in the table.
new_records = sum(
    int(s["metrics"]["numProcessedRecords"])
    for p in sensor_query.recentProgress
    for s in p["sources"]
)
total = spark.table("br_sensor_readings").count()
print(f"This run read {new_records} new record(s) from Kinesis; the table now holds {total}.")

display(
    spark.table("br_sensor_readings")
    .groupBy("_ingested_at")
    .agg(
        F.count("*").alias("records"),
        F.min("approximate_arrival_ts").alias("earliest_event"),
        F.max("approximate_arrival_ts").alias("latest_event"),
    )
    .orderBy(F.col("_ingested_at").desc())
    .limit(10)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver — one row per engine cycle
# MAGIC
# MAGIC In the medallion model, **silver** is where raw sources become clean, conformed,
# MAGIC trustworthy tables that any downstream use can build on. For us that means two
# MAGIC things: reunite the split engine data into **one row per engine operating cycle**,
# MAGIC and deduplicate the labels into **one RUL value per engine instance**. Silver
# MAGIC stays deliberately model-agnostic, with no features and no target, so a dashboard, our
# MAGIC model, or a different model all start from the same source of truth.
# MAGIC
# MAGIC Before writing a transformation, look at what bronze actually gave us.

# COMMAND ----------

# The settings arrive one flat row per engine cycle...
display(spark.table("br_engine_settings").limit(2))

# ...and each sensor record is a raw JSON string: an array of wide, fully-keyed rows.
display(spark.table("br_sensor_readings").select("data").limit(2))

# COMMAND ----------

# MAGIC %md
# MAGIC Each sensor message is a JSON array of wide rows, so we
# MAGIC parse it against an explicit schema (written out in full so you can see exactly
# MAGIC what one sensor row looks like) and `explode` it to one row per engine cycle. That
# MAGIC joins to the settings on the four-part key `(engine_family, source_flag,
# MAGIC unit_number, time_cycles)`; carrying `source_flag` in the key matters because train
# MAGIC and test engines reuse the same unit and cycle numbers and would otherwise collide.

# COMMAND ----------

# The parsed shape of one sensor message: an array of wide rows, keys then 21 sensors.
sensor_record_schema = """
  ARRAY<STRUCT<
    engine_family STRING, source_flag STRING, unit_number INT, time_cycles INT,
    sensor_1 DOUBLE,  sensor_2 DOUBLE,  sensor_3 DOUBLE,  sensor_4 DOUBLE,  sensor_5 DOUBLE,
    sensor_6 DOUBLE,  sensor_7 DOUBLE,  sensor_8 DOUBLE,  sensor_9 DOUBLE,  sensor_10 DOUBLE,
    sensor_11 DOUBLE, sensor_12 DOUBLE, sensor_13 DOUBLE, sensor_14 DOUBLE, sensor_15 DOUBLE,
    sensor_16 DOUBLE, sensor_17 DOUBLE, sensor_18 DOUBLE, sensor_19 DOUBLE, sensor_20 DOUBLE,
    sensor_21 DOUBLE
  >>
"""

silver = (
    spark.table("br_engine_settings")
    .select(
        "engine_family", "source_flag", "unit_number", "time_cycles",
        "setting_1", "setting_2", "setting_3",
    )
    # Parse + explode the sensor JSON to one row per engine cycle, then inner-join to the
    # settings and drop any cycle the streams redelivered — matching on the four-part key
    # and collapsing duplicates in a single pass, so every engine cycle appears once.
    .join(
        spark.table("br_sensor_readings")
        .select(F.explode(F.from_json("data", sensor_record_schema)).alias("reading"))
        .select("reading.*"),
        on=["engine_family", "source_flag", "unit_number", "time_cycles"],
        how="inner",
    )
    .dropDuplicates(["engine_family", "source_flag", "unit_number", "time_cycles"])
)

# Overwrite mode overwrites the data, but doesn't change the schema. overwriteSchema
# changes the schema to match the new data.
silver.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.sl_engine_cycles")

display(spark.table("sl_engine_cycles").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC These are the companion labels for the test engines. A test engine's readings are cut off partway through its life instead of at failure, so unlike a training engine its remaining life isn't visible in the data itself. This table supplies it: the true RUL at each test engine's last recorded cycle, which is what we measure the model's predictions against.

# COMMAND ----------

(
    spark.table("br_rul_labels")
    .dropDuplicates(["engine_family", "source_flag", "unit_number"])
    .select("engine_family", "source_flag", "unit_number", "terminal_rul")
    .write.mode("overwrite")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.sl_rul_labels")
)

display(spark.table("sl_rul_labels").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploring the training data
# MAGIC
# MAGIC Before engineering anything, look at the data: how much there is, whether it's
# MAGIC complete, how the sensors behave, and what predicts failure. We explore the **train**
# MAGIC engines only, the run-to-failure trajectories the model learns from, so nothing we
# MAGIC decide here is shaped by the test data we'll evaluate on.

# COMMAND ----------

# Create a training dataset
train_df = (
    spark.table("sl_engine_cycles")
    .filter("source_flag = 'train'")
)

setting_cols = [c for c in train_df.columns if c.startswith("setting_")]
sensor_cols = [c for c in train_df.columns if c.startswith("sensor_")]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data volume and coverage
# MAGIC
# MAGIC Notebooks move between Python and SQL cell by cell over the same data. Register the
# MAGIC DataFrame as a temporary view and the next cell can query it in SQL — here, to count
# MAGIC how many engines and cycles each fleet contributes.

# COMMAND ----------

# Create a temporary view
train_df.createOrReplaceTempView('sl_engine_cycles_train')

# COMMAND ----------

# MAGIC %sql
# MAGIC -- How much train data, and how is it spread across the fleets?
# MAGIC SELECT
# MAGIC   COALESCE(engine_family, 'Total') AS engine_family,
# MAGIC   COUNT(DISTINCT CONCAT(engine_family, '-', unit_number)) AS engine_instances,
# MAGIC   COUNT(*) AS cycle_rows
# MAGIC FROM sl_engine_cycles_train
# MAGIC GROUP BY ROLLUP(engine_family)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Missing values
# MAGIC
# MAGIC An LSTM can't train on missing values, so in a real dataset this is where we'd decide
# MAGIC how to impute or drop them. C-MAPSS is simulated and fully populated, so this is a
# MAGIC quick confirmation, not a cleaning step.

# COMMAND ----------

missing_counts = train_df.select(
    [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in train_df.columns]
).toPandas().T.reset_index()
missing_counts.columns = ["column", "missing_count"]

display(missing_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Identify operating conditions
# MAGIC
# MAGIC Each engine runs under a handful of distinct operating conditions set by the three
# MAGIC operational settings. To see how many, we fit Gaussian mixtures across a range of
# MAGIC cluster counts and let the BIC show where adding clusters stops paying off.

# COMMAND ----------

import plotly.express as px
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

settings_scaled = StandardScaler().fit_transform(
    train_df.select(setting_cols).toPandas()
)

bics = []
K_range = range(1, 10)

for k in K_range:
    gmm = GaussianMixture(n_components=k, random_state=42, covariance_type="full")
    gmm.fit(settings_scaled)
    bics.append(gmm.bic(settings_scaled))

px.line(
    x=list(K_range),
    y=bics,
    markers=True,
    labels={"x": "Number of operating conditions (K)", "y": "BIC (lower is better)"},
    title="GMM BIC by number of operating conditions",
).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Now tag each cycle with its condition. We round the three settings and concatenate
# MAGIC them into a single `op_condition` label, giving one stable marker per operating state.

# COMMAND ----------

train_df = train_df.withColumn(
    "op_condition",
    F.concat_ws(
        "-",
        F.format_string("%03.0f", F.round(F.col("setting_1"), 0)),
        F.format_string("%05.2f", F.round(F.col("setting_2"), 2)),
        F.format_string("%03.0f", F.round(F.col("setting_3"), 0))
    )
)
display(train_df.select("op_condition").distinct())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Identify flatline sensors
# MAGIC
# MAGIC Compute each sensor's standard deviation within each operating condition, then read
# MAGIC off which sensors barely move. A sensor that stays flat carries no information about
# MAGIC degradation, so it's a candidate to drop before modeling.

# COMMAND ----------

train_stats_df = spark.createDataFrame(
    train_df.groupBy("op_condition")
    .agg(
        *[F.stddev(c).alias(f"{c}_stddev") for c in sensor_cols],
        *[F.mean(c).alias(f"{c}_mean") for c in sensor_cols]
    ).collect()
)

display(train_stats_df)

# COMMAND ----------

plot_data = (
    train_df.groupBy("op_condition").agg(*[F.stddev(F.col(s)).alias(s) for s in sensor_cols])
    .toPandas()
    .set_index("op_condition")
    .T
)

px.imshow(
    plot_data,
    text_auto=".2f",
    aspect="auto",
    height=700,
    color_continuous_scale="Blues",
    labels={"x": "Operating condition", "y": "Sensor", "color": "Std dev"},
    title="Sensor variability by operating condition (0 = flatline)",
).show()

# COMMAND ----------

plot_data = (
    train_df
    .filter(F.col("engine_family") == "FD004")
    .filter(F.col("unit_number") == 1)
    .filter(F.col("op_condition") == "020-00.70-100")
    .select("time_cycles", "sensor_1", "sensor_9")
    .toPandas()
    .melt(id_vars="time_cycles", var_name="sensor", value_name="reading")
)

px.scatter(
    plot_data,
    x="time_cycles",
    y="reading",
    facet_row="sensor",
    title="Flatline vs active sensor",
).update_yaxes(matches=None).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### The target: Remaining Useful Life
# MAGIC
# MAGIC RUL is the number of operating cycles an engine has left before it fails, and that is
# MAGIC what the model predicts. Early cycles make it a hard thing to learn. A healthy engine's
# MAGIC readings barely change, so cycle 10 and cycle 100 look almost identical even though 90
# MAGIC cycles separate their true remaining life, and a model forced to map near-identical
# MAGIC inputs to very different targets just learns noise.
# MAGIC
# MAGIC Capping the target at 125 fixes this. Any engine with more than 125 cycles left is
# MAGIC labeled 125, a value the model reads as healthy with plenty of life remaining. The
# MAGIC countdown toward failure begins only inside the final 125 cycles.

# COMMAND ----------

RUL_CAP = 125

engine_window = Window.partitionBy("engine_family", "unit_number")
train_df = (
    train_df
    .withColumn(
        "RUL",
        F.least(
            F.max("time_cycles").over(engine_window) - F.col("time_cycles"),
            F.lit(RUL_CAP)
        ).cast("int")
    )
    .withColumn(  # scale the target to [0, 1] to improve training stability
        "RUL_norm",
        F.col("RUL") / F.lit(RUL_CAP)
    )
)

display(
    train_df
    .select("engine_family", "unit_number", "time_cycles", "RUL", "RUL_norm")
    .where("RUL < 125")
    .orderBy("engine_family", "unit_number", "time_cycles")
    .limit(5)
)

# COMMAND ----------

plot_data = (
    train_df
    .filter(F.col("engine_family") == "FD004")
    .filter(F.col("unit_number") == 1)
    .filter(F.col("op_condition") == "020-00.70-100")
    .select("time_cycles", "RUL", "sensor_9")
    .orderBy(F.desc("RUL"))
    .toPandas()
)

px.line(
    plot_data,
    x="RUL",
    y="sensor_9",
    title="Sensor 9 vs RUL",
).update_xaxes(autorange="reversed").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold — model-ready features
# MAGIC
# MAGIC Gold turns clean silver rows into what the model trains on. We're building toward an
# MAGIC **LSTM**, which reads a short window of consecutive cycles and learns the temporal
# MAGIC pattern itself, so the work here is to normalize the sensors that carry signal onto a
# MAGIC common scale and smooth them lightly to damp per-cycle noise. Smoothing is where we
# MAGIC stop; windowing the cycles into sequences is left to the training notebook, where the
# MAGIC sequence length stays a tunable knob.
# MAGIC
# MAGIC This is ordinary Spark (a grouped aggregation, a join, some arithmetic) on serverless
# MAGIC compute, so there's no cluster to size; you write the transformation and run it. The
# MAGIC same code that shapes one fleet scales to all four unchanged, and the output is a
# MAGIC governed Delta table whose lineage across bronze, silver, and gold Unity Catalog tracks
# MAGIC for you. From there it flows straight into training on the same platform, with no
# MAGIC export step.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Normalize each sensor within its condition
# MAGIC
# MAGIC A `sensor_2` reading at high thrust isn't comparable to the same sensor at idle, and
# MAGIC those between-condition differences are large enough that a single global scaler would
# MAGIC flatten them together and bury the gradual drift that signals degradation.
# MAGIC
# MAGIC So we normalize per condition. For each active sensor we compute its mean and standard
# MAGIC deviation within each `op_condition`, then z-score every reading against the statistics
# MAGIC for its own condition. An absolute reading becomes a relative one, how far this cycle
# MAGIC sits from normal for that operating state, and it reads the same whether the engine is
# MAGIC idling or at full thrust.

# COMMAND ----------

STD_THRESHOLD = 1e-4
EPSILON = 1e-8

train_stats_dict = train_stats_df.select(
    *[F.max(f"{c}_stddev").alias(f"{c}_max_stddev") for c in sensor_cols]
).first().asDict()

active_sensors = [c for c in sensor_cols if (train_stats_dict.get(f"{c}_max_stddev") or 0.0) > STD_THRESHOLD]
flatline_sensors = [c for c in sensor_cols if c not in active_sensors]

train_df = train_df.join(
    F.broadcast(train_stats_df),
    on="op_condition",
    how="left")

norm_exprs = []
for c in active_sensors:
    safe_std = F.coalesce(F.col(f"{c}_stddev"), F.lit(0.0))
    expr = ((F.col(c) - F.col(f"{c}_mean")) / (safe_std + F.lit(EPSILON))).alias(f"{c}_norm")
    norm_exprs.append(expr)

train_df = train_df.select("*", *norm_exprs)

display(train_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save the normalization statistics
# MAGIC
# MAGIC These per-condition means and standard deviations were fit on the training engines,
# MAGIC and anything normalized later has to reuse these exact numbers, or the model meets a
# MAGIC different scale than it learned on. The test engines in Lab 02 and live telemetry in
# MAGIC production both run through this same transform, so Lab 01, where the statistics are
# MAGIC fit, is where we persist them: a small governed table the later steps read from, one
# MAGIC source of truth. We keep only the active sensors' columns, so which sensors carry
# MAGIC signal is recoverable from the table itself.

# COMMAND ----------

gd_norm_stats = train_stats_df.select(
    "op_condition",
    *[F.col(f"{c}_mean") for c in active_sensors],
    *[F.col(f"{c}_stddev").alias(f"{c}_std") for c in active_sensors],
)
gd_norm_stats.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CATALOG}.{SCHEMA}.gd_norm_stats"
)

display(spark.table("gd_norm_stats").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prune redundant sensors
# MAGIC
# MAGIC An LSTM can handle correlated inputs, but redundant sensors still cost us: each added
# MAGIC feature adds work on every backward pass, and several sensors tracking the same
# MAGIC degradation let the model lean too hard on one failure mode. Fewer, more distinct
# MAGIC features train faster and generalize better. To decide what to keep, look at how each
# MAGIC normalized sensor correlates with RUL and with the other sensors.

# COMMAND ----------

corr = (
    train_df.select([f"{c}_norm" for c in active_sensors] + ["RUL_norm"])
    .toPandas()
    .corr(method="spearman")
)

px.imshow(
    corr,
    text_auto=".2f",
    aspect="auto",
    height=800,
    width=800,
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Spearman correlation: normalized sensors and RUL",
).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Two groups stand out as candidates to drop:
# MAGIC
# MAGIC - Sensors 7, 10, 12, and 16 correlate with RUL below 0.2 in either direction, close to noise.
# MAGIC - Sensors 4, 13, and 14 track sensors 11, 8, and 9 closely, so from each pair we keep
# MAGIC   the one with the stronger RUL correlation.

# COMMAND ----------

noise_sensors = ["sensor_7", "sensor_10", "sensor_12", "sensor_16"]
redundant_sensors = ["sensor_4", "sensor_13", "sensor_14"]
retained_sensors = [c for c in active_sensors if c not in noise_sensors + redundant_sensors]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Temporal smoothing
# MAGIC
# MAGIC A single cycle's reading is noisy, so we smooth each retained sensor with a moving
# MAGIC average and a rolling standard deviation over the preceding cycles, per engine. The
# MAGIC window of 15 cycles trades noise reduction against how quickly the features react to
# MAGIC real change.

# COMMAND ----------

window_size = 15
w_smooth = (
    Window.partitionBy("engine_family", "unit_number")
    .orderBy("time_cycles")
    .rowsBetween(-window_size + 1, 0)
)

for c in retained_sensors:
    train_df = train_df.withColumn(f"{c}_norm_ma_{window_size}", F.mean(f"{c}_norm").over(w_smooth))
    train_df = train_df.withColumn(f"{c}_norm_std_{window_size}", F.coalesce(F.stddev(f"{c}_norm").over(w_smooth), F.lit(0.0)))

smoothed_features = (
    [f"{c}_norm_ma_{window_size}" for c in retained_sensors]
    + [f"{c}_norm_std_{window_size}" for c in retained_sensors]
)

display(train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write the feature table
# MAGIC
# MAGIC We register the smoothed features as a **Feature Table** rather than a plain Delta
# MAGIC table. A Feature Table is a Delta table with an ML metadata layer on top, and that
# MAGIC layer lets the training and inference notebooks look features up by primary key
# MAGIC instead of re-deriving them: at scoring time you pass `(engine_family, unit_number,
# MAGIC time_cycles)` and the model pulls the matching feature row. Unity Catalog carries the
# MAGIC lineage from these features to the models built on them, and point-in-time joins keep
# MAGIC a training row from seeing features computed after its own cycle.
# MAGIC
# MAGIC Saving the individual features and not the feature vectors or window arrays allows provides some advantages. If you save an array, the Feature Store UI simply sees a column called feature_vector of type Array[Double]. Deep learning models depend entirely on the exact order of features in the input array. If you save a pre-assembled array and later drop a highly correlated sensor upstream, the array shrinks by one. Downstream models will blindly read the new array, feeding the wrong sensor data into the wrong network nodes without throwing an error. Named columns enforce strict schema mapping. Feature Tables are designed to be shared. Another data scientist might want to build an anomaly detection model using only the _std columns. If you lock all the data into a single array, they have to write custom code to unpack it before they can use it. By saving the individual smoothed_features columns, Databricks tracks the lineage of every single sensor independently.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

primary_keys = ["engine_family", "unit_number", "time_cycles"]

feature_df = train_df.select(primary_keys + smoothed_features + ["RUL_norm"])

fe.create_table(
    name="ft_engine_telemetry_features",
    primary_keys=primary_keys,
    df=feature_df
)
