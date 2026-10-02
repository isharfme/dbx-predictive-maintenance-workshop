# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v5"
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Lab 02 — Train the engine health model on GPU, track, and register it
# MAGIC
# MAGIC By the end of this notebook you'll have trained a PyTorch LSTM on a GPU, tracked the run in
# MAGIC MLflow, and registered the model to Unity Catalog behind a `@prod` alias — ready for the
# MAGIC scheduled batch scoring in Lab 03. The single featurization shared by training and serving is
# MAGIC what keeps the model from meeting differently-built features than it trained on.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Attach to serverless GPU
# MAGIC
# MAGIC This lab trains on a GPU, and on Databricks you reach one by attaching the notebook to
# MAGIC **serverless GPU** compute — no cluster to size, no drivers to install:
# MAGIC
# MAGIC 1. Click the compute selector (top right) → **Serverless GPU**.
# MAGIC 2. Open the **Environment** panel → set **Accelerator** to **A10** (Serverless GPU Small).
# MAGIC 3. Set **Base environment** to **AI v5** — it bundles `torch`, MLflow, and the Volume data loader.
# MAGIC 4. Click **Apply**.
# MAGIC
# MAGIC A recurrent model this size trains comfortably on one A10, so a single GPU is all the lab needs.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure the workshop
# MAGIC
# MAGIC MLflow is the workspace's built-in experiment tracker and model registry: it records
# MAGIC each training run and holds the versioned, governed models it produces. Two lines wire
# MAGIC it up. `set_registry_uri("databricks-uc")` sends registered models to Unity Catalog, so
# MAGIC a model is governed by the same catalog as its data. `set_experiment` points MLflow at a
# MAGIC folder in your workspace where this project's runs collect — the tracking server is part
# MAGIC of the workspace, so there's nothing to stand up.

# COMMAND ----------

import json

import mlflow

# Unity Catalog the workshop uses. Notebooks run on Databricks and can't read the
# repo's local .env, so set this to the catalog you deployed into (the `catalog`
# value in .env, or the infra default). It must be an existing catalog.
CATALOG = "pdm_ws_catalog"

CONFIG_PATH = f"/Volumes/{CATALOG}/config/outputs/deploy.json"
with open(CONFIG_PATH) as f:
    cfg = json.load(f)
SCHEMA = cfg["data_schema"]

spark.sql(f"USE {CATALOG}.{SCHEMA}")

# When runs in a Databricks notebook an experiment automatically tied to the notebook's workspace path.
# Outside a notebook (a job/script with no experiment configured) runs fall back to the Default experiment
# Setting it explicitly gives one named, predictable experiment that all runs land in regardless of
# which notebook or job produced them.
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/pdm_ws_rul")

# COMMAND ----------

feature_table = spark.table("ft_engine_telemetry_features")
feature_cols = sorted(c for c in feature_table.columns if c.endswith(("_ma_15", "_std_15")))
display(feature_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Shape the features into sequences on a Volume
# MAGIC
# MAGIC Engine degradation is a temporal process: one cycle's reading says far less than the trajectory
# MAGIC of that reading over the cycles before it. An LSTM learns from that trajectory, so the flat
# MAGIC feature table becomes a three-dimensional `[samples, time_steps, features]` tensor — a window
# MAGIC slid over each engine's cycles, labeled with the RUL at its final cycle. A sequence length of 30
# MAGIC trades enough history to show a trend against still being able to score an engine early in life.

# COMMAND ----------

# MAGIC %md
# MAGIC For single-node training, you can convert Apache Spark DataFrames into pandas DataFrames using the PySpark method toPandas(), and then optionally convert to NumPy format using the PySpark method to_numpy().
# MAGIC
# MAGIC For large Delta tables that are too big to convert with toPandas(), export the data to a Unity Catalog volume and load it directly using PyTorch or Hugging Face.
# MAGIC
# MAGIC We deliberately choose the later path to show how it may be implemented . The LSTM needs each engine's cycles contiguous and in time order we repartition data so that each file contain full time sorted set of data. 

# COMMAND ----------

# MAGIC %md
# MAGIC That contiguous, time-ordered layout is what windowing needs and what stochastic gradient descent does not: read straight through, a batch would be one engine's consecutive cycles, and because RUL falls monotonically as an engine ages, the labels would arrive sorted — both deny the optimizer the mixed batches it assumes. So the loader keeps an in-memory shuffle buffer, drawing each sample at random from the several thousand it holds so engines and RUL levels mix back together as it streams. That holds while one engine's cycles are small next to the buffer; at larger scale you'd interleave several shards at once and split a very long series into fixed-length chunks, so the buffer always spans many units.

# COMMAND ----------

train_features_path = f"/Volumes/{CATALOG}/{SCHEMA}/ops/train_features"

(
    feature_table
    .repartition(8, "engine_family", "unit_number")
    .sortWithinPartitions("engine_family", "unit_number", "time_cycles")
    .write.mode("overwrite")
    .parquet(train_features_path)
)

# COMMAND ----------

# MAGIC %md
# MAGIC Both training and serving require window built identically or the model meets differently-shaped inputs than it trained on. window_at function is shared between train and serving to ensure windowing consistency.

# COMMAND ----------

def window_at(values, end, length):
    """Return the length-`length` window of `values` ending at row `end` (inclusive).

    Left-pads with the first row when there aren't enough cycles yet, so every cycle is
    scoreable. `values` is a [T, features] array already sorted by cycle.
    """
    win = values[max(0, end - length + 1) : end + 1]
    if len(win) < length:
        pad = np.repeat(values[:1], length - len(win), axis=0)
        win = np.concatenate([pad, win], axis=0)
    return win.astype("float32")

# COMMAND ----------

# MAGIC %md
# MAGIC A production fleet's features are far larger than any single machine's memory, so we don't pull them to the driver. We export the feature table to a Unity Catalog Volume as Parquet — engines contiguous and time-ordered within each shard — and let the GPU read it back directly. That's the Databricks-recommended path for feeding PyTorch from Delta at scale: the loader streams one shard at a time, so memory is bounded by shard size rather than by the fleet, and the windows are sliced as the GPU trains. The same layout reads identically on one GPU or, unchanged, across many.
# MAGIC
# MAGIC A window of 30 cycles is long enough for the LSTM to read a degradation trend, yet short enough to still score an engine early in its life.

# COMMAND ----------

import random

import numpy as np
import pandas as pd
import torch
from serverless_gpu.data import UCVolumeDataset
from torch.utils.data import IterableDataset

L = 30

class SequenceDataset(IterableDataset):
    """Slides `window_at` over the feature shards a UCVolumeDataset streams from a Volume.

    With `label_col`, yields `(window, label)` for training. Without it, yields
    `(window, engine_family, unit_number)` for scoring — the test fleet and real inference
    input carry no label, so the ground truth is joined to the predictions afterward.
    `last_only` keeps just each engine's final cycle. A shuffle buffer breaks up the
    per-engine ordering so a training batch isn't all one engine's cycles.
    """

    def __init__(self, volume_path, label_col=None, last_only=False, shuffle_buffer=0):
        self.paths = UCVolumeDataset(volume_path)  # local-cached shard paths, sharded across workers
        self.label_col = label_col
        self.last_only = last_only
        self.shuffle_buffer = shuffle_buffer

    def _samples(self):
        for path in self.paths:
            if not path.endswith(".parquet"):
                continue
            shard = pd.read_parquet(path)
            for (family, unit), g in shard.groupby(["engine_family", "unit_number"], sort=False):
                values = g[feature_cols].to_numpy("float32")
                labels = g[self.label_col].to_numpy("float32") if self.label_col else None
                for t in ([len(g) - 1] if self.last_only else range(len(g))):
                    window = torch.from_numpy(window_at(values, t, L))
                    if labels is None:
                        yield window, family, int(unit)
                    else:
                        yield window, torch.tensor(labels[t])

    def __iter__(self):
        if not self.shuffle_buffer:
            yield from self._samples()
            return
        buf = []
        for sample in self._samples():
            buf.append(sample)
            if len(buf) >= self.shuffle_buffer:
                yield buf.pop(random.randrange(len(buf)))
        random.shuffle(buf)
        yield from buf

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the network
# MAGIC
# MAGIC `nn.LSTM` is PyTorch's recurrent layer, but on its own it returns a hidden state for
# MAGIC every time step, not a prediction. So we wrap it: a linear head reads the last step and
# MAGIC maps it to a single RUL, and the `forward` method ties the two together into one module
# MAGIC we can train and later hand to MLflow.

# COMMAND ----------

from torch import nn


class LSTMNet(nn.Module):
    def __init__(self, n_features, hidden, layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)                          # [batch, L, hidden]
        return self.head(out[:, -1, :]).squeeze(-1)    # RUL from the last time step

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fit the model and track it in MLflow
# MAGIC
# MAGIC Fitting and tracking are two different jobs. PyTorch fits the weights — the optimizer loop
# MAGIC below is what learns, reading its windows straight off the Volume through the streaming loader.
# MAGIC MLflow records what happened: `start_run` opens a run on the workspace's tracking server, and
# MAGIC the `log_` calls attach the hyperparameters, the training table as a tracked input for lineage,
# MAGIC and the loss at each epoch. Everything we log from here on — the metrics we compute later, the
# MAGIC plot, and the model itself — lands on this one run, so it reads as a single story.
# MAGIC
# MAGIC Open the **Experiments** icon in the sidebar and select this run while it trains: the
# MAGIC parameters are already there and the loss curve fills in live as the epochs log.
# MAGIC
# MAGIC > Note. Training runs for several minutes. You don't have to watch it — read ahead
# MAGIC > through the featurization and evaluation steps, then come back once the loss has
# MAGIC > settled.

# COMMAND ----------

from serverless_gpu.data import DataLoader

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

print(f"Training on {device}.")

ARCH = {"n_features": len(feature_cols), "hidden": 64, "layers": 2, "dropout": 0.2}
LR, EPOCHS, BATCH = 1e-3, 20, 512

mlflow.end_run()  # close any stale run from a previous failed attempt
mlflow.start_run()
mlflow.log_input(
    mlflow.data.from_spark(feature_table, table_name=f"{CATALOG}.{SCHEMA}.ft_engine_telemetry_features"),
    context="training",
)
mlflow.log_params({**ARCH, "lr": LR, "epochs": EPOCHS, "window": L})

net = LSTMNet(**ARCH).to(device)
opt = torch.optim.Adam(net.parameters(), lr=LR)
loss_fn = nn.MSELoss()
scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
loader = DataLoader(
            SequenceDataset(train_features_path, 
                            "RUL_norm", 
                            shuffle_buffer=8192), 
            num_workers=8,
            batch_size=BATCH,
            pin_memory=True
)

for epoch in range(EPOCHS):
    net.train()
    running, seen = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device):
            loss = loss_fn(net(xb), yb)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        running += loss.item() * len(xb)
        seen += len(xb)
    epoch_mse = running / max(seen, 1)
    mlflow.log_metric("train_mse", epoch_mse, step=epoch)   # fills the run's live loss curve
    print(f"epoch {epoch + 1:>2}/{EPOCHS}  train MSE {epoch_mse:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Featurize the test engines
# MAGIC
# MAGIC `featurize` is the one transform that turns raw settings and sensors into the model's
# MAGIC features. It exists so the identical computation runs in the two places that meet new
# MAGIC data: here, on the test engines before we evaluate, and in the production pass below, on
# MAGIC incoming telemetry before the model scores it. Both call it with the training statistics
# MAGIC in `gd_norm_stats`, so a reading is scaled the same way wherever it enters — which is
# MAGIC what makes an evaluation, and a production prediction, comparable to what the model
# MAGIC learned.
# MAGIC
# MAGIC It's a deliberate copy of Lab 01's gold cells, kept in step by hand (see `CLAUDE.md`).
# MAGIC In production this transform belongs in one function called from feature-build, training,
# MAGIC and serving, so there's a single definition to keep honest.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window


def featurize(cycles, norm_stats):
    """Turn raw engine cycles into the model's features, using the training statistics.

    Constants and column lists mirror Lab 01's gold cells exactly, on purpose — see CLAUDE.md.
    """
    EPSILON = 1e-8    # matches Lab 01
    WINDOW = 15       # smoothing window in cycles, matches Lab 01

    # gd_norm_stats only carries the sensors Lab 01 kept, so recover that set from its columns.
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
# MAGIC The test fleet stands in for production, so we featurize it with the very same `featurize`
# MAGIC the model uses at serving time and write those features to the Volume — features only, no
# MAGIC labels, exactly as real inference would see them. The held-out truth in `sl_rul_labels` is
# MAGIC brought in later, only to score the predictions, never as model input.

# COMMAND ----------

RUL_CAP = 125  # matches Lab 01's cap on the training target

test_df = spark.table("sl_engine_cycles").where("source_flag = 'test'")

# Features only — the test fleet stands in for production input, which carries no labels.
test_features = featurize(test_df, spark.table("gd_norm_stats"))
display(test_features.limit(5))

# COMMAND ----------

test_features_path = f"/Volumes/{CATALOG}/{SCHEMA}/ops/test_features"

(
    test_features
    .repartition(8, "engine_family", "unit_number")
    .sortWithinPartitions("engine_family", "unit_number", "time_cycles")
    .write.mode("overwrite")
    .parquet(test_features_path)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate on the test fleet
# MAGIC
# MAGIC We score each engine at its most recent cycle, the realistic question of how much life is
# MAGIC left given everything seen so far. The window for that is the 30 cycles ending at the
# MAGIC engine's last observed cycle. The featurized test shards stream from the Volume through the
# MAGIC same loader in `last_only` mode (one window per engine, no label); the network is still in
# MAGIC memory from training, so we score those windows directly, then join the held-out RUL to
# MAGIC measure the error.
# MAGIC
# MAGIC Two numbers describe the result. RMSE reads directly in cycles: near 15 means predictions
# MAGIC land about fifteen cycles off across a 125-cycle horizon — a reasonable starting point for
# MAGIC a compact model over all four fleets, and you improve it by comparing runs, which is what
# MAGIC the tracking is for. The C-MAPSS score is a domain-specific metric, asymmetric because
# MAGIC predicting more life than an engine has left is worse than predicting too little; its
# MAGIC absolute value means little on its own and only ranks models on the same test set. We
# MAGIC include it to show how a custom metric is computed and logged next to a standard one,
# MAGIC since your own work may turn on a metric that isn't RMSE.

# COMMAND ----------

net.eval()
rows = []
with torch.no_grad():
    for xb, family, unit in DataLoader(SequenceDataset(test_features_path, last_only=True), batch_size=BATCH):
        rul = net(xb.to(device)).cpu().numpy() * RUL_CAP  # normalized RUL back to cycles
        rows.extend(zip(family, unit.tolist(), rul.tolist()))

# The predictions carry no truth. Join the held-out labels now, only to score: `terminal_rul` is
# the RUL at each engine's last observed cycle — exactly the cycle `last_only` predicted — capped
# to the training scale.
scored = (
    spark.createDataFrame(rows, "engine_family string, unit_number int, predicted_rul double")
    .join(
        spark.table("sl_rul_labels").where("source_flag = 'test'")
        .select("engine_family", "unit_number", F.least("terminal_rul", F.lit(RUL_CAP)).alias("true_rul")),
        on=["engine_family", "unit_number"],
    )
    .toPandas()
)
pred = scored["predicted_rul"].to_numpy()
y_test = scored["true_rul"].to_numpy()

rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))

# C-MAPSS score: late predictions (pred > actual) are penalized harder than early ones.
d = pred - y_test
cmapss = float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))
print(f"Test RMSE {rmse:.2f} cycles; C-MAPSS score {cmapss:.0f}.")

# Logged onto the same run, so they sit in the Metrics tab beside the training curve.
mlflow.log_metrics({"test_rmse": rmse, "cmapss_score": cmapss})

# COMMAND ----------

# MAGIC %md
# MAGIC The scatter puts each engine's true RUL on the x-axis and the prediction on the y-axis;
# MAGIC the dashed line is a perfect model, where the two are equal. Points above it are
# MAGIC optimistic — more life predicted than the engine has — the calls the C-MAPSS score
# MAGIC penalizes. We log the plot to the run so it lands in the Artifacts tab alongside the
# MAGIC metrics.

# COMMAND ----------

import plotly.express as px

lim = float(max(y_test.max(), pred.max()))
fig = px.scatter(
    pd.DataFrame({"true_rul": y_test, "predicted_rul": pred}),
    x="true_rul", y="predicted_rul",
    labels={"true_rul": "true RUL (cycles)", "predicted_rul": "predicted RUL (cycles)"},
    title="Predicted vs. true RUL — test fleet",
)
fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim, line=dict(dash="dash"))
mlflow.log_figure(fig, "predicted_vs_true.html")
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC The model's job is to turn a featurized history into a prediction, so the `pyfunc` we log
# MAGIC carries only the windowing — the same `window_at`, so serving shapes its input exactly as
# MAGIC training did — and leaves featurization to `featurize`. Logging it on the run keeps the
# MAGIC model with the parameters and metrics that produced it, and registering it under a Unity
# MAGIC Catalog name puts the model under the same governance and lineage as the data it learned
# MAGIC from.
# MAGIC
# MAGIC Registering is also how versions accumulate. Each `log_model` with a `registered_model_name`
# MAGIC writes a new, auto-incremented version under that name — `v1` on the first run, `v2` the next —
# MAGIC and Unity Catalog keeps them all with their run, metrics, and lineage. Any version stays
# MAGIC addressable by number (`models:/<name>/1`), so you can load an earlier one to compare against
# MAGIC or roll back to; the alias step next is how you name the *current* one so callers never track
# MAGIC version numbers.

# COMMAND ----------

class PredictionModel(mlflow.pyfunc.PythonModel):
    """Featurized cycles in, one RUL (in cycles) per engine out. Carries the windowing only."""

    def __init__(self, cols, length, arch, cap):
        self.feature_cols = cols
        self.window = length
        self.arch = arch
        self.cap = cap

    def load_context(self, context):
        net = LSTMNet(**self.arch)
        net.load_state_dict(torch.load(context.artifacts["weights"], map_location="cpu"))
        net.eval()
        self.net = net

    def predict(self, context, model_input):
        keys, wins = [], []
        for key, g in model_input.sort_values("time_cycles").groupby(["engine_family", "unit_number"]):
            values = g[self.feature_cols].to_numpy("float32")
            wins.append(window_at(values, len(g) - 1, self.window))
            keys.append(key)
        with torch.no_grad():
            preds = self.net(torch.from_numpy(np.stack(wins))).numpy().ravel().astype("float64") * self.cap
        out = pd.DataFrame(keys, columns=["engine_family", "unit_number"])
        out["predicted_rul"] = preds
        return out

# COMMAND ----------

# MAGIC %md
# MAGIC `predict` calls `window_at` and constructs an `LSTMNet`, and logging the model as a
# MAGIC `PythonModel` *instance* is what carries them along. MLflow serializes that instance with
# MAGIC cloudpickle, which walks the object and packages what it finds: the constructor arguments
# MAGIC (`feature_cols`, `window`, `arch`, `cap`) are plain data, stored as they are; the
# MAGIC `PredictionModel` class is defined in the notebook, so cloudpickle serializes the class
# MAGIC itself along with the globals its methods reference. Names that resolve to an installed
# MAGIC package — `np`, `pd`, `torch` — are stored as imports and re-created from the environment
# MAGIC on load. Names that resolve to something the notebook defined — `window_at` and `LSTMNet` —
# MAGIC cloudpickle embeds by value: `window_at`'s code and `LSTMNet`'s entire class definition
# MAGIC travel inside the artifact. When Lab 03 loads the model, cloudpickle rebuilds those names,
# MAGIC so `predict`'s call to `window_at` resolves on each worker exactly as it does here.
# MAGIC
# MAGIC The windowing travels inside the artifact, so serving shapes its input exactly as training
# MAGIC did — cloudpickle captures it automatically because `predict` references it directly, and
# MAGIC dependencies you want pinned by version, like `torch`, you also name in `pip_requirements`
# MAGIC so the environment rebuilds them on load. That covers the code the model runs. `featurize`
# MAGIC runs upstream, before the model sees a row, so we keep it aligned to the model by a
# MAGIC deliberate choice at logging time — the next cell.

# COMMAND ----------

# MAGIC %md
# MAGIC That leaves `featurize`, which runs upstream — so how do you keep it aligned to the model
# MAGIC in production? The featurization note already points to one shared function called from
# MAGIC feature-build, training, and serving; the step that completes it is versioning that
# MAGIC function together with the model. `featurize` is a model dependency: the smoothing window,
# MAGIC the sensor set, and the normalization all shape the distribution the model learns on, so a
# MAGIC change to any of them calls for a fresh training run.
# MAGIC
# MAGIC Give `featurize` the same alignment the windowing already has, and ship it *with the model
# MAGIC version*. `mlflow.pyfunc.log_model` takes a `code_paths` argument that copies named modules
# MAGIC into the artifact, so `models:/…@prod` carries the exact `featurize` it registered with and
# MAGIC the scoring job runs that copy. The registered model becomes the single unit that pins the
# MAGIC transform, Unity Catalog governs the two together, and a featurization change arrives as a
# MAGIC new version to promote behind `@prod` — keeping production and its featurizer in lockstep.

# COMMAND ----------

# MAGIC %md
# MAGIC We save the weights, prime an in-memory copy on a couple of engines' featurized rows to
# MAGIC infer the model **signature**, then log and register it. The signature records the input
# MAGIC schema — the featurized columns — that MLflow enforces whenever the model is called.

# COMMAND ----------

from mlflow.models import infer_signature

MODEL_NAME = f"{CATALOG}.{SCHEMA}.engine_health_forecast"

weights_path = "/tmp/engine_health_weights.pt"
torch.save(net.cpu().state_dict(), weights_path)

# A couple of engines' featurized rows — read back from the Volume — are enough to infer the schema.
feature_input = ["engine_family", "source_flag", "unit_number", "time_cycles", "op_condition"] + feature_cols
example = spark.read.parquet(test_features_path).where("unit_number <= 2").select(*feature_input).toPandas()
primed = PredictionModel(feature_cols, L, ARCH, RUL_CAP)
primed.net = net.cpu().eval()
signature = infer_signature(example, primed.predict(None, example))

info = mlflow.pyfunc.log_model(
    name="model",
    python_model=PredictionModel(feature_cols, L, ARCH, RUL_CAP),
    artifacts={"weights": weights_path},
    signature=signature,
    input_example=example,
    pip_requirements=[
        # The endpoint serves on CPU, so pin the CPU torch build; the default PyPI wheel pulls in
        # ~4 GB of unused CUDA libraries and stretches the serving image build past 30 minutes.
        "--extra-index-url https://download.pytorch.org/whl/cpu",
        f"torch=={torch.__version__.split('+')[0]}+cpu",
        f"mlflow=={mlflow.__version__}",
        f"pandas=={pd.__version__}",
    ],
    registered_model_name=MODEL_NAME,
)
print(f"Registered version {info.registered_model_version}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Point `@prod` at the model
# MAGIC
# MAGIC Unity Catalog names versions with movable **aliases** rather than fixed stages. An alias is
# MAGIC just a label on a version — `@prod` here — and callers load `models:/<name>@prod` without
# MAGIC tracking version numbers; the names are yours to pick, not a reserved set. Because it's only a
# MAGIC pointer, rollback is the same one-line move in reverse.
# MAGIC
# MAGIC With several candidates you'd register each as `@challenger`, compare it against the current
# MAGIC `@prod` on the same test set, and move `@prod` only when it wins. This run trains a single
# MAGIC version, so we point `@prod` straight at it. Closing the run seals what we logged.

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")
client.set_registered_model_alias(MODEL_NAME, "prod", info.registered_model_version)

mlflow.end_run()
