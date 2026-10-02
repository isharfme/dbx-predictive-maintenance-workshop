# Predictive-maintenance workshop

An educational, hands-on sample workshop for anyone who wants to learn how to run
the full predictive-maintenance loop on Databricks on AWS: ingest two live data
flows, engineer features, train and evaluate a model, and score new data. The prediction target is **Remaining Useful Life (RUL)** — how
many operating cycles an engine has left before failure.

The workshop uses NASA's **C-MAPSS** jet-engine simulation dataset. C-MAPSS gives
you complete run-to-failure trajectories, which is exactly what a RUL model needs
to learn from, and it is a standard benchmark for predictive maintenance.

## How it fits together

Environment setup provisions AWS infrastructure and a local producer that replays
the dataset as two flows (static engine metadata and sensor readings). The
workshop notebooks read only those published flows on Databricks, through Unity
Catalog, so they reference UC object names rather than AWS keys.

## Prerequisites

- **Terraform** and the **AWS CLI**, authenticated to your SSO profile:
  `aws sso login --profile <your-aws-profile>`.
- **[uv](https://docs.astral.sh/uv/)** for the producer (Python ≥ 3.11).
- **Databricks CLI** with a profile for the target workspace. Pass `--profile`
  explicitly; never let a profile be auto-selected.
- **[jq](https://jqlang.github.io/jq/)**, used by `deploy.sh` to read values from
  CLI output.
- The **C-MAPSS dataset**, downloaded into `data/` — see [`data/DATASET.md`](data/DATASET.md).

Treat the deployed infrastructure as ephemeral: it is provisioned from `.env` and
can be torn down and redeployed at any time with the script.

## Configuration (`.env`)

All deployment inputs live in a `.env` file at the project root. Copy the template
and fill it in:

```bash
cp .env.sample .env
```

You must provide only three values:

- `AWS_PROFILE` — a locally authenticated AWS CLI/SSO profile.
- `AWS_REGION` — region for all AWS resources.
- `DATABRICKS_PROFILE` — the `~/.databrickscfg` profile for the target workspace
  (needed for the databricks scope). Its account ID, stored by
  `databricks auth login`, is read for the IAM role trust policy.

Everything else has a default. Leave these blank to accept it, or set them to
override:

- `OWNER_EMAIL` (`Owner` tag) — defaults to your `git config user.email`.
- `REMOVE_AFTER` (`RemoveAfter` tag) — defaults to 30 days from the deploy date.
- `WORKSHOP_PRINCIPAL` — the UC group or user granted read access to the flows;
  defaults to the deploying Databricks user. Set a group for a class.
- `BUCKET_SUFFIX` — a suffix for the S3 bucket name, only if the global name is
  already taken.
- `CATALOG` — the existing Unity Catalog the workshop reads and writes under;
  defaults to `pdm_ws_catalog`. It must already exist. Because the notebooks
  cannot read `.env`, also set the same catalog in each notebook (see below).

The name prefix (`pdm-ws`) and `Project` tag (`pdm-workshop`) are fixed in
the Terraform defaults, so they are not in `.env`. `.env` is gitignored and must
never be committed; `.env.sample` is the tracked template. Terraform stores its
outputs, so retrieve any deployed value on demand with, e.g.,
`terraform -chdir=infra/aws output`.

### Point the notebooks at your catalog

The notebooks run on Databricks from a Git folder and cannot read this local
`.env`. Each one bootstraps by reading `deploy.json` from the catalog's config
volume, so set the catalog once at the top of the first configuration cell in
`notebooks/01_build_features.py`, `02_train_inference.py`, and
`03_batch_inference.py`:

```python
CATALOG = "pdm_ws_catalog"   # your Unity Catalog
```

Use the same catalog you set in `.env` (or its default). Everything else — schema,
region, bucket, stream — the notebooks read from `deploy.json` at runtime.

## Set up the infrastructure

With `.env` in place, deploy everything with the script:

```bash
./deploy.sh                 # AWS, then Databricks Unity Catalog access
./deploy.sh -s aws          # AWS only
./deploy.sh -s databricks   # Databricks only
```

Terraform shows its plan and prompts before applying. The AWS apply also writes
the producer's device credentials to `infra/aws/certs/` (gitignored), which the
`stream` command needs. The Databricks step provisions the storage credential,
external location, service credential, and grants that let notebooks read the two
flows. Once Terraform has applied, `deploy.sh` writes the workshop notebook's
configuration so that no environment-specific names are hardcoded.

You can still run Terraform directly in `infra/aws/` or `infra/databricks/`, but
then you must pass the variables yourself with `-var` flags; there is no committed
`terraform.tfvars`.

## Run the producer

The producer reads the local `data/*.txt` files (download them first — see
[`data/DATASET.md`](data/DATASET.md)) and drives them into the deployed AWS infra.
Run it from `infra/producer/`; `uv` resolves dependencies on first use.

```bash
cd infra/producer

# Publish the engine settings (chunked TSVs) and per-fleet RUL labels to S3.
uv run python cli.py file

# Stream sensor pings to IoT Core (routed on to Kinesis).
uv run python cli.py stream

# Empty the S3 metadata prefix to start a run clean.
uv run python cli.py reset
```

All commands read the AWS profile and region from `.env` (`AWS_PROFILE`,
`AWS_REGION`) and target the `pdm-ws-*` asset names; every other setting is a
fixed default. The one flag is `--batch N`, placed before the command, which
pauses `file` and `stream` every N files or observations so you can narrate the
flow (press N then Enter for the next batch, or just Enter to send the rest):

```bash
uv run python cli.py --batch 3 file
uv run python cli.py --batch 3 stream
```

`reset` clears only the S3 prefix. Kinesis records age out through the stream's
24-hour retention; they are not purged.

## Run the workshop

Open the numbered notebooks in `notebooks/` from a Databricks Git folder connected
to this repository, on serverless compute. Set `CATALOG` at the top of each
notebook first (see [Point the notebooks at your catalog](#point-the-notebooks-at-your-catalog)).
Start with `notebooks/01_build_features.py`, which builds the bronze→silver→gold
medallion pipeline from the two AWS flows and produces the model-ready RUL datasets.

## Tear down

`deploy.sh -d` destroys in reverse order (Databricks first, then AWS) and empties
the S3 data bucket before removing it:

```bash
./deploy.sh -d                 # everything
./deploy.sh -d -s aws          # AWS only
./deploy.sh -d -s databricks   # Databricks only
```
