#!/usr/bin/env bash
#
# deploy.sh — deploy or destroy the workshop infrastructure.
#
# Usage:
#   ./deploy.sh [-s aws|databricks] [-d]
#
#   (no flags)        deploy AWS, then Databricks Unity Catalog access
#   -s <scope>        limit to one scope: "aws" or "databricks" (default: both)
#   -d                destroy instead of deploy (Databricks first, then AWS)
#   -h                show this help
#
# All inputs come from .env (copy .env.sample and fill it in). Blank optional
# values are filled with sensible defaults at run time. Terraform still shows its
# plan and prompts before applying or destroying.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$ROOT_DIR/.env"
AWS_DIR="$ROOT_DIR/infra/aws"
DATABRICKS_DIR="$ROOT_DIR/infra/databricks"

SCOPE="all"
ACTION="deploy"

usage() {
  sed -n '3,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

die() {
  echo "deploy.sh: $*" >&2
  exit 1
}

usage_error() {
  echo "deploy.sh: $*" >&2
  usage >&2
  exit 1
}

# --- parse arguments --------------------------------------------------------

while getopts ":s:dh" opt; do
  case "$opt" in
    s) SCOPE="$OPTARG" ;;
    d) ACTION="destroy" ;;
    h) usage; exit 0 ;;
    :) usage_error "option -$OPTARG needs a value" ;;
    \?) usage_error "unknown option -$OPTARG" ;;
  esac
done
shift $((OPTIND - 1))

# Reject stray positional arguments (e.g. `deploy.sh destroy` instead of -d), so
# an unrecognized word can never be silently ignored and run the default deploy.
(($# == 0)) || usage_error "unexpected argument '$1'"

case "$SCOPE" in
  all | aws | databricks) ;;
  *) die "invalid scope '$SCOPE' (use aws or databricks)" ;;
esac

do_aws=false
do_databricks=false
[[ "$SCOPE" == "all" || "$SCOPE" == "aws" ]] && do_aws=true
[[ "$SCOPE" == "all" || "$SCOPE" == "databricks" ]] && do_databricks=true

# --- load and validate .env -------------------------------------------------

[[ -f "$ENV_FILE" ]] || die "missing .env — copy .env.sample to .env and fill it in"

if git -C "$ROOT_DIR" ls-files --error-unmatch .env >/dev/null 2>&1; then
  die ".env is tracked by git; remove it from version control before deploying"
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

require() {
  local name
  for name in "$@"; do
    [[ -n "${!name:-}" ]] || die "required value $name is not set in .env"
  done
}

# A human must choose the profiles explicitly; everything else is derived or
# defaulted below.
require AWS_PROFILE AWS_REGION
if [[ "$do_databricks" == true ]]; then
  require DATABRICKS_PROFILE
fi

# --- fill blank optional values with defaults -------------------------------

# Owner tag: the local git identity unless .env overrides it.
if [[ -z "${OWNER_EMAIL:-}" ]]; then
  OWNER_EMAIL="$(git -C "$ROOT_DIR" config user.email || true)"
  [[ -n "$OWNER_EMAIL" ]] || die "OWNER_EMAIL is blank and git config user.email is unset; set one of them"
  echo "OWNER_EMAIL defaulted to $OWNER_EMAIL (git config user.email)"
fi

# RemoveAfter tag: 30 days out (the policy maximum) unless .env overrides it.
if [[ -z "${REMOVE_AFTER:-}" ]]; then
  REMOVE_AFTER="$(date -v+30d +%F 2>/dev/null || date -d '+30 days' +%F)"
  echo "REMOVE_AFTER defaulted to $REMOVE_AFTER (30 days out)"
fi

# Databricks account ID (the external ID in the IAM trust policy): read from the
# selected profile, where OAuth login stores it.
if [[ "$do_databricks" == true ]]; then
  DATABRICKS_ACCOUNT_ID="$(databricks auth describe --profile "$DATABRICKS_PROFILE" -o json \
    | jq -r '.details.configuration.account_id.value // empty')"
  [[ -n "$DATABRICKS_ACCOUNT_ID" ]] || die "no account_id on profile '$DATABRICKS_PROFILE'; run: databricks auth login --profile $DATABRICKS_PROFILE"
  echo "DATABRICKS_ACCOUNT_ID read from profile '$DATABRICKS_PROFILE': $DATABRICKS_ACCOUNT_ID"
fi

# UC principal: the deploying Databricks user unless .env names a group/user.
if [[ "$do_databricks" == true && -z "${WORKSHOP_PRINCIPAL:-}" ]]; then
  WORKSHOP_PRINCIPAL="$(databricks current-user me --profile "$DATABRICKS_PROFILE" -o json \
    | jq -r '.userName // empty')"
  [[ -n "$WORKSHOP_PRINCIPAL" ]] || die "could not determine the current Databricks user; set WORKSHOP_PRINCIPAL"
  echo "WORKSHOP_PRINCIPAL defaulted to $WORKSHOP_PRINCIPAL (current Databricks user)"
fi

# --- build Terraform -var flags from .env -----------------------------------

aws_vars=(
  -var "aws_profile=$AWS_PROFILE"
  -var "aws_region=$AWS_REGION"
  -var "owner_email=$OWNER_EMAIL"
  -var "remove_after=$REMOVE_AFTER"
)

databricks_vars=(
  -var "aws_profile=$AWS_PROFILE"
  -var "aws_region=$AWS_REGION"
  -var "owner_email=$OWNER_EMAIL"
  -var "remove_after=$REMOVE_AFTER"
  -var "databricks_profile=${DATABRICKS_PROFILE:-}"
  -var "databricks_account_id=${DATABRICKS_ACCOUNT_ID:-}"
  -var "workshop_principal=${WORKSHOP_PRINCIPAL:-}"
)

# Optional vars fall back to the modules' own defaults when left blank in .env.
# Appended only when present, so an empty list is never expanded.
optional_vars=()
[[ -n "${PREFIX:-}" ]] && optional_vars+=(-var "prefix=$PREFIX")
[[ -n "${PROJECT:-}" ]] && optional_vars+=(-var "project=$PROJECT")
[[ -n "${BUCKET_SUFFIX:-}" ]] && optional_vars+=(-var "bucket_suffix=$BUCKET_SUFFIX")
if (( ${#optional_vars[@]} )); then
  aws_vars+=("${optional_vars[@]}")
  databricks_vars+=("${optional_vars[@]}")
fi

# Catalog is declared only in the Databricks module, so it never goes to the AWS apply.
[[ -n "${CATALOG:-}" ]] && databricks_vars+=(-var "catalog=$CATALOG")

# --- Terraform actions ------------------------------------------------------

tf_apply() {
  local dir="$1"; shift
  terraform -chdir="$dir" init -input=false
  terraform -chdir="$dir" apply -input=false "$@"
}

tf_destroy() {
  local dir="$1"; shift
  terraform -chdir="$dir" init -input=false
  terraform -chdir="$dir" destroy -input=false "$@"
}

write_config_volume() {
  # Publish the environment-specific names the notebook needs as a JSON file in
  # the UC config volume, so the notebook reads UC object names at runtime
  # instead of hardcoding them. Terraform is the source of truth: names come
  # from the AWS and Databricks module outputs, not from .env. Writing a volume
  # file needs no compute, so this stays CLI-only (no SQL warehouse).
  local aws_out db_out config_path config_json tmp
  aws_out="$(terraform -chdir="$AWS_DIR" output -json)"
  db_out="$(terraform -chdir="$DATABRICKS_DIR" output -json)"

  config_path="$(jq -r '.config_volume_path.value // empty' <<<"$db_out")"
  [[ -n "$config_path" ]] || die "config_volume_path missing from databricks outputs"

  config_json="$(jq -n \
    --arg catalog "$(jq -r '.catalog.value' <<<"$db_out")" \
    --arg data_schema "$(jq -r '.data_schema.value' <<<"$db_out")" \
    --arg aws_region "$AWS_REGION" \
    --arg data_bucket "$(jq -r '.data_bucket.value' <<<"$aws_out")" \
    --arg kinesis_stream_name "$(jq -r '.kinesis_stream_name.value' <<<"$aws_out")" \
    --arg kinesis_service_credential_name "$(jq -r '.kinesis_service_credential_name.value' <<<"$db_out")" \
    --arg external_location_name "$(jq -r '.external_location_name.value' <<<"$db_out")" \
    '{
      catalog: $catalog,
      data_schema: $data_schema,
      aws_region: $aws_region,
      data_bucket: $data_bucket,
      kinesis_stream_name: $kinesis_stream_name,
      kinesis_service_credential_name: $kinesis_service_credential_name,
      external_location_name: $external_location_name
    }')"

  tmp="$(mktemp)"
  printf '%s\n' "$config_json" >"$tmp"
  echo "writing deploy config to ${config_path}/deploy.json"
  databricks fs cp --overwrite "$tmp" "dbfs:${config_path}/deploy.json" --profile "$DATABRICKS_PROFILE"
  rm -f "$tmp"
  echo "  wrote ${config_path}/deploy.json"
}

empty_data_bucket() {
  # The S3 bucket is not force_destroy; empty it so `destroy` can remove it. Its
  # name follows the same convention the Terraform modules use (see
  # infra/databricks/main.tf), so we derive it rather than read it back.
  local bucket="${PREFIX:-pdm-ws}-data${BUCKET_SUFFIX:+-$BUCKET_SUFFIX}"
  echo "emptying s3://$bucket ..."
  aws s3 rm "s3://$bucket" --recursive \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" || true
}

# --- orchestrate ------------------------------------------------------------

if [[ "$ACTION" == "deploy" ]]; then
  if [[ "$do_aws" == true ]]; then
    echo "==> deploying AWS infrastructure"
    tf_apply "$AWS_DIR" "${aws_vars[@]}"
  fi
  if [[ "$do_databricks" == true ]]; then
    echo "==> deploying Databricks Unity Catalog access"
    tf_apply "$DATABRICKS_DIR" "${databricks_vars[@]}"
    write_config_volume
  fi
  echo "==> deploy complete"
else
  # Destroy in reverse order: Databricks depends on the AWS bucket it points at.
  # No explicit config-volume cleanup: the config schema is force_destroy, so
  # terraform destroy removes it along with the managed volume and deploy.json.
  if [[ "$do_databricks" == true ]]; then
    echo "==> destroying Databricks Unity Catalog access"
    tf_destroy "$DATABRICKS_DIR" "${databricks_vars[@]}"
  fi
  if [[ "$do_aws" == true ]]; then
    echo "==> destroying AWS infrastructure"
    empty_data_bucket
    tf_destroy "$AWS_DIR" "${aws_vars[@]}"
  fi
  echo "==> destroy complete"
fi
