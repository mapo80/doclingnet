#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: regression-parity.sh [COMMAND] [options]

Commands:
  refresh         Regenerate Python golden artefacts for the configured datasets.
  test            Execute the .NET regression parity suite against the hydrated goldens.
  all             Refresh goldens and run the parity suite (default).

Options:
  --case <id>         Limit the operation to a specific catalog case (repeatable).
  --python-cli <cmd>  Override the Docling Python CLI command (default: docling).
  --version <tag>     Target golden catalog version (default: v0.12.0).
  --golden-root <dir> Custom root directory for generated goldens (default: dataset/golden).
  --skip-tests        When combined with `all`, skip the parity test execution step.
  --help              Show this message and exit.

Environment variables:
  DOCLING_PYTHON_CLI         Same as --python-cli.
  DOCLING_PARITY_GOLDEN_ROOT Same as --golden-root.
  DOCLING_PARITY_EXTRA_ARGS  Extra arguments appended to the Docling CLI invocation.
  GOLDEN_VERSION             Default value consumed by --version.
USAGE
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

PYTHON_CLI=${DOCLING_PYTHON_CLI:-docling}
GOLDEN_VERSION=${GOLDEN_VERSION:-v0.12.0}
GOLDEN_ROOT=${DOCLING_PARITY_GOLDEN_ROOT:-"${REPO_ROOT}/dataset/golden"}
SKIP_TESTS=0
COMMAND=all
SELECTED_CASES=()

while (($#)); do
  case "$1" in
    refresh|test|all)
      COMMAND="$1"
      shift
      ;;
    --case)
      if (($# < 2)); then
        echo "error: --case requires an identifier" >&2
        exit 1
      fi
      SELECTED_CASES+=("$2")
      shift 2
      ;;
    --python-cli)
      if (($# < 2)); then
        echo "error: --python-cli requires a command" >&2
        exit 1
      fi
      PYTHON_CLI="$2"
      shift 2
      ;;
    --version)
      if (($# < 2)); then
        echo "error: --version requires a tag" >&2
        exit 1
      fi
      GOLDEN_VERSION="$2"
      shift 2
      ;;
    --golden-root)
      if (($# < 2)); then
        echo "error: --golden-root requires a directory" >&2
        exit 1
      fi
      GOLDEN_ROOT="$2"
      shift 2
      ;;
    --skip-tests)
      SKIP_TESTS=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option '$1'" >&2
      usage
      exit 1
      ;;
  esac
done

VERSION_ROOT="${GOLDEN_ROOT%/}/${GOLDEN_VERSION}"
ARTIFACTS_DIR="${REPO_ROOT}/.artifacts"
RESULTS_DIR="${ARTIFACTS_DIR}/test-results"

mkdir -p "${ARTIFACTS_DIR}" "${RESULTS_DIR}"

if [ ${#SELECTED_CASES[@]} -eq 0 ]; then
  SELECTED_CASES=(
    "DLN043A-001"
    "DLN043A-002"
  )
fi

declare -A CASE_SOURCE=(
  [DLN043A-001]="dataset/2305.03393v1-pg9-img.png"
  [DLN043A-002]="dataset/amt_handbook_sample.pdf"
)

declare -A CASE_DIRECTORY=(
  [DLN043A-001]="2305.03393v1-pg9"
  [DLN043A-002]="amt_handbook_sample"
)

ensure_python_cli() {
  if ! command -v "${PYTHON_CLI}" >/dev/null 2>&1; then
    echo "error: Docling CLI '${PYTHON_CLI}' was not found on PATH. Use --python-cli or set DOCLING_PYTHON_CLI." >&2
    exit 1
  fi
}

ensure_custom_packages() {
  local custom_dir="${REPO_ROOT}/packages/custom"
  local missing=0
  for package in EasyOcrNet.1.0.0.nupkg Docling.LayoutSdk.1.0.2.nupkg TableFormerSdk.1.0.0.nupkg; do
    if [ ! -f "${custom_dir}/${package}" ]; then
      echo "error: required custom package '${package}' is missing from '${custom_dir}'." >&2
      missing=1
    fi
  done

  if [ ${missing} -ne 0 ]; then
    echo "Install the custom NuGet packages before running the regression parity suite." >&2
    exit 1
  fi
}

refresh_case() {
  local case_id="$1"
  local source_relative="${CASE_SOURCE[${case_id}]:-}"
  local directory_name="${CASE_DIRECTORY[${case_id}]:-}"

  if [ -z "${source_relative}" ] || [ -z "${directory_name}" ]; then
    echo "error: unknown catalog case '${case_id}'." >&2
    exit 1
  fi

  local source_path="${REPO_ROOT}/${source_relative}"
  if [ ! -f "${source_path}" ]; then
    echo "error: source asset '${source_relative}' for case '${case_id}' was not found." >&2
    exit 1
  fi

  local case_root="${VERSION_ROOT}/${directory_name}"
  local source_dir="${case_root}/source"
  local python_dir="${case_root}/python-cli"
  local diffs_dir="${case_root}/diffs"

  mkdir -p "${case_root}" "${source_dir}" "${python_dir}" "${diffs_dir}"

  local source_name
  source_name=$(basename "${source_path}")
  cp "${source_path}" "${source_dir}/${source_name}"

  local notes_path="${source_dir}/notes.md"
  if [ ! -f "${notes_path}" ]; then
    cat <<EOF >"${notes_path}"
# ${case_id} source notes

- TODO: document provenance details for '${source_name}'.
EOF
  fi

  rm -rf "${python_dir}"
  mkdir -p "${python_dir}"

  local -a cli=("${PYTHON_CLI}"
    --input "${source_path}"
    --output "${python_dir}"
    --assets assets
    --markdown docling.md
    --manifest manifest.json
    --telemetry telemetry.json
    --keep-debug-overlays)

  if [ -n "${DOCLING_PARITY_EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    local extra=( ${DOCLING_PARITY_EXTRA_ARGS} )
    cli+=("${extra[@]}")
  fi

  echo "[DLN-043d] Refreshing golden artefacts for ${case_id} using '${PYTHON_CLI}'."
  "${cli[@]}"
}

run_refresh() {
  ensure_python_cli
  mkdir -p "${VERSION_ROOT}"
  for case_id in "${SELECTED_CASES[@]}"; do
    refresh_case "${case_id}"
  done
}

run_tests() {
  ensure_custom_packages
  echo "[DLN-043d] Restoring .NET solution."
  dotnet restore "${REPO_ROOT}/DoclingNet.sln"
  echo "[DLN-043d] Executing regression parity tests."
  DOCLING_PARITY_GOLDEN_ROOT="${VERSION_ROOT}" dotnet test "${REPO_ROOT}/DoclingNet.sln" \
    --no-build \
    --filter "FullyQualifiedName~Docling.Tests.Tooling.RegressionParityTests" \
    --logger "trx;LogFileName=RegressionParity.trx" \
    --results-directory "${RESULTS_DIR}"
}

case "${COMMAND}" in
  refresh)
    run_refresh
    ;;
  test)
    run_tests
    ;;
  all)
    run_refresh
    if [ ${SKIP_TESTS} -eq 0 ]; then
      run_tests
    fi
    ;;
  *)
    echo "error: unsupported command '${COMMAND}'" >&2
    usage
    exit 1
    ;;
esac
