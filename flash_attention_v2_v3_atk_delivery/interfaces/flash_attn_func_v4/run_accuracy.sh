#!/usr/bin/env bash
set -euo pipefail

INTERFACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERFACE_NAME="$(basename "${INTERFACE_DIR}")"
DELIVERY_DIR="$(cd "${INTERFACE_DIR}/../.." && pwd)"
ATK="${ATK_BIN:-/usr/local/python3.11.15/bin/atk}"
CASE_FILE="${DELIVERY_DIR}/accuracy/cases/all_${INTERFACE_NAME}.json"
ARCHIVE_DIR="${DELIVERY_DIR}/accuracy/atk_runs/${INTERFACE_NAME}"

# FA_ATK_GPU_BM=1 adds the TriDao official GPU benchmark node
# (nodes_accuracy_gpu.yaml); remote GPU nodes require multi-process mode.
NODES_FILE="nodes_accuracy.yaml"
SINGLE_PROCESS="--single_process"
if [ "${FA_ATK_GPU_BM:-0}" = "1" ]; then
  NODES_FILE="nodes_accuracy_gpu.yaml"
  SINGLE_PROCESS=""
fi

cd "${INTERFACE_DIR}"
marker="$(mktemp)"
touch "${marker}"
"${ATK}" task \
  -c "${CASE_FILE}" \
  -n "${NODES_FILE}" \
  -p . \
  --task accuracy \
  --bm_device cpu \
  ${SINGLE_PROCESS} \
  --log warning \
  --save_data input \
  --save_data output
mkdir -p "${ARCHIVE_DIR}"
while IFS= read -r run_dir; do
  mv "${run_dir}" "${ARCHIVE_DIR}/"
done < <(find "${INTERFACE_DIR}/atk_output" -mindepth 1 -maxdepth 1 -type d -newer "${marker}" -print)
rm -f "${marker}"