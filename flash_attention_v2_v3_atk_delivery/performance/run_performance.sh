#!/usr/bin/env bash
set -uo pipefail

PERFORMANCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DELIVERY_DIR="$(dirname "${PERFORMANCE_DIR}")"
INTERFACES_DIR="${DELIVERY_DIR}/interfaces"
ATK="${ATK_BIN:-/usr/local/python3.11.15/bin/atk}"
STATUS_FILE="${PERFORMANCE_DIR}/reports/run_status.tsv"

interfaces=(
  flash_attn_func_v2
  flash_attn_func_v3
  flash_attn_varlen_func_v2
  flash_attn_varlen_func_v3
  flash_attn_with_kvcache_v2
  flash_attn_with_kvcache_v3
)

modes=(without_metadata with_metadata)

mkdir -p "${PERFORMANCE_DIR}/atk_runs" "${PERFORMANCE_DIR}/reports"
printf 'interface\tmode\texit_code\tatk_run\n' > "${STATUS_FILE}"

for interface in "${interfaces[@]}"; do
  interface_dir="${INTERFACES_DIR}/${interface}"
  case_file="${PERFORMANCE_DIR}/cases/all_${interface}_performance_60.json"

  for mode in "${modes[@]}"; do
    if [[ "${mode}" == "with_metadata" ]]; then
      metadata=1
    else
      metadata=0
    fi

    archive_dir="${PERFORMANCE_DIR}/atk_runs/${mode}/${interface}"
    mkdir -p "${archive_dir}"
    marker="$(mktemp)"
    touch "${marker}"

    echo "[$(date -u +'%F %T UTC')] ${interface} ${mode}: start"
    (
      cd "${interface_dir}"
      FA_ATK_USE_SCHEDULER_METADATA="${metadata}" "${ATK}" task \
        -c "${case_file}" \
        -n "${PERFORMANCE_DIR}/nodes_performance.yaml" \
        -p . \
        --task performance_e2e \
        --single_process \
        --log warning
    )
    exit_code=$?

    mapfile -t new_runs < <(find "${interface_dir}/atk_output" -mindepth 1 -maxdepth 1 -type d -newer "${marker}" -print | sort)
    rm -f "${marker}"
    archived=""
    for run_dir in "${new_runs[@]}"; do
      run_name="$(basename "${run_dir}")"
      mv "${run_dir}" "${archive_dir}/${run_name}"
      archived="atk_runs/${mode}/${interface}/${run_name}"
    done

    printf '%s\t%s\t%s\t%s\n' "${interface}" "${mode}" "${exit_code}" "${archived}" >> "${STATUS_FILE}"
    echo "[$(date -u +'%F %T UTC')] ${interface} ${mode}: exit=${exit_code} archive=${archived}"
  done
done

if awk -F '\t' 'NR > 1 && $3 != 0 { found=1 } END { exit !found }' "${STATUS_FILE}"; then
  exit 1
fi
