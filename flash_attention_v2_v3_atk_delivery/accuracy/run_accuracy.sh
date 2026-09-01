#!/usr/bin/env bash
set -euo pipefail

ACCURACY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DELIVERY_DIR="$(dirname "${ACCURACY_DIR}")"

for interface_dir in "${DELIVERY_DIR}"/interfaces/*; do
  "${interface_dir}/run_accuracy.sh"
done
