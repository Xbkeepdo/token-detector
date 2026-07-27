#!/usr/bin/env bash
set -euo pipefail
DATASET="${DATASET:-clevr_exist_9k}" exec bash "$(dirname "$0")/run_qa.sh"
