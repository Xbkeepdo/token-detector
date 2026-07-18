#!/usr/bin/env bash
set -euo pipefail
DATASET=clevr_exist_5k exec bash "$(dirname "$0")/run_qa.sh"
