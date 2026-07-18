#!/usr/bin/env bash
set -euo pipefail
DATASET=pope exec bash "$(dirname "$0")/run_qa.sh"
