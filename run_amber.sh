#!/usr/bin/env bash
set -euo pipefail
DATASET=amber_discriminative exec bash "$(dirname "$0")/run_qa.sh"
