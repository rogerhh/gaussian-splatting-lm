#!/bin/bash

set -e

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

source /scratch/roger_hsiao/miniforge3/etc/profile.d/conda.sh
source /scratch/roger_hsiao/MonoGS_jvp/sourceme.sh
python3 tests/jvp_timing.py
