#!/usr/bin/env bash
# experiments/run_all_data_collection.sh
# Run all paper data collection scripts sequentially.
# Execute from the experiments/ directory:
#   bash run_all_data_collection.sh
# Or submit as a job after sourcing the conda environment.

set -e  # exit on first error

EXPERIMENTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$EXPERIMENTS_DIR"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wind_tomo

LOG_DIR="$EXPERIMENTS_DIR/logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "WindDensity-MBIR: full paper data collection"
echo "Start: $(date)"
echo "============================================================"

run_script() {
    local script="$1"
    local label="$2"
    local data_dir
    data_dir="$(dirname "$EXPERIMENTS_DIR/$script")/data"
    mkdir -p "$data_dir"
    echo ""
    echo "------------------------------------------------------------"
    echo "[$label] Starting: $script"
    echo "Time: $(date)"
    echo "------------------------------------------------------------"
    python "$EXPERIMENTS_DIR/$script" 2>&1 | tee "$LOG_DIR/${label}.log"
    echo "[$label] Finished: $(date)"
}

# ------------------------------------------------------------------
# Fig 6 / Table 2: FBP vs MBIR comparison (100 vols, 3 geometries)
# ------------------------------------------------------------------
run_script "fig6/table2_data_collection.py" "table2"

# ------------------------------------------------------------------
# Fig 7 / Table 1: Geometry sweep NRMSE + performance (100 vols)
# ------------------------------------------------------------------
run_script "fig7/fig7_data_collection.py" "fig7"

# ------------------------------------------------------------------
# Fig 8/9: Regional NRMSE for 3v2 and 3v16 (3000 vols)
# ------------------------------------------------------------------
run_script "fig8_9/fig9_data_collection.py" "fig9"

# ------------------------------------------------------------------
# Fig 10/11: Zernike error analysis for 3v2 and 11v16 (OPD_TT only)
#   (1000 vols)
# ------------------------------------------------------------------
run_script "fig10_11/fig10_11_data_collection.py" "fig10_11"

# ------------------------------------------------------------------
# Fig 13/14/15/17: Zernike + resolution NRMSE for 7v8 geometry
#   (1000 vols Zernike, 100 vols NRMSE sweep)
# ------------------------------------------------------------------
run_script "fig13_14_15_17/fig13_14_15_17_data_collection.py" "fig13_14_15_17"

echo ""
echo "============================================================"
echo "All data collection complete: $(date)"
echo "Log files written to: $LOG_DIR"
echo "============================================================"
