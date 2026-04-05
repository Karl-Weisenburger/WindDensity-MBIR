#!/usr/bin/env bash
# experiments/run_all_visualizations.sh
# Generate all paper figures from previously collected data.
# Assumes run_all_data_collection.sh has already been run and the
# data/ subfolders contain the required .npz / .npy files.
#
# Usage (from anywhere):
#   bash experiments/run_all_visualizations.sh
#
# Figures are written to each experiment's figures/ subfolder. Scripts
# run non-interactively (MPLBACKEND=Agg), so plt.show() calls are no-ops.

set -e

EXPERIMENTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$EXPERIMENTS_DIR"

# Activate conda environment (best-effort — skip if conda not available)
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate wind_tomo || true
fi

# Force non-interactive matplotlib so plt.show() doesn't block.
export MPLBACKEND=Agg

LOG_DIR="$EXPERIMENTS_DIR/logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "WindDensity-MBIR: generating all figures"
echo "Start: $(date)"
echo "============================================================"

run_viz() {
    local script="$1"
    local label="$2"
    echo ""
    echo "------------------------------------------------------------"
    echo "[$label] $script"
    echo "------------------------------------------------------------"
    python "$EXPERIMENTS_DIR/$script" 2>&1 | tee "$LOG_DIR/${label}_viz.log"
}

# Fig 6 / Table 2
run_viz "fig6_table2/fig6_visualization.py"                "fig6"
run_viz "fig6_table2/table2_visualization.py"              "table2"

# Fig 7 / Table 1
run_viz "fig7_table1/fig7_visualization.py"                "fig7"
run_viz "fig7_table1/table1_visualization.py"              "table1"

# Figs 8 and 9
run_viz "fig8_9/fig8_visualization.py"                     "fig8"
run_viz "fig8_9/fig9_visualization.py"                     "fig9"

# Figs 10 and 11
run_viz "fig10_11/fig10_11_visualization.py"               "fig10_11"

# Fig 12
run_viz "fig12/fig12_visualization.py"                     "fig12"

# Figs 13, 14, 15, 17
run_viz "fig13_14_15_17/fig13_visualization.py"            "fig13"
run_viz "fig13_14_15_17/fig14_15_visualization.py"         "fig14_15"
run_viz "fig13_14_15_17/fig17_visualization.py"            "fig17"

# Fig 16
run_viz "fig16/fig16_visualization.py"                     "fig16"

echo ""
echo "============================================================"
echo "All figures generated: $(date)"
echo "Figures written under each experiment's figures/ subfolder."
echo "Logs: $LOG_DIR"
echo "============================================================"
