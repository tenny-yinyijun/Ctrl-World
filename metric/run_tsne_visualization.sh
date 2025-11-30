#!/bin/bash
# Quick script to re-visualize t-SNE with different parameters
# (without recomputing embeddings)

set -e  # Exit on error

echo "=================================="
echo "t-SNE Re-Visualization Script"
echo "=================================="

# Configuration - update these to match your precomputed embeddings
PKL_NAME="1126_all_data"
OUTPUT_DIR="/n/fs/tom-project/video_models/Ctrl-World/dataset_eval_tsne"
EMBEDDING_FILE="${OUTPUT_DIR}/${PKL_NAME}_tsne_embeddings.pkl"
PLOT_DIR="metric/tsne_plots"

# Check if embedding file exists
if [ ! -f "${EMBEDDING_FILE}" ]; then
    echo "Error: Embedding file not found: ${EMBEDDING_FILE}"
    echo "Please run run_tsne_analysis.sh first to compute embeddings."
    exit 1
fi

echo ""
echo "Using precomputed embeddings: ${EMBEDDING_FILE}"
echo ""

# Create visualizations with different power values to compare
POWER_VALUES=(1.5 2.0 2.5 3.0)

for power in "${POWER_VALUES[@]}"; do
    echo "Creating visualization with power=${power}..."
    python metric/visualize_tsne_precomputed.py \
        --embedding_file "${EMBEDDING_FILE}" \
        --output_dir "${PLOT_DIR}" \
        --output_prefix "${PKL_NAME}_power${power}" \
        --use_metric \
        --metric_dir dataset_eval_metrics \
        --power "${power}"
    echo ""
done

echo "=================================="
echo "Done! Visualizations saved to: ${PLOT_DIR}/"
echo "Created plots with power values: ${POWER_VALUES[*]}"
echo ""
echo "To create a single visualization with custom settings:"
echo "  python metric/visualize_tsne_precomputed.py \\"
echo "    --embedding_file ${EMBEDDING_FILE} \\"
echo "    --output_prefix custom_name \\"
echo "    --use_metric --power 2.5"
echo "=================================="
