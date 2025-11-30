#!/bin/bash
# Example script showing how to use compute_tsne_wrist.py and visualize_tsne_wrist.py
# for t-SNE analysis of evaluation samples

set -e  # Exit on error

echo "=================================="
echo "t-SNE Analysis Pipeline"
echo "=================================="

# Configuration
DATASET_PATHS=(
    "dataset_eval_samples/irom_1126_base2"
    "dataset_eval_samples/irom_1126_all2"
    # "dataset_eval_samples/irom_1126_play"
    # "dataset_eval_samples/irom_play"
)
PKL_NAME="1126_demo_only"
# PKL_NAME="1126_all_data_2play"
# PKL_NAME="1126_all_data"


OUTPUT_DIR="/n/fs/tom-project/video_models/Ctrl-World/dataset_eval_tsne"
OUTPUT_NAME="${PKL_NAME}.pkl"
FEATURE_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}"
PLOT_DIR="metric/tsne_plots"

# Step 1: Compute CLIP features
echo ""
echo "Step 1: Computing CLIP features from evaluation samples..."
echo "-----------------------------------------------------------"
python metric/compute_tsne_wrist.py \
    --dataset_paths "${DATASET_PATHS[@]}" \
    --output_dir "${OUTPUT_DIR}" \
    --output_name "${OUTPUT_NAME}" \
    --clip_model "openai/clip-vit-base-patch32" \
    --device cuda

# Step 2: Compute performance metrics
# echo ""
# echo "Step 2: Computing performance metrics (LPIPS, MSE, SSIM)..."
# echo "-----------------------------------------------------------"
# python metric/compute_metric_scores.py \
#     --dataset_paths "${DATASET_PATHS[@]}" \
#     --metrics lpips mse ssim \
#     --device cuda

# # Step 2.5: Visualize metric distributions as histograms
# echo ""
# echo "Step 2.5: Creating metric distribution histograms..."
# echo "-----------------------------------------------------------"
# for dataset_path in "${DATASET_PATHS[@]}"; do
#     dataset_name=$(basename "${dataset_path}")
#     echo "  Generating histograms for ${dataset_name}..."
#     python metric/visualize_metric_histogram.py \
#         --dataset_name "${dataset_name}" \
#         --dataset_metrics_dir dataset_eval_metrics \
#         --n_bins 100
# done

# Step 3: Compute t-SNE embeddings (one-time computation)
echo ""
echo "Step 3: Computing t-SNE embeddings..."
echo "-----------------------------------------------------------"
EMBEDDING_FILE="${OUTPUT_DIR}/${PKL_NAME}_tsne_embeddings.pkl"
python metric/compute_tsne_embeddings.py \
    --feature_file "${FEATURE_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --perplexity 30 \
    --random_state 42

# Step 4: Create t-SNE visualizations (fast, can rerun with different params)
echo ""
echo "Step 4: Creating t-SNE visualizations from precomputed embeddings..."
echo "-----------------------------------------------------------"
python metric/visualize_tsne_precomputed.py \
    --embedding_file "${EMBEDDING_FILE}" \
    --output_dir "${PLOT_DIR}" \
    --output_prefix "${PKL_NAME}" \
    --use_metric \
    --metric_dir dataset_eval_metrics \
    --power 0.8

echo ""
echo "=================================="
echo "Done! Check output in:"
echo "  CLIP Features: ${FEATURE_FILE}"
echo "  t-SNE Embeddings: ${EMBEDDING_FILE}"
echo "  Metrics: dataset_eval_metrics/"
echo "  Plots: ${PLOT_DIR}/"
echo ""
echo "To re-visualize with different parameters (without recomputing t-SNE):"
echo "  python metric/visualize_tsne_precomputed.py \\"
echo "    --embedding_file ${EMBEDDING_FILE} \\"
echo "    --output_prefix ${PKL_NAME}_custom \\"
echo "    --use_metric --power 3.0"
echo "=================================="
