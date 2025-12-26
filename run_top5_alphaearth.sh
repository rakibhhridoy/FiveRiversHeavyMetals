#!/bin/bash

# AlphaEarth Top 5 Models - Quick Runner
# Runs only the top 5 best-performing models for efficient comparative analysis

PROJECT_DIR="/Users/rakibhhridoy/Five_Rivers"
cd "$PROJECT_DIR"

echo "=========================================="
echo "AlphaEarth - Top 5 Models Analysis"
echo "=========================================="
echo ""

# Top 5 Models by performance
# RAINY: Transformer, GNN MLP AE, CNN GNN MLP PG, GNN MLP, Stacked CNN GNN MLP
# WINTER: Transformer, GNN MLP AE, Mixture of Experts, Stacked CNN GNN MLP, GNN MLP

RAINY_TOP5=(
    "Transformer CNN GNN MLP"
    "GNN MLP AE"
    "CNN GNN MLP PG"
    "GNN MLP"
    "Stacked CNN GNN MLP"
)

WINTER_TOP5=(
    "Transformer CNN GNN MLP"
    "GNN MLP AE"
    "Mixture of Experts"
    "Stacked CNN GNN MLP"
    "GNN MLP"
)

# Show usage
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: ./run_top5_alphaearth.sh [command]"
    echo ""
    echo "Commands:"
    echo "  data-both       : Prepare AlphaEarth data (Option B) for both seasons"
    echo "  rainy-b         : Top 5 models, Rainy season, Option B"
    echo "  winter-b        : Top 5 models, Winter season, Option B"
    echo "  quick           : Both seasons, top 5, Option B (RECOMMENDED)"
    echo "  test            : Quick test - 1 model only"
    echo ""
    exit 0
fi

# Function to run data prep
run_data_prep() {
    local season=$1
    echo "Running data preparation for $season season..."
    python3 RUN_ALPHAEARTH_TOP5.py \
        --season "$season" \
        --data-prep-only
}

# Function to run top 5 models (Option B only)
run_top5_models() {
    local season=$1
    echo "Running top 5 models for $season season, Option B..."
    python3 RUN_ALPHAEARTH_TOP5.py \
        --season "$season" \
        --option B \
        --model-only
}

# Execute based on command
case "$1" in
    data-both)
        run_data_prep rainy
        run_data_prep winter
        ;;
    rainy-b)
        run_top5_models rainy
        ;;
    winter-b)
        run_top5_models winter
        ;;
    quick)
        echo "Quick analysis: Top 5 models, both seasons, Option B"
        run_data_prep rainy
        run_data_prep winter
        run_top5_models rainy
        run_top5_models winter
        ;;
    test)
        echo "Quick test: 1 model (Transformer), Option B, Rainy season"
        python3 RUN_ALPHAEARTH_TOP5.py \
            --season rainy \
            --option B \
            --model "Transformer CNN GNN MLP" \
            --model-only
        ;;
    *)
        if [ -z "$1" ]; then
            echo "No command specified. Use './run_top5_alphaearth.sh --help' for usage."
        else
            echo "Unknown command: $1"
            echo "Use './run_top5_alphaearth.sh --help' for usage."
        fi
        exit 1
        ;;
esac

echo ""
echo "Done!"
