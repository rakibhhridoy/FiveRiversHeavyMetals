#!/bin/bash

# AlphaEarth Comparative Analysis Quick Runner
# Usage: ./run_alphaearth.sh [option]

PROJECT_DIR="/Users/rakibhhridoy/Five_Rivers"
cd "$PROJECT_DIR"

echo "=========================================="
echo "AlphaEarth Comparative Analysis"
echo "=========================================="
echo ""

# Show usage
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: ./run_alphaearth.sh [command]"
    echo ""
    echo "Commands:"
    echo "  data-rainy      : Prepare data for Rainy season only"
    echo "  data-winter     : Prepare data for Winter season only"
    echo "  data-both       : Prepare data for both seasons"
    echo "  test-rainy      : Test one model (Option B, Rainy) - FAST"
    echo "  test-winter     : Test one model (Option B, Winter) - FAST"
    echo "  quick           : Quick test with Option B only"
    echo "  rainy-b         : Run all models Option B (Rainy)"
    echo "  winter-b        : Run all models Option B (Winter)"
    echo "  full            : Full comparative analysis (all options, both seasons)"
    echo ""
    exit 0
fi

# Function to run data prep
run_data_prep() {
    local season=$1
    echo "Running data preparation for $season season..."
    python3 RUN_ALPHAEARTH_COMPARATIVE_ANALYSIS.py \
        --season "$season" \
        --data-prep-only
}

# Function to run models
run_models() {
    local season=$1
    local option=$2
    echo "Running models for $season season, Option $option..."
    python3 RUN_ALPHAEARTH_COMPARATIVE_ANALYSIS.py \
        --season "$season" \
        --option "$option" \
        --model-only
}

# Execute based on command
case "$1" in
    data-rainy)
        run_data_prep rainy
        ;;
    data-winter)
        run_data_prep winter
        ;;
    data-both)
        run_data_prep rainy
        run_data_prep winter
        ;;
    test-rainy)
        echo "Testing with first model, Option B, Rainy season..."
        python3 RUN_ALPHAEARTH_COMPARATIVE_ANALYSIS.py \
            --season rainy \
            --option B \
            --model "Transformer CNN GNN MLP" \
            --model-only
        ;;
    test-winter)
        echo "Testing with first model, Option B, Winter season..."
        python3 RUN_ALPHAEARTH_COMPARATIVE_ANALYSIS.py \
            --season winter \
            --option B \
            --model "Transformer CNN GNN MLP" \
            --model-only
        ;;
    quick)
        echo "Quick test: Option B only, both seasons"
        run_models rainy B
        run_models winter B
        ;;
    rainy-b)
        echo "Running all 11 models with Option B (Rainy season)"
        run_models rainy B
        ;;
    winter-b)
        echo "Running all 11 models with Option B (Winter season)"
        run_models winter B
        ;;
    full)
        echo "Running full comparative analysis..."
        echo "This will test: 11 models × 4 options × 2 seasons = 88 model runs"
        echo "Estimated time: 4-6 weeks on M1 Pro"
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            run_data_prep rainy
            run_data_prep winter
            for option in A B C D; do
                echo ""
                echo "Testing Option $option..."
                for season in rainy winter; do
                    run_models "$season" "$option"
                done
            done
        fi
        ;;
    *)
        if [ -z "$1" ]; then
            echo "No command specified. Use './run_alphaearth.sh --help' for usage."
        else
            echo "Unknown command: $1"
            echo "Use './run_alphaearth.sh --help' for usage."
        fi
        exit 1
        ;;
esac

echo ""
echo "Done!"
