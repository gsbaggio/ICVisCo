#!/bin/bash
# HiFiC 360 Run Script
# This script sets up the environment and runs HiFiC 360 training

# Activate conda environment
echo "Activating conda environment 'hific'..."
conda activate hific

# Set Python path to include both directories
export PYTHONPATH="/home/gabrielbaggio/Documentos/Trabalhos/ICVisCo/HiFiC360/models/hific:/home/gabrielbaggio/Documentos/Trabalhos/ICVisCo/compression/models:$PYTHONPATH"

# Function to run training with 360 support
run_hific_360() {
    echo "Running HiFiC 360 training..."
    echo "Conda environment: hific"
    echo "PYTHONPATH: $PYTHONPATH"
    echo ""
    
    python "/home/gabrielbaggio/Documentos/Trabalhos/ICVisCo/HiFiC360/models/hific/train.py" "$@"
}

# Function to run validation
run_validation() {
    echo "Running HiFiC 360 validation..."
    python "/home/gabrielbaggio/Documentos/Trabalhos/ICVisCo/HiFiC360/models/hific/simple_validation.py"
}

# Function to run the original validation (with imports)
run_full_validation() {
    echo "Running full HiFiC 360 validation..."
    python "/home/gabrielbaggio/Documentos/Trabalhos/ICVisCo/HiFiC360/models/hific/validate_implementation.py"
}

# Function to show usage
show_usage() {
    echo "HiFiC 360 Run Script"
    echo "Usage:"
    echo "  $0 train [training args]   - Run training with 360 support"
    echo "  $0 validate               - Run simple validation tests"
    echo "  $0 fullvalidate           - Run full validation with imports"
    echo "  $0 example                - Show example training command"
    echo ""
    echo "Example training command:"
    echo "  $0 train --config hific-360 --ckpt_dir ./checkpoints --use_lpips_360 --latitude_weight_type cosine --pole_weight 0.3"
}

# Main script logic
case "$1" in
    "train")
        shift
        run_hific_360 "$@"
        ;;
    "validate")
        run_validation
        ;;
    "fullvalidate")
        run_full_validation
        ;;
    "example")
        echo "Example HiFiC 360 training command:"
        echo ""
        echo "$0 train \\"
        echo "  --config hific-360 \\"
        echo "  --ckpt_dir ./checkpoints/hific360_test \\"
        echo "  --num_steps 10k \\"
        echo "  --batch_size 4 \\"
        echo "  --crop_size 512 \\"
        echo "  --use_lpips_360 \\"
        echo "  --latitude_weight_type cosine \\"
        echo "  --pole_weight 0.3"
        echo ""
        echo "For more options: $0 train --help"
        ;;
    *)
        show_usage
        ;;
esac
