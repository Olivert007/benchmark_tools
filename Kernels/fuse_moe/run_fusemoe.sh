#!/bin/bash
# filepath: run_fusemoe.sh

# MOE Tuning Runner Script
# This script automates MOE tuning operations with various parameter combinations

set -e  # Exit on any error

# Default configuration
DEFAULT_HIDDEN_SIZE=7168
DEFAULT_NUM_EXPERTS=256
DEFAULT_SHARD_SIZE=512
DEFAULT_TOPK=8
DEFAULT_DTYPE="float16"
DEFAULT_QUANT_TYPE="int8_w8a8"
DEFAULT_BLOCK_SHAPE="128,128"
DEFAULT_DEVICE=1
DEFAULT_SAVE_DIR="tuned_configs_8192"
DEFAULT_BATCH_SIZES="8192,16384"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

MOE Tuning Script Options:
  -b, --batch-sizes SIZES     Comma-separated batch sizes (default: $DEFAULT_BATCH_SIZES)
  -H, --hidden-size SIZE      Hidden dimension size (default: $DEFAULT_HIDDEN_SIZE)
  -e, --num-experts NUM       Number of experts (default: $DEFAULT_NUM_EXPERTS)
  -s, --shard-size SIZE       Shard intermediate size (default: $DEFAULT_SHARD_SIZE)
  -k, --topk NUM              Top-k experts per token (default: $DEFAULT_TOPK)
  -t, --type TYPE             Quantization type: fp16|int8_w8a8|int8_w8a16|fp8_w8a8 (default: $DEFAULT_QUANT_TYPE)
  -d, --dtype TYPE            Compute dtype: float16|bfloat16 (default: $DEFAULT_DTYPE)
      --block-shape SHAPE     Block shape as n,k (default: $DEFAULT_BLOCK_SHAPE)
      --device ID             GPU device ID (default: $DEFAULT_DEVICE)
      --save-dir DIR          Directory to save configs (default: $DEFAULT_SAVE_DIR)
      --python-script PATH    Path to test_fusemoe.py (default: ./test_fusemoe.py)
      --dry-run               Show commands without executing
      --preset PRESET         Use predefined parameter sets
      --help                  Show this help message

Presets:
  small       Small model configuration (hidden=2048, experts=8, shard=1024)
  medium      Medium model configuration (hidden=4096, experts=16, shard=2048)
  large       Large model configuration (hidden=7168, experts=32, shard=4096)
  xlarge      Extra large configuration (hidden=8192, experts=64, shard=8192)

Examples:
  # Basic tuning with default parameters
  $0

  # Tune specific batch sizes
  $0 --batch-sizes "128,256,512,1024"

  # Use preset configuration
  $0 --preset large --batch-sizes "64,128,256"

  # Custom configuration
  $0 --hidden-size 4096 --num-experts 16 --shard-size 2048 --topk 4

  # Dry run to see commands
  $0 --dry-run --batch-sizes "128,256"

  # Multiple quantization types
  $0 --type int8_w8a16 --dtype bfloat16

EOF
}

# Function to apply presets
apply_preset() {
    case $1 in
        "small")  HIDDEN_SIZE=2048; NUM_EXPERTS=8;  SHARD_SIZE=1024; TOPK=4 ;;
        "medium") HIDDEN_SIZE=4096; NUM_EXPERTS=16; SHARD_SIZE=2048; TOPK=6 ;;
        "large")  HIDDEN_SIZE=7168; NUM_EXPERTS=32; SHARD_SIZE=4096; TOPK=8 ;;
        "xlarge") HIDDEN_SIZE=8192; NUM_EXPERTS=64; SHARD_SIZE=8192; TOPK=8 ;;
        *)
            print_error "Unknown preset: $1"
            print_info "Available presets: small, medium, large, xlarge"
            exit 1
            ;;
    esac
    print_info "Applied $1 preset: hidden=$HIDDEN_SIZE, experts=$NUM_EXPERTS, shard=$SHARD_SIZE, topk=$TOPK"
}

# Function to validate a positive integer parameter
validate_positive_int() {
    local name="$1" value="$2"
    if ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$value" -le 0 ]; then
        print_error "Invalid $name: $value (must be a positive integer)"
        exit 1
    fi
}

# Function to validate parameters
validate_params() {
    # Validate quantization type
    case $QUANT_TYPE in
        "fp16"|"int8_w8a8"|"int8_w8a16"|"fp8_w8a8") ;;
        *) print_error "Invalid quantization type: $QUANT_TYPE"; exit 1 ;;
    esac

    # Validate dtype
    case $DTYPE in
        "float16"|"bfloat16") ;;
        *) print_error "Invalid dtype: $DTYPE"; exit 1 ;;
    esac

    # Validate numeric parameters
    validate_positive_int "hidden size" "$HIDDEN_SIZE"
    validate_positive_int "number of experts" "$NUM_EXPERTS"
    validate_positive_int "shard size" "$SHARD_SIZE"
    validate_positive_int "topk" "$TOPK"

    # Validate block shape format
    if ! [[ "$BLOCK_SHAPE" =~ ^[0-9]+,[0-9]+$ ]]; then
        print_error "Invalid block shape format: $BLOCK_SHAPE (expected: n,k)"
        exit 1
    fi

    # Check if Python script exists
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        print_error "Python script not found: $PYTHON_SCRIPT"
        exit 1
    fi
}

# Function to create log directory
setup_logging() {
    LOG_DIR="logs"
    mkdir -p "$LOG_DIR"
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="$LOG_DIR/moe_tune_${TIMESTAMP}.log"
    print_info "Logging to: $LOG_FILE"
}

# Function to run tuning command
run_tuning() {
    local cmd="python $PYTHON_SCRIPT \
        --mode tune \
        --batch-sizes \"$BATCH_SIZES\" \
        --hidden-size $HIDDEN_SIZE \
        --num-experts $NUM_EXPERTS \
        --shard-intermediate-size $SHARD_SIZE \
        --topk $TOPK \
        --type $QUANT_TYPE \
        --dtype $DTYPE \
        --block-shape \"$BLOCK_SHAPE\" \
        --device $DEVICE \
        --save-dir \"$SAVE_DIR\""        

    print_info "Executing MOE tuning with parameters:"
    echo "  Batch sizes: $BATCH_SIZES"
    echo "  Hidden size: $HIDDEN_SIZE"
    echo "  Num experts: $NUM_EXPERTS"
    echo "  Shard size: $SHARD_SIZE"
    echo "  Top-k: $TOPK"
    echo "  Quantization: $QUANT_TYPE"
    echo "  Data type: $DTYPE"
    echo "  Block shape: $BLOCK_SHAPE"
    echo "  Device: $DEVICE"
    echo "  Save directory: $SAVE_DIR"
    echo ""

    if [ "$DRY_RUN" = true ]; then
        print_warning "DRY RUN MODE - Command that would be executed:"
        echo "$cmd"
        return 0
    fi

    print_info "Starting tuning process..."

    # Execute with logging
    if eval "$cmd" 2>&1 | tee "$LOG_FILE"; then
        print_success "Tuning completed successfully!"
        print_info "Results saved to: $SAVE_DIR"
        print_info "Log saved to: $LOG_FILE"
    else
        print_error "Tuning failed! Check log: $LOG_FILE"
        exit 1
    fi
}

# Function to run multiple configurations
run_batch_tuning() {
    local configs=(
        "fp16:float16"
        "int8_w8a8:float16"
        "int8_w8a16:bfloat16"
        "fp8_w8a8:float16"
    )

    print_info "Running batch tuning with multiple configurations..."
    
    for config in "${configs[@]}"; do
        IFS=':' read -r quant dtype <<< "$config"
        
        print_info "Running configuration: $quant with $dtype"
        
        QUANT_TYPE="$quant"
        DTYPE="$dtype"
        SAVE_DIR="${DEFAULT_SAVE_DIR}_${quant}_${dtype}"
        
        validate_params
        run_tuning
        
        print_success "Completed configuration: $quant with $dtype"
        echo ""
    done
}

# Initialize variables with defaults
BATCH_SIZES="$DEFAULT_BATCH_SIZES"
HIDDEN_SIZE="$DEFAULT_HIDDEN_SIZE"
NUM_EXPERTS="$DEFAULT_NUM_EXPERTS"
SHARD_SIZE="$DEFAULT_SHARD_SIZE"
TOPK="$DEFAULT_TOPK"
QUANT_TYPE="$DEFAULT_QUANT_TYPE"
DTYPE="$DEFAULT_DTYPE"
BLOCK_SHAPE="$DEFAULT_BLOCK_SHAPE"
DEVICE="$DEFAULT_DEVICE"
SAVE_DIR="$DEFAULT_SAVE_DIR"
PYTHON_SCRIPT="./test_fusemoe.py"
DRY_RUN=false
BATCH_MODE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--batch-sizes)
            BATCH_SIZES="$2"
            shift 2
            ;;
        -H|--hidden-size)
            HIDDEN_SIZE="$2"
            shift 2
            ;;
        -e|--num-experts)
            NUM_EXPERTS="$2"
            shift 2
            ;;
        -s|--shard-size)
            SHARD_SIZE="$2"
            shift 2
            ;;
        -k|--topk)
            TOPK="$2"
            shift 2
            ;;
        -t|--type)
            QUANT_TYPE="$2"
            shift 2
            ;;
        -d|--dtype)
            DTYPE="$2"
            shift 2
            ;;
        --block-shape)
            BLOCK_SHAPE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --python-script)
            PYTHON_SCRIPT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --preset)
            apply_preset "$2"
            shift 2
            ;;
        --batch-mode)
            BATCH_MODE=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_info "MOE Tuning Runner Started"
    print_info "Timestamp: $(date)"
    
    # Setup logging
    setup_logging
    
    # Validate parameters
    validate_params
    
    # Check GPU availability
    if command -v nvidia-smi >/dev/null 2>&1; then
        print_info "GPU Information:"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits | head -5
    elif command -v rocm-smi >/dev/null 2>&1; then
        print_info "AMD GPU Information:"
        rocm-smi --showmeminfo vram --csv | head -5
    else
        print_warning "No GPU monitoring tool found (nvidia-smi or rocm-smi)"
    fi
    
    echo ""
    
    # Run tuning
    if [ "$BATCH_MODE" = true ]; then
        run_batch_tuning
    else
        run_tuning
    fi
    
    print_success "MOE Tuning Runner Completed"
}

# Trap to handle interruption
trap 'print_error "Script interrupted!"; exit 1' INT TERM

# Run main function
main "$@"