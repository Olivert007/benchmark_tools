#!/bin/bash
#
# Triton IR/ASM dump helper
# -------------------------
# Purpose:
#   Run a Python script with Triton compiler dump enabled and save:
#   - Triton cache artifacts under an output directory (TRITON_CACHE_DIR)
#   - Compiler dump artifacts under TRITON_CACHE_DIR
#
# Why separate from rocprof profiling:
#   Enabling MLIR/LLVM/AMDGCN dumps can generate huge stdout and make ATT/profiling
#   runs look "stuck". Keep dump and profiling workflows independent for reuse.
#

set -e

usage() {
    echo "Usage: $0 -f <python_script.py> -o <output_dir> [-v <level>] [-d <devices>] [-- <script args>]"
    echo ""
    echo "Arguments:"
    echo "  -f <path>     Required. Path to the Python script to execute."
    echo "  -o <path>     Required. Output directory to store dump artifacts."
    echo "  -v <level>    Optional. Set TRITON_VERBOSE (e.g., 1..4)."
    echo "  -d <devices>  Optional. Set ROCR_VISIBLE_DEVICES (e.g., '0', '0,1')."
    echo "  -h            Display this help message."
    echo ""
    echo "Environment toggles (optional):"
    echo "  TRITON_DUMP_DIR_NAME     Subdir name under -o (default: triton_ir_asm)"
    echo "  TRITON_ALWAYS_COMPILE    Default 1 for dump runs (forces compile)"
    echo ""
    echo "Examples:"
    echo "  $0 -f ./verify_conv3d_impl.py -o ./out_dump"
    echo "  $0 -f ./verify_conv3d_optimized.py -o ./out_dump -d 1"
    echo "  $0 -f ./verify_conv3d_optimized.py -o ./out_dump -- --some-arg 123"
    exit 1
}

PYTHON_FILE=""
BASE_OUTPUT_DIR=""
VERBOSE_LEVEL=""
ROCR_DEVICES=""

while getopts "f:o:v:d:h" opt; do
    case ${opt} in
        f) PYTHON_FILE=$OPTARG ;;
        o) BASE_OUTPUT_DIR=$OPTARG ;;
        v) VERBOSE_LEVEL=$OPTARG ;;
        d) ROCR_DEVICES=$OPTARG ;;
        h) usage ;;
        \?) usage ;;
    esac
done

shift $((OPTIND - 1))
PYTHON_ARGS=("$@")

if [ -z "$PYTHON_FILE" ] || [ -z "$BASE_OUTPUT_DIR" ]; then
    echo "Error: Python script path (-f) and output directory (-o) are required." >&2
    usage
fi

if ! PYTHON_FILE_ABS=$(realpath -s "$PYTHON_FILE"); then
    echo "Error: Failed to resolve path for Python script: $PYTHON_FILE" >&2
    exit 1
fi

if [ ! -f "$PYTHON_FILE_ABS" ]; then
    echo "Error: Python script not found at '$PYTHON_FILE_ABS'" >&2
    exit 1
fi

BASE_OUTPUT_DIR_ABS=$(realpath -m "$BASE_OUTPUT_DIR")
mkdir -p "$BASE_OUTPUT_DIR_ABS"

TRITON_DUMP_DIR_NAME="${TRITON_DUMP_DIR_NAME:-triton_ir_asm}"
TRITON_DUMP_DIR="$BASE_OUTPUT_DIR_ABS/$TRITON_DUMP_DIR_NAME"
mkdir -p "$TRITON_DUMP_DIR"

echo "========================================"
echo "Triton IR/ASM Dump Utility"
echo "========================================"
echo "▶️  Python Script: $PYTHON_FILE_ABS"
echo "▶️  Output Directory: $BASE_OUTPUT_DIR_ABS"
echo "▶️  TRITON_CACHE_DIR: $TRITON_DUMP_DIR"
if [ -n "$ROCR_DEVICES" ]; then
    echo "🎯 ROCR_VISIBLE_DEVICES: $ROCR_DEVICES"
fi
if [ -n "$VERBOSE_LEVEL" ]; then
    echo "🔎 TRITON_VERBOSE: $VERBOSE_LEVEL"
fi
echo "----------------------------------------"

# Optional verbosity
if [ -n "$VERBOSE_LEVEL" ]; then
    export TRITON_VERBOSE="$VERBOSE_LEVEL"
fi

# Optional GPU selection (HSA runtime)
if [ -n "$ROCR_DEVICES" ]; then
    export ROCR_VISIBLE_DEVICES="$ROCR_DEVICES"
fi

# Enable Triton compiler dumps
export TRITON_CACHE_DIR="$TRITON_DUMP_DIR"
export MLIR_ENABLE_DUMP=1
export LLVM_IR_ENABLE_DUMP=1
export AMDGCN_ENABLE_DUMP=1
export TRITON_ALWAYS_COMPILE="${TRITON_ALWAYS_COMPILE:-1}"

# Build python command
PY_CMD=(python "$PYTHON_FILE_ABS")
if [ ${#PYTHON_ARGS[@]} -gt 0 ]; then
    for arg in "${PYTHON_ARGS[@]}"; do
        PY_CMD+=("$arg")
    done
fi

echo "🚀 Executing"
echo "📄 Command: ${PY_CMD[*]}"

if "${PY_CMD[@]}"; then
    echo "✅ Triton dump run completed successfully."
else
    echo "❌ Triton dump run failed." >&2
    exit 1
fi

echo "----------------------------------------"
echo "✅ Dump artifacts directory: $TRITON_DUMP_DIR"
echo "========================================"
