#!/bin/bash
# rocprofv3_pmc_trace.sh - ROCProfiler v3 PMC Trace Utility

set -e # Exit immediately if a command exits with a non-zero status.

# --- GPU Index Mapping ---
# Map between amd-smi index and ROCR_VISIBLE_DEVICES index
# Modify these arrays according to your system configuration
# AMD_SMI_INDICES=(0 1 2 3 4)
# ROCR_VISIBLE_DEVICES_INDICES=(4 2 3 0 1)
AMD_SMI_INDICES=(0)
ROCR_VISIBLE_DEVICES_INDICES=(0)

# --- Help Message ---
show_help() {
    echo "Usage: $0 -f <python_script> -c <yaml_file> -o <base_output_dir> [--clear] [-v <level>] [-g <gpu_index>] [-h] [-- <python_args>...]"
    echo ""
    echo "Arguments:"
    echo "  -f <path>     Required. Path to the Python script to execute."
    echo "  -c <path>     Required. Path to the rocprofv3 configuration YAML file."
    echo "  -o <path>     Required. Base directory to save the rocprofv3 output files."
    echo "                The actual output will be in a subdirectory with script name, version and timestamp."
    echo "  --clear       Optional flag. If present, clears the contents of the generated output directory."
    echo "  -v <level>    Optional. Set the TRITON_VERBOSE level (e.g., 1 for basic, 4 for max)."
    echo "  -g <index>    Optional. ROCR_VISIBLE_DEVICES GPU index to use (default: 4)."
    echo "  -h            Display this help message."
    echo "  -- <args>     Optional. Arguments to pass to the Python script. Use '--' to separate"
    echo "                rocprofv3_pmc_trace.sh arguments from Python script arguments."
    echo ""
    echo "Examples:"
    echo "  $0 -f ./test_moe_new.py -c ./pmc.yaml -o ./rocprof_output -g 4"
    echo "  $0 -f ./benchmark.py -c ./pmc.yaml -o ./rocprof_output -- -b 512 --dtype bf16"
    echo "  $0 -f ./test.py -c ./pmc.yaml -o ./output -- --batch-size 256 --verbose"
    echo ""
    echo "Note: The actual output directory will include script name, Triton version and timestamp,"
    echo "      e.g., <base_output_dir>/test_moe_new_triton_v3.3.1+gitd95ec8e0_llvm_a1b2c3d4_20240819_143052/"
    echo ""
    echo "GPU Index Mapping:"
    echo "  AMD-SMI indices:              ${AMD_SMI_INDICES[*]}"
    echo "  ROCR_VISIBLE_DEVICES indices: ${ROCR_VISIBLE_DEVICES_INDICES[*]}"
}

# --- Function to get amd-smi index from ROCR_VISIBLE_DEVICES index ---
get_amd_smi_index() {
    local rocr_index="$1"
    local i
    
    for i in "${!ROCR_VISIBLE_DEVICES_INDICES[@]}"; do
        if [ "${ROCR_VISIBLE_DEVICES_INDICES[$i]}" -eq "$rocr_index" ]; then
            echo "${AMD_SMI_INDICES[$i]}"
            return 0
        fi
    done
    
    echo "Error: ROCR_VISIBLE_DEVICES index $rocr_index not found in mapping" >&2
    return 1
}

# --- Function to set GPU performance level ---
set_gpu_performance_level() {
    local amd_smi_index="$1"
    local level="$2"
    
    echo "🔧 Setting GPU $amd_smi_index performance level to '$level'..."
    
    if ! amd-smi set -g "$amd_smi_index" -l "$level"; then
        echo "❌ Error: Failed to set GPU $amd_smi_index performance level to '$level'" >&2
        return 1
    fi
    
    # Wait a moment for the setting to take effect
    sleep 2
    
    echo "✅ GPU $amd_smi_index performance level set to '$level'"
    return 0
}

# --- Function to get Triton version ---
# get_triton_version() {
#     local version
#     if version=$(pip show triton 2>/dev/null | grep "^Version:" | cut -d' ' -f2); then
#         echo "$version"
#     else
#         echo "unknown"
#     fi
# }
get_triton_version() {
    local version
    if version=$("$PYTHON_BIN" -m pip show triton 2>/dev/null | grep "^Version:" | cut -d' ' -f2); then
        echo "$version"
    else
        echo "unknown"
    fi
}

# --- Function to get Triton source location ---
# get_triton_location() {
#     local location
#     # Try to get Editable project location first (for development installs)
#     if location=$(pip show triton 2>/dev/null | grep "^Editable project location:" | cut -d' ' -f4); then
#         echo "$location"
#     # If that fails, try Location (for installed packages)
#     elif location=$(pip show triton 2>/dev/null | grep "^Location:" | cut -d' ' -f2); then
#         echo "$location"
#     else
#         echo ""
#     fi
# }

get_triton_location() {
    local location
    if location=$("$PYTHON_BIN" -m pip show triton 2>/dev/null | grep "^Editable project location:" | cut -d' ' -f4); then
        echo "$location"
    elif location=$("$PYTHON_BIN" -m pip show triton 2>/dev/null | grep "^Location:" | cut -d' ' -f2); then
        echo "$location"
    else
        echo ""
    fi
}

# --- Function to get Triton root directory ---
get_triton_root() {
    local triton_location="$1"
    if [ -z "$triton_location" ]; then
        echo ""
        return
    fi
    
    # Navigate up from the location to find triton root
    local current_path="$triton_location"
    
    # Go up directory levels until we find a directory that starts with "triton"
    while [ "$current_path" != "/" ] && [ -n "$current_path" ]; do
        local dirname=$(basename "$current_path")
        if [[ "$dirname" == triton* ]]; then
            echo "$current_path"
            return
        fi
        current_path=$(dirname "$current_path")
    done
    
    echo ""
}

# --- Function to get LLVM commit hash ---
get_llvm_commit() {
    local triton_root="$1"
    if [ -z "$triton_root" ]; then
        echo "unknown"
        return
    fi
    
    local llvm_hash_file="$triton_root/cmake/llvm-hash.txt"
    if [ -f "$llvm_hash_file" ]; then
        # Read the first line and trim whitespace
        local llvm_hash=$(head -n1 "$llvm_hash_file" | tr -d '[:space:]')
        if [ -n "$llvm_hash" ]; then
            echo "$llvm_hash"
        else
            echo "unknown"
        fi
    else
        echo "unknown"
    fi
}

# --- Function to get current timestamp ---
get_timestamp() {
    date +"%Y%m%d_%H%M%S"
}

# --- Function to extract script name without extension ---
get_script_basename() {
    local script_path="$1"
    local basename_with_ext=$(basename "$script_path")
    # Remove the extension (.py)
    echo "${basename_with_ext%.py}"
}

# --- Cleanup function to restore GPU settings ---
cleanup() {
    local exit_code=$?
    
    if [ -n "$GPU_AMD_SMI_INDEX" ]; then
        echo ""
        echo "🔄 Restoring GPU settings..."
        if set_gpu_performance_level "$GPU_AMD_SMI_INDEX" "auto"; then
            echo "✅ GPU settings restored successfully"
        else
            echo "❌ Warning: Failed to restore GPU settings. You may need to manually run:"
            echo "   amd-smi set -g $GPU_AMD_SMI_INDEX -l \"auto\""
        fi
    fi
    
    exit $exit_code
}

# Set up trap to ensure cleanup runs on script exit
trap cleanup EXIT INT TERM

# --- Argument Parsing ---
PYTHON_FILE=""
YAML_FILE=""
BASE_OUTPUT_DIR=""
CLEAR_FLAG=0
VERBOSE_LEVEL=""
# GPU_ROCR_INDEX=4  # Default GPU index
GPU_ROCR_INDEX=0  # Default GPU index (single-GPU)
GPU_G_FLAG_USED=0  # Track if -g was explicitly passed
PYTHON_ARGS=()    # Array to store Python script arguments

# Save original ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES from caller
_ORIG_ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-}"
_ORIG_HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-}"

# --- Activate specific python virtual environment ---
VENV_DIR="/opt/venv" # Modify this path as you needed
VENV_ACTIVATE="$VENV_DIR/bin/activate"
if [ -f "$VENV_ACTIVATE" ]; then
    echo "🔧 Activating Python venv: $VENV_DIR"
    # shellcheck disable=SC1090
    source "$VENV_ACTIVATE"
else
    echo "❌ Error: venv activate script not found at $VENV_ACTIVATE" >&2
    exit 1
fi

# After activation, capture python executable explicitly
PYTHON_BIN="$(which python)"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "❌ Error: Activated python not executable: $PYTHON_BIN" >&2
    exit 1
fi
echo "✅ Using Python: $PYTHON_BIN"

while [[ $# -gt 0 ]]; do
    case $1 in
        -f)
            PYTHON_FILE="$2"
            shift 2
            ;;
        -c)
            YAML_FILE="$2"
            shift 2
            ;;
        -o)
            BASE_OUTPUT_DIR="$2"
            shift 2
            ;;
        --clear)
            CLEAR_FLAG=1
            shift
            ;;
        -v)
            VERBOSE_LEVEL="$2"
            shift 2
            ;;
        -g)
            GPU_ROCR_INDEX="$2"
            GPU_G_FLAG_USED=1
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        --)
            # All remaining arguments are for the Python script
            shift
            while [[ $# -gt 0 ]]; do
                PYTHON_ARGS+=("$1")
                shift
            done
            break
            ;;
        *)
            echo "Unknown option: $1"
            echo "Note: If you want to pass arguments to the Python script, use '--' to separate them."
            echo "      Example: $0 -f script.py -c config.yaml -o output -- --arg1 value1 --arg2"
            show_help
            exit 1
            ;;
    esac
done

# --- Validation ---
if [ -z "$PYTHON_FILE" ] || [ -z "$YAML_FILE" ] || [ -z "$BASE_OUTPUT_DIR" ]; then
    echo "Error: Python script (-f), YAML file (-c), and base output directory (-o) are required." >&2
    show_help
    exit 1
fi

# --- Resolve GPU index ---
# Priority: -g flag > external ROCR_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES > default (0)
if [ "$GPU_G_FLAG_USED" -eq 0 ] && [ -n "$_ORIG_ROCR_VISIBLE_DEVICES" ]; then
    # Caller set ROCR_VISIBLE_DEVICES externally; that GPU becomes device 0 inside the runtime
    GPU_ROCR_INDEX="$_ORIG_ROCR_VISIBLE_DEVICES"
    echo "🔧 Using external ROCR_VISIBLE_DEVICES=$GPU_ROCR_INDEX (mapped to runtime device 0)"
elif [ "$GPU_G_FLAG_USED" -eq 0 ] && [ -n "$_ORIG_HIP_VISIBLE_DEVICES" ]; then
    GPU_ROCR_INDEX="$_ORIG_HIP_VISIBLE_DEVICES"
    echo "🔧 Using external HIP_VISIBLE_DEVICES=$GPU_ROCR_INDEX (mapped to runtime device 0)"
fi

# Validate GPU index against mapping table; if not found, use it directly (bypass mapping)
if GPU_AMD_SMI_INDEX=$(get_amd_smi_index "$GPU_ROCR_INDEX" 2>/dev/null); then
    : # mapping found
else
    # GPU index not in mapping table — use it directly for amd-smi as well
    GPU_AMD_SMI_INDEX="$GPU_ROCR_INDEX"
    echo "⚠️  GPU index $GPU_ROCR_INDEX not in mapping table, using it directly for amd-smi"
fi

# Resolve paths to be absolute for robustness
if ! PYTHON_FILE_ABS=$(realpath -s "$PYTHON_FILE"); then
    echo "Error: Failed to resolve path for Python script: $PYTHON_FILE" >&2
    exit 1
fi

if ! YAML_FILE_ABS=$(realpath -s "$YAML_FILE"); then
    echo "Error: Failed to resolve path for YAML file: $YAML_FILE" >&2
    exit 1
fi

if [ ! -f "$PYTHON_FILE_ABS" ]; then
    echo "Error: Python script not found at '$PYTHON_FILE_ABS'" >&2
    exit 1
fi

if [ ! -f "$YAML_FILE_ABS" ]; then
    echo "Error: YAML file not found at '$YAML_FILE_ABS'" >&2
    exit 1
fi

# Check if amd-smi is available
if ! command -v amd-smi &> /dev/null; then
    echo "Error: amd-smi command not found. Please make sure ROCm is installed and in PATH." >&2
    exit 1
fi

# Check if rocprofv3 is available
if ! command -v rocprofv3 &> /dev/null; then
    echo "Error: rocprofv3 command not found. Please make sure ROCm profiler is installed and in PATH." >&2
    exit 1
fi

# --- Set GPU performance level ---
echo "🔧 Configuring GPU settings..."
echo "   ROCR_VISIBLE_DEVICES index: $GPU_ROCR_INDEX"
echo "   AMD-SMI index: $GPU_AMD_SMI_INDEX"

if ! set_gpu_performance_level "$GPU_AMD_SMI_INDEX" "stable_std"; then
    echo "❌ Error: Failed to set GPU performance level. Exiting." >&2
    exit 1
fi

# --- Get script name, Triton version, location, and LLVM commit ---
echo "🔍 Detecting Triton version and LLVM information..."
SCRIPT_NAME=$(get_script_basename "$PYTHON_FILE_ABS")
TRITON_VERSION=$(get_triton_version)
TRITON_LOCATION=$(get_triton_location)
TRITON_ROOT=$(get_triton_root "$TRITON_LOCATION")
LLVM_COMMIT=$(get_llvm_commit "$TRITON_ROOT")
TIMESTAMP=$(get_timestamp)

# Sanitize script name and version string for filename (replace problematic characters)
SCRIPT_NAME_CLEAN=$(echo "$SCRIPT_NAME" | sed 's/[^a-zA-Z0-9._-]/_/g')
TRITON_VERSION_CLEAN=$(echo "$TRITON_VERSION" | sed 's/[^a-zA-Z0-9+._-]/_/g')
LLVM_COMMIT_SHORT="${LLVM_COMMIT:0:8}"  # Use first 8 characters of LLVM commit

# Create the output directory name
OUTPUT_DIR_NAME="${SCRIPT_NAME_CLEAN}_triton_v${TRITON_VERSION_CLEAN}_llvm_${LLVM_COMMIT_SHORT}_${TIMESTAMP}"
BASE_OUTPUT_DIR_ABS=$(realpath -m "$BASE_OUTPUT_DIR")
OUTPUT_DIR_ABS="$BASE_OUTPUT_DIR_ABS/$OUTPUT_DIR_NAME"

# --- Main Execution ---
echo "========================================"
echo "ROCProfiler v3 PMC Trace Utility"
echo "========================================"
echo "▶️  Python Script: $PYTHON_FILE_ABS"
echo "▶️  YAML File: $YAML_FILE_ABS"
echo "▶️  Script Name: $SCRIPT_NAME"
echo "▶️  Triton Version: $TRITON_VERSION"
echo "▶️  Triton Location: $TRITON_LOCATION"
echo "▶️  Triton Root: $TRITON_ROOT"
echo "▶️  LLVM Commit: $LLVM_COMMIT"
echo "▶️  LLVM Commit (short): $LLVM_COMMIT_SHORT"
echo "▶️  Timestamp: $TIMESTAMP"
echo "▶️  GPU ROCR Index: $GPU_ROCR_INDEX"
echo "▶️  GPU AMD-SMI Index: $GPU_AMD_SMI_INDEX"
echo "▶️  Base Output Directory: $BASE_OUTPUT_DIR_ABS"
echo "▶️  Actual Output Directory: $OUTPUT_DIR_ABS"
if [ ${#PYTHON_ARGS[@]} -gt 0 ]; then
    echo "▶️  Python Script Arguments: ${PYTHON_ARGS[*]}"
else
    echo "▶️  Python Script Arguments: (none)"
fi

# Create base output directory if it doesn't exist
mkdir -p "$BASE_OUTPUT_DIR_ABS"

# Create the actual output directory
mkdir -p "$OUTPUT_DIR_ABS"

# If the --clear flag is specified, clear the output directory content
if [ "$CLEAR_FLAG" -eq 1 ]; then
    echo "🧹 Clearing contents of output directory: $OUTPUT_DIR_ABS"
    rm -rf "${OUTPUT_DIR_ABS:?}"/*
fi

# --- Permissions Check ---
if [ ! -w "$OUTPUT_DIR_ABS" ]; then
    echo "❌ Error: The output directory '$OUTPUT_DIR_ABS' is not writable by the current user." >&2
    echo "   Please check the directory permissions or choose a different directory." >&2
    exit 1
fi

# --- Save build and environment info ---
BUILD_INFO_FILE="$OUTPUT_DIR_ABS/build_info.txt"
echo "📝 Saving build and environment information to: $BUILD_INFO_FILE"

cat > "$BUILD_INFO_FILE" << EOF
=== ROCProfiler v3 PMC Trace Information ===
Generated on: $(date)
Script Name: $SCRIPT_NAME
YAML File: $YAML_FILE_ABS
Triton Version: $TRITON_VERSION
Triton Location: $TRITON_LOCATION
Triton Root Directory: $TRITON_ROOT
LLVM Commit Hash: $LLVM_COMMIT
LLVM Commit (short): $LLVM_COMMIT_SHORT
Timestamp: $TIMESTAMP
Python Script: $PYTHON_FILE_ABS
Python Script Arguments: ${PYTHON_ARGS[*]}
Output Directory: $OUTPUT_DIR_ABS
GPU ROCR_VISIBLE_DEVICES Index: $GPU_ROCR_INDEX
GPU AMD-SMI Index: $GPU_AMD_SMI_INDEX

=== GPU Index Mapping ===
AMD-SMI indices:              ${AMD_SMI_INDICES[*]}
ROCR_VISIBLE_DEVICES indices: ${ROCR_VISIBLE_DEVICES_INDICES[*]}

=== ROCProfiler Version ===
$(rocprofv3 --version 2>/dev/null || echo "Failed to get rocprofv3 version")

=== AMD-SMI Information ===
$(amd-smi --version 2>/dev/null || echo "Failed to get amd-smi version")

=== Triton Package Information ===
# $(pip show triton 2>/dev/null || echo "Failed to get Triton package info")
($PYTHON_BIN -m pip show triton 2>/dev/null || echo "Failed to get Triton package info")

=== LLVM Information ===
LLVM Hash File: $TRITON_ROOT/cmake/llvm-hash.txt
LLVM Commit: $LLVM_COMMIT
EOF

if [ -f "$TRITON_ROOT/cmake/llvm-hash.txt" ]; then
    echo "LLVM Hash File Content:" >> "$BUILD_INFO_FILE"
    cat "$TRITON_ROOT/cmake/llvm-hash.txt" >> "$BUILD_INFO_FILE" 2>/dev/null || echo "Failed to read LLVM hash file" >> "$BUILD_INFO_FILE"
else
    echo "LLVM Hash File: Not found at $TRITON_ROOT/cmake/llvm-hash.txt" >> "$BUILD_INFO_FILE"
fi

cat >> "$BUILD_INFO_FILE" << EOF

=== Python Environment ===
# Python Version: $(python --version 2>&1)
# Python Executable: $(which python)
Python Version: $("$PYTHON_BIN" --version 2>&1)
Python Executable: $PYTHON_BIN
Venv Directory: $VENV_DIR

=== System Information ===
Operating System: $(uname -a)
User: $(whoami)
Working Directory: $(pwd)

=== ROCm Information ===
$(rocm-smi --version 2>/dev/null || echo "rocm-smi not found")

=== YAML File Content ===
$(cat "$YAML_FILE_ABS" 2>/dev/null || echo "Failed to read YAML file")
EOF

if [ -n "$VERBOSE_LEVEL" ]; then
    echo "TRITON_VERBOSE=$VERBOSE_LEVEL" >> "$BUILD_INFO_FILE"
fi

# Set verbose level if specified
if [ -n "$VERBOSE_LEVEL" ]; then
    export TRITON_VERBOSE="$VERBOSE_LEVEL"
    echo "🔊 TRITON_VERBOSE is set to: $VERBOSE_LEVEL"
fi

# Set ROCR_VISIBLE_DEVICES to select the physical GPU,
# then HIP_VISIBLE_DEVICES=0 because the selected GPU becomes device 0 in the filtered view.
export ROCR_VISIBLE_DEVICES="$GPU_ROCR_INDEX"
export HIP_VISIBLE_DEVICES=0
echo "🔧 ROCR_VISIBLE_DEVICES is set to: $ROCR_VISIBLE_DEVICES (physical GPU)"
echo "🔧 HIP_VISIBLE_DEVICES is set to: 0 (device 0 in filtered view)"

# Set ROCm environment variables for rocprofv3
export HSA_OVERRIDE_GFX_VERSION="11.0.0"  # Adjust if needed
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export PATH="/opt/rocm/bin:${PATH}"
echo "🔧 ROCm environment configured for rocprofv3"

# Prepare log file
LOG_FILE="$OUTPUT_DIR_ABS/rocprof_execution_log.txt"

echo "----------------------------------------"
echo "🚀 Executing ROCProfiler v3 PMC trace..."

# Build the command with Python script arguments
if [ ${#PYTHON_ARGS[@]} -gt 0 ]; then
    echo "📄 Python script arguments: ${PYTHON_ARGS[*]}"
    # Build command array for proper argument handling
    # CMD_ARGS=(
    #     rocprofv3
    #     -i "$YAML_FILE_ABS"
    #     --output-directory "$OUTPUT_DIR_ABS"
    #     --
    #     python "$PYTHON_FILE_ABS"
    # )
    CMD_ARGS=(
        rocprofv3
        -i "$YAML_FILE_ABS"
        --output-directory "$OUTPUT_DIR_ABS"
        --
        "$PYTHON_BIN" "$PYTHON_FILE_ABS"
    )
    # Add Python script arguments
    CMD_ARGS+=("${PYTHON_ARGS[@]}")
    
    # Display the command (for logging)
    FULL_CMD="ROCR_VISIBLE_DEVICES=$GPU_ROCR_INDEX ${CMD_ARGS[*]}"
    echo "📄 Command: $FULL_CMD"
else
    # No Python arguments, use simple command
    CMD_ARGS=(
        rocprofv3
        -i "$YAML_FILE_ABS"
        --output-directory "$OUTPUT_DIR_ABS"
        --
        python "$PYTHON_FILE_ABS"
    )
    FULL_CMD="ROCR_VISIBLE_DEVICES=$GPU_ROCR_INDEX ${CMD_ARGS[*]}"
    echo "📄 Command: $FULL_CMD"
fi

echo "📄 Stdout/stderr will be redirected to: $LOG_FILE"

# Execute rocprofv3 command with Python script arguments
# Use array expansion to properly handle arguments with spaces
if ROCR_VISIBLE_DEVICES="$GPU_ROCR_INDEX" "${CMD_ARGS[@]}" > "$LOG_FILE" 2>&1; then
    echo "✅ ROCProfiler v3 execution completed successfully."
else
    echo "❌ ROCProfiler v3 execution failed. Check the log file: $LOG_FILE" >&2
    exit 1
fi

echo "----------------------------------------"

# Show generated files
echo "✅ PMC profiling complete. Files are located in: $OUTPUT_DIR_ABS"

FILE_COUNT=$(find "$OUTPUT_DIR_ABS" -type f | wc -l)
echo "ℹ️  Total files generated: $FILE_COUNT"

# List the main output files
echo "📄 Generated files:"
find "$OUTPUT_DIR_ABS" -type f | head -10

if [ $FILE_COUNT -gt 10 ]; then
    echo "   ... and $((FILE_COUNT - 10)) more files"
fi

echo "========================================"
echo "✅ All processing complete!"
echo "📁 Output directory: $OUTPUT_DIR_ABS"
echo "📄 Build info: $BUILD_INFO_FILE"
echo "📄 Execution log: $LOG_FILE"
echo "ℹ️  Triton Version: $TRITON_VERSION"
echo "ℹ️  LLVM Commit: $LLVM_COMMIT"
echo "ℹ️  GPU Index (ROCR): $GPU_ROCR_INDEX"
echo "ℹ️  GPU Index (AMD-SMI): $GPU_AMD_SMI_INDEX"
echo "========================================"

# Note: GPU cleanup will be handled by the trap cleanup function