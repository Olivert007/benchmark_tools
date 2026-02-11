#!/bin/bash

# A script to run rocprofv3 with a Triton-based Python file and generate
# organized output files based on script name, Triton version, and LLVM commit.

set -e # Exit immediately if a command exits with a non-zero status.

# --- Help Message ---
usage() {
    echo "Usage: $0 -f <triton_script.py> -c <config.yaml> -o <output_dir> [-v <level>] [-d <devices>] [-p] [-m <method>] [-u <unit>] [-i <interval>] [-h] [-- <script args>]"
    echo ""
    echo "Arguments:"
    echo "  -f <path>     Required. Path to the Triton Python script to execute."
    echo "  -c <path>     Required. Path to the rocprofv3 configuration YAML file."
    echo "  -o <path>     Required. Base directory to save the rocprofv3 output files."
    echo "                The actual output will be in a subdirectory with script name, version and timestamp."
    echo "  -v <level>    Optional. Set the TRITON_VERBOSE level (e.g., 1 for basic, 4 for max)."
    echo "  -d <devices>  Optional. Set ROCR_VISIBLE_DEVICES (e.g., '0', '0,1', '1,2,3')."
    echo "  -p            Optional. Enable PC sampling (beta feature)."
    echo "  -m <method>   Optional. PC sampling method: 'host_trap' or 'stochastic' (default: host_trap)."
    echo "  -u <unit>     Optional. PC sampling unit: 'time' or 'cycles' (default: time for host_trap, cycles for stochastic)."
    echo "  -i <interval> Optional. PC sampling interval (default: 1 for time, 1048576 for cycles)."
    echo "  -h            Display this help message."
    echo ""
    echo "Examples:"
    echo "  $0 -f ./test_moe_new.py -c ./att_trace.yaml -o ./rocprof_output"
    echo "  $0 -f ./test_moe_new.py -c ./att_trace.yaml -o ./rocprof_output -v 1 -d 0,1"
    echo "  $0 -f ./test_moe_new.py -c ./att_trace.yaml -o ./rocprof_output -p -m host_trap -u time -i 1"
    echo "  $0 -f ./test_moe_new.py -c ./att_trace.yaml -o ./rocprof_output -p -m stochastic -u cycles -i 1048576"
    echo "  $0 -f ./test_moe_new.py -c ./att_trace.yaml -o ./rocprof_output -- --no-torch-baseline --repeat-runs 30"
    echo ""
    echo "Note: The actual output directory will include script name, Triton version and timestamp,"
    echo "      e.g., <output_dir>/test_moe_new_triton_v3.3.1+gitd95ec8e0_20240819_143052/"
    echo "      If PC sampling is enabled, '_pcsampling' will be appended to the directory name."
    exit 1
}

# --- Function to get Triton version ---
get_triton_version() {
    local version
    if version=$(pip show triton 2>/dev/null | grep "^Version:" | cut -d' ' -f2); then
        echo "$version"
    else
        echo "unknown"
    fi
}

# --- Function to get Triton source location ---
get_triton_location() {
    local location
    # Try to get Editable project location first (for development installs)
    if location=$(pip show triton 2>/dev/null | grep "^Editable project location:" | cut -d' ' -f4); then
        echo "$location"
    # If that fails, try Location (for installed packages)
    elif location=$(pip show triton 2>/dev/null | grep "^Location:" | cut -d' ' -f2); then
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
    # Location could be something like "/path/to/triton_repo/python"
    # We need to find the parent directory that starts with "triton"
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

# --- Argument Parsing ---
PYTHON_FILE=""
CONFIG_FILE=""
BASE_OUTPUT_DIR=""
VERBOSE_LEVEL=""
ROCR_DEVICES=""
PC_SAMPLING_ENABLED=false
PC_SAMPLING_METHOD="host_trap"
PC_SAMPLING_UNIT=""
PC_SAMPLING_INTERVAL=""

while getopts "f:c:o:v:d:pm:u:i:h" opt; do
    case ${opt} in
        f) PYTHON_FILE=$OPTARG ;;
        c) CONFIG_FILE=$OPTARG ;;
        o) BASE_OUTPUT_DIR=$OPTARG ;;
        v) VERBOSE_LEVEL=$OPTARG ;;
        d) ROCR_DEVICES=$OPTARG ;;
        p) PC_SAMPLING_ENABLED=true ;;
        m) PC_SAMPLING_METHOD=$OPTARG ;;
        u) PC_SAMPLING_UNIT=$OPTARG ;;
        i) PC_SAMPLING_INTERVAL=$OPTARG ;;
        h) usage ;;
        \?) usage ;;
    esac
done

shift $((OPTIND - 1))
PYTHON_ARGS=("$@")

# --- Strip PyTorch baseline by default (profiling focus) ---
# If user accidentally passes "--torch" to verify_conv3d_impl.py, it will add
# extra GPU work and noise. Keep it only when explicitly allowed.
ALLOW_TORCH_BASELINE="${ALLOW_TORCH_BASELINE:-0}"
if [ "$ALLOW_TORCH_BASELINE" != "1" ] && [ ${#PYTHON_ARGS[@]} -gt 0 ]; then
    FILTERED_PYTHON_ARGS=()
    REMOVED_TORCH=false
    for arg in "${PYTHON_ARGS[@]}"; do
        if [ "$arg" = "--torch" ]; then
            REMOVED_TORCH=true
            continue
        fi
        FILTERED_PYTHON_ARGS+=("$arg")
    done
    if [ "$REMOVED_TORCH" = true ]; then
        echo "⚠️  '--torch' was removed from python args (set ALLOW_TORCH_BASELINE=1 to keep it)."
    fi
    PYTHON_ARGS=("${FILTERED_PYTHON_ARGS[@]}")
fi

# --- Validation ---
if [ -z "$PYTHON_FILE" ] || [ -z "$CONFIG_FILE" ] || [ -z "$BASE_OUTPUT_DIR" ]; then
    echo "Error: Python script path (-f), config file (-c), and output directory (-o) are required." >&2
    usage
fi

# Resolve paths to be absolute for robustness
if ! PYTHON_FILE_ABS=$(realpath -s "$PYTHON_FILE"); then
    echo "Error: Failed to resolve path for Python script: $PYTHON_FILE" >&2
    exit 1
fi

if ! CONFIG_FILE_ABS=$(realpath -s "$CONFIG_FILE"); then
    echo "Error: Failed to resolve path for config file: $CONFIG_FILE" >&2
    exit 1
fi

if [ ! -f "$PYTHON_FILE_ABS" ]; then
    echo "Error: Python script not found at '$PYTHON_FILE_ABS'" >&2
    exit 1
fi

if [ ! -f "$CONFIG_FILE_ABS" ]; then
    echo "Error: Config file not found at '$CONFIG_FILE_ABS'" >&2
    exit 1
fi

# Check if rocprofv3 is available
if ! command -v rocprofv3 &> /dev/null; then
    echo "Error: rocprofv3 command not found. Please make sure ROCm profiler is installed and in PATH." >&2
    exit 1
fi

# --- PC Sampling Validation and Defaults ---
if [ "$PC_SAMPLING_ENABLED" = true ]; then
    # Validate method
    if [ "$PC_SAMPLING_METHOD" != "host_trap" ] && [ "$PC_SAMPLING_METHOD" != "stochastic" ]; then
        echo "Error: PC sampling method must be 'host_trap' or 'stochastic', got '$PC_SAMPLING_METHOD'" >&2
        exit 1
    fi
    
    # Set default unit based on method if not specified
    if [ -z "$PC_SAMPLING_UNIT" ]; then
        if [ "$PC_SAMPLING_METHOD" = "host_trap" ]; then
            PC_SAMPLING_UNIT="time"
        else
            PC_SAMPLING_UNIT="cycles"
        fi
    fi
    
    # Validate unit
    if [ "$PC_SAMPLING_UNIT" != "time" ] && [ "$PC_SAMPLING_UNIT" != "cycles" ]; then
        echo "Error: PC sampling unit must be 'time' or 'cycles', got '$PC_SAMPLING_UNIT'" >&2
        exit 1
    fi
    
    # Set default interval based on unit if not specified
    if [ -z "$PC_SAMPLING_INTERVAL" ]; then
        if [ "$PC_SAMPLING_UNIT" = "time" ]; then
            PC_SAMPLING_INTERVAL="1"
        else
            PC_SAMPLING_INTERVAL="1048576"
        fi
    fi
    
    echo "✅ PC Sampling is enabled with the following settings:"
    echo "   Method: $PC_SAMPLING_METHOD"
    echo "   Unit: $PC_SAMPLING_UNIT"
    echo "   Interval: $PC_SAMPLING_INTERVAL"
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

# Create the output filename and directory name
OUTPUT_FILE_NAME="${SCRIPT_NAME_CLEAN}_triton_v${TRITON_VERSION_CLEAN}_llvm_${LLVM_COMMIT_SHORT}_${TIMESTAMP}"
# Add PC sampling suffix if enabled
if [ "$PC_SAMPLING_ENABLED" = true ]; then
    OUTPUT_FILE_NAME="${OUTPUT_FILE_NAME}_pcsampling_${PC_SAMPLING_METHOD}"
fi
OUTPUT_DIR_NAME="$OUTPUT_FILE_NAME"
BASE_OUTPUT_DIR_ABS=$(realpath -m "$BASE_OUTPUT_DIR")
OUTPUT_DIR_ABS="$BASE_OUTPUT_DIR_ABS/$OUTPUT_DIR_NAME"

# --- Main Execution ---
echo "========================================"
echo "ROCProfiler v3 Execution Utility"
echo "========================================"
echo "▶️  Python Script: $PYTHON_FILE_ABS"
echo "▶️  Config File: $CONFIG_FILE_ABS"
echo "▶️  Script Name: $SCRIPT_NAME"
echo "▶️  Triton Version: $TRITON_VERSION"
echo "▶️  Triton Location: $TRITON_LOCATION"
echo "▶️  Triton Root: $TRITON_ROOT"
echo "▶️  LLVM Commit: $LLVM_COMMIT"
echo "▶️  LLVM Commit (short): $LLVM_COMMIT_SHORT"
echo "▶️  Timestamp: $TIMESTAMP"
if [ "$PC_SAMPLING_ENABLED" = true ]; then
    echo "▶️  PC Sampling: Enabled"
    echo "▶️  PC Method: $PC_SAMPLING_METHOD"
    echo "▶️  PC Unit: $PC_SAMPLING_UNIT"
    echo "▶️  PC Interval: $PC_SAMPLING_INTERVAL"
fi
echo "▶️  Base Output Directory: $BASE_OUTPUT_DIR_ABS"
echo "▶️  Actual Output Directory: $OUTPUT_DIR_ABS"
echo "▶️  Output File Name: $OUTPUT_FILE_NAME"

# Create base output directory if it doesn't exist
mkdir -p "$BASE_OUTPUT_DIR_ABS"

# Create the actual output directory with script name, version and timestamp
mkdir -p "$OUTPUT_DIR_ABS"

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
=== ROCProfiler v3 Execution Information ===
Generated on: $(date)
Script Name: $SCRIPT_NAME
Config File: $CONFIG_FILE_ABS
Triton Version: $TRITON_VERSION
Triton Location: $TRITON_LOCATION
Triton Root Directory: $TRITON_ROOT
LLVM Commit Hash: $LLVM_COMMIT
LLVM Commit (short): $LLVM_COMMIT_SHORT
Timestamp: $TIMESTAMP
Python Script: $PYTHON_FILE_ABS
Output Directory: $OUTPUT_DIR_ABS
Output File Name: $OUTPUT_FILE_NAME

=== PC Sampling Configuration ===
PC Sampling Enabled: $PC_SAMPLING_ENABLED
EOF

if [ "$PC_SAMPLING_ENABLED" = true ]; then
    cat >> "$BUILD_INFO_FILE" << EOF
PC Sampling Method: $PC_SAMPLING_METHOD
PC Sampling Unit: $PC_SAMPLING_UNIT
PC Sampling Interval: $PC_SAMPLING_INTERVAL
EOF
fi

cat >> "$BUILD_INFO_FILE" << EOF

=== ROCProfiler Version ===
$(rocprofv3 --version 2>/dev/null || echo "Failed to get rocprofv3 version")

=== Triton Package Information ===
$(pip show triton 2>/dev/null || echo "Failed to get Triton package info")

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
Python Version: $(python --version 2>&1)
Python Executable: $(which python)

=== Environment Variables ===
EOF

if [ -n "$VERBOSE_LEVEL" ]; then
    echo "TRITON_VERBOSE=$VERBOSE_LEVEL" >> "$BUILD_INFO_FILE"
fi

if [ -n "$ROCR_DEVICES" ]; then
    echo "ROCR_VISIBLE_DEVICES=$ROCR_DEVICES" >> "$BUILD_INFO_FILE"
fi

cat >> "$BUILD_INFO_FILE" << EOF

=== System Information ===
Operating System: $(uname -a)
User: $(whoami)
Working Directory: $(pwd)

=== ROCm Information ===
$(rocm-smi --version 2>/dev/null || echo "rocm-smi not found")

=== Config File Content ===
$(cat "$CONFIG_FILE_ABS" 2>/dev/null || echo "Failed to read config file")
EOF

# Set verbose level if specified
if [ -n "$VERBOSE_LEVEL" ]; then
    export TRITON_VERBOSE="$VERBOSE_LEVEL"
    echo "TRITON_VERBOSE is set to: $VERBOSE_LEVEL"
fi

# Set ROCR_VISIBLE_DEVICES if specified
if [ -n "$ROCR_DEVICES" ]; then
    export ROCR_VISIBLE_DEVICES="$ROCR_DEVICES"
    echo "ROCR_VISIBLE_DEVICES is set to: $ROCR_DEVICES"
fi

# Prepare log file
LOG_FILE="$OUTPUT_DIR_ABS/rocprof_execution_log.txt"

# Build rocprofv3 command
ROCPROF_CMD="rocprofv3 -i \"$CONFIG_FILE_ABS\" --output-file \"$OUTPUT_FILE_NAME\" --output-directory \"$OUTPUT_DIR_ABS\""

# Add PC sampling options if enabled
if [ "$PC_SAMPLING_ENABLED" = true ]; then
    ROCPROF_CMD="$ROCPROF_CMD --pc-sampling-beta-enabled"
    ROCPROF_CMD="$ROCPROF_CMD --pc-sampling-method $PC_SAMPLING_METHOD"
    ROCPROF_CMD="$ROCPROF_CMD --pc-sampling-unit $PC_SAMPLING_UNIT"
    ROCPROF_CMD="$ROCPROF_CMD --pc-sampling-interval $PC_SAMPLING_INTERVAL"
    # Note: output_format should be specified in the YAML config file to avoid conflicts
fi

ROCPROF_CMD="$ROCPROF_CMD -- python \"$PYTHON_FILE_ABS\""
if [ ${#PYTHON_ARGS[@]} -gt 0 ]; then
    for arg in "${PYTHON_ARGS[@]}"; do
        ROCPROF_CMD="$ROCPROF_CMD \"$arg\""
    done
fi

echo "----------------------------------------"
echo "🚀 Executing ROCProfiler v3..."
echo "📄 Command: $ROCPROF_CMD"
if [ -n "$ROCR_DEVICES" ]; then
    echo "🎯 ROCR_VISIBLE_DEVICES: $ROCR_DEVICES"
fi
if [ "$PC_SAMPLING_ENABLED" = true ]; then
    echo "🔬 PC Sampling: Enabled ($PC_SAMPLING_METHOD, $PC_SAMPLING_UNIT, interval=$PC_SAMPLING_INTERVAL)"
fi
echo "📄 Stdout/stderr will be redirected to: $LOG_FILE"

# Execute rocprofv3 command
if eval "$ROCPROF_CMD" > "$LOG_FILE" 2>&1; then
    echo "✅ ROCProfiler v3 execution completed successfully."
else
    echo "❌ ROCProfiler v3 execution failed. Check the log file: $LOG_FILE" >&2
    exit 1
fi

echo "----------------------------------------"

# Show generated files
echo "✅ Profiling complete. Files are located in: $OUTPUT_DIR_ABS"

FILE_COUNT=$(find "$OUTPUT_DIR_ABS" -type f | wc -l)
echo "ℹ️  Total files generated: $FILE_COUNT"

# List the main output files
echo "📄 Generated files:"
find "$OUTPUT_DIR_ABS" -type f -name "${OUTPUT_FILE_NAME}*" | head -10

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
if [ "$PC_SAMPLING_ENABLED" = true ]; then
    echo "🔬 PC Sampling: Enabled ($PC_SAMPLING_METHOD, $PC_SAMPLING_UNIT, interval=$PC_SAMPLING_INTERVAL)"
fi
echo "📄 Output file prefix: $OUTPUT_FILE_NAME"
echo "========================================"