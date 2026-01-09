#!/bin/bash

# =============================================================================
# Mobile-VideoGPT Streaming Inference Runner
# =============================================================================
# This script provides a convenient way to run the streaming inference system
# with configurable HuggingFace model repository and various options.
#
# Usage:
#   ./run_streaming.sh [OPTIONS]
#
# Examples:
#   ./run_streaming.sh                           # Run with default model (webcam)
#   ./run_streaming.sh --model 1.5B              # Use 1.5B model variant
#   ./run_streaming.sh --video path/to/video.mp4 # Process video file
#   ./run_streaming.sh --custom-model user/model # Use custom HF model
# =============================================================================

# Default configuration
HF_MODEL_REPO="${HF_MODEL_REPO:-EdgeVLM-Labs/mobile-videogpt-finetune-2000}"  # Default finetuned model
CONFIG_FILE="streaming_config.yaml"
DEVICE="cuda"
VIDEO_INPUT="sample_videos/test_stream.mp4"  # Default test video
MAX_FRAMES=""
SAVE_OUTPUT=false
NO_DISPLAY=false
CUSTOM_MODEL=""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
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

# Function to display usage
show_usage() {
    cat << EOF
${GREEN}Mobile-VideoGPT Streaming Inference Runner${NC}

${BLUE}Usage:${NC}
    $0 [OPTIONS]

${BLUE}Options:${NC}
    -h, --help              Show this help message
    -m, --model MODEL       Use predefined model variant (0.5B or 1.5B)
    -c, --custom MODEL      Use custom HuggingFace model repo (e.g., user/model-name)
    -v, --video PATH        Process video file instead of webcam
    -d, --device DEVICE     Device to use (cuda, cpu, mps) [default: cuda]
    -f, --max-frames N      Maximum number of frames to process
    -s, --save              Save output video to file
    -n, --no-display        Run without display (headless mode)
    --config PATH           Path to config file [default: streaming_config.yaml]
    --test                  Run unit tests instead of demo
    --inference             Run inference.py instead of demo

${BLUE}Examples:${NC}
    ${GREEN}# Run with default 0.5B model (webcam)${NC}
    $0

    ${GREEN}# Use 1.5B model variant${NC}
    $0 --model 1.5B

    ${GREEN}# Use custom HuggingFace model${NC}
    $0 --custom YourUsername/your-finetuned-model

    ${GREEN}# Process a video file${NC}
    $0 --video sample_videos/00000340.mp4

    ${GREEN}# Save output with custom model${NC}
    $0 --model 1.5B --video input.mp4 --save

    ${GREEN}# Headless processing on CPU${NC}
    $0 --video input.mp4 --device cpu --no-display --save

    ${GREEN}# Run unit tests${NC}
    $0 --test

${BLUE}Environment Variables:${NC}
    HF_MODEL_REPO           Override default HuggingFace model repository

${BLUE}Model Variants:${NC}
    0.5B                    Amshaker/Mobile-VideoGPT-0.5B
    1.5B                    Amshaker/Mobile-VideoGPT-1.5B
    finetuned (default)     EdgeVLM-Labs/mobile-videogpt-finetune-2000

EOF
}

# Parse command line arguments
MODE="demo"
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -m|--model)
            MODEL_VARIANT="$2"
            if [[ "$MODEL_VARIANT" == "0.5B" ]]; then
                HF_MODEL_REPO="Amshaker/Mobile-VideoGPT-0.5B"
            elif [[ "$MODEL_VARIANT" == "1.5B" ]]; then
                HF_MODEL_REPO="Amshaker/Mobile-VideoGPT-1.5B"
            else
                print_error "Invalid model variant: $MODEL_VARIANT. Use 0.5B or 1.5B"
                exit 1
            fi
            shift 2
            ;;
        -c|--custom)
            CUSTOM_MODEL="$2"
            HF_MODEL_REPO="$CUSTOM_MODEL"
            shift 2
            ;;
        -v|--video)
            VIDEO_INPUT="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -f|--max-frames)
            MAX_FRAMES="$2"
            shift 2
            ;;
        -s|--save)
            SAVE_OUTPUT=true
            shift
            ;;
        -n|--no-display)
            NO_DISPLAY=true
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --test)
            MODE="test"
            shift
            ;;
        --inference)
            MODE="inference"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Navigate to project root (one level up from streaming/)
cd "$(dirname "$0")/.." || exit 1

# Create logs directory if it doesn't exist
LOG_DIR="logs/streaming"
mkdir -p "$LOG_DIR"

# Create log file with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/streaming_${TIMESTAMP}.log"

print_info "Mobile-VideoGPT Streaming System"
echo "=================================="
print_info "Log file: $LOG_FILE"

# Function to log messages to both console and file
log_message() {
    echo "$1" | tee -a "$LOG_FILE"
}

# Save session configuration to log
log_message "========================================="
log_message "Mobile-VideoGPT Streaming Session"
log_message "Started: $(date)"
log_message "========================================="
log_message "Model: $HF_MODEL_REPO"
log_message "Device: $DEVICE"
log_message "Config: $CONFIG_FILE"
log_message "Mode: $MODE"
log_message "========================================="

# Check if model repo is set (use base model if empty)
if [[ -z "$HF_MODEL_REPO" ]]; then
    HF_MODEL_REPO="EdgeVLM-Labs/mobile-videogpt-finetune-2000"
    print_warning "No model specified, using default model: $HF_MODEL_REPO"
    log_message "WARNING: No model specified, using default: $HF_MODEL_REPO"
else
    print_info "Using model: ${GREEN}$HF_MODEL_REPO${NC}"
    log_message "Model: $HF_MODEL_REPO"
fi

# Execute based on mode
case $MODE in
    test)
        print_info "Running unit tests..."
        log_message "Running unit tests..."
        python -m pytest tests/test_streaming.py -v --tb=short 2>&1 | tee -a "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}
        log_message "Test completed with exit code: $EXIT_CODE"
        exit $EXIT_CODE
        ;;

    inference)
        print_info "Running inference.py..."
        log_message "Running inference.py..."
        python inference.py 2>&1 | tee -a "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}
        log_message "Inference completed with exit code: $EXIT_CODE"
        exit $EXIT_CODE
        ;;

    demo)
        # Check if demo_streaming.py exists
        if [[ ! -f "demo_streaming.py" ]]; then
            print_error "demo_streaming.py not found in project root!"
            log_message "ERROR: demo_streaming.py not found!"
            exit 1
        fi

        # Check if config file exists
        if [[ ! -f "$CONFIG_FILE" ]]; then
            print_warning "Config file not found: $CONFIG_FILE"
            print_info "Using default configuration from code"
            log_message "WARNING: Config file not found, using defaults"
        else
            print_info "Using config: $CONFIG_FILE"
            log_message "Config file: $CONFIG_FILE"
        fi

        # Build command
        CMD="python demo_streaming.py --model \"$HF_MODEL_REPO\" --device $DEVICE"

        if [[ -n "$VIDEO_INPUT" ]]; then
            if [[ ! -f "$VIDEO_INPUT" ]]; then
                print_error "Video file not found: $VIDEO_INPUT"
                log_message "ERROR: Video file not found: $VIDEO_INPUT"
                exit 1
            fi
            CMD="$CMD --video \"$VIDEO_INPUT\""
            print_info "Input: Video file - $VIDEO_INPUT"
            log_message "Input: Video file - $VIDEO_INPUT"
        else
            print_info "Input: Webcam (default)"
            log_message "Input: Webcam"
        fi

        if [[ -f "$CONFIG_FILE" ]]; then
            CMD="$CMD --config \"$CONFIG_FILE\""
        fi

        if [[ -n "$MAX_FRAMES" ]]; then
            CMD="$CMD --max-frames $MAX_FRAMES"
        fi

        if [[ "$SAVE_OUTPUT" == true ]]; then
            CMD="$CMD --save-output"
            print_info "Output will be saved to file"
            log_message "Save output: enabled"
        fi

        if [[ "$NO_DISPLAY" == true ]]; then
            CMD="$CMD --no-display"
            print_info "Running in headless mode (no display)"
            log_message "Display: disabled (headless)"
        fi

        print_info "Device: $DEVICE"
        echo ""
        print_info "Launching streaming demo..."
        log_message "========================================="
        log_message "Launching streaming demo"
        log_message "Command: $CMD"
        log_message "========================================="
        print_info "Command: $CMD"
        echo ""

        # Run the demo with output logging
        eval $CMD 2>&1 | tee -a "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}

        echo ""
        log_message "========================================="
        if [[ $EXIT_CODE -eq 0 ]]; then
            print_success "Demo completed successfully!"
            log_message "Demo completed successfully!"
            log_message "Exit code: 0"
        else
            print_error "Demo exited with code $EXIT_CODE"
            log_message "ERROR: Demo exited with code $EXIT_CODE"
        fi
        log_message "Ended: $(date)"
        log_message "========================================="

        print_info "Full log saved to: $LOG_FILE"
        exit $EXIT_CODE
        ;;
esac
