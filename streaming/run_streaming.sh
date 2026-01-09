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
HF_MODEL_REPO="${HF_MODEL_REPO:-Amshaker/Mobile-VideoGPT-0.5B}"  # Default base model
CONFIG_FILE="streaming_config.yaml"
DEVICE="cuda"
VIDEO_INPUT=""
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
    0.5B (default)          Amshaker/Mobile-VideoGPT-0.5B
    1.5B                    Amshaker/Mobile-VideoGPT-1.5B

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

print_info "Mobile-VideoGPT Streaming System"
echo "=================================="

# Check if model repo is set (use base model if empty)
if [[ -z "$HF_MODEL_REPO" ]]; then
    HF_MODEL_REPO="Amshaker/Mobile-VideoGPT-0.5B"
    print_warning "No model specified, using base model: $HF_MODEL_REPO"
else
    print_info "Using model: ${GREEN}$HF_MODEL_REPO${NC}"
fi

# Execute based on mode
case $MODE in
    test)
        print_info "Running unit tests..."
        python -m pytest tests/test_streaming.py -v --tb=short
        exit $?
        ;;

    inference)
        print_info "Running inference.py..."
        python inference.py
        exit $?
        ;;

    demo)
        # Check if demo_streaming.py exists
        if [[ ! -f "demo_streaming.py" ]]; then
            print_error "demo_streaming.py not found in project root!"
            exit 1
        fi

        # Check if config file exists
        if [[ ! -f "$CONFIG_FILE" ]]; then
            print_warning "Config file not found: $CONFIG_FILE"
            print_info "Using default configuration from code"
        else
            print_info "Using config: $CONFIG_FILE"
        fi

        # Build command
        CMD="python demo_streaming.py --model \"$HF_MODEL_REPO\" --device $DEVICE"

        if [[ -n "$VIDEO_INPUT" ]]; then
            if [[ ! -f "$VIDEO_INPUT" ]]; then
                print_error "Video file not found: $VIDEO_INPUT"
                exit 1
            fi
            CMD="$CMD --video \"$VIDEO_INPUT\""
            print_info "Input: Video file - $VIDEO_INPUT"
        else
            print_info "Input: Webcam (default)"
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
        fi

        if [[ "$NO_DISPLAY" == true ]]; then
            CMD="$CMD --no-display"
            print_info "Running in headless mode (no display)"
        fi

        print_info "Device: $DEVICE"
        echo ""
        print_info "Launching streaming demo..."
        print_info "Command: $CMD"
        echo ""

        # Run the demo
        eval $CMD
        EXIT_CODE=$?

        echo ""
        if [[ $EXIT_CODE -eq 0 ]]; then
            print_success "Demo completed successfully!"
        else
            print_error "Demo exited with code $EXIT_CODE"
        fi

        exit $EXIT_CODE
        ;;
esac
