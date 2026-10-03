#!/usr/bin/env bash
# ==============================================================================
# Mini VeeVee - Automatic Model Downloader Script
# ==============================================================================
# Usage:
#   ./download_models.sh          # Trigger remote download on RunPod ComfyUI pod
#   ./download_models.sh local    # Download models locally with curl/wget/aria2c
# ==============================================================================

set -e

COMFY_URL="${COMFY_URL:-https://7a1raczqkrw81q-8188.proxy.runpod.net}"
BASE_MODELS_DIR="${1:-/workspace/runpod-slim/ComfyUI/models}"

# If first arg is 'local', run direct local download
if [ "$1" = "local" ] || [ "$2" = "local" ]; then
    TARGET_DIR="${2:-./models}"
    [ "$1" != "local" ] && TARGET_DIR="$1"
    
    echo "=========================================================="
    echo "  Mini VeeVee: Direct Local Download"
    echo "  Target: $TARGET_DIR"
    echo "=========================================================="

    mkdir -p "$TARGET_DIR/unet/lens"
    mkdir -p "$TARGET_DIR/text_encoders"
    mkdir -p "$TARGET_DIR/vae"

    download_file() {
        local url="$1"
        local dest="$2"
        echo "[-] Downloading $(basename "$dest")..."
        if command -v aria2c >/dev/null 2>&1; then
            aria2c -x 16 -s 16 -k 1M -c -d "$(dirname "$dest")" -o "$(basename "$dest")" "$url"
        elif command -v wget >/dev/null 2>&1; then
            wget -c --show-progress -O "$dest" "$url"
        else
            curl -L -C - --progress-bar -o "$dest" "$url"
        fi
        echo "[✓] Done: $(basename "$dest")"
    }

    download_file "https://huggingface.co/Comfy-Org/Lens/resolve/main/diffusion_models/lens_bf16.safetensors" "$TARGET_DIR/unet/lens/lens_bf16.safetensors"
    download_file "https://huggingface.co/Comfy-Org/Lens/resolve/main/text_encoders/gpt_oss_20b_nvfp4.safetensors" "$TARGET_DIR/text_encoders/gpt_oss_20b_nvfp4.safetensors"
    download_file "https://huggingface.co/Comfy-Org/Lens/resolve/main/vae/flux2-vae.safetensors" "$TARGET_DIR/vae/flux2-vae.safetensors"

    echo "=========================================================="
    echo "All models downloaded successfully!"
    echo "=========================================================="
    exit 0
fi

# Default: Run the python downloader
python3 "$(dirname "$0")/download_models.py" "$@"
