#!/usr/bin/env bash
set -e

# Minimal bootstrap script to run Forge on RunPod.
# Usage inside the pod (in /workspace):
#   rsync this file to /workspace
#   bash runpod_forge_setup.sh

cd /workspace

# 1) Clone/update Forge
if [ ! -d stable-diffusion-webui-forge ]; then
  git clone https://github.com/sangoi-exe/stable-diffusion-webui-forge.git
fi
cd stable-diffusion-webui-forge

# 2) Utility for Google Drive downloads (use pip directly to avoid PEP 668 issues)
pip install -q --upgrade pip
pip install -q gdown

# 3) Download models + configs into models/Stable-diffusion
MODEL_DIR="models/Stable-diffusion"
mkdir -p "${MODEL_DIR}"
cd "${MODEL_DIR}"

# Models (skip if already present and non-empty)
if [ ! -s "cyberrealistic_v70.safetensors" ]; then
  gdown --fuzzy 'https://drive.google.com/file/d/1F5WWD78pWna0-1-XKj_-NhJcywy3Nhxx/view?usp=drive_link' \
    -O cyberrealistic_v70.safetensors
fi

if [ ! -s "cyberrealisticPony_v130.safetensors" ]; then
  gdown --fuzzy 'https://drive.google.com/file/d/1U2OKD2apzOobHIiQi3lyUpHvefXY4eO9/view?usp=drive_link' \
    -O cyberrealisticPony_v130.safetensors
fi

if [ ! -s "rillusmRealisticIL_v21.safetensors" ]; then
  if ! gdown --fuzzy 'https://drive.google.com/file/d/1C6ff1Dp4zY9M_sPRV700LD_DXdultufJ/view?usp=drive_link' \
    -O rillusmRealisticIL_v21.safetensors; then
    echo "[runpod] WARNING: failed to download rillusmRealisticIL_v21.safetensors; check Drive sharing/quotas. Skipping." >&2
  fi
fi

cd ../..

# 3b) Extra models: ESRGAN + LoRA
ESRGAN_DIR="models/ESRGAN"
mkdir -p "${ESRGAN_DIR}"
if [ ! -s "${ESRGAN_DIR}/4x-UltraSharp.pth" ]; then
  if ! gdown --fuzzy 'https://drive.google.com/file/d/1GsPCwuhPSqUxxtUB8OlNrAUK85XpDthw/view?usp=drive_link' \
    -O "${ESRGAN_DIR}/4x-UltraSharp.pth"; then
    echo "[runpod] WARNING: failed to download 4x-UltraSharp.pth; check Drive sharing/quotas. Skipping." >&2
  fi
fi

LORA_DIR="models/Lora"
mkdir -p "${LORA_DIR}"
if [ ! -s "${LORA_DIR}/BallsDeep-IL-V2.2-S.safetensors" ]; then
  if ! gdown --fuzzy 'https://drive.google.com/file/d/1cflAGSI4hRMtf3HZJJmH0DKCOOpiLVTB/view?usp=drive_link' \
    -O "${LORA_DIR}/BallsDeep-IL-V2.2-S.safetensors"; then
    echo "[runpod] WARNING: failed to download BallsDeep-IL-V2.2-S.safetensors; check Drive sharing/quotas. Skipping." >&2
  fi
fi

# 4) Tell Forge which Torch to install inside its venv
export TORCH_COMMAND="pip install torch torchvision"

# 5) Create/overwrite webui-user.sh with listen + auth
cat > webui-user.sh << 'EOF'
#!/usr/bin/env bash
# Extra options to run Forge on RunPod.
# Change usuario:senha_forte to your own credentials.
export COMMANDLINE_ARGS="--listen --port 7860 --gradio-auth sangoi:PauBemGrosso --skip-prepare-environment"
EOF
chmod +x webui-user.sh

# 6) Remove any previously installed xformers in the venv to avoid
# version mismatches with the Torch stack used on RunPod.
if [ -d "venv" ]; then
  ./venv/bin/python -m pip uninstall -y xformers || true
  # Ensure hf_transfer is available for HF_HUB_ENABLE_HF_TRANSFER=1
  ./venv/bin/python -m pip install -q hf_transfer || true
fi

# 7) Launch Forge (force allow root inside container; venv handled by webui.sh)
bash webui.sh -f
