#!/usr/bin/env bash
# Ubuntu / Linux laptop — install deps for server/ (edge). No app code changes.
set -euo pipefail
SERVER_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SERVER_DIR"

echo "=== AI CCTV server (Ubuntu) setup ==="
echo "Folder: $SERVER_DIR"

echo ""
echo "[1] System packages (needs sudo)..."
sudo apt-get update -qq
sudo apt-get install -y \
  python3-venv python3-pip \
  libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
  tesseract-ocr libzbar0

echo ""
echo "[2] Python venv..."
if [[ ! -d venv ]]; then
  python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
pip install -U pip

echo ""
echo "[3] CPU PyTorch + requirements-ubuntu.txt ..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install --upgrade-strategy only-if-needed -r requirements-ubuntu.txt

echo ""
echo "[4] .env ..."
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "  Created .env from .env.example — edit CLOUD_URL and EDGE_SYNC_SECRET"
else
  echo "  .env already exists — keeping it"
fi

if [[ ! -f models/sugar_bag_final.pt ]]; then
  echo ""
  echo "WARNING: models/sugar_bag_final.pt not found — copy .pt weights into server/models/"
fi

echo ""
echo "=== DONE ==="
echo "Start (example, port 5001 if cloud uses 5000):"
echo "  cd $SERVER_DIR"
echo "  source venv/bin/activate"
echo "  set -a && source .env && set +a"
echo "  PORT=5001 python app.py"
echo ""
