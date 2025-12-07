#!/bin/bash
# PhotonInfer Startup Script for Docker
# Starts the web server (models are already downloaded during image build)

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  PhotonInfer Web Server${NC}"
echo -e "${BLUE}========================================${NC}"

# Model paths
MODEL_DIR="/models"
MODEL_PATH="${MODEL_DIR}/model.bin"
TOKENIZER_PATH="${MODEL_DIR}/tokenizer.model"

# Verify model files exist
echo -e "${GREEN}Verifying model files...${NC}"
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model file not found at ${MODEL_PATH}${NC}"
    exit 1
fi

if [ ! -f "$TOKENIZER_PATH" ]; then
    echo -e "${RED}Error: Tokenizer file not found at ${TOKENIZER_PATH}${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Model files verified${NC}"

# Server configuration
PORT=${PORT:-5728}
STATIC_DIR="/app/web/static"

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Configuration:${NC}"
echo -e "  Port: ${PORT}"
echo -e "  Model: ${MODEL_PATH}"
echo -e "  Tokenizer: ${TOKENIZER_PATH}"
echo -e "  Static Dir: ${STATIC_DIR}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if photon_web_server exists
if [ ! -f "/app/bin/photon_web_server" ]; then
    echo -e "${RED}Error: photon_web_server not found!${NC}"
    exit 1
fi

# Start the web server
echo -e "${GREEN}Starting PhotonInfer Web Server...${NC}"
cd /app/bin
exec ./photon_web_server \
    --port ${PORT} \
    --model ${MODEL_PATH} \
    --tokenizer ${TOKENIZER_PATH} \
    --static-dir ${STATIC_DIR}
