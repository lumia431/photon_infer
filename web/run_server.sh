#!/bin/bash
# PhotonInfer Web Server Startup Script

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  PhotonInfer Web Server${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if binary exists
if [ ! -f "../build/bin/photon_web_server" ]; then
    echo -e "${RED}Error: photon_web_server not found!${NC}"
    echo -e "${RED}Please build the project first:${NC}"
    echo -e "  cd ../build"
    echo -e "  cmake --build . --target photon_web_server -j\$(nproc)"
    exit 1
fi

# Default configuration
PORT=${PORT:-5728}
MODEL_PATH=${MODEL_PATH:-"/home/lummy/.llama/checkpoints/Llama3.2-1B-Instruct/model.bin"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/home/lummy/.llama/checkpoints/Llama3.2-1B-Instruct/tokenizer.model"}
STATIC_DIR=${STATIC_DIR:-"./web/static"}

echo -e "${GREEN}Configuration:${NC}"
echo -e "  Port: ${PORT}"
echo -e "  Model: ${MODEL_PATH}"
echo -e "  Tokenizer: ${TOKENIZER_PATH}"
echo -e "  Static Dir: ${STATIC_DIR}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Run the server
cd ../build/bin
./photon_web_server \
    --port ${PORT} \
    --model ${MODEL_PATH} \
    --tokenizer ${TOKENIZER_PATH} \
    --static-dir ${STATIC_DIR}
