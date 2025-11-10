#!/bin/bash

################################################################################
# Build PhotonInfer Docker Image
################################################################################

set -e

cd "$(dirname "$0")"

echo "========================================="
echo "Building PhotonInfer Docker Image"
echo "========================================="
echo ""

docker build -f Dockerfile -t photon_infer:latest ..

echo ""
echo "========================================="
echo "Build Complete!"
echo "========================================="
echo ""
docker images photon_infer:latest
