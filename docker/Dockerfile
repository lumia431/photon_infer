# PhotonInfer Docker Image
# Based on NVIDIA CUDA development image

# Stage 1: Build stage
FROM nvidia/cuda:12.5.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    ninja-build \
    pkg-config \
    libgtest-dev \
    libgmock-dev \
    libeigen3-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install newer CMake (3.24+) for CUDA C++20 support
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.27.7/cmake-3.27.7-linux-x86_64.sh && \
    chmod +x cmake-3.27.7-linux-x86_64.sh && \
    ./cmake-3.27.7-linux-x86_64.sh --prefix=/usr/local --skip-license && \
    rm cmake-3.27.7-linux-x86_64.sh

# Install GCC 12+ for C++20 support
RUN apt-get update && apt-get install -y \
    gcc-12 \
    g++-12 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy source code
COPY . /workspace/photon_infer

# Build PhotonInfer
WORKDIR /workspace/photon_infer
RUN mkdir -p build && cd build && \
    cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DPHOTON_BUILD_CUDA=ON \
    -DPHOTON_BUILD_TESTS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89" \
    .. && \
    ninja -j$(nproc)

# Stage 2: Runtime stage
FROM nvidia/cuda:12.5.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libeigen3-dev \
    libgoogle-glog0v5 \
    libgflags2.2 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}

WORKDIR /app

# Copy built binaries and libraries from builder
COPY --from=builder /workspace/photon_infer/build/bin /app/bin
COPY --from=builder /workspace/photon_infer/build/lib /app/lib

# Set library path
ENV LD_LIBRARY_PATH=/app/lib:${LD_LIBRARY_PATH}

# Create directories
RUN mkdir -p /models /app/results

CMD ["/bin/bash"]

LABEL maintainer="Lummy"
LABEL description="PhotonInfer - High-Performance LLM Inference Engine"
LABEL version="0.1.0"
