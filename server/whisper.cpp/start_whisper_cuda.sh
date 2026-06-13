#!/bin/bash

# based on https://github.com/ggml-org/llama.cpp/discussions/16514


# Verify that nvidia-smi runs successfully
if ! nvidia-smi >/dev/null 2>&1; then
    printf "[E] 'nvidia-smi' command failed or no NVIDIA driver detected.\n"
    exit 1
fi

printf "[I] Installing whisper.cpp\n"
if [ ! -d "whisper.cpp" ]; then
    git clone https://github.com/ggml-org/whisper.cpp
    cd ./whisper.cpp
    cmake -B build-cuda -DGGML_CUDA=ON
    cmake --build build-cuda -j

    printf "[I] Downloading Whisper model...\n"
    ./models/download-ggml-model.sh large-v3-turbo-q8_0 > /dev/null 2>&1
    cd ..
else
    printf "[I] whisper.cpp already exists, skipping build\n"
fi

printf "Starting Whisper.. \n"
./whisper.cpp/build-cuda/bin/whisper-server -m ./whisper.cpp/models/ggml-large-v3-turbo-q8_0.bin --host 0.0.0.0 --port 8025 --inference-path /v1/audio/transcriptions
