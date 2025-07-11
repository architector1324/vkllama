#!/bin/bash

mkdir -p build

python -m venv ./build/venv
source ./build/venv/bin/activate

pip install --upgrade pip
pip install pyinstaller
CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python
