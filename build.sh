#!/bin/bash

mkdir -p build

python -m venv ./build/venv
source ./build/venv/bin/activate

pip install --upgrade pip
pip install pyinstaller requests psutil
CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python

cp ./src/*.py -t build/
cd build

mkdir -p ./venv/lib/python3.13/site-packages/llama_cpp/lib/
cp ./venv/lib/python3.13/site-packages/lib/libllama.so ./venv/lib/python3.13/site-packages/llama_cpp/lib/

pyinstaller --onefile --collect-all llama-cpp-python \
    --hidden-import vkllama_run \
    --hidden-import vkllama_serve \
    --hidden-import vkllama_list \
    --add-data="vkllama_run.py:." \
    --add-data="vkllama_list.py:." \
    --add-data="vkllama_serve.py:." \
    --add-binary="./venv/lib/python3.13/site-packages/llama_cpp/lib/libllama.so:llama_cpp/lib" \
    vkllama.py

mv dist/vkllama ../vkllama
