# Vulkan LLaMA

**vkllama** is a lightweight and fast HTTP server for running LLaMA-based models locally using the **Vulkan backend**, built on top of [llama-cpp-python](https://github.com/abetlen/llama-cpp-python).

The project provides an **Ollama-compatible API**, so you can use existing clients and tools designed for Ollama without any changes.

---

## Features

*   **Single Binary, Zero Runtime Dependencies**: After building, `vkllama` is a self-contained executable with no external runtime dependencies, making it highly portable.
*   **Vulkan Backend**: Leverages `llama-cpp-python` with Vulkan support for efficient LLM inference on compatible GPUs.
*   **Ollama-Compatible API**: Provides `/api/generate`, `/api/chat`, and `/api/tags` endpoints, allowing integration with existing Ollama clients and tools.
*   **Lightweight HTTP Server**: Built with Python's standard `http.server` module, ensuring minimal overhead.
*   **Comprehensive CLI**: A user-friendly command-line interface for running models, listing available models, and starting the server.
*   **Easy Setup**: Includes a `build.sh` script for environment setup and executable creation, and a Systemd service file for production deployment.

## Prerequisites

Before you start building `vkllama` from source, ensure you have the following installed on your system:

*   **Git**: For cloning the repository.
*   **Python 3.x**: (3.10+ recommended)
*   **CMake**: Required by `llama-cpp-python` for building.
*   **Vulkan-compatible GPU and Drivers**: Ensure your system has up-to-date Vulkan drivers for your GPU.

## Installation

You have two main options for getting `vkllama` running: downloading a pre-built binary or building it from source.

### Download a Pre-built Binary (Recommended)

Pre-built `vkllama` binaries are available in the [releases section of this repository](https://github.com/architector1324/vkllama/releases) (replace with your actual repo URL).

1.  **Download the latest release**:
    Download the `vkllama` executable for your operating system.

2.  **Make `vkllama` executable and globally accessible (Optional but Recommended)**:

    ```bash
    chmod +x vkllama # Make it executable
    sudo mv vkllama /usr/local/bin/ # Move to a directory in your PATH
    ```

    Now you can run `vkllama` from any directory in your terminal.

### Build from Source

This option requires development tools like Python, Git, and CMake.

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/vkllama.git # Replace with your actual repo URL if different
    cd vkllama
    ```

2.  **Build the executable**:

    The `build.sh` script sets up a Python virtual environment, installs necessary dependencies (including `llama-cpp-python` with Vulkan support), and packages the application into a single executable using PyInstaller.

    ```bash
    ./build.sh
    ```

    After a successful build, a `vkllama` executable will be created in the root of your project directory.

3.  **Make `vkllama` globally accessible (Optional but Recommended)**:

    Move the `vkllama` executable to a directory in your system's PATH, such as `/usr/local/bin/`:

    ```bash
    sudo mv vkllama /usr/local/bin/
    ```

    Now you can run `vkllama` from any directory in your terminal.

## Model Setup

`vkllama` uses GGUF-formatted models. You need to download your desired models and configure them for the server.

1.  **Create the models directory**:

    By default, `vkllama` expects your GGUF models and the `models.json` configuration file to be located in `~/.vkllama/models/`. Create this directory if it doesn't exist:

    ```bash
    mkdir -p ~/.vkllama/models/
    ```

2.  **Download GGUF models**:

    You can download GGUF models from various sources, such as [Hugging Face](https://huggingface.co/models?search=gguf). Place your downloaded `.gguf` files into the `~/.vkllama/models/` directory.

    For example, if you download `gemma-3-4b-it-Q4_K_M.gguf`, place it in `~/.vkllama/models/`.

3.  **Create `models.json`**:

    In the same `~/.vkllama/models/` directory, create a file named `models.json`. This file tells `vkllama` about the models it can serve.

    Here's an example `models.json`:

    ```json
    [
        {
            "name": "gemma3",
            "filename": "gemma-3-4b-it-Q4_K_M.gguf",
            "digest": "be49949e48422e4547b00af14179a193d3777eea7fbbd7d6e1b0861304628a01",
            "quantization_level": "Q4_K_M",
            "parameter_size": "4.3B"
        },
        {
            "name": "qwen3",
            "filename": "Qwen3-8B-Q4_K_M.gguf",
            "digest": "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
            "quantization_level": "Q4_K_M",
            "parameter_size": "8.2B"
        },
        {
            "name": "qwen3:4b",
            "filename": "Qwen3-4B-Q4_K_M.gguf",
            "digest": "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5",
            "quantization_level": "Q4_K_M",
            "parameter_size": "4.0B"
        },
        {
            "name": "gemma3n",
            "filename":"gemma-3n-E4B-it-Q4_K_M.gguf",
            "digest": "7fcb647151fa19a0750538672cf824ef6cf18f74bb86ebe5592e1ed59b4070a0",
            "quantization_level": "Q4_K_M",
            "parameter_size": "6.9B"
        }
    ]
    ```

    **Field Descriptions for `models.json`**:
    *   `name`: A unique identifier for your model, used when running or listing models (e.g., `vkllama run -m gemma3`).
    *   `filename`: The exact filename of the GGUF model file within your models directory.
    *   `digest`: (Optional, but recommended) The SHA256 hash of the model file. If provided, `vkllama` will use this value; otherwise, it will calculate it on the fly (which can take time for large files).
    *   `quantization_level`: Describes the quantization level of the model (e.g., `Q4_K_M`).
    *   `parameter_size`: Indicates the parameter count of the model (e.g., `4.3B`).

## Running the Server

You can run the `vkllama` server manually or set it up as a Systemd service for automatic management.

### Manual Server Start

To start the server manually:

```bash
vkllama serve [--host 0.0.0.0] [--port 11434] [--models ~/.vkllama/models]
```

*   `--host`: The IP address the server will bind to. (Default: `0.0.0.0`)
*   `--port`: The port the server will listen on. (Default: `11434`)
*   `--models`: The path to your models directory containing `models.json` and GGUF files. (Default: `~/.vkllama/models`)

Example:
```bash
vkllama serve
```
You will see output indicating the server has started, e.g.: `Starting vkllama server on http://0.0.0.0:11434`

### Systemd Service (Recommended for Production)

Using Systemd ensures `vkllama` starts automatically on boot, restarts if it crashes, and logs its output to the system journal.

1.  **Copy the service file**:

    ```bash
    sudo cp src/vkllama.service /etc/systemd/system/
    ```

2.  **Edit the service file**:

    You **must** edit the `vkllama.service` file to replace the placeholder `User=arch` with your actual username.

    ```bash
    sudo nano /etc/systemd/system/vkllama.service
    # OR (replace 'your_username' with your actual username)
    sudo sed -i "s/User=arch/User=$(whoami)/" /etc/systemd/system/vkllama.service
    ```

3.  **Reload Systemd, enable, and start the service**:

    ```bash
    sudo systemctl daemon-reload       # Reload Systemd to recognize the new service
    sudo systemctl enable --now vkllama.service # Enable service to start on boot and start it now
    ```

4.  **Check service status and logs**:

    ```bash
    systemctl status vkllama.service
    journalctl -u vkllama.service -f # Follow logs in real-time
    ```

## Command-Line Interface (CLI) Usage

`vkllama` provides a simple CLI to interact with the running server.

### List Available Models

To list models configured on the `vkllama` server:

```bash
vkllama list [--address 0.0.0.0:11434]
```

Example output:

```
NAME         ID           SIZE      MODIFIED
gemma3       be49949e4842 4.3 GB    3 days ago
qwen3        d98cdcbd03e1 8.2 GB    2 weeks ago
qwen3:4b     7485fe6f11af 4.0 GB    1 month ago
gemma3n      7fcb647151fa 6.9 GB    5 days ago
```

### Run an LLM Model

To generate a completion from a model:

```bash
vkllama run -m <model_name> [OPTIONS] <prompt>
```

*   `-m` / `--model`: The name of the model to use (as defined in `models.json`).
*   `--seed`: Specify a numerical seed for reproducible text generation.
*   `-s` / `--stream`: Enable streaming output (response appears word by word).
*   `-t` / `--think`: Enable advanced, iterative reasoning. *Note: This flag is currently not implemented in the server logic.*
*   `-a` / `--address`: Server host address (e.g., `localhost:11434`). (Default: `0.0.0.0:11434`)
*   `prompt`: The text prompt for the model. Enclose in quotes if it contains spaces.

Example:

```bash
vkllama run -m gemma3 "What is the capital of France?"
```

Example with streaming:

```bash
vkllama run -m qwen3 -s "Explain quantum computing in simple terms."
```

## Ollama Compatibility

As `vkllama` implements an Ollama-compatible API, you can use any client, library, or application designed to work with Ollama. Simply configure your Ollama client to point to your `vkllama` server's address and port (e.g., `http://localhost:11434`).

This allows you to leverage the `vkllama` backend with its Vulkan performance benefits while using the familiar Ollama ecosystem.
