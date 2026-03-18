# OpenVINO Experiments

A workspace for OpenVINO tooling experiments and utilities.

## Requirements

- Intel Core U-series or higher (tested on Lunar Lake 258V 32GB RAM)
- Ubuntu 24.04 or higher
- Python 3.12 or higher
- [Intel GPU Drivers](https://dgpu-docs.intel.com/driver/client/overview.html)
- [Intel NPU Drivers](https://github.com/intel/linux-npu-driver)
- [OpenVINO Runtime](https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-linux.html)
- [OpenVINO Model Server](https://docs.openvino.ai/2026/model-server/ovms_what_is_openvino_model_server.html)
- Docker (for running OVMS in a containerized environment)


## Projects

### `ov-testdata-scraper`

A CLI tool that mirrors the [OpenVINO test data bucket](https://storage.openvinotoolkit.org/data/test_data/)
to a local directory. Supports parallel downloads, random request delays, and
resume-on-interrupt. See [`ov-testdata-scraper/README.md`](ov-testdata-scraper/README.md) for full
documentation.

## Setup

Requires Python 3.12+ and [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync
```

This installs all dependencies and sub-projects in editable mode, making their CLI entry points
available in the active virtual environment. The project is configured to use CPU-only PyTorch,
avoiding unnecessary NVIDIA CUDA dependencies.

## Samples

### Text Generation

Install a OpenVINO model from Hugging Face:

```bash
hf download OpenVino/Phi-3.5-mini-instruct-int4-cw-ov
```

Run the text generation sample in `samples/text-generation`.

```bash
uv run samples/text_generation.py -d NPU -m $HF_HOME/hub/<path-to-downloaded-snapshot> "The sun is yellow because"
```

**Results:**

| Device | Model | tok/s | TTFT | Total |
|--------|-------|-------|------|-------|
| GPU | Phi-3.5 (3.8B) | 32.6 | 56ms | 3.1s |
| NPU | Phi-3.5 (3.8B) | 23.2 | 1149ms | 5.4s |
| GPU | Mistral-7B | 18.8 | 91ms | 5.4s |
| NPU | Mistral-7B | 19.3 | 2068ms | 7.2s |
| GPU | Qwen3-8B | 16.9 | 121ms | 6.0s |
| NPU | Qwen3-8B | 16.0 | 1910ms | 8.1s |


## References:

- https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes.html
