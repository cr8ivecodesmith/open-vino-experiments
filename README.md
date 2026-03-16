# OpenVINO Experiments

A workspace for OpenVINO tooling experiments and utilities.

## Requirements

- Intel Core U-series or higher (tested on Lunar Lake)
- Ubuntu 24.04 or higher
- Python 3.12 or higher
- [Intel GPU Drivers](https://dgpu-docs.intel.com/driver/client/overview.html)
- [Intel NPU Drivers](https://github.com/intel/linux-npu-driver)
- [OpenVINO Runtime](https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-linux.html)


## Projects

### `ov-testdata-scraper`

A CLI tool that mirrors the [OpenVINO test data bucket](https://storage.openvinotoolkit.org/data/test_data/)
to a local directory. Supports parallel downloads, random request delays, and
resume-on-interrupt. See [`ov-testdata-scraper/README.md`](ov-testdata-scraper/README.md) for full
documentation.

## Setup

Requires Python 3.12+ and [`uv`](https://github.com/astral-sh/uv).

> [!WARNING]
> This is a very large venv (~7GB) mainly due to nvidia and torch dependencies. Make sure you have enough
> disk space before proceeding.

```bash
uv pip install -r requirements.txt
```

This installs all sub-projects in editable mode, making their CLI entry points
available in the active virtual environment.

## Samples

### Text Generation

Install a OpenVINO model from Hugging Face:

```bash
hf download OpenVino/Phi-3.5-mini-instruct-int4-cw-ov
```

Run the text generation sample in `samples/text-generation`.

```bash
uv run samples/text-generation.py -d NPU -m $HOME_HF/hub/<path-to-downloaded-snapshot> "The sun is yellow because"
```


## References:

- https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes.html
