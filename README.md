# AutoAWQ Kernels

AutoAWQ Kernels is a new package that is split up from the [main repository](https://github.com/casper-hansen/AutoAWQ) in order to avoid compilation times.

## Requirements

- Windows: Must use WSL2.

- NVIDIA:
  - GPU: Must be compute capability 7.5 or higher.
  - CUDA Toolkit: Must be 11.8 or higher.
- AMD:
  - ROCm: Must be 5.6 or higher. [Build from source](#build-from-source)

## Install

### Install from PyPi

The package is available on PyPi with CUDA 12.4.1 wheels:

```
pip install autoawq-kernels
```

### Build from source

To build the kernels from source, you first need to setup an environment containing the necessary dependencies.

#### Build Requirements

- Python>=3.8.0
- Numpy
- Wheel
- PyTorch
- ROCm: You need to install the following packages `rocsparse-dev hipsparse-dev rocthrust-dev rocblas-dev hipblas-dev`.

#### Building process

```
pip install git+https://github.com/casper-hansen/AutoAWQ_kernels.git
```

Notes on environment variables:
- `TORCH_VERSION`: By default, we build using the current version of torch by `torch.__version__`. You can override it with `TORCH_VERSION`.
    - `CUDA_VERSION` or `ROCM_VERSION` can also be used to build for a specific version of CUDA or ROCm.
- `CC` and `CXX`: You can specify which build system to use for the C code, e.g. `CC=g++-13 CXX=g++-13 pip install -e .`
- `COMPUTE_CAPABILITIES`: You can specify specific compute capabilities to compile for: `COMPUTE_CAPABILITIES="75,80,86,87,89,90" pip install -e .`
