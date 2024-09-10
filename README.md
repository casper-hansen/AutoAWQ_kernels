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

#### GPU support

If you want to build with GPU support, you need to make sure that PyTorch is built with the same(ish) CUDA (for Nvidia's GPUs) or ROCm (for AMD's GPUs) version that is currently installed in the environment.

Additionally, to build with ROCm, you'll need to install the following packages `rocsparse-dev hipsparse-dev rocthrust-dev rocblas-dev hipblas-dev` in the environment. Preferable using your package manager.

### Building environment

The process of creating such environment is different on every machine, depending on the hardware and software.

#### Docker Build Environment

A quick and easy way to create a building environment is to use a container with CUDA 12.6 and Ubuntu 24.04 and build the kernels inside a virtual environment.

You'll need to have [Docker installed with GPU support](https://docs.docker.com/config/containers/resource_constraints/#gpu) on your machine to use this method.

```shell
# Using docker to spawn a container
docker run -it --rm nvidia/cuda:12.6.1-cudnn-devel-ubuntu24.04

# Update & Upgrade the system
apt update && apt upgrade -y

# Make sure you have nvcc installed
nvcc --version

# Make sure you have a compatible version of g++
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#host-compiler-support-policy
g++ --version

# Install git
apt install -y git

# Install python and dependencies
apt install -y python3 python3-pip python3-venv

# Create an user
useradd -m -s /bin/bash user
su - user

# Setup a virtual environment
python3 -m venv kernel-build-env
source kernel-build-env/bin/activate

# Install dependencies
pip install numpy wheel

# Torch with CUDA 12.1.1
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Idem for ROCm.

```shell
# Using docker to spawn a container
docker run -it rocm/rocm-build-ubuntu-24.04

# Update & Upgrade the system
apt update && apt upgrade -y

# Make sure you have a compatible version of g++
g++ --version

# Install git
apt install -y git

# Install rocm dependencies
apt install -y rocsparse-dev hipsparse-dev rocthrust-dev rocblas-dev hipblas-dev

# Install python and dependencies
apt install -y python3 python3-pip python3-venv

# Create an user
useradd -m -s /bin/bash user
su - user
cd

# Setup a virtual environment
python3 -m venv kernel-build-env
source kernel-build-env/bin/activate

# Install dependencies
pip install numpy wheel

# Torch with ROCm 6.1
pip install torch --index-url https://download.pytorch.org/whl/rocm6.1
```

This gives you a good idea of what it takes to setup an environment to build the kernels.

#### Manual Build Environment

You can also setup the environment manually by installing the necessary dependencies on your machine.
This is a more complex process and requires more knowledge of the system you are using.
Checkout the package manager of your distribution of choice and the instructions of the respective software for more information.

### Building process

After setting up the environment, you can clone the source code into the `AutoAWQ_kernels` directory by running the following commands.

```shell
git clone https://github.com/casper-hansen/AutoAWQ_kernels
cd AutoAWQ_kernels
```

This will clone the repository into the `AutoAWQ_kernels` directory and move into it.

Once inside the repository, you can build the package by running the following command.

```shell
pip install .
```

#### Advanced building options

You might need to add the `--no-build-isolation` flag if you get an error complaining about the missing packages you have installed.

```shell
pip install --no-build-isolation .
```

##### Environment variables

If the setup module can successfully import the `torch` package, it will automatically use it as requirement to build the kernels, unless specified otherwise by setting the `TORCH_VERSION` environment variable.

> [!CAUTION]
> This will only work if the correct `torch` package version is present in the environment.

```shell
TORCH_VERSION=2.0.0 pip install .
```

Same applies the `CUDA_VERSION` or `ROCM_VERSION` environment variables to build the kernels for a specific (installed) version of CUDA or ROCm.

There is additionally the `CC` and `CXX` environment variables to specify the name of the `g++` compiler (as found in `$PATH`) to use when building the kernels.

> [!CAUTION]
> You need to make sure that the compiler you specify is compatible with the version of CUDA you are building for.
>
> See the [Host Compiler Support Policy](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#host-compiler-support-policy) for your installed version of CUDA.
>
> It's likely the same applies for ROCm.

```shell
CC=g++-13 CXX=g++-13 pip install .
```

To add unsupported GPU computing platforms, you can set `COMPUTE_CAPABILITIES` to include your platform to the list of supported architectures.

> [!NOTE]
> The community would appreciate if you could make a pull request to add support for your platform by editing [the appropriate line of the `setup.py` file](setup.py#L14) but feel free to open an issue instead if you need help with it.

> [!WARNING]
> You cannot add an architecture that has a lower compute capability than 75, as this is the minimum supported by the kernels.

```shell
COMPUTE_CAPABILITIES="75,80,86,87,89,90" pip install .
```

Additionally, you can set `AUTOAWQ_KERNELS_VERSION` to change the version of the package you are building and `PYPI_BUILD` to disable adding the `+{gpulib-version}` suffix to the package version when building to publish on PyPi.

#### Editable mode

For developers, you can install the package in editable mode by running the command with the `-e` flag.

```shell
pip install -e .
```

This will allow you to make changes to the source code and see the changes reflected in the installed package without having to reinstall it.

#### Troubleshooting

If you are having trouble building the kernels, you can also run the `setup.py` file directly.

```shell
python setup.py build
python setup.py install
```

This will show you the output of the build process, which can help you identify issues.

Feel free to use [gist.github.com](https://gist.github.com) or [pastebin.com](https://pastebin.com) to share logs if you need help.
