# MERA-JAX Complete Docker Image

## Quick Start

    docker run -p 8888:8888 --gpus all akatsukiwork/mera-jax:complete

## Included

- Python 3.11.13
- JAX 0.6.2 (CUDA 12)
- Flax, Optax, Aim
- Jupyter Notebook (auto-start)
- MERA-JAX project

## Usage on RunPod

1. Create new Pod
2. Container Image: akatsukiwork/mera-jax:complete
3. Access HTTP Service :8888

## Features

- Auto-starts Jupyter on port 8888
- CORS enabled for RunPod proxy
- All dependencies pre-installed
- GPU-ready (CUDA 12)

## Environment Details

The image includes:
- Base: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
- Python upgraded to 3.11.13
- JAX with CUDA 12 support
- Complete MERA-JAX project from GitHub
- Jupyter Notebook with CORS fixes for RunPod
