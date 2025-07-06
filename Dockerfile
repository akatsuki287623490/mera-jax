DIA official CUDA base image (Ubuntu 22.04 + CUDA 12.4)
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

# Install basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    vim \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install Poetry
ENV POETRY_VERSION=2.1.3
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /workspace

# Install JAX GPU version (minimal dependencies only)
RUN pip install numpy==1.26.4 && \
    pip install --upgrade "jax[cuda12]"

# Verific
