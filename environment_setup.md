# MERA-JAX Environment Setup on RunPod

## Successfully Tested Environment
- Date: 2025-07-08
- Platform: RunPod A40 GPU (48GB VRAM)
- Python: 3.11.13
- JAX: 0.6.2 (CUDA 12)
- Performance: ~46 TFLOPS

## Quick Start Commands

### 1. Start Jupyter with CORS fix
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.disable_check_xsrf=True --notebook-dir=/app/mera-jax &

### 2. Set PYTHONPATH
export PYTHONPATH=/app/mera-jax:$PYTHONPATH

## Test Results
- Matrix multiplication (3000x3000): 0.0012 seconds
- GFLOPS: 45904.42
- GPU: A40 with 48GB VRAM
