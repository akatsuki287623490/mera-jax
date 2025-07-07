#!/bin/bash
echo "=== MERA-JAX 環境セットアップ開始 ==="

# 1. Poetryパスを設定
export PATH="/root/.local/share/pypoetry/venv/bin:$PATH"

# 2. プロジェクトディレクトリへ移動
cd /workspace/mera-jax

# 3. 最新コードを取得
git pull origin main

# 4. Poetry依存関係をインストール
poetry install --no-root

# 5. CUDA版JAXをインストール
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 6. 確認
echo "=== セットアップ完了 ==="
python -c "import jax; print('JAX version:', jax.__version__, 'Devices:', jax.devices())"
