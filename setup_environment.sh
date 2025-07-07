#!/bin/bash
echo "=== MERA-JAX 環境セットアップ開始 ==="

# 1. Poetryパスを設定
export PATH="/root/.local/share/pypoetry/venv/bin:$PATH"
echo 'export PATH="/root/.local/share/pypoetry/venv/bin:$PATH"' >> ~/.bashrc

# 2. プロジェクトディレクトリへ移動
cd /workspace/mera-jax

# 3. 最新コードを取得
if [ -d ".git" ]; then
    echo "既存のリポジトリを更新..."
    git pull origin main
else
    echo "リポジトリをクローン..."
    git clone https://github.com/akatsuki287623490/mera-jax.git .
fi

# 4. Poetry設定
poetry config virtualenvs.in-project true

# 5. 依存関係インストール
echo "依存関係をインストール中..."
poetry install --no-root --sync

# 6. 環境情報を表示
echo ""
echo "=== セットアップ完了 ==="
poetry env info
echo ""
echo "仮想環境を有効化するには："
echo "source /workspace/mera-jax/.venv/bin/activate"
