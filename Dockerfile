# Stage 1: Base Image
FROM nvcr.io/nvidia/jax:23.10-py3 AS base

# Stage 2: Dependency Builder
FROM base AS builder
WORKDIR /app

# Poetryをインストール
ENV POETRY_VERSION=1.8.2
RUN pip install "poetry==${POETRY_VERSION}"

# 依存関係定義ファイルのみをコピー
COPY poetry.lock pyproject.toml ./

# requirements.txtにエクスポート
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# Stage 3: Final Production Image
FROM base AS final
WORKDIR /app

# builderステージからrequirements.txtをコピー
COPY --from=builder /app/requirements.txt .

# pipで依存関係をインストール
RUN pip install --no-cache-dir -r requirements.txt

# CUDA版JAXを別途インストール
RUN pip install --no-cache-dir "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# アプリケーションのソースコードをコピー
COPY . .

# エントリーポイント
ENTRYPOINT ["python"]
