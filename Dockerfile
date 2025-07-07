FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.11
ENV POETRY_VERSION=2.1.3
ENV POETRY_HOME=/root/.local/share/pypoetry
ENV PATH="$POETRY_HOME/venv/bin:$PATH"
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

# 基本パッケージのインストール（SSHサーバーを追加）
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    curl \
    git \
    vim \
    openssh-server \
    net-tools \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# SSHの設定
RUN mkdir /var/run/sshd \
    && echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config \
    && echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config

# Pythonのデフォルト設定
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# pipのインストール
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

# Poetryのインストール
RUN curl -sSL https://install.python-poetry.org | python - --version ${POETRY_VERSION}

# 作業ディレクトリの設定
WORKDIR /workspace/mera-jax

# プロジェクトファイルのコピー
COPY pyproject.toml poetry.lock ./

# 依存関係のインストール（.venvディレクトリに作成）
RUN poetry install --no-root --no-interaction --no-ansi

# プロジェクトファイルのコピー
COPY . .

# 環境確認スクリプトの作成
RUN echo '#!/bin/bash\n\
echo "=== MERA-JAX Docker Environment v4.0 ==="\n\
echo "Python: $(python --version)"\n\
echo "Poetry: $(poetry --version)"\n\
echo "CUDA: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Not available')"\n\
source /workspace/mera-jax/.venv/bin/activate\n\
echo "JAX: $(python -c "import jax; print(jax.__version__)" 2>/dev/null || echo 'Not installed')"\n\
echo "SSH: $(service ssh status | grep -o "is running" || echo "Not running - run: service ssh start")"\n\
echo ""\n\
echo "To activate environment: source /workspace/mera-jax/.venv/bin/activate"' > /check_env.sh \
    && chmod +x /check_env.sh

# bashrcにPoetryパスと仮想環境の自動有効化を追加
RUN echo 'export PATH="$POETRY_HOME/venv/bin:$PATH"' >> ~/.bashrc \
    && echo 'cd /workspace/mera-jax && source .venv/bin/activate 2>/dev/null' >> ~/.bashrc

# デフォルトコマンド
CMD ["/bin/bash"]