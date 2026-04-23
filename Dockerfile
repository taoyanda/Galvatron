FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/conda/bin:/usr/local/cuda-12.1/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH \
    CUDA_HOME=/usr/local/cuda-12.1 \
    TORCH_CUDA_ARCH_LIST="8.0;9.0" \
    MAX_JOBS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build wget ca-certificates \
        libibverbs-dev librdmacm-dev libnuma-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO /tmp/mc.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.2-0-Linux-x86_64.sh \
    && bash /tmp/mc.sh -b -p /opt/conda && rm /tmp/mc.sh \
    && conda install -y python=3.9.2 && conda clean -afy

RUN pip install --no-cache-dir \
        torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
        --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir packaging ninja psutil pybind11 einops

RUN pip install numpy==1.26.4

RUN git clone https://github.com/Dao-AILab/flash-attention.git /opt/flash-attention \
    && cd /opt/flash-attention && git checkout v2.5.8 \
    && MAX_JOBS=16 python setup.py install \
    && cd csrc/layer_norm && MAX_JOBS=16 pip install --no-cache-dir . --no-build-isolation \
    && rm -rf /opt/flash-attention

RUN git clone https://github.com/NVIDIA/apex /opt/apex \
    && cd /opt/apex && git checkout 312acb4 \
    && MAX_JOBS=16 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
        --config-settings "--build-option=--cpp_ext" \
        --config-settings "--build-option=--cuda_ext" ./ \
    && rm -rf /opt/apex

RUN git clone https://github.com/NVIDIA/TransformerEngine.git /opt/TransformerEngine \
    && cd /opt/TransformerEngine && git checkout 7f2afaa \
    && git submodule update --init --recursive \
    && MAX_JOBS=16 NVTE_FRAMEWORK=pytorch pip install --no-cache-dir --no-build-isolation . \
    && rm -rf /opt/TransformerEngine

WORKDIR /workspace
CMD ["/bin/bash"]
