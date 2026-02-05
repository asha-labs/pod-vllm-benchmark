FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

SHELL ["/bin/bash", "-c"]

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN conda update -n base -c defaults conda -y && \
    conda install -y -c conda-forge openmpi mpi4py && \
    pip install --no-cache-dir -U deepspeed

WORKDIR /opt

RUN git clone https://github.com/modelscope/ms-swift.git

WORKDIR /opt/ms-swift

RUN pip install --no-cache-dir -e '.[all]'
RUN pip install --no-cache-dir ninja packaging wheel && \
    pip install --no-cache-dir --no-build-isolation flash-attn && \
    pip install --no-cache-dir msgspec datasets

WORKDIR /app

COPY bench_swift_finetune.py /app/bench_swift_finetune.py
COPY run_bench.sh /app/run_bench.sh
COPY data /app/data

RUN chmod +x /app/run_bench.sh

ENV MODELS_DIR=/models
ENV OUTPUT_DIR=/output
ENV HF_HOME=/models/.hf_home
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

ENTRYPOINT []

CMD ["bash", "/app/run_bench.sh"]
