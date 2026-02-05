FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

SHELL ["/bin/bash", "-c"]

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U sentence-transformers datasets accelerate hf_transfer

COPY bench_hf_reranker_finetune.py /app/bench_hf_reranker_finetune.py
COPY generate_hf_reranker_dataset.py /app/generate_hf_reranker_dataset.py
COPY run_bench.sh /app/run_bench.sh

RUN chmod +x /app/run_bench.sh

ENV MODELS_DIR=/models
ENV OUTPUT_DIR=/output
ENV HF_HOME=/models/.hf_home
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

ENTRYPOINT []

CMD ["bash", "/app/run_bench.sh"]
