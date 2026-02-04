FROM vllm/vllm-openai:latest

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir huggingface_hub transformers

COPY bench_rerankers.py /app/bench_rerankers.py

ENV MODELS_DIR=/models
ENV HF_HOME=/models/.hf_home
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

CMD ["python", "/app/bench_rerankers.py"]
