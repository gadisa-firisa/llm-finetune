# Evaluation image for llm-finetune
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Core deps for evaluation
RUN pip install --no-cache-dir \
    transformers>=4.44.0 \
    datasets>=2.18.0 \
    accelerate>=0.33.0 \
    peft>=0.12.0 \
    evaluate>=0.4.2 \
    scikit-learn>=1.5.0 \
    sentence-transformers>=3.0.1 \
    bert-score>=0.3.13 \
    rouge-score>=0.1.2 \
    qdrant-client>=1.9.0 \
    chromadb>=0.5.3

# Copy project files
COPY llm-finetune /app/llm-finetune
RUN mkdir -p /app/evals && chmod +x /app/llm-finetune/evaluation/entrypoint.sh || true

ENTRYPOINT ["/app/llm-finetune/evaluation/entrypoint.sh"]

