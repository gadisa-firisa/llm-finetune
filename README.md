# LLM Finetune

This repository contains a small end-to-end chat-style llm fine-tuning, evaluation, and a RAG chatbot.

### Fine-tuning Tasks and Datasets
- **Dialogue generation:** continue a dialogue from the first user turn, with [DailyDialog](https://huggingface.co/datasets/roskoN/dailydialog) dataset.
- **Dialogue summarization:** summarize a full chat transcript, using [SAMSum](https://huggingface.co/datasets/knkarthick/samsum) dataset.
- **Intent classification:**  classify banking text queries into different intents, using [Banking77](https://huggingface.co/datasets/PolyAI/banking77) dataset.

### Model & Stack
- **Base model:** [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
- **Training**: Supervised Fine-Tuning (SFT), with QLoRA [Parameter-Efficient Fine-Tuning](https://huggingface.co/docs/peft/).
- **Libraries:** Transformers, TRL, PEFT, Datasets, Evaluate, scikit-learn
- **RAG:** Sentence-Transformers embeddings with [Chroma](https://github.com/chroma-core/chroma) (default) or [Qdrant](https://qdrant.tech/documentation/).

## Directory Layout

```
llm-finetune/
├── chatbot/
│   ├── app.py
│   ├── data.py
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── entrypoint.sh
│   ├── model.py
│   └── templates/
│       └── index.html
├── data/
├── evaluation/
│   ├── common.py
│   ├── dialogue_eval.py
│   ├── Dockerfile
│   ├── entrypoint.sh
│   ├── intent_eval.py
│   └── summary_eval.py
├── finetune/
│   ├── dataset.py
│   ├── model.py
│   └── pyproject.toml
├── models/
├── README.md
└── requirements.txt
```

## Requirements
- python3 3.12+
- GPU is recommended.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r llm-finetune/requirements.txt
```

## Fine-tuning

Training and LoRA settings are configured in `finetune/pyproject.toml` under `training_arguments`, with per‑task specifics.

**With uv (recommended):**
- Install uv: https://docs.astral.sh/uv/getting-started/installation/

```
cd finetune

uv run model.py --task dialogue         # Dialogue generation 

uv run model.py --task summarization    # Summarization

uv run model.py --task intent            # Intent classification
```

**Without uv (regular Python):**

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
python3 model.py --task dialogue
```

Models are saved under `models/{task}/lora` by default (can be changed in TOML).

## Evaluation

Evaluation uses Hugging Face Evaluate and scikit-learn. Results are saved to `evals/*.jsonl` and `*.metrics.json`.

**Using Docker:**

```
docker build -f evaluation/Dockerfile -t llm-finetune-eval .

docker run --rm \
  --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)/evals:/app/evals" \
  llm-finetune-eval summary_eval --batch_size 16
```
**Without Docker:**
```
# Summarization (ROUGE, BERTScore)
python3 llm-finetune/evaluation/summary_eval.py \
  --base Qwen/Qwen2.5-0.5B-Instruct \
  --adapter models/summarization/lora

# Dialogue (BERTScore)
python3 llm-finetune/evaluation/dialogue_eval.py \
  --base Qwen/Qwen2.5-0.5B-Instruct \
  --adapter models/dialogue/lora

# Intent classification (Accuracy, Macro-F1)
python3 llm-finetune/evaluation/intent_eval.py \
  --base Qwen/Qwen2.5-0.5B-Instruct \
  --adapter models/intent/lora
```

## Chatbot (RAG)

Any plain text/Markdown/HTML documents in `llm-finetune/data/manuals/` can be used as the knowledge base.

Build the index and run the server:

```
# Build local Chroma index
python3 llm-finetune/chatbot/data.py --docs llm-finetune/data/manuals --backend chroma

# Or use Qdrant (requires a running Qdrant instance at QDRANT_URL)
python3 llm-finetune/chatbot/data.py --docs llm-finetune/data/manuals --backend qdrant

# Start the FastAPI app (script mode)
python3 llm-finetune/chatbot/app.py

# Open
http://localhost:8000
```

Environment variables:
- `BASE_MODEL` (default `Qwen/Qwen2.5-0.5B-Instruct`)
- `ADAPTER` (path to LoRA, e.g., `llm-finetune/models/summarization/lora`)
- `VECTOR_BACKEND` (`chroma` or `qdrant`)
- `VECTOR_DIR` (for Chroma, default `.chroma`)
- `VECTOR_COLLECTION` (default `docs`)


## Docker for Chatbot

Build the image and run with Chroma (local vector store):

```
docker build -f llm-finetune/chatbot/Dockerfile -t llm-finetune-chat .

# Ingest docs (mount docs directory and HF cache)
docker run --rm \
  -v "$(pwd)/llm-finetune/data/manuals:/app/llm-finetune/data/manuals" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -e VECTOR_BACKEND=chroma \
  llm-finetune-chat ingest

# Serve
docker run --rm -p 8000:8000 \
  -v "$(pwd)/llm-finetune/data/manuals:/app/llm-finetune/data/manuals" \
  -v "$(pwd)/llm-finetune/models:/app/llm-finetune/models" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -e BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
  -e ADAPTER=llm-finetune/models/summarization/lora \
  -e VECTOR_BACKEND=chroma \
  llm-finetune-chat
```

Or run with Qdrant via docker compose:

```
docker compose -f llm-finetune/chatbot/docker-compose.yml up -d
```
