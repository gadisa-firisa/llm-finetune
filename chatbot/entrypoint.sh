#!/usr/bin/env bash
set -euo pipefail

CMD="${1:-serve}"
shift || true

case "$CMD" in
  ingest)
    DOCS_DIR=${DOCS_DIR:-/app/llm-finetune/data/manuals}
    BACKEND=${VECTOR_BACKEND:-chroma}
    python llm-finetune/chatbot/data.py --docs "$DOCS_DIR" --backend "$BACKEND" "$@"
    ;;
  serve)
    exec python llm-finetune/chatbot/app.py
    ;;
  *)
    exec "$CMD" "$@"
    ;;
esac

