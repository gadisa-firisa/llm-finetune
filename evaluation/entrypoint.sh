#!/usr/bin/env bash
set -euo pipefail

cmd="${1:-}"; shift || true

case "$cmd" in
  summary_eval)
    python llm-finetune/evaluation/summary_eval.py "$@"
    ;;
  dialogue_eval)
    python llm-finetune/evaluation/dialogue_eval.py "$@"
    ;;
  intent_eval)
    python llm-finetune/evaluation/intent_eval.py "$@"
    ;;
  *)
    echo "Usage: <summary_eval|dialogue_eval|intent_eval> [args...]" 1>&2
    exit 1
    ;;
esac

