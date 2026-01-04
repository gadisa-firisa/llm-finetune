import argparse
import json
import os
import sys
import evaluate

sys.path.append(os.path.dirname(__file__))
from common import get_model, generate_from_messages

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "finetune"))
from finetune.dataset import load_samsum_summarization


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--out", default="evals/summary_results.jsonl")
    args = ap.parse_args()

    tok, model = get_model(args.base, args.adapter)
    ds = load_samsum_summarization("test")

    preds, refs = [], []
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for ex in ds:
            pred = generate_from_messages(tok, model, ex["messages"], max_new_tokens=args.max_new_tokens)
            preds.append(pred)
            refs.append(ex["target"].strip())
            f.write(json.dumps({"pred": preds[-1], "ref": refs[-1]}) + "\n")

    rouge = evaluate.load("rouge")
    bert = evaluate.load("bertscore")
    rouge_scores = rouge.compute(predictions=preds, references=refs)
    bert_scores = bert.compute(predictions=preds, references=refs, model_type="roberta-large")
    metrics = {**rouge_scores, "bertscore_f1": sum(bert_scores["f1"]) / len(bert_scores["f1"]) }

    with open(args.out.replace(".jsonl", ".metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
