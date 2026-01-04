import argparse
import json
import os
import sys
import re
from sklearn.metrics import accuracy_score, f1_score
sys.path.append(os.path.dirname(__file__))
from common import get_model, generate_from_messages
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "finetune"))
from finetune.dataset import load_banking77_intent, get_banking77_labels


def normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_ ]+", "", s)
    s = s.replace(" ", "_")
    return s


def best_label(text: str, labels: list[str]) -> str:
    t = normalize(text)
    for lab in labels:
        if normalize(lab) == t:
            return lab
    for lab in labels:
        if normalize(lab) in t or t in normalize(lab):
            return lab
    return labels[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", default="evals/intent_results.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=16)
    args = ap.parse_args()

    labels = get_banking77_labels().labels
    tok, model = get_model(args.base, args.adapter)
    ds = load_banking77_intent("test")

    y_true, y_pred = [], []
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for ex in ds:
            pred = generate_from_messages(tok, model, ex["messages"], max_new_tokens=args.max_new_tokens)
            pred_label = best_label(pred, labels)
            y_pred.append(pred_label)
            y_true.append(ex["target"])
            f.write(json.dumps({"pred": pred_label, "ref": ex["target"]}) + "\n")

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    with open(args.out.replace(".jsonl", ".metrics.json"), "w") as f:
        json.dump({"accuracy": acc, "macro_f1": f1m}, f, indent=2)


if __name__ == "__main__":
    main()
