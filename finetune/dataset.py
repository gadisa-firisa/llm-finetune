from dataclasses import dataclass
from typing import List, Dict, Optional
import random
from datasets import load_dataset

def sample_data(ds, sample: Optional[int], seed: int = 42):
    if sample is None or sample >= len(ds):
        return ds
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return ds.select(idx[:sample])

def process_turns(turns: List[str]) -> str:
    out = []
    roles = ["User", "Assistant"]
    for i, txt in enumerate(turns[1:], start=1):
        role = roles[i % 2]
        out.append(f"{role}: {txt.strip()}")
    return "\n".join(out).strip()


def load_dailydialog_dialogue(split: str = "train", sample: Optional[int] = None, seed: int = 42):
    ds = load_dataset("daily_dialog", split=split)
    ds = sample_data(ds, sample, seed)
    system = "You are a helpful assistant continuing everyday dialogues naturally and concisely."

    records = []
    for ex in ds:
        dialog = ex["dialog"]
        if not dialog or len(dialog) < 2:
            continue
        first_user = dialog[0].strip()
        target = process_turns(dialog)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Continue the dialogue naturally.\nUser: {first_user}"},
        ]
        records.append({"messages": messages, "target": target})
    return records


def normalize_dialogue(text: str) -> str:
    return f"Dialogue:\n{text.strip()}"


def load_samsum_summarization(split: str = "train", sample: Optional[int] = None, seed: int = 42):
    ds = load_dataset("samsum", split=split)
    ds = sample_data(ds, sample, seed)
    system = "You are a helpful assistant that writes faithful, concise summaries of dialogues."

    records = []
    for ex in ds:
        dialog = normalize_dialogue(ex["dialogue"])
        summary = ex["summary"].strip()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "Summarize the following dialogue in 2–3 sentences.\n" + dialog},
        ]
        records.append({"messages": messages, "target": summary})
    return records


@dataclass
class Banking77Labels:
    id2label: Dict[int, str]
    label2id: Dict[str, int]
    labels: List[str]


def get_banking77_labels() -> Banking77Labels:
    dstrain = load_dataset("banking77", split="train")
    names = dstrain.features["label"].names
    id2label = {i: n for i, n in enumerate(names)}
    label2id = {n: i for i, n in enumerate(names)}
    return Banking77Labels(id2label=id2label, label2id=label2id, labels=names)


def load_banking77_intent(split: str = "train", sample: Optional[int] = None, seed: int = 42):
    labels = get_banking77_labels()
    ds = load_dataset("banking77", split=split)
    ds = sample_data(ds, sample, seed)
    system = "You are a helpful assistant that classifies text into one label from a provided set."

    records = []
    label_list = ", ".join(labels.labels)
    for ex in ds:
        text = ex["text"].strip()
        tgt = labels.id2label[int(ex["label"])]
        prompt = (
            "Classify the following text into one of the intents.\n"
            f"Intents: {label_list}\n"
            f"Text: {text}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        records.append({"messages": messages, "target": tgt})
    return records

