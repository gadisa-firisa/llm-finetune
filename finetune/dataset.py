from dataclasses import dataclass
from typing import List, Dict, Optional
import random
from datasets import load_dataset

def sample_data(dataset, sample: Optional[int], seed: int = 42):
    if sample is None or sample >= len(dataset):
        return dataset
    idx = list(range(len(dataset)))
    random.Random(seed).shuffle(idx)
    return dataset.select(idx[:sample])

def process_turns(turns: List[str]) -> str:
    output = []
    roles = ["User", "Assistant"]
    for i, txt in enumerate(turns[1:], start=1):
        role = roles[i % 2]
        output.append(f"{role}: {txt.strip()}")
    return "\n".join(output).strip()


def load_dailydialog_dialogue(split: str = "train", sample: Optional[int] = None, seed: int = 42):
    dataset = load_dataset("roskoN/dailydialog", split=split)
    dataset = sample_data(dataset, sample, seed)
    system = "You are a helpful assistant continuing everyday conversations naturally and concisely."

    records = []
    for example in dataset:
        conv_turns = example["utterances"]
        if not conv_turns or len(conv_turns) < 2:
            continue
        first_user = conv_turns[0].strip()
        target = process_turns(conv_turns)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Continue the dialogue naturally.\nUser: {first_user}"},
            {"role": "assistant", "content": target},
        ]
        records.append({"messages": messages})
    return records


def normalize_dialogue(text: str) -> str:
    return f"Dialogue:\n{text.strip()}"


def load_samsum_summarization(split: str = "train", sample: Optional[int] = None, seed: int = 42):
    dataset = load_dataset("knkarthick/samsum", split=split)
    dataset = sample_data(dataset, sample, seed)
    system = "You are a helpful assistant that summarizes dialogues concisely."

    records = []
    for example in dataset:
        dialog = normalize_dialogue(example["dialogue"])
        summary = example["summary"].strip()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "Summarize the following dialogue in 2–3 sentences.\n" + dialog},
            {"role": "assistant", "content": summary},
        ]
        records.append({"messages": messages})
    return records


@dataclass
class Banking77Labels:
    id2label: Dict[int, str]
    label2id: Dict[str, int]
    labels: List[str]


def get_banking77_labels() -> Banking77Labels:
    trainset = load_dataset("mteb/banking77", split="train")
    names = trainset.features["label"].names
    id2label = {i: n for i, n in enumerate(names)}
    label2id = {n: i for i, n in enumerate(names)}
    return Banking77Labels(id2label=id2label, label2id=label2id, labels=names)


def load_banking77_intent(split: str = "train", sample: Optional[int] = None, seed: int = 42):
    labels = get_banking77_labels()
    dataset = load_dataset("mteb/banking77", split=split)
    dataset = sample_data(dataset, sample, seed)
    system = "You are a helpful assistant that classifies text into one label from a provided set."

    records = []
    label_list = ", ".join(labels.labels)
    for example in dataset:
        text = example["text"].strip()
        target = labels.id2label[int(example["label"])]
        prompt = (
            "Classify the following text into one of the intents.\n"
            f"Intents: {label_list}\n"
            f"Text: {text}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]
        records.append({"messages": messages})
    return records

