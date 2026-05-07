import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Simple defaults: edit here if needed.
DEFAULT_INPUT = Path("Data/Preprocessed Data/narrative_data_food.csv")
DEFAULT_OUTPUT = Path("Data/Model Outputs/centaur_step1")
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_MAX_PARTICIPANTS = 3
DEFAULT_MAX_SEQ_LEN = 4096
ESTIMATE_PATTERN = re.compile(r"<<(.*?)>>", re.DOTALL)


def parse_args():
    parser = argparse.ArgumentParser(description="Step 1: Compute trial-level NLL from narrative_data_food.csv")
    parser.add_argument("--full-run", action="store_true", help="Use all participants.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id.")
    parser.add_argument("--max-participants", type=int, default=DEFAULT_MAX_PARTICIPANTS, help="Quick-test size.")
    return parser.parse_args()


def load_data(full_run: bool, max_participants: int) -> pd.DataFrame:
    df = pd.read_csv(DEFAULT_INPUT)[["ID", "narrative"]].copy()
    df["ID"] = df["ID"].astype(str)
    df = df.sort_values("ID").reset_index(drop=True)
    return df if full_run else df.head(max_participants)


def load_model(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"device_map": "auto"} if device == "cuda" else {}
    if device == "cuda":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if device == "cpu":
        model.to("cpu")
    model.eval()
    return model, tokenizer, device


def estimate_spans(text: str):
    spans = []
    cursor = 0
    trial_idx = 1
    for line in text.splitlines(keepends=True):
        phase = "unknown"
        if "[TRAIN]" in line:
            phase = "training"
        elif "[TEST]" in line:
            phase = "testing"

        for m in ESTIMATE_PATTERN.finditer(line):
            spans.append(
                {
                    "trial_index": trial_idx,
                    "phase": phase,
                    "estimate_text": m.group(1).strip(),
                    "start": cursor + m.start(1),
                    "end": cursor + m.end(1),
                }
            )
            trial_idx += 1
        cursor += len(line)
    return spans


def token_positions_for_span(offsets, start_char, end_char):
    keep = []
    for i, (a, b) in enumerate(offsets):
        if b <= a:
            continue
        if b <= start_char:
            continue
        if a >= end_char:
            break
        keep.append(i)
    return keep


def participant_trial_nlls(participant_id: str, text: str, model, tokenizer, device: str):
    spans = estimate_spans(text)
    if not spans:
        raise ValueError(f"No <<...>> found for participant {participant_id}")

    enc = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=DEFAULT_MAX_SEQ_LEN,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    offsets = enc["offset_mapping"][0].tolist()

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = out.logits[0, :-1, :].cpu()
    targets = input_ids[0, 1:].cpu()
    token_nll = F.cross_entropy(logits, targets, reduction="none").numpy()

    rows = []
    for s in spans:
        token_ids = token_positions_for_span(offsets, s["start"], s["end"])
        loss_ids = [i - 1 for i in token_ids if i > 0 and (i - 1) < len(token_nll)]
        nll = float(np.sum(token_nll[loss_ids])) if loss_ids else np.nan
        rows.append(
            {
                "participant_id": participant_id,
                "trial_index": s["trial_index"],
                "phase": s["phase"],
                "estimate_text": s["estimate_text"],
                "trial_nll": nll,
                "num_tokens": len(loss_ids),
            }
        )
    return rows


def main():
    args = parse_args()
    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)

    data = load_data(args.full_run, args.max_participants)
    model, tokenizer, device = load_model(args.model)
    print(f"Device: {device} | Participants: {len(data)} | Model: {args.model}")

    rows = []
    for i, row in data.iterrows():
        print(f"[{i + 1}/{len(data)}] {row['ID']}")
        rows.extend(participant_trial_nlls(row["ID"], row["narrative"], model, tokenizer, device))
        if device == "cuda":
            torch.cuda.empty_cache()

    trial_df = pd.DataFrame(rows)
    trial_df = trial_df[trial_df["phase"] == "testing"].copy()
    summary_df = (
        trial_df.groupby("participant_id", as_index=False)
        .agg(total_trial_nll=("trial_nll", "sum"), mean_trial_nll=("trial_nll", "mean"), n_trials=("trial_index", "count"))
    )

    trial_df["model_name"] = args.model
    trial_df.to_csv(DEFAULT_OUTPUT / "trial_nlls.csv", index=False)
    summary_df.to_csv(DEFAULT_OUTPUT / "participant_summary.csv", index=False)
    print(f"Saved outputs to: {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
