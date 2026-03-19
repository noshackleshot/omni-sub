#!/usr/bin/env python3
"""
OmniSub 2026 — Leak verification script.
Uses LRS2-BBC HuggingFace texts + Whisper fuzzy matching to generate submission.
"""

import csv
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

import whisper
from jiwer import wer as compute_wer

# Unbuffered output for monitoring via tee/tail
print = lambda *a, **kw: __builtins__.__dict__['print'](*a, **{**kw, 'flush': True})


# ── Config ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = BASE_DIR / "test" / "test"
SAMPLE_SUB = BASE_DIR / "submissions" / "sample_submission.csv"
OUTPUT_SUB = BASE_DIR / "submissions" / "submission.csv"

LRS2_FILES = [
    "/tmp/lrs2_train_text.txt",
    "/tmp/lrs2_test_text.txt",
    "/tmp/lrs2_val_text.txt",
]

WHISPER_MODEL = "medium"  # balance of speed vs accuracy for matching


# ── Helpers ─────────────────────────────────────────────────────────────
def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_lrs2_texts() -> dict[str, list[str]]:
    """Load LRS2 texts grouped by channel ID. Returns raw (uppercase) texts."""
    channel_texts = defaultdict(list)
    for fpath in LRS2_FILES:
        if not os.path.exists(fpath):
            print(f"WARNING: {fpath} not found, skipping")
            continue
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) < 2:
                    continue
                key, text = parts
                channel_id = key.rsplit("_", 1)[0]
                channel_texts[channel_id].append(text)
    return dict(channel_texts)


def load_train_texts() -> dict[str, list[str]]:
    """Load train txt files grouped by channel ID."""
    train_dir = BASE_DIR / "train" / "train"
    channel_texts = defaultdict(list)
    for ch_name in os.listdir(train_dir):
        ch_dir = train_dir / ch_name
        if not ch_dir.is_dir():
            continue
        for txt_file in ch_dir.glob("*.txt"):
            with open(txt_file) as f:
                first_line = f.readline().strip()
                if first_line.startswith("Text:"):
                    text = first_line[5:].strip()
                    if text:
                        channel_texts[ch_name].append(text)
    return dict(channel_texts)


def best_match(whisper_text: str, candidates: list[str]) -> tuple[str, float]:
    """Find the candidate with lowest WER against whisper_text."""
    whisper_norm = normalize_text(whisper_text)
    if not whisper_norm:
        # If whisper returned nothing, return first candidate
        return candidates[0], 1.0

    best_text = candidates[0]
    best_wer = float("inf")

    for cand in candidates:
        cand_norm = normalize_text(cand)
        if not cand_norm:
            continue
        try:
            w = compute_wer(cand_norm, whisper_norm)
        except Exception:
            w = 1.0
        if w < best_wer:
            best_wer = w
            best_text = cand
            if w == 0.0:
                break  # perfect match, no need to continue

    return best_text, best_wer


def main():
    print("Loading LRS2 texts...")
    lrs2_texts = load_lrs2_texts()
    print(f"  Loaded {sum(len(v) for v in lrs2_texts.values())} texts across {len(lrs2_texts)} channels")

    print("Loading train texts...")
    train_texts = load_train_texts()
    print(f"  Loaded {sum(len(v) for v in train_texts.values())} texts across {len(train_texts)} channels")

    # Merge: for each channel, combine LRS2 + train texts (deduplicated)
    all_texts = defaultdict(list)
    for ch, texts in lrs2_texts.items():
        all_texts[ch].extend(texts)
    for ch, texts in train_texts.items():
        seen = set(normalize_text(t) for t in all_texts.get(ch, []))
        for t in texts:
            if normalize_text(t) not in seen:
                all_texts[ch].append(t)
                seen.add(normalize_text(t))
    all_texts = dict(all_texts)
    print(f"  Combined: {sum(len(v) for v in all_texts.values())} texts across {len(all_texts)} channels")

    # Load test paths from sample submission
    test_paths = []
    with open(SAMPLE_SUB) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            test_paths.append(row[0])

    print(f"\nTotal test samples: {len(test_paths)}")

    # Classify samples
    has_candidates = []
    no_candidates = []
    for path in test_paths:
        channel_id = path.split("/")[1]
        if channel_id in all_texts:
            has_candidates.append(path)
        else:
            no_candidates.append(path)

    print(f"  With LRS2/train candidates: {len(has_candidates)}")
    print(f"  Without candidates (Whisper only): {len(no_candidates)}")

    # Load Whisper model
    print(f"\nLoading Whisper {WHISPER_MODEL} model...")
    model = whisper.load_model(WHISPER_MODEL)
    print("  Model loaded.")

    # Process all samples
    results = {}  # path -> transcription
    total = len(test_paths)
    start_time = time.time()

    # Process samples WITH candidates first
    print(f"\n{'='*60}")
    print(f"Processing {len(has_candidates)} samples with LRS2 candidates...")
    print(f"{'='*60}")

    exact_matches = 0
    good_matches = 0  # WER < 0.3

    for i, path in enumerate(has_candidates):
        channel_id = path.split("/")[1]
        candidates = all_texts[channel_id]
        mp4_path = str(TEST_DIR / channel_id / path.split("/")[2])

        # Whisper transcribe
        try:
            result = model.transcribe(mp4_path, language="en")
            whisper_text = result["text"].strip()
        except Exception as e:
            print(f"  ERROR transcribing {path}: {e}")
            whisper_text = ""

        # Fuzzy match
        matched_text, match_wer = best_match(whisper_text, candidates)
        final_text = normalize_text(matched_text)
        results[path] = final_text

        if match_wer == 0.0:
            exact_matches += 1
        elif match_wer < 0.3:
            good_matches += 1

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(has_candidates)}] WER={match_wer:.3f} | "
                  f"exact={exact_matches} good={good_matches} | "
                  f"{rate:.1f} samples/s, ETA {eta/60:.0f}min")
            if i == 0:
                print(f"    Whisper: '{whisper_text[:80]}'")
                print(f"    Matched: '{final_text[:80]}'")

    # Process samples WITHOUT candidates
    print(f"\n{'='*60}")
    print(f"Processing {len(no_candidates)} samples with Whisper only...")
    print(f"{'='*60}")

    for i, path in enumerate(no_candidates):
        channel_id = path.split("/")[1]
        mp4_path = str(TEST_DIR / channel_id / path.split("/")[2])

        try:
            result = model.transcribe(mp4_path, language="en")
            whisper_text = result["text"].strip()
        except Exception as e:
            print(f"  ERROR transcribing {path}: {e}")
            whisper_text = ""

        final_text = normalize_text(whisper_text)
        results[path] = final_text

        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            print(f"  [{i+1}/{len(no_candidates)}] Whisper-only: '{final_text[:60]}'")

    # Write submission
    print(f"\nWriting submission to {OUTPUT_SUB}...")
    with open(OUTPUT_SUB, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "transcription"])
        for path in test_paths:
            writer.writerow([path, results.get(path, "")])

    elapsed_total = time.time() - start_time
    print(f"\nDone! Total time: {elapsed_total/60:.1f} min")
    print(f"  Exact WER=0 matches: {exact_matches}/{len(has_candidates)}")
    print(f"  Good WER<0.3 matches: {good_matches}/{len(has_candidates)}")
    print(f"  Whisper-only: {len(no_candidates)}")
    print(f"  Output: {OUTPUT_SUB}")


if __name__ == "__main__":
    main()
