# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kaggle competition submission for **OmniSub 2026** — a visual speech recognition (lip reading) task. Given video-only clips (no audio), the goal is to detect a face, extract the mouth region, and transcribe the spoken text. Evaluated by **WER** (Word Error Rate, lower is better).

- **Competition**: Private Community Prediction Competition (`omni-sub` on Kaggle). 77 entrants, 16 participants, 11 teams.
- **Deadline**: March 19, 2026, 9:00 PM. A report must be submitted via omnisub2026@mtuci.ru.
- **Submission vehicle**: Kaggle notebook (CPU-only for final stage).

### Final Stage (2026-03-18 → 2026-03-19)

On 2026-03-18 at 21:00, organizers replaced the test set with a **new 49-clip final test** after discovering the original test data had audio available online (LRS2-BBC data leak). All previous submissions were invalidated.

- **New test**: 49 clips of casual speech (vlogs/podcasts), NOT from LRS2-BBC.
- **Submission limit**: 2 attempts/day, deadline 2026-03-19 21:00.
- **Leaderboard (public, ~20% = ~10 clips)**:

| # | Team | Score |
|---|------|-------|
| 1 | Alehandreus | 0.20477 |
| 2 | Garifullin Timur | 0.39931 |
| 3 | ТекстКадр | 0.52901 |
| **4** | **Kiva Danila** | **0.54266** |
| 5-9 | ... | 0.63–1.00 |

- **Our submissions on new test**: V5 = 0.96587, **V6 = 0.54266** (current best). **2 attempts remaining**.
- **LRS2 matching is useless** on the new test — clips are not from LRS2.

## Key Entry Points

- **`notebooks/direct-match-final/notebook.ipynb`** — Current best submission for final test (score 0.54266). Lightweight CPU notebook: loads precomputed `results.json` from Kaggle dataset and writes `submission.csv`.
- **`notebooks/direct-match-final/colab_inference.ipynb`** — Full VSR inference pipeline for Colab/A100. Runs auto_avsr, three-tier scoring (direct match + channel CTC/Att/Fuzzy + global pool), uploads `results.json` to Kaggle dataset.
- **`results_raw/results.json`** — Local copy of the 49 precomputed VSR transcriptions (raw beam search output).
- **`notebooks/leak-verify/notebook.ipynb`** — Former best submission (score 0.10524, now invalidated). N-best VSR beam search + WER/CER fuzzy matching.
- **`notebooks/ctc-combined/notebook.ipynb`** — Three-signal scoring: CTC (35%) + Attention (30%) + Fuzzy (35%) with min-max normalization.
- **`notebooks/vsr-n-best/notebook.ipynb`** — Two-pass CTC + attention scoring notebook (score 0.84801, now invalidated).
- **`scripts/verify_leak.py`** — Standalone local script. Simpler Whisper-based approach (not relevant for final test).
- **`SCORES.md`** — Version-score mapping and approach comparison for all Kaggle submissions.

## Architecture: Final Stage Pipeline (direct-match-final)

### Two-step split architecture:
1. **A100 server** (or Colab) — `colab_inference.ipynb`: Run full VSR inference → upload `results.json` as Kaggle dataset (`kivadanila/omnisub-precomputed-results`)
2. **Kaggle notebook** — `notebook.ipynb`: Load `results.json` → write `submission.csv` (CPU-only, no GPU needed)

### Inference pipeline (colab_inference.ipynb):
1. **Setup**: Clone auto_avsr, load pretrained VSR model, download competition data + LRS2 texts
2. **Classification**: Split test paths into Tier 1 (exact LRS2 key match), Tier 2 (channel candidates exist), Tier 3 (global pool)
3. **VSR inference**: For Tier 2+3 clips — detect face landmarks (MediaPipe), crop mouth, run Conformer encoder, beam search (40 hypotheses)
4. **Tier 2 scoring**: Fuzzy match → if confident (<0.10 WER+CER), accept. Otherwise: CTC (25%) + Attention (15%) + Fuzzy (60%) weighted combination
5. **Tier 3 scoring**: Word-count-filtered global pool + trigram pre-filter + CTC scoring + fuzzy fallback
6. **Upload**: Save results.json → `kaggle datasets create`

### Legacy architecture (vsr-n-best, leak-verify — invalidated):
1. **Setup**: Clone auto_avsr, load pretrained VSR model (`vsr_model.pth`)
2. **Data loading**: Build candidate text pool from LRS2 dataset files + train `.txt` files, grouped by channel
3. **VSR inference**: For each test video — detect face landmarks (MediaPipe), crop mouth region, run Conformer encoder
4. **Pass 1 — CTC scoring**: Score ALL channel candidates against CTC log-probs (fast, ~1ms each)
5. **Pass 2 — Attention scoring**: Re-score top-K candidates (default 15) using batched decoder forward pass
6. **Final ranking**: Weighted combination — CTC (35%) + Attention (55%) + Text WER/CER match (10%) + exact match bonus
7. **Fallback paths**: Duration-based matching when VSR unavailable; global pool CTC scoring for clips without channel candidates

## Data Layout

### Competition Data (on Kaggle)

#### Original test (invalidated 2026-03-18):
- `train/<video_id>/<clip_id>.mp4` — 12,000 training video clips across 1,603 video_ids
- `train/<video_id>/<clip_id>.txt` — 12,000 ground-truth transcripts
- `test/<video_id>/<clip_id>.mp4` — 3,000 test video clips across 1,713 video_ids
- `sample_submission.csv` — 3,001 lines (header + 3,000 rows), columns: `path,transcription`

#### Final test (current, 49 clips):
- `test/<clip_id>.mp4` — 49 test clips (`00000.mp4`–`00048.mp4`), casual speech (vlogs/podcasts)
- `sample_submission.csv` — 50 lines (header + 49 rows), columns: `path,transcription`
- NOT from LRS2-BBC — genuine lip reading required

Train TXT format:
```
Text: <TRANSCRIPTION>
Conf: <confidence_score>

WORD START END ASDSCORE
WORD1 0.08 0.49 7.3
...
```

### Local Data

- `test/test/<video_id>/<clip_id>.mp4` — all 3,000 original test clips (double dir from zip extraction)
- `train/train/<video_id>/<clip_id>.txt` — all 12,000 transcripts
- `train/train/<video_id>/<clip_id>.mp4` — only 9 MP4s (1 video_id: 5535496873950688380)
- `results_raw/results.json` — 49 precomputed VSR transcriptions for final test
- `submissions/sample_submission.csv` — Template with `path,transcription` columns
- `submissions/` — All generated CSV submission outputs
- `assets/` — Debug frame images

### External Datasets (added to Kaggle notebooks, not competition-provided)

- `data/lrs2-texts/lrs2_{train,test,val}_text.txt` — LRS2-BBC reference texts (128,757 lines total, format: `<video_id>_<clip_id> <text>`)
- `auto-avsr-vsr-model` — Pretrained VSR model (`vsr_model.pth`)
- `notebooks/auto_avsr/` — Local reference copy of the auto_avsr framework
- `kivadanila/omnisub-precomputed-results` — Kaggle dataset containing `results.json` (uploaded from A100/Colab inference)

## Commands

```bash
# Push direct-match-final notebook to Kaggle (current best approach for final test)
cd notebooks/direct-match-final && kaggle kernels push

# Push leak-verify notebook to Kaggle (invalidated — original test only)
cd notebooks/leak-verify && kaggle kernels push

# Push vsr-n-best notebook to Kaggle (invalidated — original test only)
cd notebooks/vsr-n-best && kaggle kernels push
```

## Dependencies

Python stack: PyTorch, PyTorch Lightning, torchvision, torchaudio, OpenAI Whisper, jiwer, sentencepiece, mediapipe, scikit-image, av, espnet (bundled in auto_avsr).

The auto_avsr framework is cloned at runtime on Kaggle into `/kaggle/working/auto_avsr`. A local copy exists at `notebooks/auto_avsr/` for development reference.

## Key Scoring Parameters

### Final pipeline (colab_inference.ipynb Cell 6):
- `W_CTC = 0.25`, `W_ATT = 0.15`, `W_FUZZY = 0.60` — Tier 2 combined weights
- `FUZZY_CONFIDENT = 0.10` — threshold for accepting fuzzy match without CTC/Att
- `TOP_K = 15` — candidates promoted from CTC to attention pass
- `TEXT_MATCH_HYPS = 20` — beam search hypotheses used for text matching
- `GLOBAL_WC_WINDOW = 4` — word count tolerance for Tier 3 global pool
- `GLOBAL_ACCEPT = 0.25` — fuzzy score threshold for Tier 3 acceptance

### Legacy (vsr-n-best notebook, invalidated):
- `W_CTC = 0.35`, `W_ATT = 0.55`, `W_TEXT = 0.10`, `EXACT_BONUS = 0.5`

## Kaggle Kernel Configs

- **Direct Match Final**: `kivadanila/omnisub-direct-match` — CPU, no internet, datasets: `lrs2-texts`, `auto-avsr-vsr-model`, `omnisub-precomputed-results`
- **Leak Verify** (invalidated): `kivadanila/omnisub-leak-verify` — GPU, internet, datasets: `lrs2-texts`, `auto-avsr-vsr-model`
- **VSR N-Best** (invalidated): `kivadanila/omnisub-vsr-n-best` — GPU, internet, datasets: `lrs2-texts`, `auto-avsr-vsr-model`
- Competition: `omni-sub`

## A100 Server

Remote GPU server used for VSR inference. Credentials in `creds.txt`.
- SSH: `195.208.16.1:48280`
- Jupyter: `195.208.16.2:8888`
- **Status as of 2026-03-19: UNAVAILABLE** (connection timeout, balance was ~498₽ at 150₽/hr ≈ 3.3h)

## Notebooks Directory (16 total)

All contain `kernel-metadata.json`. Two without `notebook.ipynb`: `ctc-att-tpu`, `full-pipeline-49`.

Key notebooks for the report: `leak-verify`, `direct-match-final` (+ `colab_inference.ipynb`), `direct-match-vsr`, `direct-match`.
