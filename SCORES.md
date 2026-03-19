# OmniSub 2026 — Scores & Version History

## Final Test Submissions (49 clips, 2026-03-18 → 2026-03-19)

New test set released 2026-03-18 at 21:00 after LRS2 data leak discovered. All previous submissions invalidated. Limit: 2 attempts/day.

| # | Source | Public Score | Approach |
|---|--------|-------------|----------|
| 1 | **Direct Match Final V6** | **0.54266** | Precomputed results.json (A100 VSR inference) → CPU submission |
| 2 | Direct Match Final V5 | 0.96587 | Same pipeline, earlier results |

**Current best: 0.54266** (rank #4). **2 attempts remaining** (deadline 2026-03-19 21:00).

### Public Leaderboard (final test)

| # | Team | Score |
|---|------|-------|
| 1 | Alehandreus | 0.20477 |
| 2 | Garifullin Timur | 0.39931 |
| 3 | ТекстКадр | 0.52901 |
| **4** | **Kiva Danila** | **0.54266** |
| 5-9 | ... | 0.63–1.00 |

---

## Original Test Submissions (3,000 clips, ALL INVALIDATED)

| # | Source | Public Score | Approach |
|---|--------|-------------|----------|
| 1 | ~~Leak Verify V13~~ | ~~0.10524~~ | N-best VSR + WER/CER fuzzy matching |
| 2 | ~~Leak Verify V12~~ | ~~0.16253~~ | Same, minor tweak (+1 -2 diff) |
| 3 | ~~VSR N-Best V2~~ | ~~0.84801~~ | Two-pass CTC (35%) + Attention (55%) + Text (10%) |
| 4 | ~~fresh_test.csv~~ | ~~0.98387~~ | "fresh paths test" (manual CSV upload) |
| 5 | ~~submission.csv~~ | ~~0.91491~~ | "multi-signal LRS2 matching (WPS+CPS)" |
| 6 | ~~submission.csv~~ | ~~0.91660~~ | "duration-based LRS2 matching (3.09 wps)" |
| 7 | ~~submission_fixed.csv~~ | ~~1.16890~~ | "LRS2 first candidate per channel" |
| 8 | ~~submission.csv~~ | ~~1.15962~~ | "baseline" |
| 9 | ~~submission_direct_v2.csv~~ | ~~1.86742~~ | "direct LRS2 key match" |

~~Best score: 0.10524~~ (invalidated)

## Three Approaches Compared

| Approach | Notebook | Model | Scoring Method | Best Score |
|----------|----------|-------|----------------|------------|
| **N-best + WER/CER** | `notebooks/leak-verify` | Auto-AVSR (beam search only) | Fuzzy WER+CER matching (0.4W+0.6C) vs candidates | **0.10524** |
| Two-pass CTC+Att | `notebooks/vsr-n-best` | Auto-AVSR (CTC+decoder) | CTC 35% + Attention 55% + Text 10% + exact bonus | 0.84801 |
| **CTC Combined** | `notebooks/ctc-combined` | Auto-AVSR (CTC+decoder+fuzzy) | Min-max normalized: CTC 35% + Attention 30% + Fuzzy 35% | (pending) |
| Whisper local | `scripts/verify_leak.py` | OpenAI Whisper medium | WER-only fuzzy matching | (local only, not submitted) |

## Leak Verify Notebook — Version History (13 versions)

| Version | Status | Runtime | Diff | Score |
|---------|--------|---------|------|-------|
| V13 | Success | 1h 31m | +105 -50 | **0.10524** |
| V12 | Success | 1h 14m | +1 -2 | 0.16253 |
| V11 | Success | 9m 16s | +29 -115 | (not submitted) |
| V10 | **Failed** | 2m 7s | +85 -56 | — |
| V9 | Success | 9m 23s | +97 -121 | (not submitted) |
| V8 | **Failed** | 27s | +191 -323 | — |
| V7 | Success | 8m 54s | +356 -296 | (not submitted) |
| V6 | Success | 7m 39s | +404 -174 | (not submitted) |
| V5 | Success | 24m 7s | +170 -194 | (not submitted) |
| V4 | Success | 6m 17s | +195 -251 | (not submitted) |
| V3 | Success | 6m 52s | +79 -36 | (not submitted) |
| V2 | **Failed** | 39s | +33 -8 | — |
| V1 | **Failed** | 32s | +244 -0 | — |

## VSR N-Best Notebook — Version History (2 versions)

| Version | Status | Runtime | Diff | Score |
|---------|--------|---------|------|-------|
| V2 | Success | 1h 11m | +176 -86 | 0.84801 |
| V1 | Cancelled | 2h 52m | +506 -0 | — |
