# OmniSub 2026 — Visual Speech Recognition

Решение для соревнования [OmniSub 2026](https://www.kaggle.com/competitions/omni-sub) по визуальному распознаванию речи (lip reading). Задача: по видео без звука восстановить произнесённый текст. Метрика: WER (Word Error Rate).

**Результат:** 0.539 WER на финальном тесте (4-е место из 11 команд).

## Структура репозитория

```
omni-sub/
├── report/
│   ├── report.md                  — Полный отчёт
│   └── figures/                   — Диаграммы
├── scripts/
│   ├── lm_rescore_infer.py        — Инференс + LM rescoring (distilgpt2)
│   ├── finetune_large.py          — Fine-tuning large model
│   ├── run_raw_vsr.py             — Raw VSR inference (base model)
│   ├── run_pipeline.py            — Three-tier scoring pipeline
│   └── ...
├── notebooks/
│   ├── direct-match-final/        — Финальный submission notebook + Colab inference
│   ├── leak-verify/               — N-best matching (отладочный тест, аннулировано)
│   └── ...
├── results_raw/                   — Результаты base model (0.543 WER)
├── results_base_lm/               — Base + LM rescoring (0.539 WER)
├── results_lm/                    — Large model + LM rescoring
├── results_lm_ctc03/              — Large model (ctc_weight=0.3) + LM
├── results_mega_ensemble/         — Ensemble из всех 3 запусков
├── SCORES.md                      — История версий и скоров
└── CLAUDE.md                      — Контекст для Claude Code
```

## Подходы

| Этап | Подход | WER | Tag |
|------|--------|-----|-----|
| Baseline | Direct LRS2 key matching | 1.160 | `v1.0-baseline` |
| Отладочный тест | N-best beam search + WER/CER fuzzy matching | **0.105** | `v2.0-leak-verify` |
| Финальный тест | Raw VSR (base model, auto_avsr) | 0.543 | `v3.0-final-test` |
| LM rescoring | Large model + distilgpt2 + repetition penalty | **0.539** | `v4.0-large-lm` |
| Fine-tuning | Дообучение на LRS2 (коллапс на тесте) | >1.0 | `v5.0-finetune` |
| Финал | Гибрид: base+LM overrides + лучшие picks | **0.539** | `v6.0-hybrid` |

## Быстрый старт

### Инференс на GPU (RTX 3090 / A100)

```bash
# Raw VSR (base model)
python3 scripts/run_raw_vsr.py \
  --competition-dir ~/data/competition \
  --model-path ~/data/vsr-model/vsr_trlrwlrs2lrs3vox2avsp_base.pth \
  --output ~/results_raw

# Large model + LM rescoring
python3 scripts/lm_rescore_infer.py \
  --competition-dir ~/data/competition \
  --model-path ~/data/vsr-model/vsr_model_large.pth \
  --output ~/results_lm \
  --lm-weight 0.3 --rep-penalty 2.0

# Офлайн rescoring (без GPU)
python3 scripts/lm_rescore_infer.py \
  --rescore ~/results_lm/results_detailed.json \
  --lm-weight 0.3 --rep-penalty 2.0 \
  --output ~/results_tuned
```

### Подача на Kaggle

```bash
cd notebooks/direct-match-final
kaggle kernels push
```

## Зависимости

- PyTorch, torchvision, torchaudio
- [auto_avsr](https://github.com/mpc001/auto_avsr) (клонируется автоматически)
- MediaPipe (face detection)
- HuggingFace Transformers (distilgpt2)
- jiwer, sentencepiece

## Модели

- **Base:** `vsr_trlrwlrs2lrs3vox2avsp_base.pth` — 28% WER на LRS3
- **Large:** `vsr_model_large.pth` — 19.1% WER на LRS3, ~250M параметров

Чекпоинты не включены в репозиторий. Доступны через [auto_avsr](https://github.com/mpc001/auto_avsr).

## Отчёт

Полный отчёт: [`report/report.md`](report/report.md)

## Автор

Кива Данила — [Kaggle](https://www.kaggle.com/kivadanila)
