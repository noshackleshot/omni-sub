#!/usr/bin/env python3
"""
OmniSub 2026 — Large model + post-hoc LM rescoring + repetition penalty.

Usage:
  # Full pipeline (GPU):
  python3 lm_rescore_infer.py \
    --competition-dir ~/data/competition \
    --model-path ~/data/vsr-model/vsr_trlrwlrs2lrs3vox2avsp_base.pth \
    --output ~/results_lm

  # Skip LM rescoring:
  python3 lm_rescore_infer.py \
    --competition-dir ~/data/competition \
    --model-path ~/data/vsr-model/vsr_trlrwlrs2lrs3vox2avsp_base.pth \
    --output ~/results_lm --no-lm

  # Rescore from saved detailed results (no GPU needed):
  python3 lm_rescore_infer.py --rescore ~/results_lm/results_detailed.json \
    --lm-weight 0.3 --rep-penalty 2.0 --output ~/results_lm
"""

import os, sys, csv, re, json, time, argparse, subprocess, math
from pathlib import Path

import torch
import torchvision
import numpy as np


def norm(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ═══════════════════════════════════════════
# Repetition penalty
# ═══════════════════════════════════════════

def repetition_penalty(text, penalty_factor=2.0):
    """Penalize repeated n-grams and consecutive chunk repeats.
    Returns a penalty score >= 0 (higher = more repetitive)."""
    if penalty_factor <= 1.0:
        return 0.0

    words = text.split()
    if len(words) < 3:
        return 0.0

    penalty = 0.0

    # Bigram repetition
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    if bigrams:
        unique_bi = len(set(bigrams))
        repeat_ratio = 1.0 - unique_bi / len(bigrams)
        penalty += repeat_ratio * 2.0

    # Trigram repetition
    trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words) - 2)]
    if trigrams:
        unique_tri = len(set(trigrams))
        repeat_ratio = 1.0 - unique_tri / len(trigrams)
        penalty += repeat_ratio * 3.0

    # Consecutive chunk repeats (e.g., "i sublimed i sublimed i sublimed")
    for chunk_len in range(1, min(6, len(words) // 2 + 1)):
        max_consecutive = 1
        current = 1
        for i in range(chunk_len, len(words) - chunk_len + 1, chunk_len):
            chunk = words[i:i+chunk_len]
            prev_chunk = words[i-chunk_len:i]
            if chunk == prev_chunk:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 1
        if max_consecutive >= 3:
            penalty += (max_consecutive - 2) * 2.0

    return penalty * (penalty_factor - 1.0)


# ═══════════════════════════════════════════
# LM scoring (distilgpt2 post-hoc)
# ═══════════════════════════════════════════

class LMScorer:
    def __init__(self, model_name='distilgpt2', device='cpu'):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f'Loading LM: {model_name}...')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        self.device = device
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f'LM ready on {device}')

    @torch.no_grad()
    def score_batch(self, texts, batch_size=32):
        """Score a list of texts. Returns list of log-prob scores (higher = more likely)."""
        scores = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, return_tensors='pt', padding=True,
                                 truncation=True, max_length=128)
            input_ids = enc['input_ids'].to(self.device)
            attention_mask = enc['attention_mask'].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (B, T, V)

            # Shift: predict token t+1 from position t
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            shift_mask = attention_mask[:, 1:]

            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            # Mask padding
            token_log_probs = token_log_probs * shift_mask

            # Average log-prob per token (only over non-padding)
            lengths = shift_mask.sum(dim=1).clamp(min=1)
            avg_scores = (token_log_probs.sum(dim=1) / lengths).tolist()
            scores.extend(avg_scores)

        return scores


# ═══════════════════════════════════════════
# Offline rescoring from detailed results
# ═══════════════════════════════════════════

def rescore_from_detailed(detailed_path, lm_weight, rep_penalty_factor, output_dir):
    """Rescore saved hypotheses with different weights. No GPU needed."""
    with open(detailed_path) as f:
        detailed = json.load(f)

    results = {}
    for clip_name, clip_data in detailed.items():
        hyps = clip_data['hypotheses']
        if not hyps:
            results[clip_name] = 'a'
            continue

        vsr_scores = [h['vsr_score'] for h in hyps]
        lm_scores = [h.get('lm_score', 0.0) for h in hyps]

        # Min-max normalize
        vsr_norm = _minmax(vsr_scores)
        lm_norm = _minmax(lm_scores)

        best_score = float('-inf')
        best_text = hyps[0]['text']
        for j, h in enumerate(hyps):
            rep_pen = repetition_penalty(h['text'], rep_penalty_factor)
            combined = (1 - lm_weight) * vsr_norm[j] + lm_weight * lm_norm[j] - rep_pen
            if combined > best_score:
                best_score = combined
                best_text = h['text']

        results[clip_name] = best_text

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open(output_dir / 'submission.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'transcription'])
        for p in sorted(results.keys()):
            writer.writerow([p, results[p]])

    print(f'\n=== RESCORED (lm_weight={lm_weight}, rep_penalty={rep_penalty_factor}) ===')
    for p in sorted(results.keys()):
        print(f'  {p} → "{results[p][:70]}"')
    print(f'Saved to {output_dir}')
    return results


def _minmax(scores):
    """Min-max normalize a list of scores to [0, 1]."""
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [0.5] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


# ═══════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='Large model + LM rescoring inference')
    p.add_argument('--competition-dir', default=None)
    p.add_argument('--model-path', default=None)
    p.add_argument('--avsr-dir', default='/tmp/auto_avsr')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')

    # Beam search
    p.add_argument('--beam-size', type=int, default=40)
    p.add_argument('--ctc-weight', type=float, default=0.1,
                   help='CTC weight in beam search (0.1 for base, try 0.3 for large)')
    p.add_argument('--length-penalty', type=float, default=0.0)
    p.add_argument('--nbest', type=int, default=40,
                   help='Number of hypotheses to keep per clip')

    # LM rescoring
    p.add_argument('--lm-model', default='distilgpt2')
    p.add_argument('--lm-weight', type=float, default=0.3,
                   help='Weight for LM score in combined ranking (0.0-1.0)')
    p.add_argument('--no-lm', action='store_true', help='Skip LM rescoring')

    # Repetition penalty
    p.add_argument('--rep-penalty', type=float, default=2.0,
                   help='Repetition penalty factor (1.0=disabled)')

    # Strict loading
    p.add_argument('--strict', action='store_true',
                   help='Use strict=True for model loading (base model)')

    # Offline rescoring mode
    p.add_argument('--rescore', default=None,
                   help='Path to results_detailed.json — rescore offline, no GPU needed')

    # Upload
    p.add_argument('--upload', action='store_true',
                   help='Upload results to Kaggle dataset')

    return p.parse_args()


def main():
    args = parse_args()

    # ── Offline rescoring mode ──
    if args.rescore:
        rescore_from_detailed(args.rescore, args.lm_weight, args.rep_penalty, args.output)
        return

    if not args.competition_dir or not args.model_path:
        print('ERROR: --competition-dir and --model-path required for inference mode')
        sys.exit(1)

    device = args.device if torch.cuda.is_available() else 'cpu'

    # ── Setup auto_avsr ──
    avsr_dir = args.avsr_dir
    if not os.path.exists(avsr_dir):
        subprocess.run(['git', 'clone', '--depth', '1',
                        'https://github.com/mpc001/auto_avsr.git', avsr_dir], check=True)
    sys.path.insert(0, avsr_dir)

    from lightning import ModelModule, get_beam_search_decoder
    from datamodule.transforms import VideoTransform, TextTransform
    from preparation.detectors.mediapipe.detector import LandmarksDetector
    from preparation.detectors.mediapipe.video_process import VideoProcess

    comp_dir = Path(args.competition_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════
    # Load model (strict=False for large model)
    # ═══════════════════════════════════════════

    print('Loading VSR model...')
    mm_args = argparse.Namespace(modality='video', ctc_weight=args.ctc_weight)
    ckpt = torch.load(args.model_path, map_location='cpu')
    modelmodule = ModelModule(mm_args)

    # Remap large model checkpoint keys if needed
    # Large model has encoder.frontend.* → frontend.*, encoder.embed.* → proj_encoder.*
    model_keys = set(modelmodule.model.state_dict().keys())
    if not args.strict and any(k.startswith('encoder.frontend.') for k in ckpt):
        print('Detected large model key format — remapping...')
        remapped = {}
        for k, v in ckpt.items():
            if k.startswith('encoder.frontend.'):
                new_k = k.replace('encoder.frontend.', 'frontend.', 1)
            elif k == 'encoder.embed.0.weight':
                new_k = 'proj_encoder.weight'
            elif k == 'encoder.embed.0.bias':
                new_k = 'proj_encoder.bias'
            else:
                new_k = k
            remapped[new_k] = v
        ckpt = remapped

    if args.strict:
        modelmodule.model.load_state_dict(ckpt)
        print('Model loaded (strict=True)')
    else:
        missing, unexpected = modelmodule.model.load_state_dict(ckpt, strict=False)
        print(f'Model loaded (strict=False):')
        print(f'  Missing keys: {len(missing)}')
        print(f'  Unexpected keys: {len(unexpected)}')
        if missing:
            print(f'  Missing (first 10): {missing[:10]}')
        if unexpected:
            print(f'  Unexpected (first 10): {unexpected[:10]}')

    modelmodule.eval()
    if device == 'cuda':
        modelmodule = modelmodule.cuda()
    model = modelmodule.model
    text_transform = modelmodule.text_transform

    beam_search = get_beam_search_decoder(
        model, modelmodule.token_list,
        penalty=args.length_penalty,
        ctc_weight=args.ctc_weight,
        beam_size=args.beam_size,
    )
    del ckpt

    landmarks_detector = LandmarksDetector()
    video_process = VideoProcess(convert_gray=False)
    video_transform = VideoTransform(subset='test')

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {total_params/1e6:.1f}M params, device={device}')
    print(f'Beam search: beam_size={args.beam_size}, ctc_weight={args.ctc_weight}, '
          f'length_penalty={args.length_penalty}')

    # ── Find test paths ──
    test_dir = comp_dir / 'test'
    sample_sub = comp_dir / 'sample_submission.csv'

    test_paths = []
    prefilled = {}
    if sample_sub.exists():
        with open(sample_sub) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                test_paths.append(row[0])
                if len(row) > 1 and row[1].strip():
                    prefilled[row[0]] = norm(row[1].strip())
        print(f'Test: {len(test_paths)} paths, {len(prefilled)} pre-filled')
    else:
        test_paths = sorted([f.name for f in test_dir.glob('*.mp4')])
        print(f'Found {len(test_paths)} test clips')

    # ═══════════════════════════════════════════
    # VSR inference — collect N-best hypotheses
    # ═══════════════════════════════════════════

    print(f'\n{"="*60}')
    print(f'INFERENCE on {len(test_paths)} clips (nbest={args.nbest})')
    print(f'{"="*60}')

    all_clip_hypotheses = {}  # clip_name -> list of {text, vsr_score}
    results_top1 = {}
    start = time.time()

    for i, path in enumerate(test_paths):
        if path in prefilled:
            all_clip_hypotheses[path] = [{'text': prefilled[path], 'vsr_score': 0.0}]
            results_top1[path] = prefilled[path]
            print(f'[{i+1}/{len(test_paths)}] {path}: PREFILLED')
            continue

        mp4_path = test_dir / path if '/' not in path else comp_dir / path
        if not mp4_path.exists():
            mp4_path = test_dir / Path(path).name
        if not mp4_path.exists():
            all_clip_hypotheses[path] = [{'text': 'a', 'vsr_score': 0.0}]
            results_top1[path] = 'a'
            print(f'[{i+1}/{len(test_paths)}] {path}: NOT FOUND')
            continue

        try:
            video = torchvision.io.read_video(str(mp4_path), pts_unit='sec')[0].numpy()
            landmarks = landmarks_detector(video)
            video_cropped = video_process(video, landmarks)
            if video_cropped is None:
                all_clip_hypotheses[path] = [{'text': 'a', 'vsr_score': 0.0}]
                results_top1[path] = 'a'
                print(f'[{i+1}/{len(test_paths)}] {path}: NO FACE')
                continue

            video_tensor = torch.tensor(video_cropped).permute(0, 3, 1, 2)
            video_tensor = video_transform(video_tensor)
            if device == 'cuda':
                video_tensor = video_tensor.cuda()

            with torch.no_grad():
                x = model.frontend(video_tensor.unsqueeze(0))
                x = model.proj_encoder(x)
                enc_feat, _ = model.encoder(x, None)
                enc_feat = enc_feat.squeeze(0)

                # Beam search — collect ALL hypotheses
                nbest = beam_search(enc_feat)
                hypotheses = []
                seen = set()
                for hyp in nbest:
                    if len(hypotheses) >= args.nbest:
                        break
                    h = hyp.asdict()
                    tids = torch.tensor(list(map(int, h["yseq"][1:])))
                    text = text_transform.post_process(tids).replace("<eos>", "")
                    text_normed = norm(text)
                    if text_normed and text_normed not in seen:
                        hypotheses.append({
                            'text': text_normed,
                            'vsr_score': float(h['score']),
                        })
                        seen.add(text_normed)

                # CTC greedy fallback
                ctc_lp = model.ctc.log_softmax(enc_feat.unsqueeze(0)).squeeze(0)
                ctc_argmax = torch.argmax(ctc_lp, dim=-1)
                tokens = []
                prev = 0
                for t in ctc_argmax:
                    tv = t.item()
                    if tv != 0 and tv != prev:
                        tokens.append(tv)
                    prev = tv
                if tokens:
                    ctc_text = norm(text_transform.post_process(
                        torch.tensor(tokens)).replace("<eos>", ""))
                    if ctc_text and ctc_text not in seen:
                        # Give CTC greedy a lower score than worst beam hyp
                        worst_score = min(h['vsr_score'] for h in hypotheses) if hypotheses else 0.0
                        hypotheses.append({
                            'text': ctc_text,
                            'vsr_score': worst_score - 1.0,
                        })

            all_clip_hypotheses[path] = hypotheses if hypotheses else [{'text': 'a', 'vsr_score': 0.0}]
            results_top1[path] = hypotheses[0]['text'] if hypotheses else 'a'

        except Exception as e:
            all_clip_hypotheses[path] = [{'text': 'a', 'vsr_score': 0.0}]
            results_top1[path] = 'a'
            print(f'[{i+1}/{len(test_paths)}] {path}: ERROR {e}')
            if 'out of memory' in str(e).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        elapsed = time.time() - start
        n_hyps = len(all_clip_hypotheses[path])
        print(f'[{i+1}/{len(test_paths)}] {path}: {n_hyps} hyps, '
              f'top1="{results_top1[path][:50]}" ({elapsed:.1f}s)')

    infer_time = time.time() - start
    total_hyps = sum(len(h) for h in all_clip_hypotheses.values())
    print(f'\nInference done in {infer_time:.1f}s — {total_hyps} total hypotheses')

    # ═══════════════════════════════════════════
    # LM rescoring
    # ═══════════════════════════════════════════

    if not args.no_lm:
        print(f'\n{"="*60}')
        print(f'LM RESCORING ({args.lm_model}, weight={args.lm_weight})')
        print(f'{"="*60}')

        lm_device = device  # Keep LM on same device
        lm_scorer = LMScorer(args.lm_model, lm_device)

        # Collect all texts for batch scoring
        all_texts = []
        text_to_idx = []  # (clip_name, hyp_idx)
        for clip_name in test_paths:
            for j, h in enumerate(all_clip_hypotheses[clip_name]):
                all_texts.append(h['text'])
                text_to_idx.append((clip_name, j))

        print(f'Scoring {len(all_texts)} hypotheses...')
        lm_start = time.time()
        lm_scores = lm_scorer.score_batch(all_texts)
        lm_time = time.time() - lm_start
        print(f'LM scoring done in {lm_time:.1f}s')

        # Attach LM scores to hypotheses
        for idx, (clip_name, hyp_idx) in enumerate(text_to_idx):
            all_clip_hypotheses[clip_name][hyp_idx]['lm_score'] = lm_scores[idx]

        # Free LM memory
        del lm_scorer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        # No LM — set lm_score to 0
        for clip_name in test_paths:
            for h in all_clip_hypotheses[clip_name]:
                h['lm_score'] = 0.0

    # ═══════════════════════════════════════════
    # Combined scoring + repetition penalty
    # ═══════════════════════════════════════════

    print(f'\n{"="*60}')
    print(f'COMBINED SCORING (lm_weight={args.lm_weight}, rep_penalty={args.rep_penalty})')
    print(f'{"="*60}')

    results = {}
    for clip_name in test_paths:
        hyps = all_clip_hypotheses[clip_name]
        if not hyps or (len(hyps) == 1 and hyps[0]['text'] == 'a'):
            results[clip_name] = 'a'
            continue

        vsr_scores = [h['vsr_score'] for h in hyps]
        lm_scores_clip = [h.get('lm_score', 0.0) for h in hyps]

        vsr_norm_scores = _minmax(vsr_scores)
        lm_norm_scores = _minmax(lm_scores_clip)

        best_score = float('-inf')
        best_text = hyps[0]['text']
        best_idx = 0

        for j, h in enumerate(hyps):
            rep_pen = repetition_penalty(h['text'], args.rep_penalty)
            combined = ((1 - args.lm_weight) * vsr_norm_scores[j]
                        + args.lm_weight * lm_norm_scores[j]
                        - rep_pen)
            h['rep_penalty'] = rep_pen
            h['combined_score'] = combined

            if combined > best_score:
                best_score = combined
                best_text = h['text']
                best_idx = j

        results[clip_name] = best_text

        # Log if LM changed the pick
        if best_idx != 0:
            print(f'  {clip_name}: LM changed pick #{best_idx} '
                  f'"{best_text[:50]}" (was "{hyps[0]["text"][:50]}")')
        # Log if rep penalty fired
        if any(h.get('rep_penalty', 0) > 0.1 for h in hyps):
            max_pen = max(h.get('rep_penalty', 0) for h in hyps)
            print(f'  {clip_name}: rep_penalty fired (max={max_pen:.2f})')

    # ═══════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════

    for path in test_paths:
        if path not in results or not results[path]:
            results[path] = 'a'

    # results.json — Kaggle-compatible
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # results_detailed.json — all hypotheses + scores
    detailed = {}
    for clip_name in test_paths:
        detailed[clip_name] = {
            'best': results[clip_name],
            'hypotheses': all_clip_hypotheses[clip_name],
        }
    with open(output_dir / 'results_detailed.json', 'w') as f:
        json.dump(detailed, f, indent=2)

    # submission.csv
    with open(output_dir / 'submission.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'transcription'])
        for p in test_paths:
            writer.writerow([p, results[p]])

    # dataset-metadata.json for Kaggle upload
    with open(output_dir / 'dataset-metadata.json', 'w') as f:
        json.dump({
            "title": "OmniSub Precomputed Results",
            "id": "kivadanila/omnisub-precomputed-results",
            "licenses": [{"name": "CC0-1.0"}]
        }, f, indent=2)

    # ── Print all results ──
    print(f'\n{"="*60}')
    print(f'ALL RESULTS')
    print(f'{"="*60}')
    for p in test_paths:
        n_hyps = len(all_clip_hypotheses.get(p, []))
        print(f'  {p} ({n_hyps} hyps) → "{results[p][:70]}"')

    ok = sum(1 for v in results.values() if v and v != 'a')
    print(f'\nTotal: {len(results)}, OK: {ok}, empty/fallback: {len(results) - ok}')
    print(f'Saved to {output_dir}')
    print(f'  results.json — Kaggle-compatible (49 entries)')
    print(f'  results_detailed.json — all hypotheses + scores (for offline tuning)')
    print(f'  submission.csv — CSV submission')

    # ── Upload to Kaggle ──
    if args.upload:
        print('\nUploading to Kaggle...')
        try:
            r = subprocess.run(['kaggle', 'datasets', 'version', '-p', str(output_dir),
                                '-m', 'large model + LM rescoring', '--dir-mode', 'zip'],
                               capture_output=True, text=True, timeout=120)
            if 'not found' in (r.stderr or '').lower() or r.returncode != 0:
                r = subprocess.run(['kaggle', 'datasets', 'create', '-p', str(output_dir),
                                    '--dir-mode', 'zip'],
                                   capture_output=True, text=True, timeout=120)
            print(f'Upload: {r.stdout}')
            if r.returncode != 0:
                print(f'Upload error: {r.stderr}')
        except Exception as e:
            print(f'Upload failed: {e}')

    total_time = time.time() - start
    print(f'\nTotal time: {total_time:.1f}s')
    print('DONE')


if __name__ == '__main__':
    main()
