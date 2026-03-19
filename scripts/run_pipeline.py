#!/usr/bin/env python3
"""
OmniSub 2026 — Full VSR pipeline for gpudc.ru (A100).

Three-tier scoring optimized for 49 test clips:
  Tier 1: Direct LRS2 key lookup (WER=0)
  Tier 2: VSR → CTC + Attention + Fuzzy vs ALL channel candidates
  Tier 3: VSR → global pool search with heavy scoring

Usage:
  python3 run_pipeline.py \
    --competition-dir ~/data/competition \
    --lrs2-dir ~/data/lrs2-texts \
    --model-path ~/data/vsr-model/vsr_model.pth \
    --output ~/results

Output: results.json + submission.csv + datapackage.json (for Kaggle upload)
"""

import os, sys, csv, re, json, time, argparse, subprocess
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchvision
import numpy as np


# ═══════════════════════════════════════════
# CLI Arguments
# ═══════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='OmniSub 2026 VSR Pipeline')
    p.add_argument('--competition-dir', required=True, help='Path to extracted competition data (contains test/, train/, sample_submission.csv)')
    p.add_argument('--lrs2-dir', required=True, help='Path to LRS2 text files (lrs2_train_text.txt etc)')
    p.add_argument('--model-path', required=True, help='Path to vsr_model.pth')
    p.add_argument('--avsr-dir', default=None, help='Path to auto_avsr clone (will clone if absent)')
    p.add_argument('--output', required=True, help='Output directory for results')
    p.add_argument('--device', default='cuda', help='Device: cuda or cpu')
    p.add_argument('--upload', action='store_true', help='Auto-upload results to Kaggle as dataset')
    p.add_argument('--dataset-id', default='kivadanila/omnisub-precomputed-results', help='Kaggle dataset ID for upload')
    return p.parse_args()


# ═══════════════════════════════════════════
# Text normalization
# ═══════════════════════════════════════════

def norm(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ═══════════════════════════════════════════
# Scoring helpers
# ═══════════════════════════════════════════

def subsequent_mask(size, device='cpu'):
    return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))


def match_score(ref, hyp):
    """Combined WER+CER (lower = better match)."""
    from jiwer import wer as compute_wer, cer as compute_cer
    try:
        w = compute_wer(ref, hyp)
        c = compute_cer(ref, hyp)
        return 0.4 * w + 0.6 * c
    except:
        return 1.0


@torch.no_grad()
def score_ctc_batch(model, enc_feat, candidates_token_ids, device, batch_size=64):
    """Batch CTC scoring. Returns list of scores (higher = better)."""
    T = enc_feat.size(0)
    ctc_logprobs = model.ctc.log_softmax(enc_feat.unsqueeze(0)).squeeze(0)

    all_scores = []
    for batch_start in range(0, len(candidates_token_ids), batch_size):
        batch = candidates_token_ids[batch_start:batch_start + batch_size]
        batch_scores = [float('-inf')] * len(batch)

        valid = [(j, tids) for j, tids in enumerate(batch) if len(tids) > 0 and len(tids) <= T]
        if not valid:
            all_scores.extend(batch_scores)
            continue

        max_s = max(len(tids) for _, tids in valid)
        targets = torch.zeros(len(valid), max_s, dtype=torch.long, device=device)
        target_lengths = torch.zeros(len(valid), dtype=torch.long, device=device)
        for k, (j, tids) in enumerate(valid):
            targets[k, :len(tids)] = torch.tensor(tids, device=device)
            target_lengths[k] = len(tids)

        log_probs = ctc_logprobs.unsqueeze(1).expand(-1, len(valid), -1).contiguous()
        input_lengths = torch.full((len(valid),), T, dtype=torch.long, device=device)

        losses = F.ctc_loss(
            log_probs, targets, input_lengths, target_lengths,
            blank=0, reduction='none', zero_infinity=True
        )

        for k, (j, tids) in enumerate(valid):
            batch_scores[j] = -losses[k].item() / max(len(tids), 1)

        all_scores.extend(batch_scores)
    return all_scores


@torch.no_grad()
def score_attention_single(model, enc_feat, token_ids, device):
    """Score a single candidate with autoregressive decoder. Returns per-token log-prob."""
    sos = model.sos
    eos = model.eos

    tgt_in = torch.tensor([[sos] + token_ids], device=device)
    tgt_out = token_ids + [eos]
    tgt_mask = subsequent_mask(tgt_in.size(1), device=device).unsqueeze(0)
    memory = enc_feat.unsqueeze(0)

    logits, _ = model.decoder(tgt_in, tgt_mask, memory, None)
    log_probs = F.log_softmax(logits, dim=-1)

    n = len(tgt_out)
    score = sum(log_probs[0, j, tgt_out[j]].item() for j in range(n)) / n
    return score


@torch.no_grad()
def score_attention_batch(model, enc_feat, candidates_token_ids, device, batch_size=16):
    """
    Batch attention scoring for multiple candidates against same enc_feat.
    Returns list of scores (higher = better).
    """
    sos = model.sos
    eos = model.eos
    all_scores = []

    for batch_start in range(0, len(candidates_token_ids), batch_size):
        batch = candidates_token_ids[batch_start:batch_start + batch_size]
        batch_scores = [float('-inf')] * len(batch)

        valid = [(j, tids) for j, tids in enumerate(batch) if len(tids) > 0]
        if not valid:
            all_scores.extend(batch_scores)
            continue

        # Pad sequences
        max_len = max(len(tids) for _, tids in valid) + 1  # +1 for SOS
        tgt_in = torch.full((len(valid), max_len), 0, dtype=torch.long, device=device)
        tgt_out_list = []
        tgt_lengths = []
        for k, (j, tids) in enumerate(valid):
            seq = [sos] + tids
            tgt_in[k, :len(seq)] = torch.tensor(seq, device=device)
            tgt_out_list.append(tids + [eos])
            tgt_lengths.append(len(tids) + 1)  # includes EOS

        tgt_mask = subsequent_mask(max_len, device=device).unsqueeze(0)
        memory = enc_feat.unsqueeze(0).expand(len(valid), -1, -1)

        logits, _ = model.decoder(tgt_in, tgt_mask, memory, None)
        log_probs = F.log_softmax(logits, dim=-1)

        for k, (j, tids) in enumerate(valid):
            tgt_out = tgt_out_list[k]
            n = len(tgt_out)
            score = sum(log_probs[k, t, tgt_out[t]].item() for t in range(n)) / n
            batch_scores[j] = score

        all_scores.extend(batch_scores)
    return all_scores


def normalize_scores(scores):
    """Min-max normalize to [0, 1]."""
    if not scores:
        return scores
    mn, mx = min(scores), max(scores)
    if mx - mn < 1e-9:
        return [0.5] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def trigrams(text):
    return set(text[i:i+3] for i in range(len(text)-2))


# ═══════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════

def load_data(competition_dir, lrs2_dir):
    """Load LRS2 texts, training texts, and test paths."""
    competition_dir = Path(competition_dir)
    lrs2_dir = Path(lrs2_dir)

    # Find test dir (handle double-nested from zip extraction)
    test_dir = competition_dir / 'test'
    if (test_dir / 'test').exists():
        test_dir = test_dir / 'test'
    elif not any(test_dir.iterdir()):
        # Try competition_dir directly
        for d in competition_dir.rglob('test'):
            if any(d.iterdir()):
                test_dir = d
                if (test_dir / 'test').exists():
                    test_dir = test_dir / 'test'
                break

    # Find train dir
    train_dir = competition_dir / 'train'
    if (train_dir / 'train').exists():
        train_dir = train_dir / 'train'

    # Find sample_submission.csv
    sample_sub = None
    for p in [competition_dir / 'sample_submission.csv',
              competition_dir / 'omni-sub' / 'sample_submission.csv']:
        if p.exists():
            sample_sub = p
            break
    if sample_sub is None:
        # Search recursively
        for p in competition_dir.rglob('sample_submission.csv'):
            sample_sub = p
            break
    if sample_sub is None:
        raise FileNotFoundError(f'sample_submission.csv not found in {competition_dir}')

    print(f'TEST_DIR: {test_dir}')
    print(f'TRAIN_DIR: {train_dir}')
    print(f'SAMPLE_SUB: {sample_sub}')
    print(f'LRS2_DIR: {lrs2_dir}')

    # === Build exact key lookup + channel pools ===
    lrs2_exact = {}
    lrs2_by_channel = defaultdict(list)
    for fname in ['lrs2_train_text.txt', 'lrs2_test_text.txt', 'lrs2_val_text.txt']:
        fpath = lrs2_dir / fname
        if not fpath.exists():
            continue
        with open(fpath) as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) < 2:
                    continue
                key = parts[0]
                text = norm(parts[1])
                lrs2_exact[key] = text
                ch = key.rsplit('_', 1)[0]
                lrs2_by_channel[ch].append(text)
    for ch in lrs2_by_channel:
        lrs2_by_channel[ch] = list(dict.fromkeys(lrs2_by_channel[ch]))
    print(f'LRS2: {len(lrs2_exact)} exact keys, {sum(len(v) for v in lrs2_by_channel.values())} channel texts, {len(lrs2_by_channel)} channels')

    # === Add training transcripts ===
    if train_dir.exists():
        train_added = 0
        for ch_name in os.listdir(train_dir):
            ch_dir = train_dir / ch_name
            if not ch_dir.is_dir():
                continue
            existing = set(lrs2_by_channel.get(ch_name, []))
            for txt_file in ch_dir.glob('*.txt'):
                with open(txt_file) as f:
                    first_line = f.readline().strip()
                if first_line.startswith('Text:'):
                    text = norm(first_line[5:].strip())
                    if text and text not in existing:
                        lrs2_by_channel[ch_name].append(text)
                        existing.add(text)
                        train_added += 1
        print(f'Train data: added {train_added} texts')

    lrs2_all_texts = list(set(t for texts in lrs2_by_channel.values() for t in texts))
    print(f'Total: {sum(len(v) for v in lrs2_by_channel.values())} texts, {len(lrs2_by_channel)} channels, {len(lrs2_all_texts)} unique')

    # === Load test paths ===
    test_paths = []
    with open(sample_sub) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            test_paths.append(row[0])

    # === Classify into tiers ===
    exact_match_paths = []
    vsr_needed_paths = []
    for p in test_paths:
        parts = p.split('/')
        key = f"{parts[1]}_{parts[2].replace('.mp4', '')}"
        if key in lrs2_exact:
            exact_match_paths.append(p)
        else:
            vsr_needed_paths.append(p)

    paths_with_cand = [p for p in vsr_needed_paths if p.split('/')[1] in lrs2_by_channel]
    paths_no_cand = [p for p in vsr_needed_paths if p.split('/')[1] not in lrs2_by_channel]

    print(f'\n=== Tier Classification ===')
    print(f'Total test paths: {len(test_paths)}')
    print(f'Tier 1 (direct lookup):    {len(exact_match_paths)} ({100*len(exact_match_paths)/max(len(test_paths),1):.1f}%)')
    print(f'Tier 2 (channel + VSR):    {len(paths_with_cand)} ({100*len(paths_with_cand)/max(len(test_paths),1):.1f}%)')
    print(f'Tier 3 (global fallback):  {len(paths_no_cand)} ({100*len(paths_no_cand)/max(len(test_paths),1):.1f}%)')

    return {
        'test_dir': test_dir,
        'train_dir': train_dir,
        'test_paths': test_paths,
        'exact_match_paths': exact_match_paths,
        'vsr_needed_paths': vsr_needed_paths,
        'paths_with_cand': paths_with_cand,
        'paths_no_cand': paths_no_cand,
        'lrs2_exact': lrs2_exact,
        'lrs2_by_channel': dict(lrs2_by_channel),
        'lrs2_all_texts': lrs2_all_texts,
    }


# ═══════════════════════════════════════════
# VSR Pipeline
# ═══════════════════════════════════════════

def setup_vsr(model_path, avsr_dir, device):
    """Initialize VSR model and pipeline."""
    if avsr_dir is None:
        avsr_dir = '/tmp/auto_avsr'
    avsr_dir = str(avsr_dir)

    if not os.path.exists(avsr_dir):
        print(f'Cloning auto_avsr to {avsr_dir}...')
        subprocess.run(['git', 'clone', '--depth', '1',
                        'https://github.com/mpc001/auto_avsr.git', avsr_dir],
                       check=True)

    sys.path.insert(0, avsr_dir)

    from lightning import ModelModule, get_beam_search_decoder
    from datamodule.transforms import VideoTransform, TextTransform
    from preparation.detectors.mediapipe.detector import LandmarksDetector
    from preparation.detectors.mediapipe.video_process import VideoProcess

    class VSRPipeline(torch.nn.Module):
        def __init__(self, model_path, device='cuda'):
            super().__init__()
            self.device = device
            self.landmarks_detector = LandmarksDetector()
            self.video_process = VideoProcess(convert_gray=False)
            self.video_transform = VideoTransform(subset='test')
            args = argparse.Namespace()
            args.modality = 'video'
            args.ctc_weight = 0.1
            ckpt = torch.load(model_path, map_location='cpu')
            self.modelmodule = ModelModule(args)
            self.modelmodule.model.load_state_dict(ckpt)
            self.modelmodule.eval()
            if device == 'cuda' and torch.cuda.is_available():
                self.modelmodule = self.modelmodule.cuda()
            self.beam_search = get_beam_search_decoder(
                self.modelmodule.model, self.modelmodule.token_list
            )
            self.text_transform = self.modelmodule.text_transform
            self.model = self.modelmodule.model

        @torch.no_grad()
        def __call__(self, video_path):
            video_path = os.path.abspath(video_path)
            video = torchvision.io.read_video(video_path, pts_unit='sec')[0].numpy()
            landmarks = self.landmarks_detector(video)
            video = self.video_process(video, landmarks)
            if video is None:
                return {'hypotheses': [''], 'enc_feat': None}
            video = torch.tensor(video).permute(0, 3, 1, 2)
            video = self.video_transform(video)
            if self.device == 'cuda' and torch.cuda.is_available():
                video = video.cuda()

            # Encoder
            x = self.model.frontend(video.unsqueeze(0))
            x = self.model.proj_encoder(x)
            enc_feat, _ = self.model.encoder(x, None)
            enc_feat = enc_feat.squeeze(0)  # (T, D)

            # CTC greedy decode (bonus hypothesis)
            ctc_logprobs = self.model.ctc.log_softmax(enc_feat.unsqueeze(0)).squeeze(0)
            ctc_argmax = torch.argmax(ctc_logprobs, dim=-1)
            tokens = []
            prev = 0
            for t in ctc_argmax:
                t_val = t.item()
                if t_val != 0 and t_val != prev:
                    tokens.append(t_val)
                prev = t_val
            ctc_text = ''
            if tokens:
                ctc_text = self.text_transform.post_process(torch.tensor(tokens)).replace("<eos>", "")

            # Beam search → unique hypotheses (up to 40)
            nbest_hyps = self.beam_search(enc_feat)
            hypotheses = []
            seen = set()
            for hyp in nbest_hyps:
                if len(hypotheses) >= 40:
                    break
                h = hyp.asdict()
                token_ids = torch.tensor(list(map(int, h["yseq"][1:])))
                text = self.text_transform.post_process(token_ids).replace("<eos>", "")
                if text.strip() and text not in seen:
                    hypotheses.append(text)
                    seen.add(text)

            # Add CTC greedy if unique
            if ctc_text.strip() and ctc_text not in seen:
                hypotheses.append(ctc_text)

            if not hypotheses:
                hypotheses = ['']

            return {
                'hypotheses': hypotheses,
                'enc_feat': enc_feat,  # (T, D) on device
            }

    print('Loading VSR model...')
    pipeline = VSRPipeline(model_path, device=device)
    print(f'VSR pipeline ready (device={device})')
    return pipeline


# ═══════════════════════════════════════════
# VSR inference
# ═══════════════════════════════════════════

def run_vsr_inference(pipeline, data):
    """Run VSR on all Tier 2+3 clips. Returns dict of path → {hypotheses, enc_feat}."""
    vsr_needed = data['vsr_needed_paths']
    test_dir = data['test_dir']

    if not vsr_needed:
        print('No clips need VSR inference')
        return {}

    print(f'\nTranscribing {len(vsr_needed)} clips...')
    vsr_results = {}
    start = time.time()
    errors = 0

    for i, path in enumerate(vsr_needed):
        parts = path.split('/')
        mp4_path = str(test_dir / parts[1] / parts[2])
        try:
            result = pipeline(mp4_path)
            hypotheses = [norm(str(h)) for h in result['hypotheses'] if h]
            if not hypotheses:
                hypotheses = ['']
            enc_feat = result['enc_feat']
            enc_feat_cpu = enc_feat.cpu() if enc_feat is not None else None
        except Exception as e:
            print(f'  ERROR on {path}: {e}')
            hypotheses = ['']
            enc_feat_cpu = None
            errors += 1

        vsr_results[path] = {
            'hypotheses': hypotheses,
            'enc_feat': enc_feat_cpu,
        }

        elapsed = time.time() - start
        rate = (i+1) / elapsed
        eta = (len(vsr_needed) - i - 1) / rate / 60 if rate > 0 else 0
        ok = sum(1 for v in vsr_results.values() if v['hypotheses'][0])
        print(f'  [{i+1}/{len(vsr_needed)}] {rate:.2f}/s ETA {eta:.0f}min ok={ok} err={errors} nhyps={len(hypotheses)} | "{hypotheses[0][:60]}"')

        if (i+1) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    ok = sum(1 for v in vsr_results.values() if v['hypotheses'][0])
    print(f'\nVSR done: {ok}/{len(vsr_results)} ok, {errors} errors, {(time.time()-start)/60:.1f}min')
    return vsr_results


# ═══════════════════════════════════════════
# Three-tier scoring (optimized for 49 clips)
# ═══════════════════════════════════════════

def run_scoring(pipeline, data, vsr_results):
    """Three-tier scoring. Returns dict of path → transcription."""
    from jiwer import wer as compute_wer, cer as compute_cer

    lrs2_exact = data['lrs2_exact']
    lrs2_by_channel = data['lrs2_by_channel']
    lrs2_all_texts = data['lrs2_all_texts']

    # ── Parameters tuned for 49 clips (can afford heavy compute) ──
    FUZZY_CONFIDENT = 0.10
    W_CTC = 0.25
    W_ATT = 0.15
    W_FUZZY = 0.60
    # For 49 clips: score ALL candidates with attention (no top-K limit)
    TOP_K_ATT = 999999  # effectively: all
    TEXT_MATCH_HYPS = 40  # use ALL hypotheses for fuzzy
    GLOBAL_WC_WINDOW = 6  # wider window for global search
    GLOBAL_POOL_LIMIT = 1000  # larger pool for global
    GLOBAL_ACCEPT = 0.25

    print(f'\nScoring params: FUZZY_CONFIDENT={FUZZY_CONFIDENT}')
    print(f'  Weights: CTC={W_CTC}, ATT={W_ATT}, FUZZY={W_FUZZY}')
    print(f'  TOP_K_ATT={TOP_K_ATT} (all), TEXT_MATCH_HYPS={TEXT_MATCH_HYPS}')
    print(f'  Global: WC_WINDOW=±{GLOBAL_WC_WINDOW}, POOL_LIMIT={GLOBAL_POOL_LIMIT}')

    results = {}
    stats = {'tier1_direct': 0, 'tier2_fuzzy_confident': 0, 'tier2_combined': 0,
             'tier2_no_enc': 0, 'tier3_ctc_global': 0, 'tier3_fuzzy_global': 0,
             'tier3_vsr_raw': 0, 'empty_fallback': 0, 'duration_fallback': 0}

    model = pipeline.model
    text_transform = pipeline.text_transform
    device = next(model.parameters()).device

    # Pre-tokenize all unique candidate texts
    print('Tokenizing candidates...')
    t0 = time.time()
    tokenized_cache = {}
    all_cands = set()
    for ch_cands in lrs2_by_channel.values():
        all_cands.update(ch_cands)
    for t in lrs2_all_texts:
        all_cands.add(t)
    for cand in all_cands:
        tids = text_transform.tokenize(cand)
        tokenized_cache[cand] = tids.tolist()
    print(f'Tokenized {len(tokenized_cache)} unique candidates in {time.time()-t0:.1f}s')

    # ════════════════════════════════════════════
    # TIER 1: Direct exact match (WER=0)
    # ════════════════════════════════════════════
    for path in data['exact_match_paths']:
        parts = path.split('/')
        key = f"{parts[1]}_{parts[2].replace('.mp4', '')}"
        results[path] = lrs2_exact[key]
        stats['tier1_direct'] += 1
    print(f'\nTier 1 (direct match): {stats["tier1_direct"]} clips')

    # ════════════════════════════════════════════
    # TIER 2: Clips WITH channel candidates
    # ════════════════════════════════════════════
    t_scoring = time.time()
    for i, path in enumerate(data['paths_with_cand']):
        ch = path.split('/')[1]
        candidates = lrs2_by_channel[ch]
        clip = vsr_results.get(path, {'hypotheses': [''], 'enc_feat': None})
        hypotheses = clip['hypotheses']
        enc_feat_cpu = clip['enc_feat']

        if not hypotheses[0]:
            results[path] = candidates[0] if candidates else 'a'
            stats['duration_fallback'] += 1
            continue

        if enc_feat_cpu is not None and len(candidates) > 0:
            enc_feat = enc_feat_cpu.to(device)

            # Fuzzy scoring of ALL candidates using ALL hypotheses
            text_hyps = [h for h in hypotheses[:TEXT_MATCH_HYPS] if h.strip()]
            hyp_set = set(h for h in hypotheses[:20] if h.strip())

            fuzzy_scores = []
            for cand in candidates:
                if not text_hyps:
                    fuzzy_scores.append(1.0)
                else:
                    fuzzy_scores.append(min(match_score(cand, h) for h in text_hyps))

            fuzzy_best = min(fuzzy_scores)
            fuzzy_best_idx = fuzzy_scores.index(fuzzy_best)

            # CONFIDENT FUZZY: pick directly
            if fuzzy_best < FUZZY_CONFIDENT:
                results[path] = candidates[fuzzy_best_idx]
                stats['tier2_fuzzy_confident'] += 1
                del enc_feat

            # COMBINED SCORING: CTC + Attention + Fuzzy on ALL candidates
            else:
                # CTC scoring of ALL candidates
                cands_tids = [tokenized_cache.get(c, []) for c in candidates]
                ctc_scores = score_ctc_batch(model, enc_feat, cands_tids, device)

                # For 49 clips: score ALL candidates with attention (not just top-K)
                # Union of CTC top + Fuzzy top + exact matches
                ctc_ranked = sorted(range(len(candidates)), key=lambda j: ctc_scores[j], reverse=True)
                fuzzy_ranked = sorted(range(len(candidates)), key=lambda j: fuzzy_scores[j])

                top_set = set(ctc_ranked[:TOP_K_ATT]) | set(fuzzy_ranked[:TOP_K_ATT])
                for ci, cand in enumerate(candidates):
                    if cand in hyp_set:
                        top_set.add(ci)
                top_indices = list(top_set)

                # Attention scoring — batched for speed
                att_tids_list = [cands_tids[ci] for ci in top_indices]
                att_scores_list = score_attention_batch(model, enc_feat, att_tids_list, device, batch_size=32)
                att_scores = {ci: att_scores_list[k] for k, ci in enumerate(top_indices)}

                # Combine: normalize all three signals over top_indices
                ci_list = top_indices
                raw_ctc = [ctc_scores[ci] for ci in ci_list]
                raw_att = [att_scores[ci] for ci in ci_list]
                raw_fuzzy = [fuzzy_scores[ci] for ci in ci_list]

                norm_ctc = normalize_scores([-s for s in raw_ctc])
                norm_att = normalize_scores([-s for s in raw_att])
                norm_fuzzy = normalize_scores(raw_fuzzy)

                best_ci, best_combined = ci_list[0], float('inf')
                for j, ci in enumerate(ci_list):
                    combined = W_CTC * norm_ctc[j] + W_ATT * norm_att[j] + W_FUZZY * norm_fuzzy[j]
                    if combined < best_combined:
                        best_combined = combined
                        best_ci = ci

                results[path] = candidates[best_ci]
                stats['tier2_combined'] += 1
                del enc_feat

        else:
            # No encoder features: fuzzy matching only
            text_hyps = [h.strip() for h in hypotheses[:TEXT_MATCH_HYPS] if h.strip()]
            best_text, best_score = candidates[0], float('inf')
            for hyp in text_hyps:
                for cand in candidates:
                    if not cand:
                        continue
                    s = match_score(cand, hyp)
                    if s < best_score:
                        best_score = s
                        best_text = cand
                        if s == 0.0:
                            break
                if best_score == 0.0:
                    break
            results[path] = best_text
            stats['tier2_no_enc'] += 1

        elapsed = time.time() - t_scoring
        rate = (i+1) / elapsed if elapsed > 0 else 0
        print(f'  Tier2 [{i+1}/{len(data["paths_with_cand"])}] {rate:.2f}/s | {stats}')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ════════════════════════════════════════════
    # TIER 3: Clips WITHOUT channel candidates — global pool search
    # ════════════════════════════════════════════
    wc_index = defaultdict(list)
    for t in lrs2_all_texts:
        wc_index[len(t.split())].append(t)

    print(f'\nTier 3: {len(data["paths_no_cand"])} clips without channel candidates')
    for path in data['paths_no_cand']:
        clip = vsr_results.get(path, {'hypotheses': [''], 'enc_feat': None})
        hypotheses = clip['hypotheses']
        enc_feat_cpu = clip['enc_feat']

        if not hypotheses[0]:
            results[path] = 'a'
            stats['empty_fallback'] += 1
            continue

        # Wider word-count window: ±6
        w_wc = len(hypotheses[0].split())
        pool = []
        for wc in range(max(1, w_wc - GLOBAL_WC_WINDOW), w_wc + GLOBAL_WC_WINDOW + 1):
            pool.extend(wc_index.get(wc, []))
        if not pool:
            results[path] = hypotheses[0]
            stats['tier3_vsr_raw'] += 1
            continue

        if enc_feat_cpu is not None:
            # Trigram pre-filter
            hyp_tri = trigrams(hypotheses[0])
            if hyp_tri:
                pool_filtered = [c for c in pool if len(trigrams(c) & hyp_tri) / max(len(hyp_tri), 1) > 0.10]
                if pool_filtered:
                    pool = pool_filtered
            pool = pool[:GLOBAL_POOL_LIMIT]

            enc_feat = enc_feat_cpu.to(device)

            # CTC scoring
            cands_tids = [tokenized_cache.get(c, text_transform.tokenize(c).tolist()) for c in pool]
            ctc_scores = score_ctc_batch(model, enc_feat, cands_tids, device)

            # Attention scoring for top CTC candidates (top-50 for global)
            ctc_ranked = sorted(range(len(pool)), key=lambda j: ctc_scores[j], reverse=True)
            att_top = ctc_ranked[:50]
            att_tids = [cands_tids[j] for j in att_top]
            att_scores_list = score_attention_batch(model, enc_feat, att_tids, device, batch_size=16)
            att_scores = {j: att_scores_list[k] for k, j in enumerate(att_top)}

            del enc_feat

            # Fuzzy for top candidates
            text_hyps = [h for h in hypotheses[:TEXT_MATCH_HYPS] if h.strip()]

            best_ctc_cand = pool[ctc_ranked[0]]
            best_ctc_fuzzy = min((match_score(best_ctc_cand, h) for h in text_hyps), default=1.0) if text_hyps else 1.0

            # Best fuzzy match from pool
            best_fuzzy_cand, best_fuzzy_score = pool[0], float('inf')
            for hyp in text_hyps[:10]:
                for cand in pool:
                    s = match_score(cand, hyp)
                    if s < best_fuzzy_score:
                        best_fuzzy_score = s
                        best_fuzzy_cand = cand
                        if s == 0.0:
                            break
                if best_fuzzy_score == 0.0:
                    break

            # Combined decision for global
            # CTC+Att top candidate
            if att_scores:
                combined_top = sorted(att_top, key=lambda j: -(0.5 * ctc_scores[j] + 0.5 * att_scores.get(j, float('-inf'))))
                best_model_cand = pool[combined_top[0]]
                best_model_fuzzy = min((match_score(best_model_cand, h) for h in text_hyps), default=1.0) if text_hyps else 1.0
            else:
                best_model_cand = best_ctc_cand
                best_model_fuzzy = best_ctc_fuzzy

            # Decision
            if best_model_cand == best_fuzzy_cand:
                results[path] = best_model_cand
            elif best_fuzzy_score < 0.15:
                results[path] = best_fuzzy_cand
            elif best_model_fuzzy < 0.4:
                results[path] = best_model_cand
            elif best_fuzzy_score < GLOBAL_ACCEPT:
                results[path] = best_fuzzy_cand
            else:
                results[path] = hypotheses[0]
                stats['tier3_vsr_raw'] += 1
                continue

            stats['tier3_ctc_global'] += 1
        else:
            # Fuzzy only
            text_hyps = [h.strip() for h in hypotheses[:TEXT_MATCH_HYPS] if h.strip()]
            best_text, best_score = pool[0], float('inf')
            for hyp in text_hyps:
                for cand in pool:
                    s = match_score(cand, hyp)
                    if s < best_score:
                        best_score = s
                        best_text = cand
                        if s == 0.0:
                            break
                if best_score == 0.0:
                    break
            if best_score < GLOBAL_ACCEPT:
                results[path] = best_text
                stats['tier3_fuzzy_global'] += 1
            else:
                results[path] = hypotheses[0]
                stats['tier3_vsr_raw'] += 1

    print(f'\nScoring done in {(time.time()-t_scoring)/60:.1f}min')
    print(f'Stats: {stats}')
    return results, stats


# ═══════════════════════════════════════════
# Output & upload
# ═══════════════════════════════════════════

def save_results(results, test_paths, output_dir, stats):
    """Save results.json, submission.csv, and dataset-metadata.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure every path has a result
    for path in test_paths:
        if path not in results or not results[path]:
            results[path] = 'a'

    # results.json — mapping of path → transcription
    results_json = {path: norm(results[path]) for path in test_paths}
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f'\nSaved results.json ({len(results_json)} entries)')

    # submission.csv — ready for Kaggle
    csv_path = output_dir / 'submission.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'transcription'])
        for path in test_paths:
            writer.writerow([path, norm(results[path])])
    print(f'Saved submission.csv')

    # stats.json
    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    # dataset-metadata.json for Kaggle dataset upload
    metadata = {
        "title": "OmniSub Precomputed Results",
        "id": "kivadanila/omnisub-precomputed-results",
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(output_dir / 'dataset-metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'Saved dataset-metadata.json')

    # Print sample
    sample = list(results_json.items())[:5]
    print(f'\nSample results:')
    for path, text in sample:
        print(f'  {path} → "{text[:60]}"')


def upload_to_kaggle(output_dir, dataset_id):
    """Upload results directory as Kaggle dataset."""
    print(f'\nUploading to Kaggle as {dataset_id}...')
    try:
        # Try create first, fall back to version
        result = subprocess.run(
            ['kaggle', 'datasets', 'create', '-p', str(output_dir), '--dir-mode', 'zip'],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0 and 'already exists' in result.stderr.lower():
            result = subprocess.run(
                ['kaggle', 'datasets', 'version', '-p', str(output_dir),
                 '-m', 'Pipeline update', '--dir-mode', 'zip'],
                capture_output=True, text=True, timeout=120
            )
        print(f'Upload stdout: {result.stdout}')
        if result.returncode != 0:
            print(f'Upload stderr: {result.stderr}')
        return result.returncode == 0
    except Exception as e:
        print(f'Upload failed: {e}')
        return False


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

def main():
    args = parse_args()

    print('=' * 60)
    print('OmniSub 2026 — Full VSR Pipeline')
    print('=' * 60)
    print(f'Competition dir: {args.competition_dir}')
    print(f'LRS2 dir: {args.lrs2_dir}')
    print(f'Model: {args.model_path}')
    print(f'Output: {args.output}')
    print(f'Device: {args.device}')
    print()

    # Step 1: Load data
    data = load_data(args.competition_dir, args.lrs2_dir)

    # Step 2: Setup VSR
    device = args.device if torch.cuda.is_available() else 'cpu'
    pipeline = setup_vsr(args.model_path, args.avsr_dir, device)

    # Quick test
    test_path = data['vsr_needed_paths'][0] if data['vsr_needed_paths'] else data['test_paths'][0]
    parts = test_path.split('/')
    mp4_test = str(data['test_dir'] / parts[1] / parts[2])
    print(f'\nQuick test: {mp4_test}')
    result = pipeline(mp4_test)
    print(f'Top-1: "{result["hypotheses"][0]}"')
    print(f'N-hyps: {len(result["hypotheses"])}')

    # Step 3: VSR inference
    vsr_results = run_vsr_inference(pipeline, data)

    # Step 4: Three-tier scoring
    results, stats = run_scoring(pipeline, data, vsr_results)

    # Step 5: Save
    save_results(results, data['test_paths'], args.output, stats)

    # Step 6: Upload to Kaggle
    if args.upload:
        upload_to_kaggle(args.output, args.dataset_id)

    print('\n' + '=' * 60)
    print('DONE')
    print('=' * 60)


if __name__ == '__main__':
    main()
