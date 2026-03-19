#!/usr/bin/env python3
"""
OmniSub 2026 — Pipeline v2 for new 49-clip test set.

New test format: 00000.mp4 - 00048.mp4 (no video_id/clip_id structure).
All clips go through: VSR inference → Global pool CTC + Attention + Fuzzy scoring.

Usage:
  python3 run_pipeline_v2.py \
    --competition-dir ~/data/competition \
    --lrs2-dir ~/data/lrs2-texts \
    --model-path ~/data/vsr-model/vsr_model.pth \
    --output ~/results
"""

import os, sys, csv, re, json, time, argparse, subprocess
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchvision
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description='OmniSub 2026 VSR Pipeline v2 (49 clips)')
    p.add_argument('--competition-dir', required=True)
    p.add_argument('--lrs2-dir', required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--avsr-dir', default=None)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--upload', action='store_true')
    return p.parse_args()


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
    from jiwer import wer as compute_wer, cer as compute_cer
    try:
        w = compute_wer(ref, hyp)
        c = compute_cer(ref, hyp)
        return 0.4 * w + 0.6 * c
    except:
        return 1.0


@torch.no_grad()
def score_ctc_batch(model, enc_feat, candidates_token_ids, device, batch_size=64):
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
        losses = F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                            blank=0, reduction='none', zero_infinity=True)
        for k, (j, tids) in enumerate(valid):
            batch_scores[j] = -losses[k].item() / max(len(tids), 1)
        all_scores.extend(batch_scores)
    return all_scores


@torch.no_grad()
def score_attention_batch(model, enc_feat, candidates_token_ids, device, batch_size=16):
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
        max_len = max(len(tids) for _, tids in valid) + 1
        tgt_in = torch.full((len(valid), max_len), 0, dtype=torch.long, device=device)
        tgt_out_list = []
        for k, (j, tids) in enumerate(valid):
            seq = [sos] + tids
            tgt_in[k, :len(seq)] = torch.tensor(seq, device=device)
            tgt_out_list.append(tids + [eos])
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
    if not scores:
        return scores
    mn, mx = min(scores), max(scores)
    if mx - mn < 1e-9:
        return [0.5] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def trigrams(text):
    return set(text[i:i+3] for i in range(len(text)-2))


# ═══════════════════════════════════════════
# Data loading (new format)
# ═══════════════════════════════════════════

def load_data(competition_dir, lrs2_dir):
    competition_dir = Path(competition_dir)
    lrs2_dir = Path(lrs2_dir)

    # Test dir: files directly in test/
    test_dir = competition_dir / 'test'
    if not test_dir.exists():
        # Maybe test files are directly in competition_dir
        test_dir = competition_dir

    # Train dir
    train_dir = competition_dir / 'train'
    if (train_dir / 'train').exists():
        train_dir = train_dir / 'train'

    # Sample submission
    sample_sub = competition_dir / 'sample_submission.csv'
    assert sample_sub.exists(), f'sample_submission.csv not found in {competition_dir}'

    print(f'TEST_DIR: {test_dir}')
    print(f'TRAIN_DIR: {train_dir}')
    print(f'SAMPLE_SUB: {sample_sub}')

    # Load test paths and pre-filled answers
    test_paths = []
    prefilled = {}
    with open(sample_sub) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            path = row[0]
            test_paths.append(path)
            if len(row) > 1 and row[1].strip():
                prefilled[path] = norm(row[1].strip())
    print(f'Test: {len(test_paths)} clips, {len(prefilled)} pre-filled')

    # Load ALL LRS2 texts (no channel grouping needed — global search)
    all_texts = set()
    for fname in ['lrs2_train_text.txt', 'lrs2_test_text.txt', 'lrs2_val_text.txt']:
        fpath = lrs2_dir / fname
        if not fpath.exists():
            continue
        with open(fpath) as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) < 2:
                    continue
                all_texts.add(norm(parts[1]))
    print(f'LRS2: {len(all_texts)} unique texts')

    # Add training transcripts
    if train_dir.exists():
        train_added = 0
        for ch_name in os.listdir(train_dir):
            ch_dir = train_dir / ch_name
            if not ch_dir.is_dir():
                continue
            for txt_file in ch_dir.glob('*.txt'):
                with open(txt_file) as f:
                    first_line = f.readline().strip()
                if first_line.startswith('Text:'):
                    text = norm(first_line[5:].strip())
                    if text:
                        all_texts.add(text)
                        train_added += 1
        print(f'Train: added {train_added} texts')

    all_texts = list(all_texts)
    print(f'Total candidate pool: {len(all_texts)} unique texts')

    # Build word-count index for fast filtering
    wc_index = defaultdict(list)
    for t in all_texts:
        wc_index[len(t.split())].append(t)

    return {
        'test_dir': test_dir,
        'test_paths': test_paths,
        'prefilled': prefilled,
        'all_texts': all_texts,
        'wc_index': dict(wc_index),
    }


# ═══════════════════════════════════════════
# VSR Pipeline setup
# ═══════════════════════════════════════════

def setup_vsr(model_path, avsr_dir, device):
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
            x = self.model.frontend(video.unsqueeze(0))
            x = self.model.proj_encoder(x)
            enc_feat, _ = self.model.encoder(x, None)
            enc_feat = enc_feat.squeeze(0)
            # CTC greedy
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
            # Beam search
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
            if ctc_text.strip() and ctc_text not in seen:
                hypotheses.append(ctc_text)
            if not hypotheses:
                hypotheses = ['']
            return {'hypotheses': hypotheses, 'enc_feat': enc_feat}

    print('Loading VSR model...')
    pipeline = VSRPipeline(model_path, device=device)
    print(f'VSR pipeline ready (device={device})')
    return pipeline


# ═══════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════

def main():
    args = parse_args()

    print('=' * 60)
    print('OmniSub 2026 — Pipeline v2 (49 clips, global search)')
    print('=' * 60)

    # Load data
    data = load_data(args.competition_dir, args.lrs2_dir)

    # Setup VSR
    device = args.device if torch.cuda.is_available() else 'cpu'
    pipeline = setup_vsr(args.model_path, args.avsr_dir, device)
    model = pipeline.model
    text_transform = pipeline.text_transform

    # Pre-tokenize ALL candidate texts
    print('\nTokenizing candidate pool...')
    t0 = time.time()
    tokenized_cache = {}
    for cand in data['all_texts']:
        tids = text_transform.tokenize(cand)
        tokenized_cache[cand] = tids.tolist()
    print(f'Tokenized {len(tokenized_cache)} texts in {time.time()-t0:.1f}s')

    # Scoring parameters (tuned for 49 clips with heavy compute)
    W_CTC = 0.25
    W_ATT = 0.15
    W_FUZZY = 0.60
    WC_WINDOW = 6       # ±6 words for word-count filter
    TRIGRAM_THRESH = 0.10
    POOL_LIMIT = 2000    # max candidates after filtering
    ATT_TOP = 100        # attention-score top CTC candidates
    TEXT_MATCH_HYPS = 40 # use all hypotheses for fuzzy

    print(f'Weights: CTC={W_CTC}, ATT={W_ATT}, FUZZY={W_FUZZY}')
    print(f'WC_WINDOW=±{WC_WINDOW}, POOL_LIMIT={POOL_LIMIT}, ATT_TOP={ATT_TOP}')

    results = {}
    stats = {'prefilled': 0, 'combined': 0, 'fuzzy_only': 0, 'vsr_raw': 0, 'empty': 0}

    # Process each clip
    total_start = time.time()
    for i, path in enumerate(data['test_paths']):
        clip_start = time.time()

        # Pre-filled answers: keep as-is
        if path in data['prefilled']:
            results[path] = data['prefilled'][path]
            stats['prefilled'] += 1
            print(f'[{i+1}/{len(data["test_paths"])}] {path}: PREFILLED "{results[path][:50]}"')
            continue

        # VSR inference
        mp4_path = str(data['test_dir'] / path)
        try:
            result = pipeline(mp4_path)
            hypotheses = [norm(str(h)) for h in result['hypotheses'] if h]
            if not hypotheses:
                hypotheses = ['']
            enc_feat = result['enc_feat']
        except Exception as e:
            print(f'  ERROR on {path}: {e}')
            hypotheses = ['']
            enc_feat = None

        if not hypotheses[0]:
            results[path] = 'a'
            stats['empty'] += 1
            print(f'[{i+1}/{len(data["test_paths"])}] {path}: EMPTY (VSR failed)')
            continue

        # Build candidate pool: word-count filter + trigram filter
        w_wc = len(hypotheses[0].split())
        pool = []
        for wc in range(max(1, w_wc - WC_WINDOW), w_wc + WC_WINDOW + 1):
            pool.extend(data['wc_index'].get(wc, []))

        if not pool:
            results[path] = hypotheses[0]
            stats['vsr_raw'] += 1
            print(f'[{i+1}/{len(data["test_paths"])}] {path}: VSR_RAW (no pool) "{hypotheses[0][:50]}"')
            continue

        # Trigram pre-filter
        hyp_tri = trigrams(hypotheses[0])
        if hyp_tri and len(pool) > POOL_LIMIT:
            pool_filtered = [c for c in pool if len(trigrams(c) & hyp_tri) / max(len(hyp_tri), 1) > TRIGRAM_THRESH]
            if pool_filtered:
                pool = pool_filtered
        if len(pool) > POOL_LIMIT:
            pool = pool[:POOL_LIMIT]

        text_hyps = [h for h in hypotheses[:TEXT_MATCH_HYPS] if h.strip()]

        if enc_feat is not None:
            enc_feat_dev = enc_feat if enc_feat.device.type == device else enc_feat.to(device)

            # CTC scoring of entire pool
            cands_tids = [tokenized_cache.get(c, []) for c in pool]
            ctc_scores = score_ctc_batch(model, enc_feat_dev, cands_tids, device)

            # Attention scoring of top CTC candidates
            ctc_ranked = sorted(range(len(pool)), key=lambda j: ctc_scores[j], reverse=True)
            att_indices = ctc_ranked[:ATT_TOP]
            att_tids = [cands_tids[j] for j in att_indices]
            att_scores_list = score_attention_batch(model, enc_feat_dev, att_tids, device, batch_size=32)
            att_scores = {j: att_scores_list[k] for k, j in enumerate(att_indices)}

            # Fuzzy scoring of top CTC candidates + top fuzzy candidates
            fuzzy_scores = {}
            # Score top CTC candidates
            for j in ctc_ranked[:200]:
                if text_hyps:
                    fuzzy_scores[j] = min(match_score(pool[j], h) for h in text_hyps)
                else:
                    fuzzy_scores[j] = 1.0

            # Also find best fuzzy matches from full pool (quick scan)
            best_fuzzy_cand_idx = None
            best_fuzzy_score = float('inf')
            for hyp in text_hyps[:5]:
                for j, cand in enumerate(pool):
                    if j not in fuzzy_scores:
                        fuzzy_scores[j] = min(match_score(cand, h) for h in text_hyps[:3])
                    if fuzzy_scores[j] < best_fuzzy_score:
                        best_fuzzy_score = fuzzy_scores[j]
                        best_fuzzy_cand_idx = j
                    if best_fuzzy_score == 0.0:
                        break
                if best_fuzzy_score == 0.0:
                    break

            # Combined scoring on candidates that have all three scores
            scored_indices = [j for j in att_indices if j in fuzzy_scores]

            if scored_indices:
                raw_ctc = [ctc_scores[j] for j in scored_indices]
                raw_att = [att_scores.get(j, float('-inf')) for j in scored_indices]
                raw_fuzzy = [fuzzy_scores[j] for j in scored_indices]

                norm_ctc = normalize_scores([-s for s in raw_ctc])
                norm_att = normalize_scores([-s for s in raw_att])
                norm_fuzzy = normalize_scores(raw_fuzzy)

                best_j, best_combined = scored_indices[0], float('inf')
                for k, j in enumerate(scored_indices):
                    combined = W_CTC * norm_ctc[k] + W_ATT * norm_att[k] + W_FUZZY * norm_fuzzy[k]
                    if combined < best_combined:
                        best_combined = combined
                        best_j = j

                # Also consider best fuzzy if it's very good
                if best_fuzzy_score < 0.10 and best_fuzzy_cand_idx is not None:
                    results[path] = pool[best_fuzzy_cand_idx]
                else:
                    results[path] = pool[best_j]
                stats['combined'] += 1
            else:
                # Fallback to fuzzy best
                if best_fuzzy_cand_idx is not None and best_fuzzy_score < 0.5:
                    results[path] = pool[best_fuzzy_cand_idx]
                    stats['fuzzy_only'] += 1
                else:
                    results[path] = hypotheses[0]
                    stats['vsr_raw'] += 1

            del enc_feat_dev
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        else:
            # No encoder features: fuzzy only
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
            if best_score < 0.5:
                results[path] = best_text
                stats['fuzzy_only'] += 1
            else:
                results[path] = hypotheses[0]
                stats['vsr_raw'] += 1

        elapsed = time.time() - clip_start
        print(f'[{i+1}/{len(data["test_paths"])}] {path}: {elapsed:.1f}s pool={len(pool)} hyps={len(hypotheses)} | "{results[path][:60]}"')

    total_time = time.time() - total_start
    print(f'\n{"="*60}')
    print(f'Done in {total_time/60:.1f}min')
    print(f'Stats: {stats}')
    print(f'{"="*60}')

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure every path has a result
    for path in data['test_paths']:
        if path not in results or not results[path]:
            results[path] = 'a'

    # results.json
    results_json = {path: norm(results[path]) for path in data['test_paths']}
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f'\nSaved results.json ({len(results_json)} entries)')

    # submission.csv
    with open(output_dir / 'submission.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'transcription'])
        for path in data['test_paths']:
            writer.writerow([path, norm(results[path])])
    print(f'Saved submission.csv')

    # stats.json
    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    # dataset-metadata.json for Kaggle upload
    with open(output_dir / 'dataset-metadata.json', 'w') as f:
        json.dump({
            "title": "OmniSub Precomputed Results",
            "id": "kivadanila/omnisub-precomputed-results",
            "licenses": [{"name": "CC0-1.0"}]
        }, f, indent=2)

    # Print all results
    print(f'\n=== ALL RESULTS ===')
    for path in data['test_paths']:
        print(f'  {path} → "{results_json[path][:70]}"')

    # Upload
    if args.upload:
        print(f'\nUploading to Kaggle...')
        try:
            r = subprocess.run(['kaggle', 'datasets', 'create', '-p', str(output_dir), '--dir-mode', 'zip'],
                               capture_output=True, text=True, timeout=120)
            if r.returncode != 0 and 'already exists' in r.stderr.lower():
                r = subprocess.run(['kaggle', 'datasets', 'version', '-p', str(output_dir),
                                    '-m', 'v2 pipeline results', '--dir-mode', 'zip'],
                                   capture_output=True, text=True, timeout=120)
            print(f'Upload: {r.stdout}')
            if r.returncode != 0:
                print(f'Upload error: {r.stderr}')
        except Exception as e:
            print(f'Upload failed: {e}')

    print('\nDONE')


if __name__ == '__main__':
    main()
