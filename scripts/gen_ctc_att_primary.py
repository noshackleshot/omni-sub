#!/usr/bin/env python3
"""Generate ctc-att-primary notebook."""
import json

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.split("\n")})

def code(source):
    lines = source.split("\n")
    # Add \n to all lines except last
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src})

md("# OmniSub 2026 — CTC-ATT Primary Scoring")

# ── Cell 1: Install & Setup ──
code(r"""# ── Cell 1: Install & clone auto_avsr ──
import os, subprocess, glob

!pip install -q jiwer sentencepiece scikit-image av
!pip install -q mediapipe==0.10.14
!pip install -q 'Pillow>=10.0'

import mediapipe as mp
print(f'mediapipe {mp.__version__}, solutions={hasattr(mp, "solutions")}')

AVSR_DIR = '/kaggle/working/auto_avsr'
if not os.path.exists(AVSR_DIR):
    !git clone --depth 1 https://github.com/mpc001/auto_avsr.git {AVSR_DIR}

MODEL_PATH = None
for c in glob.glob('/kaggle/input/**/vsr_model.pth', recursive=True):
    if os.path.getsize(c) > 1e6:
        MODEL_PATH = c
        break
if not MODEL_PATH:
    for c in glob.glob('/kaggle/input/**/*.pth', recursive=True):
        if os.path.getsize(c) > 1e8:
            MODEL_PATH = c
            break

print(f'Model: {MODEL_PATH} ({os.path.getsize(MODEL_PATH)/1e6:.0f} MB)' if MODEL_PATH else 'Model NOT FOUND')
print('Setup OK')""")

# ── Cell 2: Load Data ──
code(r"""# ── Cell 2: Discover paths, load LRS2 + competition train texts ──
import os, csv, re, json, subprocess, time, sys, argparse
from pathlib import Path
from collections import defaultdict

def find_file(root, name, max_depth=4):
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.replace(root, '').count(os.sep)
        if depth >= max_depth:
            dirnames.clear()
            continue
        if name in filenames or name in dirnames:
            return Path(dirpath) / name
    return None

SAMPLE_SUB = find_file('/kaggle/input', 'sample_submission.csv')
LRS2_DIR = find_file('/kaggle/input', 'lrs2_train_text.txt').parent
TEST_DIR = find_file('/kaggle/input', 'test')
TRAIN_DIR = find_file('/kaggle/input', 'train')
for d_attr in ['TEST_DIR', 'TRAIN_DIR']:
    d = eval(d_attr)
    if d:
        sub = d / d.name
        if sub.exists(): exec(f'{d_attr} = sub')
OUTPUT = Path('/kaggle/working/submission.csv')
print(f'TEST: {TEST_DIR}\nTRAIN: {TRAIN_DIR}\nLRS2: {LRS2_DIR}')

def norm(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Load LRS2 texts ──
lrs2_by_channel = defaultdict(list)
for fname in ['lrs2_train_text.txt', 'lrs2_test_text.txt', 'lrs2_val_text.txt']:
    fpath = LRS2_DIR / fname
    if not fpath.exists(): continue
    with open(fpath) as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) < 2: continue
            ch = parts[0].rsplit('_', 1)[0]
            lrs2_by_channel[ch].append(norm(parts[1]))
for ch in lrs2_by_channel:
    lrs2_by_channel[ch] = list(dict.fromkeys(lrs2_by_channel[ch]))
print(f'LRS2: {sum(len(v) for v in lrs2_by_channel.values())} texts, {len(lrs2_by_channel)} channels')

# ── Load competition train texts ──
train_by_channel = defaultdict(list)
if TRAIN_DIR and TRAIN_DIR.exists():
    for ch_name in os.listdir(TRAIN_DIR):
        ch_dir = TRAIN_DIR / ch_name
        if not ch_dir.is_dir(): continue
        for txt_file in ch_dir.glob('*.txt'):
            with open(txt_file) as f:
                line = f.readline().strip()
            if not line.startswith('Text:'): continue
            text = norm(line[5:].strip())
            if text:
                train_by_channel[ch_name].append(text)
    for ch in train_by_channel:
        train_by_channel[ch] = list(dict.fromkeys(train_by_channel[ch]))
    print(f'Train: {sum(len(v) for v in train_by_channel.values())} texts, {len(train_by_channel)} channels')

# ── Merge train texts into candidate pool ──
merged_count = 0
new_channels = 0
for ch, texts in train_by_channel.items():
    if ch not in lrs2_by_channel:
        new_channels += 1
    existing = set(lrs2_by_channel.get(ch, []))
    for t in texts:
        if t not in existing:
            lrs2_by_channel[ch].append(t)
            existing.add(t)
            merged_count += 1
print(f'Merged {merged_count} new texts ({new_channels} new channels) into candidate pool')

# ── Rebuild global pool ──
lrs2_all_texts = list(set(t for texts in lrs2_by_channel.values() for t in texts))
print(f'Total pool: {len(lrs2_all_texts)} unique texts, {len(lrs2_by_channel)} channels')

# ── Load test paths and compute coverage ──
test_paths = []
with open(SAMPLE_SUB) as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader: test_paths.append(row[0])
paths_with_cand = [p for p in test_paths if p.split('/')[1] in lrs2_by_channel]
paths_no_cand = [p for p in test_paths if p.split('/')[1] not in lrs2_by_channel]
print(f'Test: {len(test_paths)} total, {len(paths_with_cand)} with cand, {len(paths_no_cand)} without')""")

# ── Cell 3: VSR Pipeline ──
code(r"""# ── Cell 3: VSR Pipeline (n-best=40 + CTC greedy + enc_feat float16) ──
import torch, torchvision, numpy as np
import torch.nn.functional as F

AVSR_DIR = '/kaggle/working/auto_avsr'
sys.path.insert(0, AVSR_DIR)

VSR_OK = False
pipeline = None

if MODEL_PATH and os.path.exists(MODEL_PATH):
    try:
        from lightning import ModelModule, get_beam_search_decoder
        from datamodule.transforms import VideoTransform, TextTransform
        from preparation.detectors.mediapipe.detector import LandmarksDetector
        from preparation.detectors.mediapipe.video_process import VideoProcess

        class VSRPipeline(torch.nn.Module):
            def __init__(self, model_path, device='cuda', n_best=40):
                super().__init__()
                self.device = device
                self.n_best = n_best
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
                _read = torchvision.io.read_video(video_path, pts_unit='sec')
                video = _read[0].numpy()
                fps = _read[2].get('video_fps', 25.0) if len(_read) > 2 else 25.0
                duration = len(video) / fps if fps > 0 else None

                landmarks = self.landmarks_detector(video)
                video = self.video_process(video, landmarks)
                if video is None:
                    return {'hypotheses': [''], 'enc_feat': None, 'duration': duration}

                video = torch.tensor(video).permute(0, 3, 1, 2)
                video = self.video_transform(video)
                if self.device == 'cuda' and torch.cuda.is_available():
                    video = video.cuda()

                # Run encoder
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

                # Beam search -> unique n-best hypotheses
                nbest_hyps = self.beam_search(enc_feat)
                hypotheses = []
                seen = set()
                for hyp in nbest_hyps:
                    if len(hypotheses) >= self.n_best:
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

                return {
                    'hypotheses': hypotheses if hypotheses else [''],
                    'enc_feat': enc_feat.cpu().half(),  # float16 to save memory
                    'duration': duration
                }

        print('Loading VSR model...')
        pipeline = VSRPipeline(MODEL_PATH, device='cuda' if torch.cuda.is_available() else 'cpu', n_best=40)
        print('VSR pipeline ready')

        parts = test_paths[0].split('/')
        mp4_test = str(TEST_DIR / parts[1] / parts[2])
        print(f'Test: {mp4_test}')
        result = pipeline(mp4_test)
        print(f'Top-1: "{result["hypotheses"][0]}"')
        print(f'N-hyps: {len(result["hypotheses"])}, enc_feat: {result["enc_feat"].shape}')
        VSR_OK = True

    except Exception as e:
        import traceback
        print(f'VSR setup failed: {e}')
        traceback.print_exc()
        VSR_OK = False
else:
    print('No model — skipping VSR')

print(f'\nVSR_OK: {VSR_OK}')""")

# ── Cell 4: Transcribe ──
code(r"""# ── Cell 4: Transcribe all test videos (n-best=40, enc_feat float16) ──
vsr_results = {}
if VSR_OK:
    print(f'Transcribing {len(test_paths)} videos (n-best=40)...')
    start = time.time()
    errors = 0
    for i, path in enumerate(test_paths):
        parts = path.split('/')
        mp4_path = str(TEST_DIR / parts[1] / parts[2])
        try:
            result = pipeline(mp4_path)
            hypotheses = [norm(str(h)) for h in result['hypotheses'] if h]
            vsr_results[path] = {
                'hypotheses': hypotheses if hypotheses else [''],
                'enc_feat': result['enc_feat'],  # float16 on CPU
                'duration': result.get('duration')
            }
        except:
            vsr_results[path] = {'hypotheses': [''], 'enc_feat': None, 'duration': None}
            errors += 1
        if (i+1) % 100 == 0 or i == 0:
            elapsed = time.time() - start
            rate = (i+1) / elapsed
            eta = (len(test_paths) - i - 1) / rate / 60 if rate > 0 else 0
            ok = sum(1 for v in vsr_results.values() if v['hypotheses'][0])
            hyps = vsr_results.get(path, {'hypotheses': ['']})
            print(f'  [{i+1}/{len(test_paths)}] {rate:.2f}/s ETA {eta:.0f}min ok={ok} err={errors} nhyps={len(hyps["hypotheses"])} | "{hyps["hypotheses"][0][:50]}"')
        if (i+1) % 500 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    ok = sum(1 for v in vsr_results.values() if v['hypotheses'][0])
    nhyps_avg = sum(len(v['hypotheses']) for v in vsr_results.values()) / max(len(vsr_results), 1)
    enc_mem = sum(v['enc_feat'].nelement() * v['enc_feat'].element_size()
                  for v in vsr_results.values() if v['enc_feat'] is not None) / 1e6
    print(f'\nDone: {ok}/{len(vsr_results)} ok, {errors} err, avg_hyps={nhyps_avg:.1f}, enc_mem={enc_mem:.0f}MB, {(time.time()-start)/60:.1f}min')
else:
    print('VSR not available')""")

# ── Cell 5: Scoring Functions ──
code(r"""# ── Cell 5: Scoring functions — CTC batch, Attention batch, Rank normalize ──
from jiwer import wer as compute_wer, cer as compute_cer

def match_score(ref, hyp):
    # Combined WER+CER score (lower = better)
    try:
        w = compute_wer(ref, hyp)
        c = compute_cer(ref, hyp)
        return 0.4 * w + 0.6 * c
    except:
        return 1.0

@torch.no_grad()
def score_ctc_batch(model, enc_feat, candidates_token_ids, device, batch_size=128):
    # Batch CTC scoring. enc_feat: (T, D) single clip, float.
    # Returns list of scores (higher = better match).
    T = enc_feat.size(0)
    ctc_logprobs = model.ctc.log_softmax(enc_feat.unsqueeze(0)).squeeze(0)  # (T, V)

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
def score_attention_batch(model, enc_feat, candidates_token_ids, device, batch_size=32):
    # Batch attention (teacher-forcing) scoring. enc_feat: (T, D) single clip, float.
    # Returns list of scores (higher = better match).
    sos = model.sos
    eos = model.eos
    ignore_id = model.ignore_id  # -1

    all_scores = []
    for batch_start in range(0, len(candidates_token_ids), batch_size):
        batch = candidates_token_ids[batch_start:batch_start + batch_size]
        batch_scores = [float('-inf')] * len(batch)

        valid = [(j, tids) for j, tids in enumerate(batch) if len(tids) > 0]
        if not valid:
            all_scores.extend(batch_scores)
            continue

        # Build padded decoder input/output using ESPnet conventions
        # ys_in: [SOS, t1, t2, ...], padded with EOS (same as add_sos_eos)
        # ys_out: [t1, t2, ..., EOS], padded with ignore_id (-1)
        max_len = max(len(tids) for _, tids in valid) + 1  # +1 for SOS/EOS
        ys_in = torch.full((len(valid), max_len), eos, dtype=torch.long, device=device)
        ys_out = torch.full((len(valid), max_len), ignore_id, dtype=torch.long, device=device)

        for k, (j, tids) in enumerate(valid):
            ys_in[k, 0] = sos
            for t_idx, tid in enumerate(tids):
                ys_in[k, t_idx + 1] = tid
            # ys_out: [t1, ..., tn, EOS, ignore_id, ...]
            for t_idx, tid in enumerate(tids):
                ys_out[k, t_idx] = tid
            ys_out[k, len(tids)] = eos

        # Build mask: causal + padding (using target_mask logic)
        # ys_in != ignore_id gives padding mask, then AND with subsequent_mask
        ys_mask = (ys_in != ignore_id)  # all True since we pad with EOS, not ignore_id
        # But we need to mask positions beyond each sequence's actual length
        for k, (j, tids) in enumerate(valid):
            seq_len = len(tids) + 1  # SOS + tokens
            ys_mask[k, seq_len:] = False
        causal = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool, device=device)).unsqueeze(0)
        tgt_mask = ys_mask.unsqueeze(-2) & causal  # (B, max_len, max_len)

        # Expand encoder features
        memory = enc_feat.unsqueeze(0).expand(len(valid), -1, -1)

        # Forward through decoder
        logits, _ = model.decoder(ys_in, tgt_mask, memory, None)  # (B, max_len, odim)
        log_probs = F.log_softmax(logits, dim=-1)

        # Score: mean log-prob over non-padding target positions
        for k, (j, tids) in enumerate(valid):
            n = len(tids) + 1  # number of target tokens (including EOS)
            score = 0.0
            for pos in range(n):
                target_id = ys_out[k, pos].item()
                if target_id >= 0:
                    score += log_probs[k, pos, target_id].item()
            batch_scores[j] = score / max(n, 1)

        all_scores.extend(batch_scores)
    return all_scores

def rank_normalize(scores, higher_is_better=True):
    # Rank-based normalization to [0, 1]. Best = 1.0, worst = 0.0.
    n = len(scores)
    if n <= 1:
        return [1.0] * n
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=higher_is_better)
    result = [0.0] * n
    for rank, idx in enumerate(ranked):
        result[idx] = 1.0 - rank / (n - 1)
    return result

print('Scoring functions ready')""")

# ── Cell 6: Score & Select ──
code(r"""# ── Cell 6: CTC-ATT Primary scoring — NEVER fall back to raw VSR ──

W_ATT = 0.55
W_CTC = 0.35
W_FUZZY = 0.10
TEXT_MATCH_HYPS = 10  # hypotheses used for fuzzy matching

HAS_VSR = bool(vsr_results) and sum(1 for v in vsr_results.values() if v['hypotheses'][0]) > 100
print(f'HAS_VSR: {HAS_VSR}, W_ATT={W_ATT}, W_CTC={W_CTC}, W_FUZZY={W_FUZZY}')

results = {}
stats = {'att_ctc_fuzzy': 0, 'fuzzy_only': 0, 'global_ctc': 0, 'empty': 0, 'duration': 0}

if HAS_VSR:
    model = pipeline.model
    text_transform = pipeline.text_transform
    device = next(model.parameters()).device
    start = time.time()

    # Pre-tokenize all unique candidates
    print('Tokenizing candidates...')
    tokenized_cache = {}
    all_cands = set()
    for ch_cands in lrs2_by_channel.values():
        all_cands.update(ch_cands)
    for t in lrs2_all_texts:
        all_cands.add(t)
    for cand in all_cands:
        tids = text_transform.tokenize(cand)
        tokenized_cache[cand] = tids.tolist()
    print(f'Tokenized {len(tokenized_cache)} unique candidates in {time.time()-start:.1f}s')

    # ════════════════════════════════════════════════════
    # Clips WITH channel candidates: CTC + ATT + Fuzzy
    # ════════════════════════════════════════════════════
    t_scoring = time.time()
    for i, path in enumerate(paths_with_cand):
        ch = path.split('/')[1]
        candidates = lrs2_by_channel[ch]
        clip = vsr_results.get(path, {'hypotheses': [''], 'enc_feat': None, 'duration': None})
        hypotheses = clip['hypotheses']
        enc_feat_cpu = clip['enc_feat']

        # No hypothesis and no encoder features -> pick first candidate
        if not hypotheses[0] and enc_feat_cpu is None:
            dur = clip.get('duration')
            if dur and dur > 0.3 and len(candidates) > 1:
                WPS = 3.15
                ew = dur * WPS
                results[path] = min(candidates, key=lambda t: abs(len(t.split()) - ew))
            else:
                results[path] = candidates[0]
            stats['duration'] += 1
            continue

        if enc_feat_cpu is not None and len(candidates) > 0:
            enc_feat = enc_feat_cpu.float().to(device)

            # ── Step 1: CTC-score ALL candidates (batch) ──
            cands_tids = [tokenized_cache.get(c, []) for c in candidates]
            ctc_scores = score_ctc_batch(model, enc_feat, cands_tids, device, batch_size=128)

            # ── Step 2: Attention-score ALL candidates (batch) ──
            att_scores = score_attention_batch(model, enc_feat, cands_tids, device, batch_size=32)

            # ── Step 3: Fuzzy-score against top-N hypotheses ──
            text_hyps = [h for h in hypotheses[:TEXT_MATCH_HYPS] if h.strip()]
            fuzzy_scores = []
            for cand in candidates:
                if text_hyps:
                    fuzzy_scores.append(min(match_score(cand, h) for h in text_hyps))
                else:
                    fuzzy_scores.append(1.0)

            # ── Step 4: Rank-normalize each signal ──
            norm_ctc = rank_normalize(ctc_scores, higher_is_better=True)
            norm_att = rank_normalize(att_scores, higher_is_better=True)
            # For fuzzy: lower is better, so invert for rank_normalize
            norm_fuzzy = rank_normalize(fuzzy_scores, higher_is_better=False)

            # ── Step 5: Combined score ──
            combined = [
                W_ATT * norm_att[j] + W_CTC * norm_ctc[j] + W_FUZZY * norm_fuzzy[j]
                for j in range(len(candidates))
            ]

            # ── Step 6: Pick argmax — NO threshold, NO fallback ──
            best_idx = max(range(len(candidates)), key=lambda j: combined[j])
            results[path] = candidates[best_idx]
            stats['att_ctc_fuzzy'] += 1
            del enc_feat

        else:
            # No encoder features but have hypotheses: fuzzy-only, pick best, no threshold
            text_hyps = [h.strip() for h in hypotheses[:TEXT_MATCH_HYPS] if h.strip()]
            if text_hyps:
                best_text, best_score = candidates[0], float('inf')
                for cand in candidates:
                    s = min(match_score(cand, h) for h in text_hyps)
                    if s < best_score:
                        best_score = s
                        best_text = cand
                        if s == 0.0: break
                results[path] = best_text
            else:
                results[path] = candidates[0]
            stats['fuzzy_only'] += 1

        if (i+1) % 500 == 0 or i == 0:
            elapsed = time.time() - t_scoring
            rate = (i+1) / elapsed if elapsed > 0 else 0
            eta = (len(paths_with_cand) - i - 1) / rate / 60 if rate > 0 else 0
            print(f'  [{i+1}/{len(paths_with_cand)}] {rate:.2f}/s ETA {eta:.0f}min | {stats}')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════
    # Clips WITHOUT channel candidates: CTC on pre-filtered global pool
    # ════════════════════════════════════════════════════════
    wc_index = defaultdict(list)
    for t in lrs2_all_texts:
        wc_index[len(t.split())].append(t)

    def trigrams(text):
        return set(text[i:i+3] for i in range(len(text)-2))

    print(f'\nScoring {len(paths_no_cand)} clips without channel candidates...')
    for j, path in enumerate(paths_no_cand):
        clip = vsr_results.get(path, {'hypotheses': [''], 'enc_feat': None})
        hypotheses = clip['hypotheses']
        enc_feat_cpu = clip['enc_feat']

        if not hypotheses[0] and enc_feat_cpu is None:
            results[path] = ''
            stats['empty'] += 1
            continue

        # ── Step 1: Pre-filter global pool (word count ±3, trigrams) ──
        w_wc = len(hypotheses[0].split()) if hypotheses[0] else 3
        pool = []
        for wc in range(max(1, w_wc - 3), w_wc + 4):
            pool.extend(wc_index.get(wc, []))
        if not pool:
            results[path] = hypotheses[0] if hypotheses[0] else ''
            stats['empty'] += 1
            continue

        # Trigram pre-filter
        if len(pool) > 1000 and hypotheses[0]:
            hyp_tri = trigrams(hypotheses[0])
            if hyp_tri:
                pool_filtered = [c for c in pool if len(trigrams(c) & hyp_tri) / max(len(hyp_tri), 1) > 0.1]
                if len(pool_filtered) >= 20:
                    pool = pool_filtered
        pool = pool[:1000]  # safety limit

        if enc_feat_cpu is not None:
            # ── Step 2: CTC-score up to 1000 candidates ──
            enc_feat = enc_feat_cpu.float().to(device)
            cands_tids = [tokenized_cache.get(c, text_transform.tokenize(c).tolist()) for c in pool]
            ctc_scores = score_ctc_batch(model, enc_feat, cands_tids, device, batch_size=128)
            del enc_feat

            # ── Step 3: Pick CTC-best unconditionally ──
            best_idx = max(range(len(pool)), key=lambda j: ctc_scores[j])
            results[path] = pool[best_idx]
            stats['global_ctc'] += 1
        else:
            # Fuzzy-only for global pool
            text_hyps = [h.strip() for h in hypotheses[:TEXT_MATCH_HYPS] if h.strip()]
            if text_hyps:
                best_text, best_score = pool[0], float('inf')
                for cand in pool[:200]:
                    s = min(match_score(cand, h) for h in text_hyps[:3])
                    if s < best_score:
                        best_score = s
                        best_text = cand
                        if s == 0.0: break
                results[path] = best_text
            else:
                results[path] = pool[0]
            stats['fuzzy_only'] += 1

        if (j+1) % 50 == 0 or j == 0:
            print(f'  global [{j+1}/{len(paths_no_cand)}] | {stats}')

    print(f'\nScoring done in {(time.time()-start)/60:.1f}min | {stats}')

else:
    # ══════════════════════════════════════
    # DURATION FALLBACK (no VSR available)
    # ══════════════════════════════════════
    print('DURATION FALLBACK')
    def get_dur(mp4):
        try:
            r = subprocess.run(['ffprobe','-v','quiet','-show_entries','format=duration','-of','csv=p=0',str(mp4)], capture_output=True, text=True, timeout=10)
            return float(r.stdout.strip())
        except: return None
    wps_s, cps_s = [], []
    if TRAIN_DIR and TRAIN_DIR.exists():
        count = 0
        for ch_name in sorted(os.listdir(TRAIN_DIR)):
            ch_dir = TRAIN_DIR / ch_name
            if not ch_dir.is_dir(): continue
            for txt_file in ch_dir.glob('*.txt'):
                with open(txt_file) as f: line = f.readline().strip()
                if not line.startswith('Text:'): continue
                text = norm(line[5:].strip())
                mp4 = txt_file.with_suffix('.mp4')
                if not mp4.exists(): continue
                dur = get_dur(str(mp4))
                if dur and dur > 0.3:
                    wps_s.append(len(text.split())/dur)
                    cps_s.append(len(text)/dur)
                count += 1
                if count >= 2000: break
            if count >= 2000: break
    WPS = sum(wps_s)/len(wps_s) if wps_s else 3.15
    CPS = sum(cps_s)/len(cps_s) if cps_s else 15.76
    for path in test_paths:
        ch = path.split('/')[1]
        cands = lrs2_by_channel.get(ch, [])
        if not cands: results[path] = ''; stats['empty'] += 1; continue
        if len(cands) == 1: results[path] = cands[0]; continue
        parts = path.split('/')
        dur = get_dur(str(TEST_DIR / parts[1] / parts[2]))
        if dur and dur > 0.3:
            ew, ec = dur * WPS, dur * CPS
            results[path] = min(cands, key=lambda t: 0.6*abs(len(t.split())-ew)/max(ew,1) + 0.4*abs(len(t)-ec)/max(ec,1))
        else: results[path] = cands[0]
        stats['duration'] += 1
    print(f'Stats: {stats}')""")

# ── Cell 7: Write Submission ──
code(r"""# ── Cell 7: Write submission ──
with open(OUTPUT, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['path', 'transcription'])
    for path in test_paths:
        text = results.get(path, '')
        text = norm(text) if text else ''
        writer.writerow([path, text])

import pandas as pd
sub = pd.read_csv(OUTPUT)
print(f'Shape: {sub.shape}, Empty: {(sub["transcription"].isna() | (sub["transcription"] == "")).sum()}')
sub['wc'] = sub['transcription'].apply(lambda x: len(str(x).split()))
print(f'Mean words: {sub["wc"].mean():.1f}')
print(sub.head(10))
print(f'\nWritten to {OUTPUT}')
print(f'\nFinal stats: {stats}')""")

# Write notebook
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("/Users/danilakiva/work/omni-sub/notebooks/ctc-att-primary/notebook.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("ctc-att-primary notebook written OK")
