#!/usr/bin/env python3
"""Generate finetune-vsr notebook v2 — online training, no disk saves."""
import json

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.split("\n")})

def code(source):
    lines = source.split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src})

md("# OmniSub 2026 — Fine-Tune VSR + CTC-ATT Scoring (v2)")

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
import os, csv, re, json, subprocess, time, sys, argparse, random
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
CHECKPOINT_DIR = Path('/kaggle/working/checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True)
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

# ── Load competition train texts + GT for fine-tuning ──
train_by_channel = defaultdict(list)
train_gt = []  # list of (mp4_abs_path, gt_text_normalized)
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
                mp4_path = str(txt_file.with_suffix('.mp4'))
                if os.path.exists(mp4_path):
                    train_gt.append((mp4_path, text))
    for ch in train_by_channel:
        train_by_channel[ch] = list(dict.fromkeys(train_by_channel[ch]))
    print(f'Train: {sum(len(v) for v in train_by_channel.values())} texts, {len(train_by_channel)} channels')
    print(f'Train GT with video: {len(train_gt)} pairs')

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
print(f'Merged {merged_count} new texts ({new_channels} new channels)')

lrs2_all_texts = list(set(t for texts in lrs2_by_channel.values() for t in texts))
print(f'Total pool: {len(lrs2_all_texts)} unique texts, {len(lrs2_by_channel)} channels')

# ── Load test paths ──
test_paths = []
with open(SAMPLE_SUB) as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader: test_paths.append(row[0])
paths_with_cand = [p for p in test_paths if p.split('/')[1] in lrs2_by_channel]
paths_no_cand = [p for p in test_paths if p.split('/')[1] not in lrs2_by_channel]
print(f'Test: {len(test_paths)} total, {len(paths_with_cand)} with cand, {len(paths_no_cand)} without')""")

# ── Cell 3: Initialize model + preprocessing tools ──
code(r"""# ── Cell 3: Load model + preprocessing tools ──
import torch, torchvision, numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

AVSR_DIR = '/kaggle/working/auto_avsr'
sys.path.insert(0, AVSR_DIR)

from lightning import ModelModule, get_beam_search_decoder
from datamodule.transforms import VideoTransform, TextTransform
from preparation.detectors.mediapipe.detector import LandmarksDetector
from preparation.detectors.mediapipe.video_process import VideoProcess

landmarks_detector = LandmarksDetector()
video_process = VideoProcess(convert_gray=False)
video_transform_train = VideoTransform(subset='train')
video_transform_test = VideoTransform(subset='test')
text_transform = TextTransform()

# Load model
args = argparse.Namespace()
args.modality = 'video'
args.ctc_weight = 0.1
ckpt = torch.load(MODEL_PATH, map_location='cpu')
modelmodule = ModelModule(args)
modelmodule.model.load_state_dict(ckpt)
model = modelmodule.model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Save original checkpoint as fallback
ORIG_CKPT = str(CHECKPOINT_DIR / 'original.pth')
torch.save(ckpt, ORIG_CKPT)
del ckpt

# Freeze frontend (ResNet-18 visual encoder)
for p in model.frontend.parameters():
    p.requires_grad = False
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f'Frontend frozen. Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M params')

model = model.to(device)
print('Model loaded and ready for fine-tuning')""")

# ── Cell 4: Fine-Tune (online — process video on the fly, no disk) ──
code(r"""# ── Cell 4: Fine-tune — online training (no disk saves) ──
# Process each video on-the-fly: read -> crop mouth -> forward -> backprop
# This avoids the disk space issue entirely.

model.train()
model.frontend.eval()  # keep frozen BN in eval

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-6
)
scaler = GradScaler()

NUM_EPOCHS = 2
GRAD_ACCUM = 4
LOG_EVERY = 200
best_loss = float('inf')
best_ckpt = None

print(f'Fine-tuning: {NUM_EPOCHS} epochs, {len(train_gt)} samples/epoch, grad_accum={GRAD_ACCUM}')

for epoch in range(NUM_EPOCHS):
    model.train()
    model.frontend.eval()

    random.shuffle(train_gt)
    epoch_loss = 0.0
    epoch_ctc_l = 0.0
    epoch_att_l = 0.0
    epoch_acc = 0.0
    n_ok = 0
    n_err = 0

    start_epoch = time.time()
    optimizer.zero_grad()

    for idx, (mp4_path, gt_text) in enumerate(train_gt):
        try:
            # ── Preprocess on the fly ──
            _read = torchvision.io.read_video(mp4_path, pts_unit='sec')
            video_np = _read[0].numpy()
            landmarks = landmarks_detector(video_np)
            video_cropped = video_process(video_np, landmarks)
            if video_cropped is None:
                n_err += 1
                continue

            video_tensor = torch.tensor(video_cropped).permute(0, 3, 1, 2)
            video_tensor = video_transform_train(video_tensor)  # augmentations
            video_tensor = video_tensor.unsqueeze(0).to(device)  # (1, T, C, H, W)
            seq_len = torch.tensor([video_tensor.size(1)], dtype=torch.long, device=device)

            # Tokenize ground truth
            token_ids = text_transform.tokenize(gt_text)
            targets = token_ids.unsqueeze(0).to(device)  # (1, S)

            # ── Forward ──
            with autocast():
                loss, loss_ctc, loss_att, acc = model(video_tensor.squeeze(0).unsqueeze(0), seq_len, targets)
                loss_scaled = loss / GRAD_ACCUM

            scaler.scale(loss_scaled).backward()

            if (n_ok + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_ctc_l += loss_ctc.item()
            epoch_att_l += loss_att.item()
            epoch_acc += acc
            n_ok += 1

        except Exception as e:
            if 'out of memory' in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                optimizer.zero_grad()
            n_err += 1
            continue

        if (idx + 1) % LOG_EVERY == 0:
            avg_loss = epoch_loss / max(n_ok, 1)
            avg_acc = epoch_acc / max(n_ok, 1)
            elapsed = time.time() - start_epoch
            rate = (idx + 1) / elapsed
            eta = (len(train_gt) - idx - 1) / rate / 60 if rate > 0 else 0
            print(f'  Ep{epoch+1} [{idx+1}/{len(train_gt)}] loss={avg_loss:.4f} acc={avg_acc:.4f} ok={n_ok} err={n_err} {rate:.1f}/s ETA {eta:.0f}min')

    # Final grad step
    if n_ok % GRAD_ACCUM != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    if n_ok > 0:
        avg_loss = epoch_loss / n_ok
        avg_ctc = epoch_ctc_l / n_ok
        avg_att = epoch_att_l / n_ok
        avg_acc = epoch_acc / n_ok
        elapsed = (time.time() - start_epoch) / 60
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}: loss={avg_loss:.4f} ctc={avg_ctc:.4f} att={avg_att:.4f} acc={avg_acc:.4f} ok={n_ok} err={n_err} ({elapsed:.1f}min)')

        ckpt_path = str(CHECKPOINT_DIR / f'finetune_epoch{epoch+1}.pth')
        torch.save(model.state_dict(), ckpt_path)
        print(f'  Saved: {ckpt_path} ({os.path.getsize(ckpt_path)/1e6:.0f}MB)')
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt = ckpt_path

print(f'\nFine-tuning complete. Best: {best_ckpt} (loss={best_loss:.4f})')
if torch.cuda.is_available():
    torch.cuda.empty_cache()""")

# ── Cell 5: Inference with Fine-Tuned Model ──
code(r"""# ── Cell 5: Inference with fine-tuned model ──

FINETUNE_CKPT = best_ckpt if best_ckpt else ORIG_CKPT
print(f'Loading checkpoint: {FINETUNE_CKPT}')

# Reload model with fine-tuned weights
args_p = argparse.Namespace()
args_p.modality = 'video'
args_p.ctc_weight = 0.1
mm = ModelModule(args_p)
ft_state = torch.load(FINETUNE_CKPT, map_location='cpu')
mm.model.load_state_dict(ft_state)
mm.eval()
if torch.cuda.is_available():
    mm = mm.cuda()
ft_model = mm.model
ft_beam_search = get_beam_search_decoder(ft_model, mm.token_list)
ft_text_transform = mm.text_transform
ft_device = next(ft_model.parameters()).device
del ft_state

print('Fine-tuned model loaded for inference')

# Transcribe all test videos
vsr_results = {}
print(f'Transcribing {len(test_paths)} test videos...')
start = time.time()
errors = 0

for i, path in enumerate(test_paths):
    parts = path.split('/')
    mp4_path = str(TEST_DIR / parts[1] / parts[2])
    try:
        _read = torchvision.io.read_video(mp4_path, pts_unit='sec')
        video_np = _read[0].numpy()
        fps = _read[2].get('video_fps', 25.0) if len(_read) > 2 else 25.0
        duration = len(video_np) / fps if fps > 0 else None

        landmarks = landmarks_detector(video_np)
        video_cropped = video_process(video_np, landmarks)
        if video_cropped is None:
            vsr_results[path] = {'hypotheses': [''], 'enc_feat': None, 'duration': duration}
            errors += 1
            continue

        video_tensor = torch.tensor(video_cropped).permute(0, 3, 1, 2)
        video_tensor = video_transform_test(video_tensor)
        if torch.cuda.is_available():
            video_tensor = video_tensor.cuda()

        with torch.no_grad():
            x = ft_model.frontend(video_tensor.unsqueeze(0))
            x = ft_model.proj_encoder(x)
            enc_feat, _ = ft_model.encoder(x, None)
            enc_feat = enc_feat.squeeze(0)

            # CTC greedy
            ctc_lp = ft_model.ctc.log_softmax(enc_feat.unsqueeze(0)).squeeze(0)
            ctc_argmax = torch.argmax(ctc_lp, dim=-1)
            tokens = []
            prev = 0
            for t in ctc_argmax:
                t_val = t.item()
                if t_val != 0 and t_val != prev:
                    tokens.append(t_val)
                prev = t_val
            ctc_text = ''
            if tokens:
                ctc_text = ft_text_transform.post_process(torch.tensor(tokens)).replace("<eos>", "")

            # Beam search
            nbest_hyps = ft_beam_search(enc_feat)
            hypotheses = []
            seen = set()
            for hyp in nbest_hyps:
                if len(hypotheses) >= 40:
                    break
                h = hyp.asdict()
                tids = torch.tensor(list(map(int, h["yseq"][1:])))
                text = ft_text_transform.post_process(tids).replace("<eos>", "")
                if text.strip() and text not in seen:
                    hypotheses.append(text)
                    seen.add(text)
            if ctc_text.strip() and ctc_text not in seen:
                hypotheses.append(ctc_text)

        hyps_norm = [norm(str(h)) for h in hypotheses if h]
        vsr_results[path] = {
            'hypotheses': hyps_norm if hyps_norm else [''],
            'enc_feat': enc_feat.cpu().half(),
            'duration': duration
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
        print(f'  [{i+1}/{len(test_paths)}] {rate:.2f}/s ETA {eta:.0f}min ok={ok} err={errors} | "{hyps["hypotheses"][0][:50]}"')
    if (i+1) % 500 == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()

ok = sum(1 for v in vsr_results.values() if v['hypotheses'][0])
nhyps_avg = sum(len(v['hypotheses']) for v in vsr_results.values()) / max(len(vsr_results), 1)
enc_mem = sum(v['enc_feat'].nelement() * v['enc_feat'].element_size()
              for v in vsr_results.values() if v['enc_feat'] is not None) / 1e6
print(f'\nDone: {ok}/{len(vsr_results)} ok, {errors} err, avg_hyps={nhyps_avg:.1f}, enc_mem={enc_mem:.0f}MB, {(time.time()-start)/60:.1f}min')""")

# ── Cell 6: CTC-ATT Scoring ──
code(r"""# ── Cell 6: CTC-ATT Primary scoring with fine-tuned model ──
from jiwer import wer as compute_wer, cer as compute_cer

def match_score(ref, hyp):
    try:
        w = compute_wer(ref, hyp)
        c = compute_cer(ref, hyp)
        return 0.4 * w + 0.6 * c
    except:
        return 1.0

@torch.no_grad()
def score_ctc_batch(mdl, enc_feat, candidates_token_ids, dev, batch_size=128):
    T = enc_feat.size(0)
    ctc_logprobs = mdl.ctc.log_softmax(enc_feat.unsqueeze(0)).squeeze(0)
    all_scores = []
    for bs in range(0, len(candidates_token_ids), batch_size):
        batch = candidates_token_ids[bs:bs + batch_size]
        batch_scores = [float('-inf')] * len(batch)
        valid = [(j, tids) for j, tids in enumerate(batch) if len(tids) > 0 and len(tids) <= T]
        if not valid:
            all_scores.extend(batch_scores)
            continue
        max_s = max(len(tids) for _, tids in valid)
        targets = torch.zeros(len(valid), max_s, dtype=torch.long, device=dev)
        target_lengths = torch.zeros(len(valid), dtype=torch.long, device=dev)
        for k, (j, tids) in enumerate(valid):
            targets[k, :len(tids)] = torch.tensor(tids, device=dev)
            target_lengths[k] = len(tids)
        log_probs = ctc_logprobs.unsqueeze(1).expand(-1, len(valid), -1).contiguous()
        input_lengths = torch.full((len(valid),), T, dtype=torch.long, device=dev)
        losses = F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                            blank=0, reduction='none', zero_infinity=True)
        for k, (j, tids) in enumerate(valid):
            batch_scores[j] = -losses[k].item() / max(len(tids), 1)
        all_scores.extend(batch_scores)
    return all_scores

@torch.no_grad()
def score_attention_batch(mdl, enc_feat, candidates_token_ids, dev, batch_size=32):
    sos = mdl.sos
    eos = mdl.eos
    ignore_id = mdl.ignore_id
    all_scores = []
    for bs in range(0, len(candidates_token_ids), batch_size):
        batch = candidates_token_ids[bs:bs + batch_size]
        batch_scores = [float('-inf')] * len(batch)
        valid = [(j, tids) for j, tids in enumerate(batch) if len(tids) > 0]
        if not valid:
            all_scores.extend(batch_scores)
            continue
        max_len = max(len(tids) for _, tids in valid) + 1
        ys_in = torch.full((len(valid), max_len), eos, dtype=torch.long, device=dev)
        ys_out = torch.full((len(valid), max_len), ignore_id, dtype=torch.long, device=dev)
        for k, (j, tids) in enumerate(valid):
            ys_in[k, 0] = sos
            for t_idx, tid in enumerate(tids):
                ys_in[k, t_idx + 1] = tid
            for t_idx, tid in enumerate(tids):
                ys_out[k, t_idx] = tid
            ys_out[k, len(tids)] = eos
        ys_mask = (ys_in != ignore_id)
        for k, (j, tids) in enumerate(valid):
            seq_len = len(tids) + 1
            ys_mask[k, seq_len:] = False
        causal = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool, device=dev)).unsqueeze(0)
        tgt_mask = ys_mask.unsqueeze(-2) & causal
        memory = enc_feat.unsqueeze(0).expand(len(valid), -1, -1)
        logits, _ = mdl.decoder(ys_in, tgt_mask, memory, None)
        log_probs = F.log_softmax(logits, dim=-1)
        for k, (j, tids) in enumerate(valid):
            n = len(tids) + 1
            score = 0.0
            for pos in range(n):
                target_id = ys_out[k, pos].item()
                if target_id >= 0:
                    score += log_probs[k, pos, target_id].item()
            batch_scores[j] = score / max(n, 1)
        all_scores.extend(batch_scores)
    return all_scores

def rank_normalize(scores, higher_is_better=True):
    n = len(scores)
    if n <= 1:
        return [1.0] * n
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=higher_is_better)
    result = [0.0] * n
    for rank, idx in enumerate(ranked):
        result[idx] = 1.0 - rank / (n - 1)
    return result

# ── Score all clips ──
W_ATT = 0.55
W_CTC = 0.35
W_FUZZY = 0.10
TEXT_MATCH_HYPS = 10

HAS_VSR = bool(vsr_results) and sum(1 for v in vsr_results.values() if v['hypotheses'][0]) > 100
print(f'HAS_VSR: {HAS_VSR}, W_ATT={W_ATT}, W_CTC={W_CTC}, W_FUZZY={W_FUZZY}')

results = {}
stats = {'att_ctc_fuzzy': 0, 'fuzzy_only': 0, 'global_ctc': 0, 'empty': 0, 'duration': 0}

if HAS_VSR:
    start = time.time()

    # Pre-tokenize
    print('Tokenizing candidates...')
    tokenized_cache = {}
    all_cands = set()
    for ch_cands in lrs2_by_channel.values():
        all_cands.update(ch_cands)
    for t in lrs2_all_texts:
        all_cands.add(t)
    for cand in all_cands:
        tids = ft_text_transform.tokenize(cand)
        tokenized_cache[cand] = tids.tolist()
    print(f'Tokenized {len(tokenized_cache)} candidates')

    # ── Clips WITH channel candidates ──
    t_scoring = time.time()
    for i, path in enumerate(paths_with_cand):
        ch = path.split('/')[1]
        candidates = lrs2_by_channel[ch]
        clip = vsr_results.get(path, {'hypotheses': [''], 'enc_feat': None, 'duration': None})
        hypotheses = clip['hypotheses']
        enc_feat_cpu = clip['enc_feat']

        if not hypotheses[0] and enc_feat_cpu is None:
            dur = clip.get('duration')
            if dur and dur > 0.3 and len(candidates) > 1:
                ew = dur * 3.15
                results[path] = min(candidates, key=lambda t: abs(len(t.split()) - ew))
            else:
                results[path] = candidates[0]
            stats['duration'] += 1
            continue

        if enc_feat_cpu is not None and len(candidates) > 0:
            enc_feat = enc_feat_cpu.float().to(ft_device)
            cands_tids = [tokenized_cache.get(c, []) for c in candidates]
            ctc_scores = score_ctc_batch(ft_model, enc_feat, cands_tids, ft_device, batch_size=128)
            att_scores = score_attention_batch(ft_model, enc_feat, cands_tids, ft_device, batch_size=32)
            text_hyps = [h for h in hypotheses[:TEXT_MATCH_HYPS] if h.strip()]
            fuzzy_scores = []
            for cand in candidates:
                if text_hyps:
                    fuzzy_scores.append(min(match_score(cand, h) for h in text_hyps))
                else:
                    fuzzy_scores.append(1.0)
            norm_ctc = rank_normalize(ctc_scores, higher_is_better=True)
            norm_att = rank_normalize(att_scores, higher_is_better=True)
            norm_fuzzy = rank_normalize(fuzzy_scores, higher_is_better=False)
            combined = [W_ATT * norm_att[j] + W_CTC * norm_ctc[j] + W_FUZZY * norm_fuzzy[j]
                        for j in range(len(candidates))]
            best_idx = max(range(len(candidates)), key=lambda j: combined[j])
            results[path] = candidates[best_idx]
            stats['att_ctc_fuzzy'] += 1
            del enc_feat
        else:
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

    # ── Clips WITHOUT channel candidates ──
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

        w_wc = len(hypotheses[0].split()) if hypotheses[0] else 3
        pool = []
        for wc in range(max(1, w_wc - 3), w_wc + 4):
            pool.extend(wc_index.get(wc, []))
        if not pool:
            results[path] = hypotheses[0] if hypotheses[0] else ''
            stats['empty'] += 1
            continue

        if len(pool) > 1000 and hypotheses[0]:
            hyp_tri = trigrams(hypotheses[0])
            if hyp_tri:
                pf = [c for c in pool if len(trigrams(c) & hyp_tri) / max(len(hyp_tri), 1) > 0.1]
                if len(pf) >= 20:
                    pool = pf
        pool = pool[:1000]

        if enc_feat_cpu is not None:
            enc_feat = enc_feat_cpu.float().to(ft_device)
            cands_tids = [tokenized_cache.get(c, ft_text_transform.tokenize(c).tolist()) for c in pool]
            ctc_scores = score_ctc_batch(ft_model, enc_feat, cands_tids, ft_device, batch_size=128)
            del enc_feat
            best_idx = max(range(len(pool)), key=lambda j: ctc_scores[j])
            results[path] = pool[best_idx]
            stats['global_ctc'] += 1
        else:
            text_hyps = [h.strip() for h in hypotheses[:TEXT_MATCH_HYPS] if h.strip()]
            if text_hyps:
                best_text, best_score = pool[0], float('inf')
                for cand in pool[:200]:
                    s = min(match_score(cand, h) for h in text_hyps[:3])
                    if s < best_score:
                        best_score = s
                        best_text = cand
                results[path] = best_text
            else:
                results[path] = pool[0]
            stats['fuzzy_only'] += 1

        if (j+1) % 50 == 0 or j == 0:
            print(f'  global [{j+1}/{len(paths_no_cand)}] | {stats}')

    print(f'\nScoring done in {(time.time()-start)/60:.1f}min | {stats}')

else:
    print('DURATION FALLBACK')
    def get_dur(mp4):
        try:
            r = subprocess.run(['ffprobe','-v','quiet','-show_entries','format=duration','-of','csv=p=0',str(mp4)], capture_output=True, text=True, timeout=10)
            return float(r.stdout.strip())
        except: return None
    for path in test_paths:
        ch = path.split('/')[1]
        cands = lrs2_by_channel.get(ch, [])
        if not cands: results[path] = ''; stats['empty'] += 1; continue
        if len(cands) == 1: results[path] = cands[0]; continue
        parts = path.split('/')
        dur = get_dur(str(TEST_DIR / parts[1] / parts[2]))
        if dur and dur > 0.3:
            ew = dur * 3.15
            results[path] = min(cands, key=lambda t: abs(len(t.split()) - ew))
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
print(f'Final stats: {stats}')
print(f'Model used: {FINETUNE_CKPT}')""")

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

with open("/Users/danilakiva/work/omni-sub/notebooks/finetune-vsr/notebook.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("finetune-vsr v2 notebook written OK")
