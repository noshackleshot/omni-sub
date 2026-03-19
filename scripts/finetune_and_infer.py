#!/usr/bin/env python3
"""
OmniSub 2026 — Attempt 3: Fine-tune pretrained VSR on competition train data,
then inference on test clips. Raw model output, no pool matching.

Usage:
  python3 finetune_and_infer.py \
    --competition-dir ~/data/competition \
    --model-path ~/data/vsr-model/vsr_model.pth \
    --output ~/results_finetuned

  # Skip fine-tuning, use existing checkpoint:
  python3 finetune_and_infer.py \
    --competition-dir ~/data/competition \
    --model-path ~/data/vsr-model/vsr_model.pth \
    --output ~/results_finetuned \
    --checkpoint ~/results_finetuned/checkpoints/finetune_epoch3.pth
"""

import os, sys, csv, re, json, time, argparse, subprocess, random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from torch.cuda.amp import autocast, GradScaler


def norm(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def parse_args():
    p = argparse.ArgumentParser(description='Fine-tune VSR + inference')
    p.add_argument('--competition-dir', required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--avsr-dir', default='/tmp/auto_avsr')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--max-samples', type=int, default=0, help='Max training samples per epoch (0=all)')
    p.add_argument('--checkpoint', default=None, help='Skip fine-tuning, use this checkpoint')
    p.add_argument('--upload', action='store_true', help='Upload results to Kaggle dataset')
    return p.parse_args()


def main():
    args = parse_args()
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
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    # ── Load model ──
    print('Loading VSR model...')
    mm_args = argparse.Namespace(modality='video', ctc_weight=0.1)
    ckpt = torch.load(args.model_path, map_location='cpu')
    modelmodule = ModelModule(mm_args)
    modelmodule.model.load_state_dict(ckpt)
    model = modelmodule.model
    text_transform = modelmodule.text_transform
    del ckpt

    landmarks_detector = LandmarksDetector()
    video_process = VideoProcess(convert_gray=False)
    video_transform_train = VideoTransform(subset='train')
    video_transform_test = VideoTransform(subset='test')

    # ── Freeze frontend (ResNet-18) ──
    for p in model.frontend.parameters():
        p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Frontend frozen. Trainable: {trainable/1e6:.1f}M / {total_params/1e6:.1f}M')
    model = model.to(device)

    # ── Load training data (video-text pairs) ──
    train_dir = comp_dir / 'train'
    if (train_dir / 'train').exists():
        train_dir = train_dir / 'train'

    train_gt = []
    if train_dir.exists():
        for ch_name in os.listdir(train_dir):
            ch_dir = train_dir / ch_name
            if not ch_dir.is_dir():
                continue
            for txt_file in ch_dir.glob('*.txt'):
                with open(txt_file) as f:
                    line = f.readline().strip()
                if not line.startswith('Text:'):
                    continue
                text = norm(line[5:].strip())
                if not text:
                    continue
                mp4_path = str(txt_file.with_suffix('.mp4'))
                if os.path.exists(mp4_path):
                    train_gt.append((mp4_path, text))

    print(f'Training data: {len(train_gt)} video-text pairs')

    # ═══════════════════════════════════════════
    # Phase 1: Fine-tune
    # ═══════════════════════════════════════════

    best_ckpt = args.checkpoint

    if not best_ckpt:
        if len(train_gt) == 0:
            print('WARNING: No training data found, skipping fine-tuning')
        else:
            # Apply max-samples limit
            if args.max_samples > 0 and args.max_samples < len(train_gt):
                random.shuffle(train_gt)
                train_gt = train_gt[:args.max_samples]
                print(f'Limited to {args.max_samples} samples')

            print(f'\n{"="*60}')
            print(f'FINE-TUNING: {args.epochs} epochs, {len(train_gt)} samples/epoch')
            print(f'LR={args.lr}, grad_accum={args.grad_accum}')
            print(f'{"="*60}')

            model.train()
            model.frontend.eval()

            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr, weight_decay=1e-6
            )
            scaler = GradScaler()

            best_loss = float('inf')
            LOG_EVERY = min(200, max(50, len(train_gt) // 10))

            for epoch in range(args.epochs):
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
                        video_np = torchvision.io.read_video(mp4_path, pts_unit='sec')[0].numpy()
                        landmarks = landmarks_detector(video_np)
                        video_cropped = video_process(video_np, landmarks)
                        if video_cropped is None:
                            n_err += 1
                            continue

                        video_tensor = torch.tensor(video_cropped).permute(0, 3, 1, 2)
                        video_tensor = video_transform_train(video_tensor)
                        video_tensor = video_tensor.unsqueeze(0).to(device)
                        seq_len = torch.tensor([video_tensor.size(1)], dtype=torch.long, device=device)

                        token_ids = text_transform.tokenize(gt_text)
                        targets = token_ids.unsqueeze(0).to(device)

                        with autocast():
                            loss, loss_ctc, loss_att, acc = model(
                                video_tensor, seq_len, targets
                            )
                            loss_scaled = loss / args.grad_accum

                        scaler.scale(loss_scaled).backward()

                        if (n_ok + 1) % args.grad_accum == 0:
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
                        print(f'  Ep{epoch+1} [{idx+1}/{len(train_gt)}] loss={avg_loss:.4f} '
                              f'acc={avg_acc:.4f} ok={n_ok} err={n_err} '
                              f'{rate:.1f}/s ETA {eta:.0f}min')

                # Final gradient step for remainder
                if n_ok % args.grad_accum != 0:
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
                    print(f'\nEpoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f} '
                          f'ctc={avg_ctc:.4f} att={avg_att:.4f} acc={avg_acc:.4f} '
                          f'ok={n_ok} err={n_err} ({elapsed:.1f}min)')

                    ckpt_path = str(checkpoint_dir / f'finetune_epoch{epoch+1}.pth')
                    torch.save(model.state_dict(), ckpt_path)
                    print(f'  Saved: {ckpt_path} ({os.path.getsize(ckpt_path)/1e6:.0f}MB)')
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_ckpt = ckpt_path

            print(f'\nFine-tuning complete. Best: {best_ckpt} (loss={best_loss:.4f})')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ═══════════════════════════════════════════
    # Phase 2: Inference with fine-tuned model
    # ═══════════════════════════════════════════

    if best_ckpt:
        print(f'\nLoading fine-tuned checkpoint: {best_ckpt}')
        ft_state = torch.load(best_ckpt, map_location='cpu')
        mm2 = ModelModule(mm_args)
        mm2.model.load_state_dict(ft_state)
        mm2.eval()
        if device == 'cuda':
            mm2 = mm2.cuda()
        model = mm2.model
        beam_search = get_beam_search_decoder(model, mm2.token_list)
        text_transform = mm2.text_transform
        del ft_state
    else:
        print('Using original pretrained model (no fine-tuned checkpoint)')
        model.eval()
        beam_search = get_beam_search_decoder(model, modelmodule.token_list)

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

    # ── Run inference ──
    print(f'\n{"="*60}')
    print(f'INFERENCE on {len(test_paths)} clips')
    print(f'{"="*60}')

    results = {}
    start = time.time()

    for i, path in enumerate(test_paths):
        if path in prefilled:
            results[path] = prefilled[path]
            print(f'[{i+1}/{len(test_paths)}] {path}: PREFILLED')
            continue

        mp4_path = test_dir / path if '/' not in path else comp_dir / path
        if not mp4_path.exists():
            mp4_path = test_dir / Path(path).name
        if not mp4_path.exists():
            results[path] = 'a'
            print(f'[{i+1}/{len(test_paths)}] {path}: NOT FOUND')
            continue

        try:
            video = torchvision.io.read_video(str(mp4_path), pts_unit='sec')[0].numpy()
            landmarks = landmarks_detector(video)
            video_cropped = video_process(video, landmarks)
            if video_cropped is None:
                results[path] = 'a'
                print(f'[{i+1}/{len(test_paths)}] {path}: NO FACE')
                continue

            video_tensor = torch.tensor(video_cropped).permute(0, 3, 1, 2)
            video_tensor = video_transform_test(video_tensor)
            if device == 'cuda':
                video_tensor = video_tensor.cuda()

            with torch.no_grad():
                x = model.frontend(video_tensor.unsqueeze(0))
                x = model.proj_encoder(x)
                enc_feat, _ = model.encoder(x, None)
                enc_feat = enc_feat.squeeze(0)

                # Beam search
                nbest = beam_search(enc_feat)
                hypotheses = []
                seen = set()
                for hyp in nbest:
                    if len(hypotheses) >= 10:
                        break
                    h = hyp.asdict()
                    tids = torch.tensor(list(map(int, h["yseq"][1:])))
                    text = text_transform.post_process(tids).replace("<eos>", "")
                    if text.strip() and text not in seen:
                        hypotheses.append(norm(text))
                        seen.add(norm(text))

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
                        hypotheses.append(ctc_text)

            results[path] = hypotheses[0] if hypotheses else 'a'

        except Exception as e:
            results[path] = 'a'
            print(f'[{i+1}/{len(test_paths)}] {path}: ERROR {e}')
            continue

        elapsed = time.time() - start
        print(f'[{i+1}/{len(test_paths)}] {path}: "{results[path][:60]}" ({elapsed:.1f}s)')

    total_time = time.time() - start
    print(f'\nInference done in {total_time:.1f}s')

    # ── Save results ──
    for path in test_paths:
        if path not in results or not results[path]:
            results[path] = 'a'

    results_json = {p: norm(results[p]) for p in test_paths}

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    with open(output_dir / 'submission.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'transcription'])
        for p in test_paths:
            writer.writerow([p, results_json[p]])

    with open(output_dir / 'dataset-metadata.json', 'w') as f:
        json.dump({
            "title": "OmniSub Precomputed Results",
            "id": "kivadanila/omnisub-precomputed-results",
            "licenses": [{"name": "CC0-1.0"}]
        }, f, indent=2)

    print(f'\n=== ALL RESULTS ===')
    for p in test_paths:
        print(f'  {p} → "{results_json[p][:70]}"')

    ok = sum(1 for v in results_json.values() if v and v != 'a')
    print(f'\nTotal: {len(results_json)}, OK: {ok}, empty/fallback: {len(results_json) - ok}')
    print(f'Saved to {output_dir}')

    # ── Upload to Kaggle ──
    if args.upload:
        print('\nUploading to Kaggle...')
        try:
            r = subprocess.run(['kaggle', 'datasets', 'version', '-p', str(output_dir),
                                '-m', 'fine-tuned VSR results', '--dir-mode', 'zip'],
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

    print('DONE')


if __name__ == '__main__':
    main()
