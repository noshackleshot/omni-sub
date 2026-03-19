#!/usr/bin/env python3
"""
OmniSub 2026 — Fine-tune large VSR model on competition train data.
Attention-only loss, frozen frontend. Saves checkpoints for lm_rescore_infer.py.

Usage:
  python3 finetune_large.py \
    --competition-dir ~/data/competition \
    --model-path ~/data/vsr-model/vsr_model_large.pth \
    --output ~/finetune_large \
    --epochs 3 --lr 3e-5
"""

import os, sys, re, json, time, argparse, subprocess, random
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
    p = argparse.ArgumentParser(description='Fine-tune large VSR model')
    p.add_argument('--competition-dir', required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--avsr-dir', default='/tmp/auto_avsr')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--grad-clip', type=float, default=5.0)
    p.add_argument('--max-samples', type=int, default=0,
                   help='Max training samples per epoch (0=all)')
    p.add_argument('--val-split', type=float, default=0.05,
                   help='Fraction of data for validation (0=no val)')
    p.add_argument('--loss-mode', default='att',
                   choices=['att', 'combined'],
                   help='att=attention-only loss, combined=ctc+att')
    p.add_argument('--ctc-weight', type=float, default=0.1,
                   help='CTC weight when loss-mode=combined')
    p.add_argument('--warmup-steps', type=int, default=100,
                   help='Linear warmup steps')
    p.add_argument('--save-every', type=int, default=500,
                   help='Save checkpoint every N optimizer steps')
    return p.parse_args()


def remap_large_checkpoint(ckpt):
    """Remap large model checkpoint keys to match E2E model."""
    if not any(k.startswith('encoder.frontend.') for k in ckpt):
        return ckpt
    print('Remapping large model checkpoint keys...')
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
    return remapped


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'

    # ── Setup auto_avsr ──
    avsr_dir = args.avsr_dir
    if not os.path.exists(avsr_dir):
        subprocess.run(['git', 'clone', '--depth', '1',
                        'https://github.com/mpc001/auto_avsr.git', avsr_dir], check=True)
    sys.path.insert(0, avsr_dir)

    from lightning import ModelModule
    from datamodule.transforms import VideoTransform, TextTransform
    from preparation.detectors.mediapipe.detector import LandmarksDetector
    from preparation.detectors.mediapipe.video_process import VideoProcess

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    # ═══════════════════════════════════════════
    # Load large model
    # ═══════════════════════════════════════════

    print('Loading large VSR model...')
    mm_args = argparse.Namespace(modality='video', ctc_weight=args.ctc_weight)
    ckpt = torch.load(args.model_path, map_location='cpu')
    ckpt = remap_large_checkpoint(ckpt)
    modelmodule = ModelModule(mm_args)
    missing, unexpected = modelmodule.model.load_state_dict(ckpt, strict=False)
    print(f'Model loaded: {len(missing)} missing, {len(unexpected)} unexpected')
    if missing:
        print(f'  WARNING: Missing keys: {missing[:5]}')

    model = modelmodule.model
    text_transform = modelmodule.text_transform
    del ckpt

    # ── Freeze frontend (ResNet-18) ──
    for p in model.frontend.parameters():
        p.requires_grad = False

    if args.loss_mode == 'att':
        # Also freeze CTC head
        for p in model.ctc.parameters():
            p.requires_grad = False
        print('Frozen: frontend + CTC head (att-only training)')
    else:
        print('Frozen: frontend only (combined training)')

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M')

    model = model.to(device)

    landmarks_detector = LandmarksDetector()
    video_process = VideoProcess(convert_gray=False)
    video_transform_train = VideoTransform(subset='train')

    # ═══════════════════════════════════════════
    # Load training data
    # ═══════════════════════════════════════════

    comp_dir = Path(args.competition_dir)
    train_dir = comp_dir / 'train'
    if (train_dir / 'train').exists():
        train_dir = train_dir / 'train'

    train_data = []
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
                train_data.append((mp4_path, text))

    print(f'Training data: {len(train_data)} video-text pairs')

    if len(train_data) == 0:
        print('ERROR: No training data found!')
        sys.exit(1)

    # Split val
    random.shuffle(train_data)
    if args.val_split > 0 and len(train_data) > 100:
        n_val = max(50, int(len(train_data) * args.val_split))
        val_data = train_data[:n_val]
        train_data = train_data[n_val:]
        print(f'Split: {len(train_data)} train, {len(val_data)} val')
    else:
        val_data = []

    if args.max_samples > 0 and args.max_samples < len(train_data):
        train_data = train_data[:args.max_samples]
        print(f'Limited to {args.max_samples} train samples')

    # ═══════════════════════════════════════════
    # Fine-tuning
    # ═══════════════════════════════════════════

    print(f'\n{"="*60}')
    print(f'FINE-TUNING: {args.epochs} epochs, {len(train_data)} samples/epoch')
    print(f'LR={args.lr}, grad_accum={args.grad_accum}, loss={args.loss_mode}')
    print(f'warmup={args.warmup_steps} steps, save_every={args.save_every}')
    print(f'{"="*60}')

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-6, betas=(0.9, 0.98)
    )
    scaler = GradScaler()

    global_step = 0  # optimizer steps
    best_val_loss = float('inf')
    best_ckpt = None
    LOG_EVERY = min(200, max(50, len(train_data) // 10))

    for epoch in range(args.epochs):
        model.train()
        model.frontend.eval()
        if args.loss_mode == 'att':
            model.ctc.eval()

        random.shuffle(train_data)
        epoch_loss = 0.0
        epoch_att = 0.0
        epoch_ctc = 0.0
        epoch_acc = 0.0
        n_ok = 0
        n_err = 0

        start_epoch = time.time()
        optimizer.zero_grad()

        for idx, (mp4_path, gt_text) in enumerate(train_data):
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
                    loss_combined, loss_ctc, loss_att, acc = model(
                        video_tensor, seq_len, targets
                    )

                    if args.loss_mode == 'att':
                        loss = loss_att
                    else:
                        loss = loss_combined

                    loss_scaled = loss / args.grad_accum

                scaler.scale(loss_scaled).backward()

                if (n_ok + 1) % args.grad_accum == 0:
                    # Warmup LR
                    if global_step < args.warmup_steps:
                        lr_scale = (global_step + 1) / args.warmup_steps
                        for pg in optimizer.param_groups:
                            pg['lr'] = args.lr * lr_scale

                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1

                    # Save periodic checkpoint
                    if args.save_every > 0 and global_step % args.save_every == 0:
                        ckpt_path = str(ckpt_dir / f'step_{global_step}.pth')
                        torch.save(model.state_dict(), ckpt_path)
                        print(f'  Saved checkpoint: {ckpt_path} ({os.path.getsize(ckpt_path)/1e6:.0f}MB)')

                epoch_loss += loss.item()
                epoch_att += loss_att.item()
                epoch_ctc += loss_ctc.item()
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
                avg_att = epoch_att / max(n_ok, 1)
                avg_acc = epoch_acc / max(n_ok, 1)
                elapsed = time.time() - start_epoch
                rate = (idx + 1) / elapsed
                eta = (len(train_data) - idx - 1) / rate / 60 if rate > 0 else 0
                cur_lr = optimizer.param_groups[0]['lr']
                print(f'  Ep{epoch+1} [{idx+1}/{len(train_data)}] loss={avg_loss:.4f} '
                      f'att={avg_att:.4f} acc={avg_acc:.4f} '
                      f'ok={n_ok} err={n_err} lr={cur_lr:.2e} '
                      f'{rate:.1f}/s ETA {eta:.0f}min')

        # Final gradient step for remainder
        if n_ok % args.grad_accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1

        # Epoch summary
        if n_ok > 0:
            avg_loss = epoch_loss / n_ok
            avg_att = epoch_att / n_ok
            avg_ctc = epoch_ctc / n_ok
            avg_acc = epoch_acc / n_ok
            elapsed = (time.time() - start_epoch) / 60
            print(f'\nEpoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f} '
                  f'att={avg_att:.4f} ctc={avg_ctc:.4f} acc={avg_acc:.4f} '
                  f'ok={n_ok} err={n_err} ({elapsed:.1f}min) step={global_step}')

            # Save epoch checkpoint
            ckpt_path = str(ckpt_dir / f'epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f'  Saved: {ckpt_path} ({os.path.getsize(ckpt_path)/1e6:.0f}MB)')

        # ── Validation ──
        if val_data:
            model.eval()
            val_loss = 0.0
            val_att = 0.0
            val_n = 0
            print(f'\n  Validating on {len(val_data)} clips...')

            with torch.no_grad():
                for mp4_path, gt_text in val_data:
                    try:
                        video_np = torchvision.io.read_video(mp4_path, pts_unit='sec')[0].numpy()
                        landmarks = landmarks_detector(video_np)
                        video_cropped = video_process(video_np, landmarks)
                        if video_cropped is None:
                            continue

                        video_tensor = torch.tensor(video_cropped).permute(0, 3, 1, 2)
                        video_transform_val = VideoTransform(subset='test')
                        video_tensor = video_transform_val(video_tensor)
                        video_tensor = video_tensor.unsqueeze(0).to(device)
                        seq_len = torch.tensor([video_tensor.size(1)], dtype=torch.long, device=device)
                        token_ids = text_transform.tokenize(gt_text)
                        targets = token_ids.unsqueeze(0).to(device)

                        with autocast():
                            loss_c, loss_ctc_v, loss_att_v, acc_v = model(
                                video_tensor, seq_len, targets
                            )

                        val_loss += loss_att_v.item()
                        val_att += loss_att_v.item()
                        val_n += 1
                    except:
                        continue

            if val_n > 0:
                avg_val = val_loss / val_n
                print(f'  Val: att_loss={avg_val:.4f} (n={val_n})')

                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    best_ckpt = str(ckpt_dir / 'best.pth')
                    torch.save(model.state_dict(), best_ckpt)
                    print(f'  New best! Saved to {best_ckpt}')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Final save ──
    final_ckpt = str(ckpt_dir / 'final.pth')
    torch.save(model.state_dict(), final_ckpt)
    print(f'\nFine-tuning complete. Steps={global_step}')
    print(f'Final checkpoint: {final_ckpt}')
    if best_ckpt:
        print(f'Best val checkpoint: {best_ckpt} (val_loss={best_val_loss:.4f})')

    # Save training config
    config = {
        'model_path': args.model_path,
        'epochs': args.epochs,
        'lr': args.lr,
        'loss_mode': args.loss_mode,
        'ctc_weight': args.ctc_weight,
        'grad_accum': args.grad_accum,
        'warmup_steps': args.warmup_steps,
        'total_steps': global_step,
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'best_val_loss': best_val_loss if val_data else None,
    }
    with open(output_dir / 'train_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print('DONE')


if __name__ == '__main__':
    main()
