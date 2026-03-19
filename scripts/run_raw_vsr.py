#!/usr/bin/env python3
"""
OmniSub 2026 — Attempt 2: Raw VSR inference (no pool matching).
Runs pretrained auto-AVSR beam search on test clips, outputs top-1 hypothesis directly.

Usage:
  python3 run_raw_vsr.py \
    --competition-dir ~/data/competition \
    --model-path ~/data/vsr-model/vsr_model.pth \
    --output ~/results_raw
"""

import os, sys, csv, re, json, time, argparse, subprocess
from pathlib import Path

import torch
import torchvision
import numpy as np


def norm(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def parse_args():
    p = argparse.ArgumentParser(description='Raw VSR inference — no pool matching')
    p.add_argument('--competition-dir', required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--avsr-dir', default='/tmp/auto_avsr')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
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
    from datamodule.transforms import VideoTransform
    from preparation.detectors.mediapipe.detector import LandmarksDetector
    from preparation.detectors.mediapipe.video_process import VideoProcess

    # ── Load model ──
    print('Loading VSR model...')
    mm_args = argparse.Namespace(modality='video', ctc_weight=0.1)
    ckpt = torch.load(args.model_path, map_location='cpu')
    modelmodule = ModelModule(mm_args)
    modelmodule.model.load_state_dict(ckpt)
    modelmodule.eval()
    if device == 'cuda':
        modelmodule = modelmodule.cuda()
    model = modelmodule.model
    beam_search = get_beam_search_decoder(model, modelmodule.token_list)
    text_transform = modelmodule.text_transform
    del ckpt

    landmarks_detector = LandmarksDetector()
    video_process = VideoProcess(convert_gray=False)
    video_transform = VideoTransform(subset='test')
    print(f'Model ready on {device}')

    # ── Find test paths ──
    comp_dir = Path(args.competition_dir)
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
        print(f'Sample submission: {len(test_paths)} paths, {len(prefilled)} pre-filled')
    else:
        test_paths = sorted([f.name for f in test_dir.glob('*.mp4')])
        print(f'Found {len(test_paths)} test clips in {test_dir}')

    # ── VSR inference on each clip ──
    results = {}
    start = time.time()

    for i, path in enumerate(test_paths):
        if path in prefilled:
            results[path] = prefilled[path]
            print(f'[{i+1}/{len(test_paths)}] {path}: PREFILLED "{results[path][:50]}"')
            continue

        # Resolve mp4 file path
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
            video_tensor = video_transform(video_tensor)
            if device == 'cuda':
                video_tensor = video_tensor.cuda()

            with torch.no_grad():
                x = model.frontend(video_tensor.unsqueeze(0))
                x = model.proj_encoder(x)
                enc_feat, _ = model.encoder(x, None)
                enc_feat = enc_feat.squeeze(0)

                # Beam search — collect top hypotheses
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

                # CTC greedy as fallback
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
                    ctc_text = norm(text_transform.post_process(torch.tensor(tokens)).replace("<eos>", ""))
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
    print(f'\nDone in {total_time:.1f}s')

    # ── Save results ──
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

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
                                '-m', 'raw VSR results', '--dir-mode', 'zip'],
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
