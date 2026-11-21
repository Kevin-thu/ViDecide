"""Distance of every DINOv2 patch of a video to its nearest patch in a bank of expert frames (24 x 24 per frame).

Used when post-training rollouts are scored (verifier.py rollouts): the rule checker flags frames whose patches are
unlike every expert frame, and the fork selection looks for the frame where this distance peaks.

    python scripts/robotwin/appearance_scores.py --videos JOBS.json --out DIR      # DIR/<key>.npy per job

Needs checkpoints/dinov2-base and the released bank checkpoints/videcide/robotwin_verifier/appearance_bank_dinov2.pt.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
DINO = 'checkpoints/dinov2-base'
BANK = 'checkpoints/videcide/robotwin_verifier/appearance_bank_dinov2.pt'   # DINOv2 patch features of expert frames
RES = 336
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def scores(frames, model, bank, batch=32):
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(frames), batch):
        x = torch.from_numpy(np.stack([cv2.resize(f, (RES, RES), interpolation=cv2.INTER_AREA) for f in frames[i:i + batch]]))
        x = ((x.permute(0, 3, 1, 2).float() / 255 - MEAN) / STD).to(device)
        h = torch.nn.functional.normalize(model(pixel_values=x).last_hidden_state[:, 1:].float(), dim=-1)
        for f in h:
            out.append((1 - (f @ bank.T).max(1).values).reshape(24, 24).cpu().numpy().astype(np.float16))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--videos', type=Path, required=True, help='JSON list of jobs {key, video, eos}')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--strip-eos-overlay', action='store_true')
    a = ap.parse_args()
    import scene
    from transformers import AutoModel
    if a.device == 'cpu':
        torch.set_num_threads(4)
    model = AutoModel.from_pretrained(DINO).to(a.device).eval()
    bank = torch.load(BANK, map_location=a.device, weights_only=False).float()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.strip_eos_overlay:
        scene.cache_contract(a.out, {'preprocess': scene.PREPROCESS_VERSION, 'backbone': DINO, 'bank': BANK})
    for job in json.loads(a.videos.read_text()):
        dst = a.out / f"{job['key']}.npy"
        if dst.exists():
            continue
        t0 = time.time(); frames = scene.read_visible(job['video'])
        eos = job.get('eos') if 'eos' in job else scene.find_eos_frame(frames)
        if eos is not None:
            frames = frames[:eos]
        if a.strip_eos_overlay:
            frames = scene.remove_eos_overlay(frames)
        s = scores(frames, model, bank); np.save(dst, s)
        f = np.sort(s.reshape(len(s), -1).astype(np.float32), 1)[:, -3]
        print(json.dumps({'key': job['key'], 'frames': len(s), 'max': round(float(f.max()), 3), 'seconds': round(time.time() - t0, 1)}), flush=True)


if __name__ == '__main__':
    main()
