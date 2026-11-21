"""Appearance classifier of the RoboTwin-Plan verifier: a logistic-regression head on frozen DINOv3 ViT-L/16 patch
features of a frame and its temporal difference flags local visual defects of the arm and the blocks (Sec. B of the
supplement). A video fails if any visible frame before the EOS frame scores above the threshold.

    python scripts/robotwin/local_quality_binary.py check --out checkpoints/videcide/robotwin_verifier/appearance --video V --result R.json

check scores one video with the released head (<out>/head.npz). extract and fit trained the head from annotated
frames (<out>/JOBS.json, <out>/DEFECT_REGIONS.json; the annotations are not released); videos scores the jobs of
<out>/VIDEO_JOBS.json. combine_with_rules() merges the verdict with the rule checker (state_checker.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

VERSION = 'appearance_dinov3'
BACKBONE = 'checkpoints/dinov3-vitl16'
RES = 384
GRID = 24


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def read_video(path):
    import scene
    raw = scene.read_visible(str(path))
    eos = scene.find_eos_frame(raw)
    frames = scene.remove_eos_overlay(raw[:eos] if eos is not None else raw)
    if not len(frames):
        raise ValueError('No visible pre-EOS frames')
    return frames, eos


def load_backbone():
    import torch
    from transformers import AutoModel
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    return AutoModel.from_pretrained(BACKBONE).cpu().eval()


def encode(model, frames):
    import torch
    x = torch.from_numpy(np.stack([cv2.resize(f, (RES, RES), interpolation=cv2.INTER_AREA)
                                   for f in frames])).permute(0, 3, 1, 2).float() / 255
    x = (x - torch.tensor([.485, .456, .406])[None, :, None, None]) / torch.tensor([.229, .224, .225])[None, :, None, None]
    with torch.inference_mode():
        h = model(pixel_values=x).last_hidden_state[:, -GRID * GRID:].float()
        return torch.nn.functional.normalize(h, dim=-1).numpy()


def descriptors(tokens, frame):
    h = np.asarray(tokens, dtype=np.float32)
    prev, now, nxt = h[max(0, frame - 1)], h[frame], h[min(len(h) - 1, frame + 1)]
    return np.concatenate([now, np.abs(now - (prev + nxt) * .5)], axis=-1) * 32


def frame_scores(logits):
    a = np.asarray(logits, dtype=np.float32).reshape(-1, GRID, GRID)
    z = (a[:, :-1, :-1] + a[:, 1:, :-1] + a[:, :-1, 1:] + a[:, 1:, 1:]) * .25
    flat = z.reshape(len(z), -1)
    return flat.max(1), flat.argmax(1)


def extract_job(job, out):
    out = Path(out)
    dest = out / 'features' / f'{job["key"]}_{job["frame"]}.npz'
    sha = hashlib.sha256(Path(job['video']).read_bytes()).hexdigest()
    import scene
    contract = dict(version=VERSION, video_sha256=sha, frame=job['frame'], backbone=BACKBONE,
                    resolution=RES, eos_preprocess=scene.PREPROCESS_VERSION)
    if dest.exists():
        cached = np.load(dest)
        if json.loads(str(cached['contract'])) != contract:
            raise ValueError(f'Cache contract mismatch: {dest}')
        return job['key']
    frames, eos = read_video(job['video'])
    t = job['frame']
    if not 0 <= t < len(frames):
        raise ValueError(f'Annotation outside visible prefix: {job["key"]}:{t}')
    ids = np.clip([t - 1, t, t + 1], 0, len(frames) - 1)
    h = encode(_MODEL, frames[ids])
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, tokens=h.astype(np.float16), contract=json.dumps(contract), eos=-1 if eos is None else eos)
    return job['key']


_MODEL = None


def init():
    global _MODEL
    _MODEL = load_backbone()


def extract_call(args):
    return extract_job(*args)


def fit(out):
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    out = Path(out)
    jobs = json.loads((out / 'JOBS.json').read_text())
    rois = json.loads((out / 'DEFECT_REGIONS.json').read_text())['regions']
    regions = {r['key']: r for r in rois}
    rng = np.random.default_rng(20261025)
    features, patches = {}, {}
    yy, xx = np.mgrid[:GRID, :GRID]
    for j in jobs:
        key = f'{j["key"]}_{j["frame"]}'
        h = np.load(out / 'features' / f'{key}.npz')['tokens'].astype(np.float32)
        features[key] = descriptors(h, 1)
        if j['role'] != 'train':
            continue
        if j['label'] == 1:
            x0, y0, x1, y1 = regions[j['key']]['box_normalized']
            ids = np.flatnonzero(((xx + .5) / GRID >= x0) & ((xx + .5) / GRID < x1)
                                 & ((yy + .5) / GRID >= y0) & ((yy + .5) / GRID < y1))
            if len(ids) < 2:
                raise ValueError('Defect annotation needs multiple supported patches')
        else:
            ids = np.sort(rng.choice(GRID * GRID, 96, replace=False))
        patches[key] = ids

    train = [j for j in jobs if j['role'] == 'train']

    def train_head(exclude=None):
        rows = [j for j in train if j['group'] != exclude]
        x, y, w = [], [], []
        types = [(j['label'], j['domain']) for j in rows]
        counts = {t: types.count(t) for t in set(types)}
        for j in rows:
            key = f'{j["key"]}_{j["frame"]}'
            ids = patches[key]
            mass = .5 if j['label'] else .25
            weight = mass / counts[(j['label'], j['domain'])] / len(ids)
            x.append(features[key][ids]); y.extend([j['label']] * len(ids)); w.extend([weight] * len(ids))
        x = np.concatenate(x); w = np.asarray(w) * len(y)
        model = LogisticRegression(C=.01, solver='liblinear', max_iter=500, tol=1e-3, random_state=20261025)
        model.fit(x, y, sample_weight=w)
        return model

    cv, folds = [], {}
    for j in train:
        if j['domain'] != 'generated':
            continue
        if j['group'] not in folds:
            folds[j['group']] = train_head(j['group'])
        head = folds[j['group']]
        s, loc = frame_scores(head.decision_function(features[f'{j["key"]}_{j["frame"]}']))
        cv.append(dict(key=j['key'], frame=j['frame'], group=j['group'], label=j['label'], score=float(s[0])))
    scores = sorted(set(r['score'] for r in cv))
    choices = [scores[0] - .001] + [(a + b) / 2 for a, b in zip(scores, scores[1:])] + [scores[-1] + .001]
    def metric(t):
        bad = [r for r in cv if r['label'] == 1]; good = [r for r in cv if r['label'] == 0]
        recall = sum(r['score'] > t for r in bad) / len(bad)
        specificity = sum(r['score'] <= t for r in good) / len(good)
        return ((recall + specificity) / 2, specificity, -t)
    threshold = max(choices, key=metric)
    head = train_head()
    rows = []
    for j in jobs:
        if j['role'] == 'train':
            continue
        s, loc = frame_scores(head.decision_function(features[f'{j["key"]}_{j["frame"]}']))
        rows.append(dict(j, score=float(s[0]), success=bool(s[0] <= threshold), patch=[int(loc[0] // (GRID - 1)), int(loc[0] % (GRID - 1))]))
    for r in cv:
        r['success'] = r['score'] <= threshold
    np.savez_compressed(out / 'head.npz', weight=head.coef_.astype(np.float32), intercept=head.intercept_.astype(np.float32), threshold=threshold)
    report = dict(version=VERSION, threshold=threshold, cv_calibration=cv, holdout=rows,
                  backbone_frozen=True, source_puzzle_groups_excluded=True)
    write(out / 'FRAME_RESULTS.json', report)
    print('FIT_COMPLETE', threshold, metric(threshold), flush=True)


def infer_tokens(tokens, head):
    if not len(tokens) or not np.isfinite(tokens).all():
        raise ValueError('Need finite visible-frame tokens')
    if not all(np.isfinite(head[k]).all() for k in ['weight', 'intercept', 'threshold']):
        raise ValueError('Nonfinite fitted head')
    values, locations = [], []
    for t in range(len(tokens)):
        x = descriptors(tokens, t)
        s, loc = frame_scores(x @ head['weight'].T + head['intercept'])
        values.append(float(s[0])); locations.append(int(loc[0]))
    values = np.asarray(values)
    worst = int(values.argmax())
    loc = locations[worst]
    flagged = np.flatnonzero(values > float(head['threshold'])).tolist()
    return dict(success=bool(values[worst] <= float(head['threshold'])),
                score=float(values[worst]), threshold=float(head['threshold']), failure_frame=flagged[0] if flagged else None,
                peak_frame=worst, patch=[loc // (GRID - 1), loc % (GRID - 1)], scores=values.tolist(),
                flagged_frames=flagged)


def video_call(args):
    job, out = args; out = Path(out)
    dest = out / 'videos' / f'{job["key"]}.json'
    if dest.exists():
        return job['key']
    head = np.load(out / 'head.npz')
    frames, eos = read_video(job['video'])
    tokens = np.concatenate([encode(_MODEL, frames[t:t + 4]) for t in range(0, len(frames), 4)]).astype(np.float16).astype(np.float32)
    verdict = infer_tokens(tokens, head)
    write(dest, dict(job, **verdict, version=VERSION, frames_scored=len(frames), eos=eos,
                     no_post_eos_frames_used=True, appearance_only=True))
    return job['key']


def check_video(video, artifact):
    global _MODEL
    if _MODEL is None:
        init()
    artifact = Path(artifact)
    head = np.load(artifact / 'head.npz')
    frames, eos = read_video(video)
    tokens = np.concatenate([encode(_MODEL, frames[t:t + 4]) for t in range(0, len(frames), 4)]).astype(np.float16).astype(np.float32)
    result = infer_tokens(tokens, head)
    y, x = result['patch']
    result.update(version=VERSION, video=str(video), frames_scored=len(frames), eos=eos,
                  region_normalized=[x / GRID, y / GRID, (x + 2) / GRID, (y + 2) / GRID],
                  no_post_eos_frames_used=True, appearance_only=True)
    return result


def combine_with_rules(rule_result, appearance):
    if appearance.get('version') != VERSION or type(appearance.get('success')) is not bool:
        raise ValueError('Need a versioned binary appearance result')
    if appearance.get('no_post_eos_frames_used') is not True:
        raise ValueError('Appearance evidence must exclude post-EOS frames')
    if not np.isfinite([appearance['score'], appearance['threshold']]).all():
        raise ValueError('Nonfinite appearance evidence')
    violations = [dict(v) for v in rule_result['violations'] if v['type'] != 'unlike_expert_picture']
    if not appearance['success']:
        violations.append(dict(type='appearance_defect', frame=appearance['failure_frame'],
                               peak_frame=appearance['peak_frame'], patch=appearance['patch'],
                               score=appearance['score'], threshold=appearance['threshold']))
    violations.sort(key=lambda v: v.get('frame', 0))
    return dict(rule_result, success=not violations, violations=violations,
                violation=violations[0]['type'] if violations else None,
                checker=rule_result.get('checker', '') + '+' + VERSION,
                local_appearance=appearance)


def main():
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=['extract', 'fit', 'videos', 'check'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--video', type=Path)
    p.add_argument('--result', type=Path)
    a = p.parse_args()
    if a.mode == 'check':
        if not a.video or not a.result:
            p.error('check requires --video and --result')
        result = check_video(a.video, a.out)
        write(a.result, result)
        print(json.dumps({k: v for k, v in result.items() if k != 'scores'})); return
    if a.mode == 'fit':
        fit(a.out); return
    name = 'JOBS.json' if a.mode == 'extract' else 'VIDEO_JOBS.json'
    jobs = json.loads((a.out / name).read_text())
    with ProcessPoolExecutor(a.workers, initializer=init, mp_context=mp.get_context('spawn')) as pool:
        fun = extract_call if a.mode == 'extract' else video_call
        for i, key in enumerate(pool.map(fun, [(j, a.out) for j in jobs])):
            print(a.mode, i + 1, len(jobs), key, flush=True)
    write(a.out / (a.mode.upper() + '_COMPLETE.json'), dict(complete=True, jobs=len(jobs), version=VERSION))


if __name__ == '__main__':
    main()
