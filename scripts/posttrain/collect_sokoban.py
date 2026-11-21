"""Collect the CFT data of Sokoban: roll out the SFT model on the training puzzles, then branch from failures.

    python scripts/posttrain/collect_sokoban.py --run-dir outputs/sft/sokoban_k1_stage2 --out outputs/cft_collect/sokoban \
        [--shard i --shards n]

Every puzzle of the manifests (default: the 9,500 VideoRLVR and the 5,000 harder training puzzles) is rolled out
--roots times; the strict verifier finds the failures, src/videcide/training/sokoban_fork.py chooses the fork point of
each failure, and four continuations are resampled from it. Prepare the training manifests with
scripts/posttrain/prepare_sokoban_cft.py.
"""
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import Counter
import argparse, hashlib, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from videcide.generation.common import (continuation_seed, default_checkpoint, load_generator_state_dict, load_run_config,
                                        rollout_seed_fields, row_seed, with_seed_fields)
from videcide.generation.rollout_engine import RolloutEngine, RolloutJob
from videcide.tasks.video_parser import find_eos_frame, parse_video
from videcide.tasks.video import write_mp4
from videcide.paths import data_path, manifest_path

MAX_LATENTS = 80


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp'); tmp.write_text(json.dumps(value, default=str)); os.replace(tmp, path)


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def select(item):
    record, out = item
    from videcide.training.sokoban_fork import select_fork, trace
    try:
        frames = np.load(record['frame_path'])['frames']; tr = trace(frames, record['source'])
        fork = select_fork(frames, record['source'], floor=1, tr=tr)
        write(Path(out) / 'forks' / (record['key'] + '.json'), {**record, 'selection': fork})
        return fork['status']
    except Exception as e:
        write(Path(out) / 'forks' / (record['key'] + '.json'), {**record, 'selection': {'status': 'error', 'error': repr(e)}})
        return 'error'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, nargs='+', default=[manifest_path('sokoban', 'train'), manifest_path('sokoban', 'hard_train')],
                    help='training puzzles (default: data/VideoRLVR/Sokoban/train.jsonl and hard_train.jsonl)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--batch', type=int, default=12); ap.add_argument('--roots', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--stage', choices=['roots', 'branches', 'all'], default='all')
    ap.add_argument('--run-dir', type=Path, required=True, help='SFT run (resolved_config.json) or released model folder')
    ap.add_argument('--checkpoint', type=Path, help='default: the final checkpoint of --run-dir')
    ap.add_argument('--root-seed', type=int, default=20261041); ap.add_argument('--branch-seed', type=int, default=20261051)
    ap.add_argument('--backoffs', default='', help='also branch from fork points moved this many latents earlier '
                    '(comma-separated; the paper\'s Sokoban data adds successful continuations of --backoffs 12, see '
                    'prepare_sokoban_cft.py arms --extra-positives)')
    ap.add_argument('--topup', type=int, default=0, help='resample this many more continuations at fork points whose four '
                    'continuations all failed')
    ap.add_argument('--no-fail-frames', action='store_true', help='keep no frames of failed continuations (saves disk)')
    a = ap.parse_args()
    from videcide.generation import wan_setup as P   # needs the Causal-Forcing checkout and a GPU
    from omegaconf import OmegaConf
    for name in ['roots/latents', 'roots/frames', 'roots/records', 'branches/latents', 'branches/frames', 'branches/records', 'forks']:
        (a.out / name).mkdir(parents=True, exist_ok=True)
    config = load_run_config(a.run_dir); config.guidance_scale = 1.0
    a.checkpoint = a.checkpoint or default_checkpoint(a.run_dir, config)
    rows = [json.loads(l) for m in a.manifest for l in open(m)]
    rows = with_seed_fields(rows, manifest_path('sokoban', 'train'), manifest_path('sokoban', 'hard_train'))
    rows.sort(key=lambda r: r['sample_id']); rows = [r for i, r in enumerate(rows) if i % a.shards == a.shard]
    if a.limit:
        rows = rows[:a.limit]
    device = torch.device('cuda'); dtype = torch.bfloat16
    gen = P.WanDiffusionWrapper(**OmegaConf.to_container(config.model_kwargs, resolve=True), is_causal=True)
    st = load_generator_state_dict(a.checkpoint); gen.load_state_dict(st, strict=True); del st
    gen = gen.to(device=device, dtype=dtype).eval(); name = str(config.model_kwargs.model_name)
    text = P.WanTextEncoder(model_name=name).to(device=device, dtype=dtype).eval(); vae = P.WanVAEWrapper(model_name=name).to(device=device, dtype=dtype).eval()
    pipe = P.CausalDiffusionInferencePipeline(config, device='cuda', generator=gen, text_encoder=text, vae=vae)
    pipe.frame_seq_length = 144; pipe.sampling_steps = 5; pipe.disable_tqdm = True
    runner = SimpleNamespace(pipeline=pipe, trainer=SimpleNamespace(device=device, dtype=dtype), generators=None)
    eos_only = lambda frames, source, include_trace=False: {'outcome': 'incomplete', 'eos_detected': find_eos_frame(frames) is not None}
    engine = RolloutEngine(runner, stop_on='eos', evaluate=eos_only)
    began = time.time(); state_path = a.out / f'status.{a.stage}.shard{a.shard}of{a.shards}.json'
    status = {'pid': os.getpid(), 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'checkpoint': str(a.checkpoint), 'cfg': 1.0,
              'manifest': [str(m) for m in a.manifest], 'puzzles': len(rows), 'expected_roots': a.roots * len(rows),
              'roots_completed': 0, 'branches_completed': 0}
    pool = ThreadPoolExecutor(8)
    size = lambda job: job.source['grid_rows'] * job.source['grid_cols']

    def persist(result, stage):
        job = result.job; key = job.key
        m = parse_video(result.frames, job.source, 'sokoban')
        payload = {'latents': result.latents.clone(), 'seed': job.seed, 'noise_seed': job.noise_seed, 'history_length': len(job.history), 'meta': job.meta}
        if stage == 'branches':
            assert torch.equal(result.latents[:len(job.history)], job.history.to(result.latents))
            payload['shared_prefix_sha256'] = sha(result.latents[:len(job.history)])
        torch.save(payload, a.out / stage / 'latents' / (key + '.pt'))
        keep = not (stage == 'branches' and a.no_fail_frames and not m['success'])
        if keep:
            np.savez_compressed(a.out / stage / 'frames' / (key + '.npz'), frames=result.frames)
        rec = {'key': key, **job.meta, 'seed': job.seed, 'noise_seed': job.noise_seed, 'history_length': len(job.history),
               'latent_path': str(a.out / stage / 'latents' / (key + '.pt')), 'frame_path': str(a.out / stage / 'frames' / (key + '.npz')) if keep else None,
               'source': job.source, 'metrics': {k: m.get(k) for k in ('success', 'solved', 'hallucination', 'hall_frame', 'illegal_transitions', 'eos_frame')},
               'success': bool(m['success']), 'generated_latents': len(result.latents)}
        write(a.out / stage / 'records' / (key + '.json'), rec); return rec

    def flush(pending, field, label):
        for f in pending: f.result()
        status[field] += len(pending); pending.clear(); write(state_path, status); print(label, status[field], flush=True)

    if a.stage in ('roots', 'all'):
        status['phase'] = 'roots'; write(state_path, status); jobs = []
        for r in rows:
            src = json.loads(data_path(r['source_path']).read_text()); src.update(render_theme='classic', prompt=r['prompt'])
            todo = [k for k in range(a.roots) if not (a.out / 'roots/records' / f"{r['sample_id']}_r{k}.json").exists()]
            status['roots_completed'] += a.roots - len(todo)
            if not todo:
                continue
            _, pixels = P.first_frame_tensor(data_path(r['native_video']), 384, 'cuda', dtype)
            with torch.no_grad():
                first = vae.encode_to_latent(pixels)[:, :1].cpu()
            for k in todo:
                seed = row_seed(a.root_seed + k, r)
                jobs.append(RolloutJob(f"{r['sample_id']}_r{k}", first[0], r['prompt'], src, seed, MAX_LATENTS, history_key=r['sample_id'],
                                       noise_seed=seed, noise_latents=MAX_LATENTS - 1,
                                       meta={'sample_id': r['sample_id'], 'rollout': k, 'source_path': r['source_path'],
                                             **rollout_seed_fields(r, k)}))
        pending = []
        def on_root(res):
            pending.append(pool.submit(persist, res, 'roots'))
            if len(pending) >= 32: flush(pending, 'roots_completed', 'ROOTS')
        engine.run(jobs, batch_size=a.batch, on_result=on_root, order=size); flush(pending, 'roots_completed', 'ROOTS')
    if a.stage in ('branches', 'all'):
        status['phase'] = 'fork_selection'; write(state_path, status)
        ids = {r['sample_id'] for r in rows}
        recs = [json.loads(p.read_text()) for p in (a.out / 'roots/records').glob('*.json')]
        recs = [r for r in recs if r['sample_id'] in ids]; fails = [r for r in recs if not r['success']]
        todo = [(r, str(a.out)) for r in fails if not (a.out / 'forks' / (r['key'] + '.json')).exists()]
        with ProcessPoolExecutor(16, mp_context=__import__('multiprocessing').get_context('spawn')) as ex:
            list(ex.map(select, todo, chunksize=1))
        forks = [json.loads((a.out / 'forks' / (r['key'] + '.json')).read_text()) for r in fails]
        write(a.out / f'fork_qa.shard{a.shard}.json', {'roots': len(recs), 'failures': len(fails),
              'statuses': Counter(f['selection']['status'] for f in forks), 'kinds': Counter(f['selection'].get('kind') for f in forks)})
        ok = sorted([f for f in forks if f['selection']['status'] == 'ok'], key=lambda f: hashlib.sha256(f['key'].encode()).hexdigest())
        by = {}
        for f in sorted(ok, key=lambda f: (f['sample_id'], f['rollout'])): by.setdefault(f['sample_id'], []).append(f)
        ok = [f for v in by.values() for f in v[:2]]
        status['phase'] = 'branches'; write(state_path, status); jobs = []
        for f in ok:
            root = torch.load(f['latent_path'], map_location='cpu', weights_only=False)
            rules = [('sel', f['selection'])]
            if a.backoffs:
                from videcide.training.sokoban_fork import select_fork, trace
                frames = np.load(f['frame_path'])['frames']; tr = trace(frames, f['source'])
                for k in map(int, a.backoffs.split(',')):
                    alt = select_fork(frames, f['source'], floor=1, tr=tr, override_fatal=max(1, f['selection']['fatal_frame'] - 4 * k))
                    if alt.get('status') == 'ok':
                        rules.append((f'k{k}', {**alt, 'backoff_target_latents': k}))
            for rule, sel in rules:
                p = sel['prefix']
                if p >= MAX_LATENTS - 1:
                    continue
                hist = root['latents'][:p].clone(); h = sha(hist)
                if rule == 'sel':
                    noise = torch.randn((1, MAX_LATENTS - 1, *root['latents'].shape[1:]), device=device, dtype=dtype,
                                        generator=torch.Generator(device=device).manual_seed(root['noise_seed']))
                    torch.save({'failed_fork_noise': noise[0, p - 1].cpu().clone(), 'shared_prefix_sha256': h, 'prefix': p,
                                'root_noise_seed': root['noise_seed']}, a.out / 'branches/latents' / (f['key'] + '_fork_noise.pt'))
                    del noise
                for b in range(4):
                    key = f"{f['key']}_{rule}_b{b}"
                    if (a.out / 'branches/records' / (key + '.json')).exists():
                        continue
                    seed = continuation_seed(a.branch_seed + b, f)
                    jobs.append(RolloutJob(key, hist, f['source']['prompt'], f['source'], seed, MAX_LATENTS, history_key=f"{f['key']}_{rule}",
                                           noise_seed=seed, noise_latents=MAX_LATENTS - p,
                                           meta={'sample_id': f['sample_id'], 'rollout': f['rollout'], 'branch': b, 'rule': rule, 'prefix': p,
                                                 'loss_start': p, 'root_key': f['key'], 'root_latent_path': f['latent_path'],
                                                 'shared_prefix_sha256': h, 'fork_selection': sel}))
        pending = []
        def on_branch(res):
            pending.append(pool.submit(persist, res, 'branches'))
            if len(pending) >= 32: flush(pending, 'branches_completed', 'BRANCHES')
        engine.run(jobs, batch_size=a.batch, on_result=on_branch, order=size); flush(pending, 'branches_completed', 'BRANCHES')
        if a.topup:
            jobs = []
            for f in ok:
                recs4 = [a.out / 'branches/records' / f"{f['key']}_sel_b{b}.json" for b in range(4)]
                if not all(r.exists() for r in recs4) or any(json.loads(r.read_text())['success'] for r in recs4):
                    continue
                root = torch.load(f['latent_path'], map_location='cpu', weights_only=False); sel = f['selection']; p = sel['prefix']
                if p >= MAX_LATENTS - 1:
                    continue
                hist = root['latents'][:p].clone(); h = sha(hist)
                for b in range(4, 4 + a.topup):
                    key = f"{f['key']}_sel_b{b}"
                    if (a.out / 'branches/records' / (key + '.json')).exists():
                        continue
                    seed = continuation_seed(a.branch_seed + b, f)
                    jobs.append(RolloutJob(key, hist, f['source']['prompt'], f['source'], seed, MAX_LATENTS, history_key=f"{f['key']}_sel",
                                           noise_seed=seed, noise_latents=MAX_LATENTS - p,
                                           meta={'sample_id': f['sample_id'], 'rollout': f['rollout'], 'branch': b, 'rule': 'sel', 'prefix': p,
                                                 'loss_start': p, 'root_key': f['key'], 'root_latent_path': f['latent_path'],
                                                 'shared_prefix_sha256': h, 'fork_selection': sel, 'topup': True}))
            status['phase'] = 'topup'; status['topup_jobs'] = len(jobs); write(state_path, status)
            engine.run(jobs, batch_size=a.batch, on_result=on_branch, order=size); flush(pending, 'branches_completed', 'BRANCHES')
    pool.shutdown(); status.update(phase='complete', ended=time.time(), engine_stats=engine.stats); write(state_path, status)
    print('COMPLETE', json.dumps(status, default=str), flush=True)


if __name__ == '__main__':
    main()
