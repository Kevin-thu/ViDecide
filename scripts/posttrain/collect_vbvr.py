"""Collect the CFT data of VBVR-MultiStep: roll out the SFT model on training puzzles, then branch from failures.

    python scripts/posttrain/collect_vbvr.py --run-dir outputs/sft/vbvr_k1 --out outputs/cft_collect/vbvr [--shard i --shards n]

Each training puzzle of the selected sub-tasks (--tasks, the weakest ones of the SFT model) is rolled out --roots times;
scripts/eval/score_vbvr.py (``rollouts``) scores the rollouts and selects the fork point of every failure
(src/videcide/training/vbvr_fork.py), and four continuations are resampled from each fork point. Prepare the
training manifests with scripts/posttrain/prepare_vbvr_cft.py.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from videcide.generation.common import continuation_seed, rollout_seed_fields, row_seed, with_seed_fields
from videcide.generation.rollout_engine import RolloutEngine, RolloutJob
from videcide.tasks.video_parser import find_eos_frame
from videcide.tasks.video import write_mp4
from videcide.paths import data_path, manifest_path

MANIFEST = manifest_path('vbvr', 'train')
VPY = sys.executable   # scripts/eval/score_vbvr.py runs in this environment (not the RoboTwin-Plan verifier's)
MAX_LATENTS = 80


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp'); tmp.write_text(json.dumps(value, default=str)); os.replace(tmp, path)


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def score(out, stage, keys):
    kf = Path(out) / f'score_keys.{stage}.{os.getpid()}.txt'; kf.write_text('\n'.join(keys))
    subprocess.run([VPY, str(Path(__file__).resolve().parents[1] / 'eval' / 'score_vbvr.py'), 'rollouts', '--out', str(out), '--stage', stage,
                    '--keys', str(kf), '--workers', os.environ.get('VBVR_SCORE_WORKERS', '40')], check=True,
                   env={**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--batch', type=int, default=12); ap.add_argument('--roots', type=int, default=2)
    ap.add_argument('--tasks', default='hanoi,sliding,sokoban', help='sub-tasks to collect (the weakest ones of the SFT model)')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--stage', choices=['roots', 'branches', 'all'], default='all')
    ap.add_argument('--checkpoint', type=Path, default=None)
    ap.add_argument('--root-seed', type=int, default=20261061); ap.add_argument('--branch-seed', type=int, default=20261071)
    ap.add_argument('--run-dir', type=Path, required=True, help='SFT run directory (resolved_config.json)'); ap.add_argument('--per-task', type=int, default=0)
    ap.add_argument('--manifest', type=Path, default=MANIFEST)
    ap.add_argument('--reserve-gib', type=float, default=0)
    ap.add_argument('--early-stop', action='store_true')
    ap.add_argument('--check-interval', type=int, default=4, help='latents between early-stop checks')
    a = ap.parse_args()
    from videcide.generation import wan_setup as P   # needs the Causal-Forcing checkout and a GPU
    if a.checkpoint is None:
        a.checkpoint = a.run_dir / ('checkpoint_model_%06d/model.pt' % json.loads((a.run_dir / 'resolved_config.json').read_text())['max_iters'])
    from omegaconf import OmegaConf
    for name in ['roots/latents', 'roots/videos', 'roots/records', 'branches/latents', 'branches/videos', 'branches/records', 'forks']:
        (a.out / name).mkdir(parents=True, exist_ok=True)
    config = OmegaConf.create(json.loads((a.run_dir / 'resolved_config.json').read_text())); config.guidance_scale = 1.0
    tasks = {'vbvr_' + t for t in a.tasks.split(',')}
    rows = [r for r in map(json.loads, open(a.manifest)) if r['split'] == 'train' and r['task'] in tasks]
    rows = with_seed_fields(rows, MANIFEST)
    rows.sort(key=lambda r: r['sample_id'])
    if a.per_task:
        rows = [r for t in sorted(tasks) for r in [q for q in rows if q['task'] == t][:a.per_task]]
    rows = [r for i, r in enumerate(rows) if i % a.shards == a.shard]
    if a.limit:
        rows = rows[:a.limit]
    device = torch.device('cuda'); dtype = torch.bfloat16
    gen = P.WanDiffusionWrapper(**OmegaConf.to_container(config.model_kwargs, resolve=True), is_causal=True)
    st = torch.load(a.checkpoint, map_location='cpu', weights_only=False, mmap=True); gen.load_state_dict(st['generator'], strict=True); del st
    gen = gen.to(device=device, dtype=dtype).eval(); name = str(config.model_kwargs.model_name)
    text = P.WanTextEncoder(model_name=name).to(device=device, dtype=dtype).eval(); vae = P.WanVAEWrapper(model_name=name).to(device=device, dtype=dtype).eval()
    pipe = P.CausalDiffusionInferencePipeline(config, device='cuda', generator=gen, text_encoder=text, vae=vae)
    pipe.frame_seq_length = 144; pipe.sampling_steps = 5; pipe.disable_tqdm = True
    runner = SimpleNamespace(pipeline=pipe, trainer=SimpleNamespace(device=device, dtype=dtype), generators=None)
    eos_only = lambda frames, source, include_trace=False: {'outcome': 'incomplete', 'eos_detected': find_eos_frame(frames) is not None}
    engine = RolloutEngine(runner, stop_on='eos', evaluate=eos_only)
    if a.early_stop:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
        from videcide.tasks import vbvr_checkers as VC
        from videcide.training.vbvr_fork import death
        refs = {}

        def early(frames, source, include_trace=False):
            eos = find_eos_frame(frames) is not None
            sid = source['sample_id']; task = source['task'].replace('vbvr_', '')
            if sid not in refs:
                refs[sid] = (VC.puzzle(source), VC.read(data_path(source['native_video'])))
            meta, ref = refs[sid]
            if task not in ('hanoi', 'sliding', 'sokoban') or eos:
                return {'outcome': 'incomplete', 'eos_detected': eos}
            if task == 'sliding':
                vis, _ = VC.content(frames)
                snaps, viol, _ = VC._slide_replay(vis, meta, ref); n, _ = VC._slide_geom(meta, ref)
                target = {tuple(o['target_position']): o['value'] for o in meta['objects'] if o.get('value')}
                changes, prev = [], None
                for i, st in snaps:
                    key = sorted((list(q), v) for q, v in st.items())
                    if key != prev:
                        changes.append((i, key)); prev = key
                m = {'violation': viol[0] if viol else None, 'violation_frame': viol[1] if viol else None, 'changes': changes, 'n': n,
                     'target_state': sorted((list(q), v) for q, v in target.items()), 'budget': meta['solution_length']}
            else:
                m = VC.CHECK[task](frames, meta, ref)
            d = death(task, dict(m, success=False))
            why = 'violation:' + str(m['violation']) if m.get('violation') else (d[0] if d and d[0] in ('detour', 'unsolvable') else None)
            return {'outcome': 'rule_violation' if why else 'incomplete', 'eos_detected': eos, 'early_stop': why}
        engine = RolloutEngine(runner, stop_on='rule', evaluate=early, interval=a.check_interval)
    state_path = a.out / f'status.{a.stage}.shard{a.shard}of{a.shards}.json'
    status = {'pid': os.getpid(), 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'checkpoint': str(a.checkpoint), 'cfg': 1.0,
              'tasks': sorted(tasks), 'puzzles': len(rows), 'expected_roots': a.roots * len(rows), 'roots_completed': 0, 'branches_completed': 0}
    pool = ThreadPoolExecutor(8)
    size = lambda job: job.source['native_frames']

    def persist(result, stage):
        if a.reserve_gib:
            import shutil
            assert shutil.disk_usage(a.out).free > a.reserve_gib*2**30, 'Data disk reserve reached; inspect before recovery'
        job = result.job; key = job.key
        payload = {'latents': result.latents.clone(), 'seed': job.seed, 'noise_seed': job.noise_seed, 'history_length': len(job.history), 'meta': job.meta}
        if stage == 'branches':
            assert torch.equal(result.latents[:len(job.history)], job.history.to(result.latents))
            payload['shared_prefix_sha256'] = sha(result.latents[:len(job.history)])
        torch.save(payload, a.out / stage / 'latents' / (key + '.pt'))
        write_mp4(a.out / stage / 'videos' / (key + '.mp4'), result.frames)
        rec = {'key': key, **job.meta, 'seed': job.seed, 'noise_seed': job.noise_seed, 'history_length': len(job.history),
               'latent_path': str(a.out / stage / 'latents' / (key + '.pt')), 'mp4': str(a.out / stage / 'videos' / (key + '.mp4')),
               'row': job.source, 'generated_latents': len(result.latents), 'success': None,
               'early_stop': (result.verdict or {}).get('early_stop')}
        write(a.out / stage / 'records' / (key + '.json'), rec); return key

    def flush(pending, field, label, done):
        for f in pending: done.append(f.result())
        status[field] += len(pending); pending.clear(); write(state_path, status); print(label, status[field], flush=True)

    if a.stage in ('roots', 'all'):
        status['phase'] = 'roots'; write(state_path, status); jobs = []
        for r in rows:
            todo = [k for k in range(a.roots) if not (a.out / 'roots/records' / f"{r['sample_id']}_r{k}.json").exists()]
            status['roots_completed'] += a.roots - len(todo)
            if not todo:
                continue
            _, pixels = P.first_frame_tensor(data_path(r['native_video']), 384, 'cuda', dtype)
            with torch.no_grad():
                first = vae.encode_to_latent(pixels)[:, :1].cpu()
            for k in todo:
                seed = row_seed(a.root_seed + k, r)
                jobs.append(RolloutJob(f"{r['sample_id']}_r{k}", first[0], r['prompt'], r, seed, MAX_LATENTS, history_key=r['sample_id'],
                                       noise_seed=seed, noise_latents=MAX_LATENTS - 1,
                                       meta={'sample_id': r['sample_id'], 'task': r['task'], 'rollout': k, 'source_path': r['source_path'],
                                             **rollout_seed_fields(r, k)}))
        pending, done = [], []
        def on_root(res):
            pending.append(pool.submit(persist, res, 'roots'))
            if len(pending) >= 32: flush(pending, 'roots_completed', 'ROOTS', done)
        engine.run(jobs, batch_size=a.batch, on_result=on_root, order=size); flush(pending, 'roots_completed', 'ROOTS', done)
    if a.stage in ('branches', 'all'):
        ids = {r['sample_id'] for r in rows}
        recs = [json.loads(p.read_text()) for p in (a.out / 'roots/records').glob('*.json')]
        recs = [r for r in recs if r['sample_id'] in ids]
        status['phase'] = 'score_roots'; write(state_path, status)
        unscored = [r['key'] for r in recs if r['success'] is None or not (a.out / 'forks' / (r['key'] + '.json')).exists() and not r['success']]
        if unscored:
            score(a.out, 'roots', unscored)
            recs = [json.loads((a.out / 'roots/records' / (r['key'] + '.json')).read_text()) for r in recs]
        fails = [r for r in recs if not r['success']]
        forks = [json.loads((a.out / 'forks' / (r['key'] + '.json')).read_text()) for r in fails]
        ok = [f for f in forks if f['selection']['status'] == 'ok']
        by = {}
        for f in sorted(ok, key=lambda f: (f['sample_id'], f['rollout'])): by.setdefault(f['sample_id'], []).append(f)
        ok = [f for v in by.values() for f in v[:2]]
        status.update(phase='branches', root_failures=len(fails), forks_ok=len(ok)); write(state_path, status); jobs = []
        for f in ok:
            root = torch.load(f['latent_path'], map_location='cpu', weights_only=False); sel = f['selection']; p = sel['prefix']
            if p >= MAX_LATENTS - 1:
                continue
            hist = root['latents'][:p].clone(); h = sha(hist)
            noise = torch.randn((1, MAX_LATENTS - 1, *root['latents'].shape[1:]), device=device, dtype=dtype,
                                generator=torch.Generator(device=device).manual_seed(root['noise_seed']))
            torch.save({'failed_fork_noise': noise[0, p - 1].cpu().clone(), 'shared_prefix_sha256': h, 'prefix': p,
                        'root_noise_seed': root['noise_seed']}, a.out / 'branches/latents' / (f['key'] + '_fork_noise.pt'))
            del noise
            for b in range(4):
                key = f"{f['key']}_sel_b{b}"
                if (a.out / 'branches/records' / (key + '.json')).exists():
                    continue
                seed = continuation_seed(a.branch_seed + b, f)
                jobs.append(RolloutJob(key, hist, f['row']['prompt'], f['row'], seed, MAX_LATENTS, history_key=f"{f['key']}_sel",
                                       noise_seed=seed, noise_latents=MAX_LATENTS - p,
                                       meta={'sample_id': f['sample_id'], 'task': f['task'], 'rollout': f['rollout'], 'branch': b, 'rule': 'sel',
                                             'prefix': p, 'loss_start': p, 'root_key': f['key'], 'root_latent_path': f['latent_path'],
                                             'shared_prefix_sha256': h, 'fork_selection': sel}))
        pending, done = [], []
        def on_branch(res):
            pending.append(pool.submit(persist, res, 'branches'))
            if len(pending) >= 32: flush(pending, 'branches_completed', 'BRANCHES', done)
        engine.run(jobs, batch_size=a.batch, on_result=on_branch, order=size); flush(pending, 'branches_completed', 'BRANCHES', done)
        status['phase'] = 'score_branches'; write(state_path, status)
        keys = [f"{f['key']}_sel_b{b}" for f in ok for b in range(4)]
        keys = [k for k in keys if (a.out / 'branches/records' / (k + '.json')).exists()
                and json.loads((a.out / 'branches/records' / (k + '.json')).read_text())['success'] is None]
        if keys:
            score(a.out, 'branches', keys)
    pool.shutdown(); status.update(phase='complete', ended=time.time(), engine_stats=engine.stats,
                                   peak_allocated_bytes=torch.cuda.max_memory_allocated()); write(state_path, status)
    print('COMPLETE', json.dumps(status, default=str), flush=True)


if __name__ == '__main__':
    main()
