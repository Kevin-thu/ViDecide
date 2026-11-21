"""Collect the CFT data of FlowFree: roll out the SFT model on training puzzles, then branch from failures.

    python scripts/posttrain/collect_flowfree.py --run-dir outputs/sft/flowfree_k1_stage2 --out OUT --media MEDIA \
        --shard i --shards n                                        # the 9,500 training puzzles of the SFT run
    ... --manifest data/VideoRLVR/FlowFree/hard_train.jsonl        # the 5,000 harder training puzzles

Every puzzle is rolled out --roots times; the strict verifier finds the failures, src/videcide/training/fork_selection.py
chooses the fork point of each failure, and four continuations are resampled from it. Records go to OUT, latents and
frames to MEDIA. Prepare the training manifests with scripts/posttrain/prepare_flowfree_cft.py.
"""
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import Counter
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from videcide.generation.common import continuation_seed, rollout_seed_fields, row_seed, with_seed_fields
from videcide.generation.rollout_engine import RolloutEngine, RolloutJob
from videcide.tasks.video_parser import find_eos_frame, parse_video
from videcide.tasks.flow_replay import evaluate_flowfree
from videcide.training.fork_selection import select_fork
from videcide.tasks.video import write_mp4
from videcide.paths import data_path, manifest_path


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, default=lambda x: x.tolist(), indent=1))
    tmp.replace(path)


def sha(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def select(item):
    record, media, output, mb, *rest = (*item, 0) if len(item) == 3 else item
    nosolver = bool(rest and rest[0])
    fdir = ('forks' if mb == 0 else f'forks_mb{mb}') + ('_nosolver' if nosolver else '')
    try:
        import cv2
        cv2.setNumThreads(1)
        frames = np.load(Path(media) / 'roots/frames' / (record['key'] + '.npz'))['frames']
        source = record['source']
        trace = evaluate_flowfree(frames, source, include_trace=True,
                                  require_adjacent_color_switch=bool(source.get('continuous_global_walk', False)))['trace']
        fork = select_fork(frames, source, floor=1, trace=trace, min_backoff=mb, use_solver=not nosolver)
        if fork['status'] == 'ok':
            last = 4 * (fork['prefix'] - 1)
            assert not any(e.get('local_violation') and e['frame'] <= last for e in trace)
            assert fork['seam']['clean'] and (nosolver or fork['fork_state'] == 'solvable')
        write(Path(output) / fdir / (record['key'] + '.json'), {**record, 'selection': fork})
        return fork['status']
    except Exception as e:
        write(Path(output) / fdir / (record['key'] + '.json'), {**record, 'selection': {'status': 'error', 'error': repr(e)}})
        return 'error'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--media', type=Path, required=True)
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--shards', type=int, default=4)
    ap.add_argument('--batch', type=int, default=12)
    ap.add_argument('--stage', choices=['roots', 'branches', 'all'], default='all')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--roots', type=int, default=2)
    ap.add_argument('--run-dir', type=Path, required=True, help='SFT run directory (resolved_config.json) of the policy')
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--expected-step', type=int, help='refuse a different policy checkpoint step')
    ap.add_argument('--guidance', type=float, default=1.0, help='CFG scale (1.0 = no guidance, as in the paper)')
    ap.add_argument('--root-seed', type=int, default=20261025, help='seed of rollout k: root seed + k + the seed_offset of the '
                    'puzzle (hash of its sample id if the manifest has none); the paper\'s collection')
    ap.add_argument('--branch-seed', type=int, default=20261102, help='seed of continuation b of rollout k: branch seed + b + '
                    'continuation_seed_offsets[k] of the puzzle (hash of the rollout key if the manifest has none)')
    ap.add_argument('--manifest', type=Path, nargs='+', help='puzzles of these manifests (e.g. data/VideoRLVR/FlowFree/hard_train.jsonl); '
                    'default: the 9,500 training rows of the SFT run')
    ap.add_argument('--train-ids', type=Path, help='JSON list of sample ids, a subset of the training rows of the run')
    ap.add_argument('--validate-only', action='store_true', help='check the configuration and the data on the CPU and exit')
    ap.add_argument('--no-branch-frames', action='store_true',
                    help='keep only latents and records of failed branches (their frames take much disk); successes keep frames')
    ap.add_argument('--branches', type=int, default=4)
    ap.add_argument('--fork-min-backoff', type=int, default=0)
    ap.add_argument('--fork-no-solver', action='store_true')
    ap.add_argument('--branch-tag', default='')
    ap.add_argument('--branch-seed-offset', type=int, default=0, help='added to the contract branch seed for tagged branches only')
    a = ap.parse_args()
    if a.manifest and a.train_ids:
        ap.error('--manifest and --train-ids are mutually exclusive')
    config = SimpleNamespace(**json.loads((a.run_dir / 'resolved_config.json').read_text()))
    if a.checkpoint is None:
        a.checkpoint = a.run_dir / f'checkpoint_model_{int(config.max_iters):06d}/model.pt'
    config.guidance_scale = float(a.guidance)
    if a.manifest:
        rows = [json.loads(l) for m in a.manifest for l in open(m)]
    else:
        rows = [r for r in json.loads(Path(config.metadata_path).read_text()) if r['split'] == 'train']
        assert len(rows) == 9500 and len({r['sample_id'] for r in rows}) == 9500
        if a.train_ids:
            ids = json.loads(a.train_ids.read_text())
            assert isinstance(ids, list) and ids and len(ids) == len(set(ids))
            assert set(ids) <= {r['sample_id'] for r in rows}, 'Pilot includes non-training IDs'
            rows = [r for r in rows if r['sample_id'] in set(ids)]
    rows.sort(key=lambda r: r['sample_id'])
    rows = with_seed_fields(rows, manifest_path('flowfree', 'train'), manifest_path('flowfree', 'hard_train'))
    rows = [r for i, r in enumerate(rows) if i % a.shards == a.shard]
    if a.limit:
        rows = rows[:a.limit]
    selected_ids = {r['sample_id'] for r in rows}
    if a.validate_only:
        flags = Counter()
        for row in rows:
            source = json.loads(data_path(row['source_path']).read_text())
            flags[bool(source.get('continuous_global_walk', False))] += 1
        print(json.dumps(dict(cpu_contract_only=True, GPU_or_process_launch=False,
              run_dir=str(a.run_dir), checkpoint=str(a.checkpoint),
              checkpoint_ready=a.checkpoint.is_file(), expected_step=a.expected_step,
              puzzles=len(rows), expected_roots=a.roots * len(rows), source_rule_flags=dict(flags),
              source_mode='manifest' if a.manifest else 'training_subset' if a.train_ids else 'full_training',
              metadata_sha256=hashlib.sha256(Path(config.metadata_path).read_bytes()).hexdigest())))
        return
    a.out.mkdir(parents=True, exist_ok=True)
    if True:
        contract = dict(run_dir=str(a.run_dir.resolve()),
                        checkpoint=str(a.checkpoint.resolve()), expected_step=a.expected_step,
                        metadata_sha256=hashlib.sha256(Path(config.metadata_path).read_bytes()).hexdigest(),
                        train_ids_sha256=hashlib.sha256(a.train_ids.read_bytes()).hexdigest() if a.train_ids else None,
                        manifest_sha256=[hashlib.sha256(m.read_bytes()).hexdigest() for m in a.manifest] if a.manifest else None,
                        root_seed=a.root_seed, branch_seed=a.branch_seed, roots=a.roots,
                        guidance=float(config.guidance_scale), steps=5, native_latent_budget=80)
        path = a.out / 'COLLECTION_CONTRACT.json'
        if path.exists():
            assert json.loads(path.read_text()) == contract, 'Collection input contract changed; use a new output'
        else:
            assert not any((a.out / 'roots/records').glob('*.json')), 'roots without a collection contract cannot be resumed'
            write(path, contract)
    BR = ('branches' if a.fork_min_backoff == 0 else f'branches_mb{a.fork_min_backoff}') + ('_nosolver' if a.fork_no_solver else '') + a.branch_tag
    for name in ['roots/latents', 'roots/frames', 'roots/videos', f'{BR}/latents', f'{BR}/frames', f'{BR}/videos']:
        (a.media / name).mkdir(parents=True, exist_ok=True)
    from videcide.generation import wan_setup as P
    import lmdb
    from omegaconf import OmegaConf
    config = OmegaConf.create(vars(config))
    device = torch.device('cuda'); dtype = torch.bfloat16
    generator = P.WanDiffusionWrapper(**OmegaConf.to_container(config.model_kwargs, resolve=True), is_causal=True)
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if a.expected_step is not None:
        assert int(state['step']) == a.expected_step, 'Policy checkpoint step mismatch'
    generator.load_state_dict(state['generator'], strict=True); del state
    generator = generator.to(device=device, dtype=dtype).eval()
    text_encoder = P.WanTextEncoder(model_name=str(config.model_kwargs.model_name)).to(device=device, dtype=dtype).eval()
    vae = P.WanVAEWrapper(model_name=str(config.model_kwargs.model_name)).to(device=device, dtype=dtype).eval()
    pipe = P.CausalDiffusionInferencePipeline(config, device='cuda', generator=generator, text_encoder=text_encoder, vae=vae)
    pipe.frame_seq_length = 144; pipe.sampling_steps = 5; pipe.disable_tqdm = True
    runner = SimpleNamespace(pipeline=pipe, trainer=SimpleNamespace(device=device, dtype=dtype), generators=None)

    def eos_only(frames, source, include_trace=False):
        return {'outcome': 'incomplete', 'eos_detected': find_eos_frame(frames) is not None}

    engine = RolloutEngine(runner, stop_on='eos', evaluate=eos_only)
    env = lmdb.open(str(config.data_path), readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        shape = tuple(map(int, txn.get(b'latents_shape').decode().split()))[2:]

    def first_latent(row):
        if not a.manifest:
            with env.begin() as txn:
                raw = txn.get(f"latents_{row['row_idx']}_data".encode())
            return torch.from_numpy(np.frombuffer(raw, dtype=np.float16, count=int(np.prod(shape))).reshape((1,) + shape).copy()).to(dtype)
        _, pixels = P.first_frame_tensor(data_path(row['native_video']), 384, 'cuda', dtype)
        with torch.no_grad():
            return vae.encode_to_latent(pixels)[:, :1].cpu()[0]

    def source_of(row):
        src = json.loads(data_path(row['source_path']).read_text())
        src.update(render_theme=row['theme'], prompt=row['prompt']); return src
    began = time.time()
    state_path = a.out / f'status.shard{a.shard}.json'
    status = {'pid': os.getpid(), 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'started': began, 'checkpoint': str(a.checkpoint),
              'cfg': float(config.guidance_scale), 'root_seed': a.root_seed, 'branch_seed': a.branch_seed,
              'puzzles': len(rows), 'expected_roots': a.roots * len(rows), 'roots_completed': 0, 'branches_completed': 0}
    save_pool = ThreadPoolExecutor(8)

    def persist(result, stage):
        job = result.job; key = job.key
        metrics = parse_video(result.frames, job.source, 'flowfree')
        payload = {'latents': result.latents.clone(), 'seed': job.seed, 'noise_seed': job.noise_seed,
                   'noise_latents': job.noise_latents, 'noise_offset': job.noise_offset,
                   'history_length': len(job.history), 'meta': job.meta}
        if stage == BR:
            assert torch.equal(result.latents[:len(job.history)], job.history.to(result.latents))
            payload['shared_prefix_sha256'] = sha(result.latents[:len(job.history)])
            assert payload['shared_prefix_sha256'] == job.meta['shared_prefix_sha256']
        torch.save(payload, a.media / stage / 'latents' / (key + '.pt'))
        keep_frames = not (stage == BR and a.no_branch_frames and not metrics['success'])
        if keep_frames:
            np.savez_compressed(a.media / stage / 'frames' / (key + '.npz'), frames=result.frames)
        if int(hashlib.sha256(job.meta['sample_id'].encode()).hexdigest()[:8], 16) % 100 == 0:   # videos of 1% of the puzzles
            write_mp4(a.media / stage / 'videos' / (key + '.mp4'), result.frames)
        record = {'key': key, **job.meta, 'seed': job.seed, 'noise_seed': job.noise_seed,
                  'noise_latents': job.noise_latents, 'history_length': len(job.history),
                  'noise_offset': job.noise_offset,
                  'latent_path': str(a.media / stage / 'latents' / (key + '.pt')),
                  'frame_path': str(a.media / stage / 'frames' / (key + '.npz')) if keep_frames else None,
                  'source': job.source, 'metrics': metrics, 'success': bool(metrics['success']),
                  'generated_latents': len(result.latents), 'elapsed': time.time() - began}
        write(a.out / stage / 'records' / (key + '.json'), record)
        return record

    if a.stage in ['roots', 'all']:
        status['phase'] = 'roots'; write(state_path, status)
        jobs = []
        for row in rows:
            if all((a.out / 'roots/records' / (row['sample_id'] + f'_r{k}.json')).exists() for k in range(a.roots)):
                status['roots_completed'] += a.roots; continue
            first = first_latent(row); src = source_of(row)
            for k in range(a.roots):
                key = row['sample_id'] + f'_r{k}'
                if (a.out / 'roots/records' / (key + '.json')).exists():
                    status['roots_completed'] += 1; continue
                seed = row_seed(a.root_seed + k, row)
                jobs.append(RolloutJob(key, first, row['prompt'], src, seed, 80, history_key=row['sample_id'],
                    noise_seed=seed, noise_latents=79, meta={'sample_id': row['sample_id'], 'rollout': k, 'source_path': row['source_path'],
                                                             **rollout_seed_fields(row, k)}))
        pending = []
        def on_root(result):
            pending.append(save_pool.submit(persist, result, 'roots'))
            if len(pending) >= 32:
                for future in pending: future.result()
                status['roots_completed'] += len(pending);pending.clear();write(state_path,status)
                print('ROOTS', status['roots_completed'], '/', status['expected_roots'], flush=True)
        engine.run(jobs, batch_size=a.batch, on_result=on_root, order=lambda job: job.source['grid_size'])
        for future in pending: future.result()
        status['roots_completed'] += len(pending)
        assert status['roots_completed'] == status['expected_roots']
        status['phase'] = 'fork_selection';write(state_path,status)
    if a.stage in ['branches', 'all']:
        records = [json.loads(p.read_text()) for p in (a.out / 'roots/records').glob('*.json')]
        records = [r for r in records if r['sample_id'] in selected_ids]
        assert len(records) == a.roots * len(rows), 'Cannot branch incomplete root collection'
        failures = [r for r in records if not r['success']]
        fdir = ('forks' if a.fork_min_backoff == 0 else f'forks_mb{a.fork_min_backoff}') + ('_nosolver' if a.fork_no_solver else '')
        todo = [(r, str(a.media), str(a.out), a.fork_min_backoff, a.fork_no_solver) for r in failures if not (a.out / fdir / (r['key'] + '.json')).exists()]
        with ProcessPoolExecutor(32, mp_context=__import__('multiprocessing').get_context('spawn')) as pool:
            statuses = list(pool.map(select, todo, chunksize=1))
        assert 'error' not in statuses, Counter(statuses)
        forks = [json.loads((a.out / fdir / (r['key'] + '.json')).read_text()) for r in failures]
        counts = Counter(r['selection']['status'] for r in forks)
        write(a.out / f'fork_qa.shard{a.shard}.json', {'n': len(forks), 'statuses': counts,
            'selector_sha256': hashlib.sha256(Path(__file__).parents[2].joinpath('src/videcide/training/fork_selection.py').read_bytes()).hexdigest()})
        groups = {}
        for r in sorted(forks, key=lambda r: (r['sample_id'],r['rollout'])):
            if r['selection']['status'] == 'ok':groups.setdefault(r['sample_id'],[]).append(r)
        status['phase'] = 'branches';write(state_path,status)
        bcontract = dict(branches=a.branches)
        if a.fork_min_backoff:
            bcontract['fork_min_backoff'] = a.fork_min_backoff
        if a.fork_no_solver:
            bcontract['fork_no_solver'] = True
        if a.branch_tag:
            bcontract.update(branch_tag=a.branch_tag, branch_seed=a.branch_seed, branch_seed_offset=a.branch_seed_offset)
        bpath = a.out / f'{BR}_CONTRACT.json'
        if bpath.exists():
            assert json.loads(bpath.read_text()) == bcontract, 'Branch sampler changed; use a new output'
        elif a.branches != 4 or a.fork_min_backoff or a.fork_no_solver or a.branch_tag:
            write(bpath, bcontract)
        jobs = []
        for sid, candidates in groups.items():
            for row in candidates[:2]:
                p = row['selection']['prefix']; root = torch.load(row['latent_path'], map_location='cpu', weights_only=False)
                history = root['latents'][:p].clone(); history_hash = sha(history)
                noise = torch.randn((1,79,*shape),device=device,dtype=dtype,generator=torch.Generator(device=device).manual_seed(root['noise_seed']))
                extra = {'root_noise_from_prefix': noise[0, p-1:].cpu().clone()}   # prepare_flowfree_cft.py reads the noise at the divergence step
                torch.save({'failed_fork_noise':noise[0,p-1].cpu().clone(),'shared_prefix_sha256':history_hash,
                    'prefix':p,'root_noise_seed':root['noise_seed'],**extra},a.media/BR/'latents'/(row['key']+'_fork_noise.pt'))
                del noise
                for b in range(a.branches):
                    key = row['key']+f'_b{b}'
                    if (a.out/BR/'records'/(key+'.json')).exists():continue
                    seed = continuation_seed(a.branch_seed+a.branch_seed_offset+b,row)
                    jobs.append(RolloutJob(key,history,row['source']['prompt'],row['source'],seed,80,history_key=row['key'],
                        noise_seed=seed,noise_latents=80-p,meta={'sample_id':sid,'rollout':row['rollout'],'branch':b,'prefix':p,
                        'loss_start':p,'root_key':row['key'],'root_latent_path':row['latent_path'],
                        'shared_prefix_sha256':history_hash,'fork_selection':row['selection']}))
        pending = []
        def on_branch(result):
            pending.append(save_pool.submit(persist,result,BR))
            if len(pending)>=32:
                for future in pending:future.result()
                status['branches_completed']+=len(pending);pending.clear();write(state_path,status)
                print('BRANCHES',status['branches_completed'],flush=True)
        engine.run(jobs,batch_size=a.batch,on_result=on_branch,order=lambda job: job.source['grid_size'])
        for future in pending:future.result()
        status['branches_completed']+=len(pending)
    save_pool.shutdown();env.close()
    status.update(phase='complete',ended=time.time(),engine_stats=engine.stats)
    write(state_path,status);print('COMPLETE',status,flush=True)


if __name__ == '__main__':
    main()
