"""VBVR-MultiStep CFT data: training manifests of CFT and its baselines from rollout collections (Sec. 4.4).

Pipeline (scripts/posttrain/collect_vbvr.py rolls out the SFT model):
    1. collect_vbvr.py --stage roots --out ROOTS             roll out every training puzzle of the selected sub-tasks
    2. prepare_vbvr_cft.py forks --src ROOTS --out COLLECT   re-check the roots with the shortest-solution requirement,
                                                              choose the fork point of every failure, and write the
                                                              rows of the puzzles with a fork to COLLECT/manifest.jsonl
    3. collect_vbvr.py --stage branches --out COLLECT --manifest COLLECT/manifest.jsonl   resample four continuations
    4. prepare_vbvr_cft.py --collect COLLECT --out OUT --sft-run RUN   gate the successful branches (strict + shortest),
                                                              find the divergence step and write positives.json and
                                                              metadata_{cft,decision_rft,sample_rft,sft}.json
    5. prepare_vbvr_cft.py noise OUT/positives.json          rebuild the noise of the failed rollout at the divergence
                                                              step of every positive (one GPU)
Train with configs/cft/vbvr.yaml and metadata_path=OUT/metadata_<method>.json (row counts in OUT/PREPARED.json).
"""
import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARENT = None   # SFT run of the policy (--sft-run)
BATCH, ANCHOR_SHARE = 3, 0.25


def lat(f):
    return 0 if f == 0 else (f - 1) // 4 + 1


def verify(job):
    rec, root = job
    import numpy as np
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT / 'src'))
    from videcide.tasks import vbvr_checkers as V
    out = {'key': rec['key'], 'ok': False}
    if not rec.get('success'):
        return out
    p = rec['prefix']
    a = torch.load(root['latent_path'], map_location='cpu', weights_only=False, mmap=True)['latents']
    b = torch.load(rec['latent_path'], map_location='cpu', weights_only=False, mmap=True)['latents']
    if not torch.equal(a[:p], b[:p]):
        out['why'] = 'prefix'; return out
    if hashlib.sha256(a[:p].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() != rec['shared_prefix_sha256']:
        out['why'] = 'sha'; return out
    bf = V.read(rec['mp4']); m = V.check(rec['row'], bf)
    if not m['success']:
        out['why'] = 'strict_recheck'; return out
    task = rec['task'].replace('vbvr_', '')
    meta = V.puzzle(rec['row'])
    if task == 'hanoi':
        opt = m['moves'] <= m['optimal_moves']
    elif task == 'sliding':
        opt = len(m['changes']) - 1 <= meta['solution_length']
    else:
        opt = True
    if not opt:
        out['why'] = 'not_optimal'; return out
    rf = V.read(root['mp4']); n = min(len(rf), len(bf))
    f = next((f for f in range(4 * (p - 1) + 1, n)
              if int((np.abs(rf[f].astype(np.int16) - bf[f].astype(np.int16)).max(-1) > 60).sum()) > 150), None)
    d = max(p, lat(f)) if f is not None else p
    d = min(d, rec['generated_latents'] - 1)
    out.update(ok=True, D=d, P=p, generated=rec['generated_latents'], eos=m.get('eos_frame'))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--collect', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--sft-run', type=Path, required=True, help='SFT run of the policy (resolved_config.json, final checkpoint)')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    PARENT = a.sft_run
    sft = json.loads((PARENT / 'resolved_config.json').read_text()); start = int(sft['max_iters'])
    checkpoint = PARENT / f'checkpoint_model_{start:06d}/model.pt'; assert checkpoint.is_file()
    for f in a.collect.glob('status.branches.*.json'):
        st = json.loads(f.read_text())
        assert st.get('phase') == 'complete', f'{f.name} incomplete'
        assert Path(st['checkpoint']).resolve() == checkpoint.resolve(), f'{f.name}: parent mismatch'
    roots = {r['key']: r for r in (json.loads(f.read_text()) for f in (a.collect / 'roots/records').glob('*.json'))}
    branches = [json.loads(f.read_text()) for f in (a.collect / 'branches/records').glob('*.json')]
    assert all(b['success'] is not None for b in branches), 'unscored branches'
    with ProcessPoolExecutor(48) as ex:
        res = list(ex.map(verify, [(b, roots[b['root_key']]) for b in branches], chunksize=8))
    info = {r['key']: r for r in res}
    positives = sorted([b for b in branches if info[b['key']]['ok']], key=lambda b: b['key'])
    rejections = Counter(r.get('why') for r in res if not r['ok'] and r.get('why'))
    groups = defaultdict(list)
    for r in roots.values():
        groups[r['sample_id']].append(r)
    mixed = {s for s, rs in groups.items() if 0 < sum(bool(r['success']) for r in rs) < len(rs)}
    root_pool = sorted([r for r in roots.values() if r['success'] and r['sample_id'] in mixed], key=lambda r: r['key'])
    meta = json.loads(Path(sft['metadata_path']).read_text())
    train = [r for r in meta if r['split'] == 'train']; val = [r for r in meta if r['split'] == 'val']
    assert len(train) == 30000 and len(val) == 300
    noise_dir = a.out / 'failed_noise'
    (a.out / 'positives.json').write_text(json.dumps([
        {'key': b['key'], 'root_key': b['root_key'], 'root_latent_path': roots[b['root_key']]['latent_path'],
         'root_noise_seed': roots[b['root_key']]['noise_seed'], 'P': info[b['key']]['P'], 'D': info[b['key']]['D'],
         'collector_fork_noise': str(Path(b['latent_path']).parent / (b['root_key'] + '_fork_noise.pt')),
         'out': str(noise_dir / (b['key'] + '.pt'))} for b in positives], indent=0))

    def row(rec, kind):
        d = info[rec['key']]['D'] if kind == 'branch' else 1
        out = {'sample_id': f"{kind}_{rec['key']}", 'original_sample_id': rec['sample_id'], 'training_noise_group_id': rec['key'],
               'task': 'vbvr', 'subtask': rec['task'], 'split': 'train', 'prompt': rec['row']['prompt'],
               'source_path': roots[rec.get('root_key', rec['key'])]['source_path'], 'latent_path': rec['latent_path'],
               'actual_latent_frames': rec['generated_latents'], 'validity_mask_start': d, 'validity_mask_end': rec['generated_latents'],
               'rft_kind': kind, 'collection_key': rec['key']}
        if kind == 'branch':
            out.update(shared_prefix_sha256=rec['shared_prefix_sha256'], failed_noise_path=str(noise_dir / (rec['key'] + '.pt')),
                       divergence_latent=d, branch_prefix=rec['prefix'])
        return out
    n_pos = len(positives); assert n_pos > 0
    rows_per_epoch = math.ceil(n_pos / (1 - ANCHOR_SHARE) / (2 * BATCH * 50)) * 2 * BATCH * 50
    rng0 = random.Random(20261030)
    root_rows = rng0.sample(root_pool, min(n_pos, len(root_pool)))
    while root_pool and len(root_rows) < n_pos:
        extra = list(root_pool); rng0.shuffle(extra); root_rows += extra[:n_pos - len(root_rows)]
    pools = {'branch': [row(b, 'branch') for b in positives], 'root': [row(r, 'root') for r in root_rows], 'none': []}
    arms = {'cft': 'branch', 'decision_rft': 'branch', 'sample_rft': 'root', 'sft': 'none'}   # method -> rows of the method
    steps = rows_per_epoch * a.epochs // BATCH; half = rows_per_epoch // 2 // BATCH
    named = [start + half * k for k in range(1, 2 * a.epochs + 1)]
    report = {'roots': len(roots), 'roots_strict_optimal': sum(bool(r['success']) for r in roots.values()), 'branches': len(branches),
              'branches_strict': sum(bool(b['success']) for b in branches), 'positives': n_pos,
              'positives_by_task': Counter(b['task'] for b in positives), 'positive_puzzles': len({b['sample_id'] for b in positives}),
              'rejections': rejections, 'D_minus_P': Counter(min(info[b['key']]['D'] - info[b['key']]['P'], 4) for b in positives),
              'mixed_puzzles': len(mixed), 'root_pool': len(root_pool), 'rows_per_epoch': rows_per_epoch, 'updates': steps,
              'checkpoints': named, 'parent_checkpoint': str(checkpoint), 'arms': {}}
    for arm, pool in arms.items():
        if pool == 'root' and not pools['root']:
            report['arms'][arm] = 'skipped: no mixed puzzle'; continue
        rng = random.Random(20261030); pos = pools[pool]; n_anchor = rows_per_epoch - len(pos); assert n_anchor > 0
        anchors = []
        while len(anchors) < n_anchor * a.epochs:
            p = list(train); rng.shuffle(p); anchors += p
        rows = []
        for e in range(a.epochs):
            rows += [{**r, 'sample_id': f"{r['sample_id']}_e{e}", 'epoch_copy': e} for r in pos]
            rows += [{**x, 'base_row_idx': x['row_idx'], 'rft_kind': 'anchor', 'epoch_copy': e} for x in anchors[e * n_anchor:(e + 1) * n_anchor]]
        rng.shuffle(rows)
        rows += [{**x, 'base_row_idx': x['row_idx'], 'rft_kind': 'anchor'} for x in val]
        for i, r in enumerate(rows):
            r['row_idx'] = i
        manifest = a.out / f'metadata_{arm}.json'; manifest.write_text(json.dumps(rows))
        n_train = sum(r['split'] == 'train' for r in rows); assert n_train == rows_per_epoch * a.epochs
        report['arms'][arm] = {'train_rows': n_train, 'expected_train_rows': n_train, 'metadata_path': str(manifest),
                               'resume_lr_schedule_steps': steps, 'max_iters': start + steps, 'kinds': Counter(r['rft_kind'] for r in rows if r['split'] == 'train'),
                               'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}
    (a.out / 'PREPARED.json').write_text(json.dumps(report, indent=1, default=dict)); print(json.dumps(report, indent=1, default=dict))


from multiprocessing import Pool

FORK_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(FORK_ROOT / 'src'))

from videcide.paths import manifest_path
from videcide.tasks import vbvr_checkers as V
import videcide.training.vbvr_fork as VF
from videcide.training.vbvr_fork import select_fork

VF.BACK_STATES.update({'premature_eos': 4, 'unsolvable': 6})

SRC = OUT = None   # set by the forks sub-command (--src, --out)
TASKS = ('hanoi', 'sliding', 'sokoban')   # the three weakest sub-tasks of the SFT model


def optimal(task, m, meta):
    if task == 'hanoi':
        return m['moves'] <= m['optimal_moves']
    if task == 'sliding':
        return len(m['changes']) - 1 <= meta['solution_length']
    return True


def work(path):
    r = json.loads(Path(path).read_text()); task = r['task'].replace('vbvr_', '')
    if task == 'hanoi' and r['success'] and r['metrics']['moves'] <= r['metrics']['optimal_moves']:
        return r, None, 'strict+optimal'
    frames = V.read(r['mp4'])
    m = V.check(r['row'], frames)
    meta = V.puzzle(r['row'])
    opt = bool(m['success']) and optimal(task, m, meta)
    if opt:
        return r, None, 'strict+optimal'
    if task == 'sliding':
        m['budget'] = meta['solution_length']
    mm = dict(m, success=False)
    sel = select_fork(frames, task, mm, floor=1)
    return r, {'metrics': {k: v for k, v in m.items() if k not in ('changes',)}, 'strict': bool(m['success']),
               'optimal': opt, 'selection': sel}, 'strict_nonoptimal' if m['success'] else 'strict_fail'


def forks_main():
    global SRC, OUT
    ap = argparse.ArgumentParser(); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--src', type=Path, required=True, help='collection whose roots are re-checked (collect_vbvr.py --out)')
    ap.add_argument('--out', type=Path, required=True, help='new collection folder: re-checked roots, forks and manifest.jsonl')
    a = ap.parse_args(); SRC, OUT = a.src, a.out
    for d in ('roots/records', 'forks'):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    paths = []
    for t in TASKS:
        ps = sorted((SRC / 'roots/records').glob(f'vbvr_{t}_*.json'))
        paths += ps[:a.limit] if a.limit else ps
    stats = defaultdict(Counter); prefix = defaultdict(list)
    with Pool(64) as pool:
        for r, fk, status in pool.imap_unordered(work, map(str, paths), chunksize=8):
            t = r['task'].replace('vbvr_', ''); stats[t][status] += 1
            rec = dict(r, success=fk is None, success_definition='strict_and_optimal')
            (OUT / 'roots/records' / f"{r['key']}.json").write_text(json.dumps(rec))
            if fk is not None:
                sel = fk['selection']; stats[t]['fork_' + sel['status']] += 1
                if sel['status'] == 'ok':
                    stats[t]['kind:' + sel['kind']] += 1; prefix[t].append(sel['prefix'])
                (OUT / 'forks' / f"{r['key']}.json").write_text(json.dumps({**rec, **fk}))
    summary = {}
    for t in TASKS:
        p = sorted(prefix[t])
        summary[t] = dict(stats[t], prefix_eq1=round(sum(x == 1 for x in p) / max(1, len(p)), 3),
                          prefix_median=p[len(p) // 2] if p else None)
        print(t, json.dumps(summary[t]), flush=True)
    (OUT / 'FORK_STATS.json').write_text(json.dumps(summary, indent=1))
    ids = {json.loads(f.read_text())['sample_id'] for f in (OUT / 'forks').glob('*.json')}
    with open(OUT / 'manifest.jsonl', 'w') as f:
        for r in map(json.loads, open(manifest_path('vbvr', 'train'))):
            if r['sample_id'] in ids:
                f.write(json.dumps(r) + '\n')
    print('manifest rows', len(ids), flush=True)


def forks_command():
    forks_main()


import torch

MAX_LATENTS = 80


def noise_main():
    pos = json.loads(Path(sys.argv[1]).read_text()); dev = torch.device('cuda'); dtype = torch.bfloat16
    by_root = {}
    for p in pos:
        by_root.setdefault(p['root_key'], []).append(p)
    checked = written = 0
    for rk, items in by_root.items():
        shape = torch.load(items[0]['root_latent_path'], map_location='cpu', weights_only=False, mmap=True)['latents'].shape[1:]
        seed = items[0]['root_noise_seed']
        noise = torch.randn((1, MAX_LATENTS - 1, *shape), device=dev, dtype=dtype,
                            generator=torch.Generator(device=dev).manual_seed(seed))[0].cpu()
        ref = torch.load(items[0]['collector_fork_noise'], map_location='cpu', weights_only=False)
        P = int(ref['prefix']); assert int(ref['root_noise_seed']) == int(seed)
        assert torch.equal(noise[P - 1], ref['failed_fork_noise']), f'noise rebuild mismatch for {rk}'
        checked += 1
        for p in items:
            d = int(p['D']); assert d >= int(p['P']) == P
            out = Path(p['out']); out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'failed_fork_noise': noise[d - 1].clone(), 'divergence_latent': d, 'branch_prefix': P, 'root_noise_seed': seed}, out)
            written += 1
    print(json.dumps({'roots_checked_bit_exact': checked, 'noise_files': written}))


def noise_command():
    noise_main()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ('forks', 'noise'):
        {'forks': forks_command, 'noise': noise_command}[sys.argv.pop(1)]()
    else:
        main()
