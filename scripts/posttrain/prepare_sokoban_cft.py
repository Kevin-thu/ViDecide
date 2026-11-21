"""Sokoban CFT data: training manifests of CFT and its baselines from a rollout collection (Sec. 4.4).

    python scripts/posttrain/prepare_sokoban_cft.py all --collect outputs/cft_collect/sokoban --out outputs/cft_data/sokoban

runs every step in order (the collection comes from scripts/posttrain/collect_sokoban.py; experts and noise need one
GPU). The steps, each a sub-command:
    experts    render and encode the expert videos of the 5,000 harder training puzzles with the Wan2.2 VAE (anchors),
               into data/latents/sokoban_hard_train/experts (puzzles already encoded are skipped)
    arms       per-method manifests from the collection: metadata_{cft,decision_rft,sample_rft,sft}.json in --out
                   arms --collect COLLECT --out OUT/arms
    sft-hard   continued-SFT manifest whose experts also cover the 5,000 harder puzzles (metadata_sft_hard.json)
                   sft-hard --train OUT/arms
    decisions  divergence step d of every successful branch (first latent where branch and failed rollout differ)
                   decisions --train OUT/arms --collect COLLECT --out OUT/arms/decision_latents.json
    noise      noise of the failed rollout at d (the CFT target noise)
                   noise --train OUT/arms --roots COLLECT/roots/records --out OUT/decision_noise
    (default)  final manifests: branch rows with d and the noise path, anchors from the 14,500 experts
                   --src OUT/arms --out OUT
The expert latents of the 9,500 training and 500 validation videos are those of SFT (data/latents/sokoban/native,
built with scripts/benchmarks/build_latents.py). Train with configs/cft/sokoban.yaml and
metadata_path=OUT/metadata_<method>.json (rft_mode: cft, decision_rft, sample_rft or sft; the number of training rows
of every manifest is in OUT/PREPARED.json).
"""
import argparse, hashlib, json, random
from collections import Counter
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks')); sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from videcide.paths import data_path, latents_dir, manifest_path

HARD = latents_dir('sokoban_hard_train', 'experts')               # expert latents of the harder puzzles (experts step)
SFT_METADATA = latents_dir('sokoban', 'native') / 'metadata.json'   # 9,500 train / 500 val experts (LMDB)


def load_metadata(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--src', type=Path, required=True, help='output of the arms / sft-hard / decisions steps')
    ap.add_argument('--arms', default='cft,decision_rft,sample_rft,sft')
    ap.add_argument('--expert-metadata', type=Path, default=SFT_METADATA, help='metadata of the SFT expert latents')
    ap.add_argument('--experts', type=Path, default=HARD, help='expert latents of the harder puzzles (experts step)')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    dec = json.loads((a.src / 'decision_latents.json').read_text()); prep = json.loads((a.src / 'PREPARED.json').read_text())
    rows_per_epoch, epochs = prep['rows_per_epoch'], 2
    hard_rows = {r['sample_id']: r for r in json.loads((a.src / 'metadata_sft_hard.json').read_text()) if r['rft_kind'] == 'root'}
    meta = load_metadata(a.expert_metadata)
    train = [r for r in meta if r['split'] == 'train']; val = [r for r in meta if r['split'] == 'val']
    hard = [{k: v for k, v in r.items() if k not in ('sample_id', 'epoch_copy', 'row_idx')} | {'sample_id': r['sample_id'].rsplit('_e', 1)[0], 'rft_kind': 'anchor'}
            for r in hard_rows.values()]
    hard = list({r['sample_id']: r for r in hard}.values())
    if len(hard) < 5000:   # a small collection: its sft-hard manifest samples only some of the harder experts
        seen = {r['sample_id'] for r in hard}
        hard += [{k: v for k, v in r.items() if k != 'sample_id'} | {'sample_id': r['sample_id'], 'rft_kind': 'anchor'}
                 for r in hard_experts(a.experts) if r['sample_id'] not in seen]
    assert len(hard) == 5000 and len(train) == 9500
    pool = [('lmdb', x) for x in train] + [('pt', x) for x in hard]
    report = {'arms': {}}
    for arm, src, mode in [(x, x, x) for x in a.arms.split(',')]:
        old = json.loads((a.src / f'metadata_{src}.json').read_text())
        pos = [r for r in old if r['split'] == 'train' and r['rft_kind'] != 'anchor']
        for r in pos:
            if r['rft_kind'] == 'branch':
                v = dec[r['collection_key']]; assert v['p'] == r['validity_mask_start']
                r['divergence_latent'] = v['d']; r['decision_rule'] = v['rule']
                r['failed_noise_path'] = str(a.out / 'decision_noise' / f"{r['collection_key']}.pt")
        n_anchor = rows_per_epoch * epochs - len(pos); assert n_anchor % epochs == 0
        rng = random.Random(20261017); stream = []
        while len(stream) < n_anchor:
            p = list(pool); rng.shuffle(p); stream += p
        anchors = []
        for i, (kind, x) in enumerate(stream[:n_anchor]):
            e = i * epochs // n_anchor
            anchors.append({**x, 'base_row_idx': x['row_idx'], 'rft_kind': 'anchor', 'epoch_copy': e} if kind == 'lmdb'
                           else {**x, 'sample_id': f"{x['sample_id']}_a{i}", 'epoch_copy': e})
        rows = pos + anchors; rng.shuffle(rows)
        rows += [{**x, 'base_row_idx': x['row_idx'], 'rft_kind': 'anchor'} for x in val]
        for i, r in enumerate(rows):
            r['row_idx'] = i
        man = a.out / f'metadata_{arm}.json'; man.write_text(json.dumps(rows))
        n_train = sum(r['split'] == 'train' for r in rows); assert n_train == rows_per_epoch * epochs
        tr = [r for r in rows if r['split'] == 'train']
        report['arms'][arm] = {'train_rows': n_train, 'expected_train_rows': n_train, 'metadata_path': str(man),
                               'kinds': Counter(r['rft_kind'] for r in tr),
                               'anchors_hard': sum(r['rft_kind'] == 'anchor' and bool(r.get('latent_path')) for r in tr),
                               'decision_shift': Counter(min(r['divergence_latent'] - r['validity_mask_start'], 5) for r in tr if r['rft_kind'] == 'branch'),
                               'manifest_sha256': hashlib.sha256(man.read_bytes()).hexdigest()}
    (a.out / 'PREPARED.json').write_text(json.dumps(report, indent=1, default=dict)); print(json.dumps(report, indent=1, default=dict))


import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

def decision_one(key, collect, roots, extra_collect=None):
    """Divergence latent d of one successful branch (keys with the k12_ prefix come from --extra-collect)."""
    from videcide.training.sokoban_fork import trace
    if key.startswith('k12_'):
        brp = f'{extra_collect}/branches/records/{key[4:]}.json'
    else:
        brp = f'{collect}/branches/records/{key}.json'
    rootdir = f'{roots}/'
    br = json.load(open(brp)); root = json.load(open(rootdir + br['root_key'] + '.json'))
    p = br['prefix']; n = br['generated_latents']
    try:
        sb = trace(np.load(br['frame_path'])['frames'], br['source'])['parse'].per_frame_state
        sr = trace(np.load(root['frame_path'])['frames'], br['source'])['parse'].per_frame_state
        d, found = p, False
        for f in range(4 * (p - 1) + 1, min(len(sb), len(sr))):
            if sb[f] is not None and sr[f] is not None and sb[f] != sr[f]:
                d, found = (f - 1) // 4 + 1, True; break
        rule = 'first_state_difference'
        if not found:
            cands = br['fork_selection'].get('candidates', {})
            vis = [v[1] for k, v in cands.items() if k in ('hallucination', 'illegal_move', 'premature_eos')]
            if vis and (min(vis) - 1) // 4 + 1 > p:
                d, rule = (min(vis) - 1) // 4 + 1, 'root_visible_failure'
            else:
                rule = 'fork_latent'
        d = min(d, n - 1)
        return key, {'p': p, 'd': d, 'n': n, 'rule': rule, 'root_key': br['root_key'], 'root_latent_path': root['latent_path']}
    except Exception as e:
        return key, {'p': p, 'd': p, 'n': n, 'root_key': br['root_key'], 'root_latent_path': root['latent_path'], 'error': repr(e)}


def decisions_command():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True, help='decision_latents.json to write')
    ap.add_argument('--train', required=True, help='output folder of the arms step')
    ap.add_argument('--collect', required=True, help='collection folder of the branches (collect_sokoban.py --out)')
    ap.add_argument('--roots', help='<collection>/roots/records of the failed rollouts (default: <collect>/roots/records)')
    ap.add_argument('--extra-collect', help='collection of the extra positives (keys k12_*), if the arms step used them')
    a = ap.parse_args()
    roots = a.roots or f'{a.collect}/roots/records'
    rows = json.load(open(f'{a.train}/metadata_cft.json'))
    keys = sorted({r['collection_key'] for r in rows if r['rft_kind'] == 'branch'})
    from functools import partial
    with ProcessPoolExecutor(64) as ex:
        res = dict(ex.map(partial(decision_one, collect=a.collect, roots=roots, extra_collect=a.extra_collect), keys, chunksize=8))
    a.out.write_text(json.dumps(res))
    import collections
    c = collections.Counter(v['d'] - v['p'] for v in res.values())
    print(len(res), 'positives; d - p:', sorted(c.items())[:10], 'errors', sum('error' in v for v in res.values()),
          'rules', collections.Counter(v.get('rule') for v in res.values()))


import torch

def noise_main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--train', required=True, help='output folder of the arms step (with decision_latents.json)')
    ap.add_argument('--roots', required=True, help='<collection>/roots/records')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    dec = json.load(open(f'{a.train}/decision_latents.json'))
    rows = {r['collection_key']: r for r in json.load(open(f'{a.train}/metadata_cft.json')) if r['rft_kind'] == 'branch'}
    checked = 0
    for key, v in dec.items():
        root = json.load(open(f"{a.roots}/{v['root_key']}.json")); d = v['d']
        noise = torch.randn((1, 79, 48, 24, 24), device='cuda', dtype=torch.bfloat16,
                            generator=torch.Generator(device='cuda').manual_seed(root['noise_seed']))[0, d - 1].cpu().clone()
        if d == v['p']:
            old = torch.load(rows[key]['failed_noise_path'], map_location='cpu', weights_only=False)['failed_fork_noise']
            assert torch.equal(old, noise), key; checked += 1
        torch.save({'failed_fork_noise': noise, 'latent': d, 'root_noise_seed': root['noise_seed']}, a.out / f'{key}.pt')
    print('written', len(dec), 'checked against the collector at d == p:', checked)


def noise_command():
    noise_main()


import os

os.environ.setdefault('OMP_NUM_THREADS', '1')


def expert_render(row):
    from build_latents import render as R
    video, _ = R('sokoban', str(data_path(row['source_path'])))
    assert len(video) == row['native_frames'], (row['sample_id'], len(video), row['native_frames'])
    return row, video


def experts_main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, default=HARD)
    ap.add_argument('--batch', type=int, default=4); ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--model-name', default='checkpoints/Wan2.2-TI2V-5B'); a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = list(map(json.loads, open(manifest_path('sokoban', 'hard_train'))))
    rows = [r for r in rows if not (a.out / f"{r['sample_id']}.pt").exists()]
    rows.sort(key=lambda r: r['native_frames'])
    print('to encode', len(rows), flush=True)
    from videcide.generation import wan_setup as P
    name = a.model_name
    vae = P.WanVAEWrapper(model_name=name).to(device='cuda', dtype=torch.bfloat16).eval()
    done = 0; pending = []

    def flush():
        nonlocal done
        x = torch.from_numpy(np.stack([v for _, v in pending])).to('cuda').permute(0, 4, 1, 2, 3).to(torch.bfloat16).div_(127.5).sub_(1.0)
        with torch.no_grad():
            lat = vae.encode_to_latent(x).to(torch.float16).cpu()
        for (row, _), z in zip(pending, lat):
            assert z.shape[0] == (row['native_frames'] - 1) // 4 + 1
            torch.save({'latents': z.clone(), 'sample_id': row['sample_id'], 'native_frames': row['native_frames']}, a.out / f"{row['sample_id']}.pt")
        done += len(pending); pending.clear(); print('encoded', done, flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for row, video in ex.map(expert_render, rows, chunksize=4):
            if pending and len(video) != len(pending[0][1]):
                flush()
            pending.append((row, video))
            if len(pending) >= a.batch:
                flush()
    if pending:
        flush()
    print('COMPLETE', done, flush=True)


def experts_command():
    experts_main()


ARMS_BATCH, ANCHOR_SHARE = 3, 5_042 / 20_100   # expert anchors fill about a quarter of every batch of 3


def verify_branch(job):
    """Gate of one branch: it shares the exact latent prefix of its root, its fork noise exists, a success re-passes."""
    rec, root = job
    import torch
    torch.set_num_threads(1)
    p = rec['prefix']
    a_ = torch.load(root['latent_path'], map_location='cpu', weights_only=False, mmap=True)['latents']
    b_ = torch.load(rec['latent_path'], map_location='cpu', weights_only=False, mmap=True)['latents']
    assert torch.equal(a_[:p], b_[:p]), rec['key']
    h = hashlib.sha256(a_[:p].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    assert h == rec['shared_prefix_sha256'], rec['key']
    noise = Path(rec['latent_path']).parent / (rec['root_key'] + '_fork_noise.pt')   # saved next to the branch latents
    assert noise.is_file(), noise
    if rec['success']:
        from videcide.tasks.video_parser import parse_video
        assert parse_video(np.load(rec['frame_path'])['frames'], rec['source'], 'sokoban')['success'], rec['key']
    return rec['key'], str(noise)


def arms_main():
    """Per-method manifests from a collection: successful branches (CFT / decision-level RFT), successful whole roots
    of mixed puzzles (sample-level RFT) or none (continued SFT), each topped up with expert anchors to the same number
    of rows (two epochs over the branch positives at a quarter of expert rows)."""
    from collections import defaultdict
    ap = argparse.ArgumentParser(description=arms_main.__doc__)
    ap.add_argument('--collect', type=Path, required=True, help='collect_sokoban.py output (roots/, branches/)')
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--partial', action='store_true', help='incomplete collection: keep fork groups with 4 or 8 branches')
    ap.add_argument('--extra-positives', type=Path, help='JSON {"kept": [branch record files]} of extra successful branches, '
                    'keyed k12_*: the paper adds the 332 successful continuations of collect_sokoban.py --backoffs 12 whose first '
                    'move change lies at most one latent after the fork')
    ap.add_argument('--expert-metadata', type=Path, default=SFT_METADATA)
    ap.add_argument('--start-step', type=int, default=47_500, help='SFT steps of the start model')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    if not a.partial:
        for f in a.collect.glob('status.*.json'):
            assert json.loads(f.read_text()).get('phase') == 'complete', f'{f} is not complete (pass --partial)'
    roots = {r['key']: r for r in (json.loads(f.read_text()) for f in (a.collect / 'roots/records').glob('*.json'))}
    branches = [json.loads(f.read_text()) for f in (a.collect / 'branches/records').glob('*.json')]
    by_root = defaultdict(list)
    for b in branches:
        by_root[b['root_key']].append(b)
    if a.partial:   # a stopped top-up keeps its first 4 branches; fewer than 4 -> dropped
        by_root = {k: (v if len(v) in (4, 8) else [b for b in v if b['branch'] < 4]) for k, v in by_root.items()}
    incomplete = [k for k, v in by_root.items() if len(v) not in (4, 8)]   # 8 = topped up (first 4 all failed)
    assert a.partial or not incomplete, incomplete[:5]
    branches = [b for k, v in by_root.items() if len(v) in (4, 8) for b in v]
    with ProcessPoolExecutor(16) as ex:
        noise = dict(ex.map(verify_branch, [(b, roots[b['root_key']]) for b in branches], chunksize=8))
    if a.extra_positives:
        extra = [json.loads(Path(f).read_text()) for f in json.loads(a.extra_positives.read_text())['kept']]
        with ProcessPoolExecutor(16) as ex:
            extra_noise = dict(ex.map(verify_branch, [(b, {'latent_path': b['root_latent_path']}) for b in extra], chunksize=8))
        for b in extra:
            noise['k12_' + b['key']] = extra_noise[b['key']]
            b['key'] = 'k12_' + b['key']
        assert all(b['success'] and b['root_key'] in roots for b in extra)
        branches = branches + extra
    positives = sorted([b for b in branches if b['success']], key=lambda b: b['key'])
    groups = defaultdict(list)
    for r in roots.values():
        groups[r['sample_id']].append(r)
    mixed = {s_ for s_, rs in groups.items() if 0 < sum(r['success'] for r in rs) < len(rs)}
    root_pool = sorted([r for r in roots.values() if r['success'] and r['sample_id'] in mixed], key=lambda r: r['key'])
    meta = load_metadata(a.expert_metadata)
    train = [r for r in meta if r['split'] == 'train']; val = [r for r in meta if r['split'] == 'val']
    assert len(train) == 9500 and len(val) == 500

    def row(rec, kind):
        start = rec['prefix'] if kind == 'branch' else 1
        out = {'sample_id': f"{kind}_{rec['key']}", 'original_sample_id': rec['sample_id'], 'training_noise_group_id': rec['key'],
               'task': 'sokoban', 'split': 'train', 'prompt': rec['source']['prompt'],
               'source_path': roots[rec.get('root_key', rec['key'])]['source_path'],
               'latent_path': rec['latent_path'], 'actual_latent_frames': rec['generated_latents'], 'validity_mask_start': start,
               'validity_mask_end': rec['generated_latents'], 'rft_kind': kind, 'collection_key': rec['key']}
        if kind == 'branch':
            out.update(shared_prefix_sha256=rec['shared_prefix_sha256'], failed_noise_path=noise[rec['key']])
        return out
    # whole half-epochs of 50-update checkpoints; at least one for a small collection
    rows_per_epoch = max(1, round(len(positives) / (1 - ANCHOR_SHARE) / (2 * ARMS_BATCH * 50))) * 2 * ARMS_BATCH * 50
    n_pos = len(positives)
    pools = {'branch': [row(b, 'branch') for b in positives],
             'root': [row(r, 'root') for r in random.Random(20261015).sample(root_pool, min(n_pos, len(root_pool)))], 'none': []}
    arms = {'cft': 'branch', 'decision_rft': 'branch', 'sample_rft': 'root', 'sft': 'none'}   # method -> rows of the method
    steps = rows_per_epoch * a.epochs // ARMS_BATCH; half = rows_per_epoch // 2 // ARMS_BATCH
    report = {'start_step': a.start_step, 'roots': len(roots), 'root_success': sum(r['success'] for r in roots.values()),
              'forks_branched': len(by_root), 'branches': len(branches), 'branch_positives': n_pos,
              'extra_k12_positives': sum(b['key'].startswith('k12_') for b in positives),
              'branch_positive_puzzles': len({b['sample_id'] for b in positives}), 'mixed_puzzles': len(mixed),
              'root_pool': len(root_pool), 'rows_per_epoch': rows_per_epoch, 'updates': steps,
              'checkpoints': [a.start_step + half * k for k in range(1, 2 * a.epochs + 1)],
              'incomplete_fork_groups': len(incomplete), 'arms': {}}
    for arm, pool in arms.items():
        rng = random.Random(20261015); pos = pools[pool]; n_anchor = rows_per_epoch - len(pos); assert n_anchor > 0
        anchors = []
        while len(anchors) < n_anchor * a.epochs:
            q = list(train); rng.shuffle(q); anchors += q
        rows = []
        for e in range(a.epochs):
            rows += [{**r, 'sample_id': f"{r['sample_id']}_e{e}", 'epoch_copy': e} for r in pos]
            rows += [{**x, 'base_row_idx': x['row_idx'], 'rft_kind': 'anchor', 'epoch_copy': e} for x in anchors[e * n_anchor:(e + 1) * n_anchor]]
        rng.shuffle(rows)
        rows += [{**x, 'base_row_idx': x['row_idx'], 'rft_kind': 'anchor'} for x in val]
        for i_, r in enumerate(rows):
            r['row_idx'] = i_
        manifest = a.out / f'metadata_{arm}.json'; manifest.write_text(json.dumps(rows))
        n_train = sum(r['split'] == 'train' for r in rows); assert n_train == rows_per_epoch * a.epochs
        report['arms'][arm] = {'train_rows': n_train, 'kinds': Counter(r['rft_kind'] for r in rows if r['split'] == 'train'),
                               'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}
    (a.out / 'PREPARED.json').write_text(json.dumps(report, indent=1, default=dict)); print(json.dumps(report, indent=1, default=dict))


def hard_experts(experts):
    """Rows of the encoded expert videos of the 5,000 harder training puzzles (experts step)."""
    import torch
    hard = []
    for r in map(json.loads, open(manifest_path('sokoban', 'hard_train'))):
        path = experts / f"{r['sample_id']}.pt"; n = torch.load(path, map_location='cpu', weights_only=False)['latents'].shape[0]
        assert n == (r['native_frames'] - 1) // 4 + 1
        hard.append({'sample_id': f"expert_{r['sample_id']}", 'original_sample_id': r['sample_id'], 'training_noise_group_id': r['sample_id'],
                     'task': 'sokoban', 'split': 'train', 'prompt': r['prompt'], 'source_path': r['source_path'],
                     'latent_path': str(path), 'actual_latent_frames': n, 'validity_mask_start': 1, 'validity_mask_end': n,
                     'rft_kind': 'root', 'collection_key': r['sample_id']})
    return hard


def sft_hard_main():
    """Continued-SFT manifest on the full Sokoban expert pool: the 9,500 original experts (LMDB) and the 5,000 harder
    ones (encoded by the experts step), with the same number of rows as the other arms (metadata_sft_hard.json)."""
    import torch
    ap = argparse.ArgumentParser(description=sft_hard_main.__doc__)
    ap.add_argument('--train', type=Path, required=True, help='output folder of the arms step')
    ap.add_argument('--experts', type=Path, default=HARD, help='encoded expert latents of the harder puzzles')
    ap.add_argument('--expert-metadata', type=Path, default=SFT_METADATA)
    ap.add_argument('--epochs', type=int, default=2); a = ap.parse_args()
    rows_per_epoch = json.loads((a.train / 'PREPARED.json').read_text())['rows_per_epoch']
    meta = load_metadata(a.expert_metadata)
    train = [r for r in meta if r['split'] == 'train']; val = [r for r in meta if r['split'] == 'val']
    assert len(train) == 9500 and len(val) == 500
    hard = hard_experts(a.experts)
    assert len(hard) == 5000
    pool = [('lmdb', x) for x in train] + [('hard', x) for x in hard]
    rng = random.Random(20261016); stream = []
    while len(stream) < rows_per_epoch * a.epochs:
        q = list(pool); rng.shuffle(q); stream += q
    rows = []
    for e in range(a.epochs):
        for kind, x in stream[e * rows_per_epoch:(e + 1) * rows_per_epoch]:
            rows.append({**x, 'base_row_idx': x['row_idx'], 'rft_kind': 'anchor', 'epoch_copy': e} if kind == 'lmdb'
                        else {**x, 'sample_id': f"{x['sample_id']}_e{e}", 'epoch_copy': e})
    rng.shuffle(rows)
    rows += [{**x, 'base_row_idx': x['row_idx'], 'rft_kind': 'anchor'} for x in val]
    for i_, r in enumerate(rows):
        r['row_idx'] = i_
    out = a.train / 'metadata_sft_hard.json'; out.write_text(json.dumps(rows))
    n_train = sum(r['split'] == 'train' for r in rows); assert n_train == rows_per_epoch * a.epochs
    kinds = Counter(('hard' if r['rft_kind'] == 'root' else 'orig') for r in rows if r['split'] == 'train')
    print(json.dumps({'train_rows': n_train, 'by_source': kinds, 'unique_hard': len({r['original_sample_id'] for r in rows if r['rft_kind'] == 'root'}),
                      'manifest_sha256': hashlib.sha256(out.read_bytes()).hexdigest()}, default=dict))


def all_main():
    """Every step in order: experts, arms (OUT/arms), sft-hard, decisions, noise (OUT/decision_noise) and the final
    manifests (OUT/metadata_<method>.json)."""
    ap = argparse.ArgumentParser(description=all_main.__doc__)
    ap.add_argument('--collect', type=Path, required=True, help='collect_sokoban.py output (roots/, branches/)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--experts', type=Path, default=HARD, help='expert latents of the harder puzzles (experts step)')
    ap.add_argument('--partial', action='store_true', help='arms: incomplete collection (see arms --help)')
    a = ap.parse_args()
    arms = a.out / 'arms'
    steps = [('experts', experts_main, ['--out', a.experts]),
             ('arms', arms_main, ['--collect', a.collect, '--out', arms] + (['--partial'] if a.partial else [])),
             ('sft-hard', sft_hard_main, ['--train', arms, '--experts', a.experts]),
             ('decisions', decisions_command, ['--train', arms, '--collect', a.collect, '--out', arms / 'decision_latents.json']),
             ('noise', noise_main, ['--train', arms, '--roots', a.collect / 'roots/records', '--out', a.out / 'decision_noise']),
             ('manifests', main, ['--src', arms, '--out', a.out])]
    for name, step, argv in steps:
        print(f'[prepare] {name}', flush=True)
        sys.argv = [sys.argv[0], *map(str, argv)]
        step()


SUBCOMMANDS = {'arms': arms_main, 'sft-hard': sft_hard_main, 'decisions': decisions_command, 'noise': noise_command,
               'experts': experts_command, 'all': all_main}


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in SUBCOMMANDS:
        SUBCOMMANDS[sys.argv.pop(1)]()
    else:
        main()
