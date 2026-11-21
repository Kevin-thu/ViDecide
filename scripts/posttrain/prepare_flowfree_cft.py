"""FlowFree CFT data: training manifests of CFT and its baselines from rollout collections (Sec. 4.4).

Steps:
    forks OUT:MEDIA [...]   choose the fork point of every failed rollout of collect_flowfree.py (also done there)
    (default)               gate the branches, find the divergence step, and write metadata_cft.json,
                            metadata_decision_rft.json (the same rows), metadata_sample_rft.json and metadata_sft.json
The expert anchors are the SFT training videos and the 5,000 harder training puzzles, whose latents are built with
    python scripts/benchmarks/build_latents.py lengths --task flowfree --set hard_train   (then encode, merge)
Train with configs/cft/flowfree.yaml and metadata_path=<out>/metadata_<method>.json (see prepared_protocol.json for
the number of training rows of each manifest).
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from videcide.paths import latents_dir as latents_folder


PARENT = None   # SFT run of the policy (--sft-run)
START = 47500


def mixed_successful_roots(roots):
    groups = defaultdict(list)
    for r in roots:
        groups[r['sample_id']].append(r)
    return [r for rows in groups.values() if 0 < sum(x['success'] for x in rows) < len(rows)
            for r in rows if r['success']]


def verify(job):
    record, root_record, media = job
    import numpy as np
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
    from videcide.tasks.video_parser import parse_video
    p = record['prefix']; assert 1 <= p < record['generated_latents']
    root = torch.load(root_record['latent_path'], map_location='cpu', weights_only=False, mmap=True)
    branch = torch.load(record['latent_path'], map_location='cpu', weights_only=False, mmap=True)
    assert torch.equal(branch['latents'][:p], root['latents'][:p]), record['key']
    digest = hashlib.sha256(root['latents'][:p].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    assert digest == record['shared_prefix_sha256']
    noise = torch.load(Path(record['failed_noise_path']), map_location='cpu', weights_only=False)
    assert noise['prefix'] == p and noise['root_noise_seed'] == root['noise_seed'] and noise['shared_prefix_sha256'] == digest
    if record['success']:
        assert parse_video(np.load(record['frame_path'])['frames'], record['source'], 'flowfree')['success'], record['key']
    return record['key']


def divergence(job):
    record, root_record, out = job
    import numpy as np
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
    from videcide.tasks.flow_replay import evaluate_flowfree
    from videcide.training.fork_selection import move_start_frame
    src = record['source']; p = record['prefix']
    walks = []
    for path in (root_record['frame_path'], record['frame_path']):
        frames = np.load(path)['frames']
        trace = evaluate_flowfree(frames, src, include_trace=True, require_adjacent_color_switch=True)['trace']
        walks.append((frames, trace, [((int(r), int(c), int(col)), int(e['frame'])) for e in trace for r, c, col in e.get('added', [])]))
    k = 0
    while k < min(len(walks[0][2]), len(walks[1][2])) and walks[0][2][k][0] == walks[1][2][k][0]:
        k += 1
    starts = [move_start_frame(fr, src, tr, mv[k][1]) for fr, tr, mv in walks if k < len(mv)]
    assert starts, record['key']
    d = max(p, (max(1, min(starts)) - 1) // 4 + 1)
    sel = record['fork_selection'] if isinstance(record['fork_selection'], dict) else __import__('ast').literal_eval(record['fork_selection'])
    bf = int(sel['boundary_frame']); mv0 = walks[0][2]
    fatal = next((i for i, (_, f) in enumerate(mv0) if f >= bf), len(mv0))
    assert d < record['generated_latents'], record['key']
    noise = torch.load(record['fork_noise_file'], map_location='cpu', weights_only=False)
    assert noise['prefix'] == p and 'root_noise_from_prefix' in noise
    path = Path(out) / 'failed_noise' / f"{record['key']}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'failed_fork_noise': noise['root_noise_from_prefix'][d - p].clone(), 'divergence_latent': d, 'branch_prefix': p,
                'root_noise_seed': noise['root_noise_seed'], 'first_differing_move': k}, path)
    return record['key'], d, str(path), fatal - k


def hard_anchors(latents_dir, out):
    """Expert anchors of the harder training puzzles: their latents (build_latents.py --set hard_train) as .pt files."""
    import lmdb
    import numpy as np
    import torch
    meta = json.loads((latents_dir / 'metadata.json').read_text())
    env = lmdb.open(str(latents_dir / 'lmdb'), readonly=True, lock=False, readahead=False)
    (out / 'hard_anchor_latents').mkdir(parents=True, exist_ok=True); rows = []
    with env.begin() as txn:
        for m in meta:
            n = m['actual_latent_frames']
            raw = txn.get(f"latents_{m['row_idx']}_data".encode())
            lat = torch.from_numpy(np.frombuffer(raw, dtype=np.float16).reshape(n, 48, 24, 24).copy())
            path = out / 'hard_anchor_latents' / f"{m['sample_id']}.pt"
            if not path.exists():
                torch.save({'latents': lat}, path)
            rows.append({**m, 'split': 'train', 'latent_path': str(path), 'stored_latent_frames': n})
    return rows


def main():
    global PARENT, START
    ap = argparse.ArgumentParser()
    ap.add_argument('--collection', action='append', required=True, help='OUT:MEDIA[:FORKDIR:BRANCHDIR]')
    ap.add_argument('--max-moves-before-fatal', type=int, default=None, help='keep positives whose D is <= K root moves before the fatal move')
    ap.add_argument('--extra-positives', action='append', default=[],
                    help='DIR:OUT:MEDIA: successful continuations in DIR/records branched from the collection OUT:MEDIA')
    ap.add_argument('--hard-latents', type=Path, action='append', default=None,
                    help='latents of the harder training puzzles (default: data/latents/flowfree_hard_train/native)')
    ap.add_argument('--branches', type=int, default=4)
    ap.add_argument('--fork-min-backoff', type=int, default=0, help='0: forks/ and branches/ of the collection; k: forks_mb<k>/ '
                    'and branches_mb<k>/ (collect_flowfree.py --fork-min-backoff k)')
    ap.add_argument('--fork-no-solver', action='store_true', help='ablation: forks and branches chosen without the solver '
                    '(collect_flowfree.py --fork-no-solver)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--sft-run', type=Path, required=True, help='SFT run of the policy (resolved_config.json): expert anchors')
    ap.add_argument('--start-step', type=int, default=47500, help='SFT steps of the policy checkpoint')
    a = ap.parse_args()
    PARENT, START = a.sft_run, a.start_step
    config = json.loads((PARENT / 'resolved_config.json').read_text())
    metadata = json.loads(Path(config['metadata_path']).read_text())
    train = [r for r in metadata if r['split'] == 'train']; val = [r for r in metadata if r['split'] == 'val']
    assert len(train) == 9500 and len(val) == 500
    roots, branches, jobs, report = [], [], [], {'collections': {}, 'fork_min_backoff': a.fork_min_backoff}
    sfx = '_nosolver' if a.fork_no_solver else ''
    FD0 = ('forks' if a.fork_min_backoff == 0 else f'forks_mb{a.fork_min_backoff}') + sfx
    BD0 = ('branches' if a.fork_min_backoff == 0 else f'branches_mb{a.fork_min_backoff}') + sfx
    seen_outs = set()
    for spec in a.collection:
        parts = spec.split(':')
        out, media = map(Path, parts[:2])
        FD, BD = (parts[2], parts[3]) if len(parts) == 4 else (FD0, BD0)
        rs = [json.loads(f.read_text()) for f in sorted((out / 'roots/records').glob('*.json'))]
        by_key = {r['key']: r for r in rs}; expected = defaultdict(list)
        for r in rs:
            if r['success']:
                continue
            sel = json.loads((out / FD / (r['key'] + '.json')).read_text())['selection']
            assert sel['status'] != 'error'
            if sel['status'] == 'ok':
                assert sel['seam']['clean'] and (a.fork_no_solver or sel['fork_state'] == 'solvable')
                expected[r['sample_id']].append(r['key'])
        chosen = {k for v in expected.values() for k in sorted(v, key=lambda k: by_key[k]['rollout'])[:2]}
        bs = [json.loads(f.read_text()) for f in sorted((out / BD / 'records').glob('*.json'))]
        bs = [b for b in bs if b['root_key'] in chosen and b['branch'] < a.branches]
        missing = chosen - {b['root_key'] for b in bs}
        assert not missing, f'{out}: {len(missing)} chosen forks without branches'
        jobs += [(b, by_key[b['root_key']], str(media / BD)) for b in bs]
        for b in bs:
            b['failed_noise_path'] = str(media / BD / 'latents' / (b['root_key'] + '_fork_noise.pt'))
            b['fork_noise_file'] = b['failed_noise_path']
            b['key'] = f"{b['key']}_{BD}"; b['branch_dir'] = BD
        if str(out) not in seen_outs:
            roots += rs; seen_outs.add(str(out))
        branches += bs
        report['collections'][f'{out}:{BD}'] = dict(roots=len(rs), root_success=sum(r['success'] for r in rs), forks=len(chosen),
                                               branches=len(bs), positives=sum(b['success'] for b in bs))
    for spec in a.extra_positives:
        vd, out, media = map(Path, spec.split(':'))
        by_key = {r['key']: r for r in roots}
        for rec in sorted((vd / 'records').glob('*.json')):
            r = json.loads(rec.read_text())
            if not r['success']:
                continue
            root = by_key[r['root_key']]; fork = json.loads((out / 'forks' / (r['root_key'] + '.json')).read_text())
            import torch
            hist = torch.load(root['latent_path'], map_location='cpu', weights_only=False)['latents'][:r['prefix']]
            b = {**r, 'key': f"{r['root_key']}_x{vd.name}_b{r['branch']}", 'latent_path': str(vd / 'frames' / f"{r['key']}.pt"),
                 'frame_path': str(vd / 'frames' / f"{r['key']}.npz"), 'source': root['source'],
                 'shared_prefix_sha256': hashlib.sha256(hist.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                 'failed_noise_path': str(media / 'branches/latents' / (r['root_key'] + '_fork_noise.pt')),
                 'fork_selection': fork['selection']}
            branches.append(b); jobs.append((b, root, str(media / 'branches')))
        report.setdefault('extra_positives', {})[str(vd)] = sum(1 for b in branches if f'_x{vd.name}_' in b['key'])
    with ProcessPoolExecutor(16) as pool:
        verified = list(pool.map(verify, jobs, chunksize=8))
    assert len(verified) == len(branches)
    positives = [b for b in branches if b['success']]
    by_key = {r['key']: r for r in roots}
    if True:
        with ProcessPoolExecutor(16) as pool:
            dv = {k: (d, path, mbf) for k, d, path, mbf in pool.map(divergence, [(b, by_key[b['root_key']], str(a.out)) for b in positives], chunksize=4)}
        for b in positives:
            b['divergence_latent'], b['failed_noise_path'], b['moves_before_fatal'] = dv[b['key']]
        report['divergence_minus_fork'] = dict(sorted(Counter(b['divergence_latent'] - b['prefix'] for b in positives).items()))
        report['moves_before_fatal_all'] = dict(sorted(Counter(min(b['moves_before_fatal'], 20) for b in positives).items()))
        report['moves_before_fatal_by_dir'] = {d_: dict(sorted(Counter(min(b['moves_before_fatal'], 20) for b in positives if b['branch_dir'] == d_).items()))
                                               for d_ in sorted({b['branch_dir'] for b in positives})}
        if a.max_moves_before_fatal is not None:
            positives = [b for b in positives if b['moves_before_fatal'] <= a.max_moves_before_fatal]
            report['max_moves_before_fatal'] = a.max_moves_before_fatal; report['positives_after_filter'] = len(positives)
    root_pool = mixed_successful_roots(roots)
    anchors = list(train)
    for latents_dir in a.hard_latents or [latents_folder('flowfree_hard_train', 'native')]:
        anchors += hard_anchors(latents_dir, a.out)
    rng = random.Random(20261026); rng.shuffle(positives); rng.shuffle(anchors); rng.shuffle(root_pool)

    def anchor_row(anchor, idx):
        row = {**anchor, 'row_idx': idx, 'rft_kind': 'anchor', 'split': 'train'}
        if not anchor.get('latent_path'):
            row['base_row_idx'] = anchor['row_idx']
        return row

    def positive_row(r, idx, tag):
        branch = 'root_key' in r and tag != 'root'
        parent = by_key[r['root_key']] if branch else r
        row = {'row_idx': idx, 'sample_id': f'{tag}_{idx}_{r["key"]}', 'original_sample_id': r['sample_id'],
               'training_noise_group_id': r['key'], 'task': 'flowfree', 'split': 'train', 'prompt': r['source']['prompt'],
               'source_path': parent['source_path'], 'latent_path': r['latent_path'], 'actual_latent_frames': r['generated_latents'],
               'validity_mask_start': r.get('divergence_latent', r['prefix']) if branch else 1, 'validity_mask_end': r['generated_latents'],
               'rft_kind': 'branch' if branch else 'root', 'collection_key': r['key']}
        if branch:
            row['shared_prefix_sha256'] = r['shared_prefix_sha256']; row['failed_noise_path'] = r['failed_noise_path']
            row['divergence_latent'] = r.get('divergence_latent', r['prefix']); row['branch_prefix'] = r['prefix']
            assert Path(row['failed_noise_path']).is_file()
        return row

    def build(pool, tag):
        units = max(len(pool), len(anchors)); rows = []
        for i in range(units):
            rows.append(anchor_row(anchors[i % len(anchors)], len(rows)))
            for j in range(3):
                rows.append(positive_row(pool[(3 * i + j) % len(pool)], len(rows), tag))
        for v in val:
            rows.append({**v, 'row_idx': len(rows), 'base_row_idx': v['row_idx'], 'rft_kind': 'anchor'})
        return rows, units

    if not positives:
        sys.exit('no successful continuation in the collections: collect more puzzles (or more rollouts per puzzle)')
    a.out.mkdir(parents=True, exist_ok=True)
    main_rows, units = build(positives, 'positive')
    root_rows, runits = build(root_pool, 'root') if root_pool else (None, 0)
    sft_rows = [anchor_row(x, i) for i, x in enumerate(anchors)]
    sft_rows += [{**v, 'row_idx': len(sft_rows) + i, 'base_row_idx': v['row_idx'], 'rft_kind': 'anchor'} for i, v in enumerate(val)]
    files, expected_rows = {}, {}
    for method, rows, n_train in (('cft', main_rows, 4 * units), ('decision_rft', main_rows, 4 * units),
                                  ('sample_rft', root_rows, 4 * runits), ('sft', sft_rows, len(anchors))):
        if rows is None:     # sample-level RFT needs a puzzle with both a successful and a failed rollout
            expected_rows[method] = 'skipped: no puzzle with both a successful and a failed rollout'
            continue
        manifest = a.out / f'metadata_{method}.json'
        manifest.write_text(json.dumps(rows)); files[manifest.name] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        expected_rows[method] = dict(metadata_path=str(manifest), expected_train_rows=n_train)
    report['training_manifests'] = expected_rows
    report.update(positives=len(positives), positive_puzzles=len({b['sample_id'] for b in positives}),
                  positive_by_grid=dict(Counter(b['source']['grid_size'] for b in positives)),
                  branches=len(branches), verified=len(verified), sample_rft_roots=len(root_pool),
                  anchors=len(anchors), hard_anchors=len(anchors) - len(train), units=units, root_units=runits,
                  manifests=files, parent=str(PARENT / f'checkpoint_model_{START:06d}/model.pt'),
                  free_storage_bytes=shutil.disk_usage('.').free)
    (a.out / 'prepared_protocol.json').write_text(json.dumps(report, indent=2)); print(json.dumps(report), flush=True)


from collect_flowfree import select

def forks_command():
    ap = argparse.ArgumentParser(description='fork points of the failed rollouts of FlowFree collections')
    ap.add_argument('sets', nargs='+', help='OUT:MEDIA pairs of collect_flowfree.py (--out and --media)')
    SETS = [tuple(map(Path, x.split(':'))) for x in ap.parse_args().sets]
    for mb in (0, 3):
        fdir = 'forks' if mb == 0 else f'forks_mb{mb}'
        todo = []
        for out, media in SETS:
            for f in (out / 'roots/records').glob('*.json'):
                r = json.loads(f.read_text())
                if not r['success'] and not (out / fdir / f'{r["key"]}.json').exists():
                    todo.append((r, str(media), str(out), mb))
        print(fdir, 'todo', len(todo), flush=True)
        with ProcessPoolExecutor(48, mp_context=__import__('multiprocessing').get_context('spawn')) as pool:
            st = list(pool.map(select, todo, chunksize=2))
        print(fdir, Counter(st), flush=True)
        for out, _ in SETS:
            c = Counter(json.loads(f.read_text())['selection']['status'] for f in (out / fdir).glob('*.json'))
            print(' ', out.name, fdir, dict(c), flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ('forks',):
        {'forks': forks_command}[sys.argv.pop(1)]()
    else:
        main()
