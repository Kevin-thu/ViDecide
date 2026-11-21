"""RoboTwin-Plan CFT data: training manifests of CFT and its baselines from rollout collections (Sec. 4.4 and C.4).

Steps, after scripts/posttrain/collect_robotwin.py (several --collection folders are merged):
    recheck --collection DIR [DIR ...] --recheck-dir R [--keys FILE] [--gpu-part I --parts N]
        judge the collected videos again with the full verifier (the rules checked at collection time plus the DINOv3
        appearance classifier of scripts/robotwin/verifier.py). It covers the branches that succeeded at collection
        time or failed only its appearance check (every other rule is unchanged) and both roots of every puzzle with
        a failed root. Runs on the CPU (with CUDA_VISIBLE_DEVICES=''), or on GPU I of N with --gpu-part I --parts N.
    build --collection DIR [DIR ...] --recheck-dir R --sft-run RUN --out OUT
        positives: successful branches of failed roots in the post-training sub-tasks (--subtasks, by default the four
        with the lowest success), whose final arrangement the colour reader confirms. The divergence step D is the
        first K=8 segment from the fork on in which the branch and the root visibly differ, or in which one of them
        stops (EOS) and the other does not, or in which the root breaks an appearance rule. Writes one training
        manifest per method (expert videos fill a quarter of the rows): metadata_cft.json and metadata_decision_rft.json
        (the positives), metadata_sample_rft.json (successful roots of puzzles with one successful and one failed
        root), metadata_sft.json (expert videos only), and positives.json, PREPARED.json.
    noise OUT/positives.json
        (GPU) rebuild the noise of the failed root for segment D, bit-exact with the collection, into
        OUT/failed_noise/.
Train with configs/cft/robotwin.yaml and metadata_path=OUT/metadata_<method>.json (training rows in PREPARED.json).
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
import os
import random
import sys
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
for p in (HERE.parent / 'robotwin', HERE.parents[1] / 'src'):
    sys.path.insert(0, str(p))

import scene
import verifier as R

BATCH, BLOCK = 3, 8             # CFT batch (rows per update) and segment length K of the policy
MAX_LATENTS = 49                # collection cap (collect_robotwin.py)
ANCHOR_SHARE = 0.25             # share of expert-video rows in every manifest
SEED = 20261030                 # row sampling and shuffling
TASKS = ('blocksworld/row_2-4', 'blocksworld/triangle_2-6', 'hanoi/triangle_2-5', 'sort/nobuffer_4')
KEEP = {'plan_change_D_is_fork_block', 'plan_change_D_later_block', 'no_stop_at_goal', 'stopped_early', 'fix_appearance'}
METHODS = {'cft': 'branch', 'decision_rft': 'branch', 'sample_rft': 'root', 'sft': 'none'}   # method -> pool of rows
APPEARANCE_ONLY = {'unlike_expert_picture'}   # the collection-time appearance check, replaced by the classifier

COLLECTIONS: list[Path] = []    # --collection
SFT_RUN = None                  # --sft-run: SFT run of the policy (resolved_config.json, final checkpoint)
REC = None                      # --recheck-dir: verdicts of the recheck step
NORM = None                     # REC / 'normalized'


def as_dict(x):
    return x if isinstance(x, dict) else ast.literal_eval(x)


def record(latent_path):
    return json.loads(Path(latent_path.replace('/latents/', '/records/').replace('.pt', '.json')).read_text())


def load(collect: Path):
    roots = {}
    for f in (collect / 'roots/records').glob('*.json'):
        r = json.loads(f.read_text()); roots[r['key']] = r
    branches = [b for b in (json.loads(f.read_text()) for f in (collect / 'branches/records').glob('*.json')) if b['root_key'] in roots]
    return roots, branches


def load_all():
    roots, branches = {}, []
    for c in COLLECTIONS:
        r, b = load(c); roots.update(r); branches += b
    return roots, branches


def verify(job):
    """Prefix and noise of a branch match its root; for a success, the colour reader confirms the goal (True/False)."""
    rec, root = job
    import torch
    torch.set_num_threads(1)
    p = rec['prefix']
    a = torch.load(root['latent_path'], map_location='cpu', weights_only=False, mmap=True)['latents']
    b = torch.load(rec['latent_path'], map_location='cpu', weights_only=False, mmap=True)['latents']
    assert torch.equal(a[:p], b[:p]), rec['key']
    h = hashlib.sha256(a[:p].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    assert h == rec['shared_prefix_sha256'], rec['key']
    noise = Path(rec['latent_path']).parent / (rec['root_key'] + '_fork_noise.pt')
    tn = torch.load(noise, map_location='cpu', weights_only=False)
    assert tn['prefix'] == p and tn['shared_prefix_sha256'] == h and tuple(tn['failed_fork_block_noise'].shape)[0] == BLOCK, rec['key']
    colour = None
    if rec['success'] and rec['subtask'] in TASKS:
        import final_colour_verify as F
        row = as_dict(rec['row'])
        final = F.read_final(row, rec['mp4'])
        colour = final['certain'] and final['final'] == [list(s) for s in scene.Geometry(row).info['goal']]
    return rec['key'], str(noise), colour


# ----------------------------------------------------------------------------------------------------------- recheck
def job_of(rec, kind):
    return dict(key=rec['key'], kind=kind, video=rec['mp4'], subtask=rec.get('subtask'), rules=rec['metrics'])


def recheck_jobs():
    """Roots of every puzzle with a failed root, and the branches whose collection verdict can change."""
    roots, branches = load_all()
    roots = {k: r for k, r in roots.items() if r.get('subtask') in TASKS and r.get('metrics')}
    forked = {r['sample_id'] for r in roots.values() if not r['success']}
    out = [job_of(r, 'root') for r in sorted(roots.values(), key=lambda r: r['key'])
           if r['sample_id'] in forked and Path(r['mp4']).exists()]
    for b in sorted(branches, key=lambda b: b['key']):
        if b.get('subtask') in TASKS and b.get('metrics') and Path(b['mp4']).exists():
            types = {v['type'] for v in b['metrics'].get('violations', [])}
            if b['success'] or types <= APPEARANCE_ONLY:
                out.append(job_of(b, 'branch'))
    return out


def jobs_from_keys(path):
    """Jobs of the root / branch keys listed in a file."""
    want = set(Path(path).read_text().split())
    roots, branches = load_all()
    recs = [(r, 'root') for r in roots.values()] + [(b, 'branch') for b in branches]
    return [job_of(r, kind) for r, kind in recs if r['key'] in want and r.get('metrics')]


def _recheck_worker(out):
    R.OUT = Path(out)          # worker processes start fresh: set the output folder of verifier.one
    R.init_threads()


def recheck_main(a):
    REC.mkdir(parents=True, exist_ok=True)
    if not (REC / 'head.npz').exists():
        (REC / 'head.npz').symlink_to(R.ART.resolve() / 'head.npz')
    js = jobs_from_keys(a.keys) if a.keys else recheck_jobs()
    if a.gpu_part is not None:
        return gpu_part(js, a.gpu_part, a.parts)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        sys.exit("the CPU recheck runs with CUDA_VISIBLE_DEVICES=''; use --gpu-part I --parts N on GPUs")
    todo = sorted([j for j in js if not (REC / 'combined' / f'{j["key"]}.json').exists()], key=lambda j: j['kind'] != 'root')
    print('jobs', len(js), Counter(j['kind'] for j in js), 'todo', len(todo), flush=True)
    with ProcessPoolExecutor(int(os.environ.get('RECHECK_WORKERS', 24)), initializer=_recheck_worker, initargs=(str(REC),),
                             mp_context=__import__('multiprocessing').get_context('spawn')) as pool:
        for i, f in enumerate(as_completed([pool.submit(R.one, j) for j in todo])):
            f.result()
            if i % 200 == 0:
                print(i, flush=True)
    verdicts = {}
    for j in js:
        c = REC / 'combined' / f'{j["key"]}.json'
        if c.exists():
            n = R.normalize(json.loads(c.read_text())); R.save(NORM / f'{j["key"]}.json', n)
            verdicts[j['key']] = dict(kind=j['kind'], subtask=j['subtask'], collection=bool(j['rules']['success']),
                                      final=bool(n['success']), violation=n['violation'])
    counts = Counter(f"{v['kind']} collection={v['collection']} final={v['final']}" for v in verdicts.values())
    R.save(REC / 'SUMMARY.json', dict(n=len(js), judged=len(verdicts), counts=counts, verdicts=verdicts))
    print(json.dumps(counts, indent=1))


def gpu_part(js, part, parts):
    L = R.L
    mine = [j for j in js if zlib.crc32(j['key'].encode()) % parts == part
            and not (REC / 'combined' / f'{j["key"]}.json').exists()]
    print('gpu part', part, 'todo', len(mine), flush=True)
    model = R.gpu_init(); head = np.load(REC / 'head.npz'); cpu = False; n = 0
    for j in mine:
        dest = REC / 'videos' / f'{j["key"]}.json'
        if not dest.exists():
            res = R.gpu_infer(model, j['video'], head)
            if any(abs(x - res['threshold']) < R.GUARD for x in res['scores']):   # close to the threshold: exact CPU path
                if not cpu:
                    L.init(); L.write = R.save; cpu = True
                L.video_call((j, REC))
            else:
                R.save(dest, dict(j, **res))
        c = REC / 'combined' / f'{j["key"]}.json'
        if not c.exists():
            R.save(c, L.combine_with_rules(j['rules'], json.loads(dest.read_text())))
        n += 1
        if n % 200 == 0:
            print('gpu part', part, n, '/', len(mine), flush=True)
    print('gpu part', part, 'done', n, flush=True)


# ------------------------------------------------------------------------------------------------------------- noise
def noise_main(path):
    import torch
    pos = json.loads(Path(path).read_text()); dev = torch.device('cuda'); dtype = torch.bfloat16
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
        assert torch.equal(noise[P - 1:P - 1 + BLOCK], ref['failed_fork_block_noise']), f'noise rebuild mismatch for {rk}'
        checked += 1
        for p in items:
            d = int(p['D']); assert d >= int(p['P']) == P and (d - P) % BLOCK == 0 and d - 1 + BLOCK <= MAX_LATENTS - 1, p['key']
            out = Path(p['out']); out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'failed_fork_block_noise': noise[d - 1:d - 1 + BLOCK].clone(), 'failed_fork_noise': noise[d - 1].clone(),
                        'divergence_latent': d, 'branch_prefix': P, 'root_noise_seed': seed}, out)
            written += 1
    print(json.dumps({'roots_checked_bit_exact': checked, 'noise_files': written}))


# ------------------------------------------------------------------------------------------------------------- build
def lat(f):
    return 0 if f == 0 else (f - 1) // 4 + 1


def states(m):
    return [tuple(map(tuple, s['state'])) for s in m['rest_states']]


def info(r):
    b = record(r['latent_path']); root = record(b['root_latent_path']); s = b['fork_selection']; P = b['prefix']
    sa, sb = states(b['metrics']), states(root['metrics'])
    if sa == sb:
        div = None
    else:
        i = next((i for i in range(min(len(sa), len(sb))) if sa[i] != sb[i]), min(len(sa), len(sb)))
        f = b['metrics']['rest_states'][i - 1]['end'] + 1 if i > 0 else 0
        div = (lat(f) - P) // 8 if lat(f) >= P else -1
    death = s['death'].split(':')[0] if s['death'] in ('appearance_peak', 'detour') else 'violation'
    return dict(row=r, b=b, root=root, P=P, div=div, death=death, kind=s['death'], same=sa == sb,
                root_goal=bool(root['metrics']['goal_reached']), death_frame=s.get('death_frame'))


def block_start(L, P):
    return P + 8 * ((L - P) // 8) if L >= P else None


def first_violation(n):
    vs = [v for v in n['violations'] if v.get('frame') is not None]
    return min(vs, key=lambda v: int(v['frame'])) if vs else None


def layout(fr):
    import cv2
    hsv = cv2.cvtColor(fr, cv2.COLOR_RGB2HSV)
    return np.where((hsv[..., 1] > 110) & (hsv[..., 2] > 80), (hsv[..., 0].astype(np.int16) + 8) // 16 % 12 + 1, 0).astype(np.int8)


def divergence(job):
    """Per-frame difference of the coloured pixels of root and branch, and the first frame where they visibly differ."""
    key, rmp4, bmp4, P, re_, be = job
    rf, bf = scene.read(rmp4), scene.read(bmp4)
    n = min(len(rf), len(bf), (re_ + 1) if re_ is not None else 10 ** 6, (be + 1) if be is not None else 10 ** 6)
    s0 = 1 if P <= 1 else 4 * (P - 1) + 1
    d = [0] * s0 + [int((layout(rf[f]) != layout(bf[f])).sum()) for f in range(s0, n)]
    thr = max(1.8 * max(d[s0:s0 + 8] or [0]), 800)
    f_vis = next((f for f in range(s0, n - 1) if d[f] > thr and d[f + 1] > thr), None)
    return key, d, f_vis, thr, border_onset(rf)


def border_onset(fr):
    for f in range(len(fr)):
        x = fr[f].astype(np.int16)
        b = np.concatenate([x[:10].reshape(-1, 3), x[-10:].reshape(-1, 3), x[:, :10].reshape(-1, 3), x[:, -10:].reshape(-1, 3)])
        if ((b[:, 0] > 150) & (b[:, 1] < 90) & (b[:, 2] < 90)).mean() >= 0.1:
            return f
    return None


def sample(x, div=None):
    """Category and divergence step D of a successful branch whose root fails (None: no usable decision)."""
    rn, bn = NORM / f"{x['root']['key']}.json", NORM / f"{x['b']['key']}.json"
    if not (rn.exists() and bn.exists()):
        return None
    rv, bv = json.loads(rn.read_text()), json.loads(bn.read_text())
    if rv['success'] or not bv['success']:
        return None
    P = x['P']; rm, bm = x['root']['metrics'], x['b']['metrics']
    re_, be = rm.get('eos_frame'), bm.get('eos_frame')
    sa, sb = states(bm), states(rm)
    v = first_violation(rv)
    d, f_vis, thr, onset = div if div else (None, None, None, None)
    phys = [u for u in rv['violations'] if u.get('frame') is not None and u['type'] not in ('appearance_defect', 'ended_not_goal', 'no_eos')]
    if onset is not None and phys and all(int(u['frame']) >= onset - 1 for u in phys) and \
            not any(u['type'] in ('appearance_defect', 'ended_not_goal', 'no_eos') for u in rv['violations']):
        return None
    i = next((i for i in range(min(len(sa), len(sb))) if sa[i] != sb[i]), None)
    INF = 10 ** 6
    if i is not None:
        f_move = int(bm['rest_states'][i - 1]['end']) + 1 if i > 0 else 0
        f = f_vis if f_vis is not None else f_move; kind = 'plan_change'
        why = f'first visible divergence at frame {f} (latent {lat(f)})' if f_vis is not None else f'first different move starts at frame {f}'
    elif len(sa) < len(sb) and (be or INF) < (re_ or INF):
        f = min(f_vis if f_vis is not None else INF, be); kind = 'no_stop_at_goal'
        why = f'branch stops (EOS) at frame {be}, root keeps moving; divergence at frame {f} (latent {lat(f)})'
    elif len(sa) > len(sb) and (re_ or INF) < (be or INF):
        f = min(f_vis if f_vis is not None else INF, re_); kind = 'stopped_early'
        why = f'root stops (EOS) at frame {re_} before the goal; divergence at frame {f} (latent {lat(f)})'
    elif v is not None and sa == sb and be is not None and be <= int(v['frame']) and (re_ or INF) > be:
        f = min(f_vis if f_vis is not None else INF, be); kind = 'no_stop_at_goal'
        why = f"branch stops (EOS) at frame {be}; root goes on, {v['type']} at frame {v['frame']}; divergence at frame {f} (latent {lat(f)})"
    elif v is not None and sa == sb:
        f = int(v['frame'])
        kind = 'fix_appearance' if v['type'] == 'appearance_defect' else (
            'fix_physical_at_end' if (re_ or INF) - f <= 12 else 'fix_physical_mid')
        why = f"same plan; root's first violation {v['type']} at frame {f} (latent {lat(f)})"
    else:
        return None
    D = block_start(max(lat(f), P), P)
    cat = kind + ('_D_is_fork_block' if kind == 'plan_change' and D == P else '_D_later_block' if kind == 'plan_change' else '')
    return dict(x, D=D, why=why, cat=cat, rv=rv, bv=bv, diff=d, f_vis=f_vis, thr=thr)


def verdict(key):
    c = REC / 'combined' / f'{key}.json'
    if not c.exists():
        return None
    n = NORM / f'{key}.json'
    if not n.exists():
        R.save(n, R.normalize(json.loads(c.read_text())))
    return json.loads(n.read_text())


def native_end(rec):
    eos = rec['metrics']['eos_frame']
    assert eos is not None and eos >= 0
    return min(rec['generated_latents'], 1 + BLOCK * math.ceil(eos / (4 * BLOCK)))


def gate(job):
    rec, root = job
    return verify((dict(rec, success=True), root))


def colour(rec):
    import final_colour_verify as F
    row = as_dict(rec['row']); final = F.read_final(row, rec['mp4'])
    return rec['key'], bool(final['certain'] and final['final'] == [list(s) for s in scene.Geometry(row).info['goal']])


def build_main(a):
    a.out.mkdir(parents=True, exist_ok=True)
    roots, branches = load_all()
    by_root = defaultdict(list)
    for b in branches:
        by_root[b['root_key']].append(b)
    branches = [b for k, v in by_root.items() if len(v) == 4 for b in v if b['subtask'] in TASKS]
    why = Counter(); cand = []
    for b in branches:
        bv = verdict(b['key'])
        if bv is None:
            why['branch not re-judged (failed a rule at collection time)'] += 1; continue
        if not bv['success']:
            why['branch fails'] += 1; continue
        rv = verdict(b['root_key'])
        if rv is None and a.dry_run:
            why['DRY RUN: root not judged yet'] += 1; continue
        assert rv is not None, f"root {b['root_key']} of a successful branch was not re-judged"
        if rv['success']:
            why['root succeeds'] += 1; continue
        cand.append(b)
    print('candidates (branch ok, root fails):', len(cand), dict(why), flush=True)
    with Pool(32) as pool:
        g = {k: (n, c) for k, n, c in pool.map(gate, [(b, roots[b['root_key']]) for b in cand], chunksize=4)}
    cand = [b for b in cand if g[b['key']][1] is True]; why['colour not certain'] = len(g) - len(cand)
    with Pool(48) as pool:
        div = {k: (d, f, t, o) for k, d, f, t, o in pool.map(divergence, [(b['key'], roots[b['root_key']]['mp4'], b['mp4'], b['prefix'],
               roots[b['root_key']]['metrics'].get('eos_frame'), b['metrics'].get('eos_frame')) for b in cand], chunksize=4)}
    positives, cats = [], Counter()
    for b in cand:
        s = sample(info({'latent_path': b['latent_path']}), div[b['key']])
        if s is None:
            why['dropped: EOS-border misread only / no usable decision'] += 1; continue
        cats[s['cat']] += 1
        if s['cat'] not in KEEP:
            why['dropped: ' + s['cat']] += 1; continue
        if native_end(b) < s['D'] + BLOCK:
            why['dropped: native EOS before the end of block D'] += 1; continue
        positives.append(dict(b, D=s['D'], cat=s['cat'], why=s['why']))
    positives.sort(key=lambda b: b['key'])
    noise_dir = a.out / 'failed_noise'
    (a.out / 'positives.json').write_text(json.dumps([
        {'key': b['key'], 'root_key': b['root_key'], 'root_latent_path': roots[b['root_key']]['latent_path'],
         'root_noise_seed': roots[b['root_key']]['noise_seed'], 'P': b['prefix'], 'D': b['D'], 'cat': b['cat'],
         'collector_fork_noise': g[b['key']][0], 'out': str(noise_dir / (b['key'] + '.pt'))} for b in positives], indent=0))
    puz = defaultdict(list)
    for r in roots.values():
        if r['subtask'] in TASKS:
            v = verdict(r['key'])
            if v is not None:
                puz[r['sample_id']].append((r, bool(v['success'])))
    mixed = [rs for rs in puz.values() if len(rs) == 2 and sum(ok for _, ok in rs) == 1]
    pool_ok = [r for rs in mixed for r, ok in rs if ok]
    with Pool(32) as pool:
        col = dict(pool.map(colour, pool_ok, chunksize=4))
    root_pool = sorted([r for r in pool_ok if col[r['key']] and native_end(r) > 1], key=lambda r: r['key'])
    sft = json.loads((SFT_RUN / 'resolved_config.json').read_text())
    start = int(sft['max_iters'])
    meta = json.loads(Path(sft['metadata_path']).read_text())
    train = [r for r in meta if r['split'] == 'train']; val = [r for r in meta if r['split'] == 'val']
    assert train and val, 'the SFT metadata needs train and val rows (expert videos)'

    def row(rec, kind, D=None):
        r = as_dict(rec['row']); begin = D if kind == 'branch' else 1
        out = {'sample_id': f"{kind}_{rec['key']}", 'original_sample_id': rec['sample_id'], 'training_noise_group_id': rec['key'],
               'task': 'robotwin', 'subtask': rec['subtask'], 'split': 'train', 'prompt': r['prompt'], 'source_path': r['source_path'],
               'latent_path': rec['latent_path'], 'actual_latent_frames': native_end(rec), 'stored_latent_frames': rec['generated_latents'],
               'native_eos_frame': rec['metrics']['eos_frame'], 'validity_mask_start': begin, 'validity_mask_end': native_end(rec),
               'rft_kind': kind, 'collection_key': rec['key']}
        if kind == 'branch':
            out.update(divergence_latent=D, fork_block=BLOCK, shared_prefix_sha256=rec['shared_prefix_sha256'], branch_prefix=rec['prefix'],
                       failed_noise_path=str(noise_dir / (rec['key'] + '.pt')), category=rec['cat'])
        return out
    n_pos = len(positives); assert n_pos > 0
    rows_per_epoch = math.ceil(n_pos / (1 - ANCHOR_SHARE) / (2 * BATCH * 50)) * 2 * BATCH * 50
    rng0 = random.Random(SEED)
    roots_used = rng0.sample(root_pool, min(n_pos, len(root_pool)))
    while root_pool and len(roots_used) < n_pos:
        extra = list(root_pool); rng0.shuffle(extra); roots_used += extra[:n_pos - len(roots_used)]
    pools = {'branch': [row(b, 'branch', b['D']) for b in positives], 'root': [row(r, 'root') for r in roots_used], 'none': []}
    steps = rows_per_epoch * a.epochs // BATCH; half = rows_per_epoch // 2 // BATCH
    named = [start + 2 * half * k for k in range(1, a.epochs + 1)]
    report = {'candidates_branch_ok_root_fails': len(g), 'positives': n_pos, 'categories_before_drops': cats,
              'positive_categories': Counter(b['cat'] for b in positives), 'positives_by_task': Counter(b['subtask'] for b in positives),
              'positive_puzzles': len({b['sample_id'] for b in positives}), 'D_minus_P_blocks': Counter((b['D'] - b['prefix']) // BLOCK for b in positives),
              'excluded': why, 'mixed_puzzles': len(mixed), 'sample_rft_pool': len(root_pool), 'rows_per_epoch': rows_per_epoch,
              'updates': steps, 'checkpoints': named,
              'sft_checkpoint': str(SFT_RUN / f'checkpoint_model_{start:06d}/model.pt'), 'methods': {}}
    for method, pool_name in METHODS.items():
        rng = random.Random(SEED); pos = pools[pool_name]; n_anchor = rows_per_epoch - len(pos); assert n_anchor > 0
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
        manifest = a.out / f'metadata_{method}.json'; manifest.write_text(json.dumps(rows))
        n_train = sum(r['split'] == 'train' for r in rows); assert n_train == rows_per_epoch * a.epochs
        report['methods'][method] = {'train_rows': n_train, 'expected_train_rows': n_train, 'metadata_path': str(manifest),
                                     'resume_lr_schedule_steps': steps, 'max_iters': start + steps,
                                     'kinds': Counter(r['rft_kind'] for r in rows if r['split'] == 'train'),
                                     'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}
    (a.out / 'PREPARED.json').write_text(json.dumps(report, indent=1, default=dict)); print(json.dumps(report, indent=1, default=dict))


def main():
    global COLLECTIONS, SFT_RUN, REC, NORM, TASKS
    argv = sys.argv[1:]
    command = argv.pop(0) if argv and argv[0] in ('recheck', 'build', 'noise') else 'build'
    if command == 'noise':
        if len(argv) != 1:
            sys.exit('usage: prepare_robotwin_cft.py noise OUT/positives.json')
        return noise_main(argv[0])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--collection', type=Path, nargs='+', required=True, help='output folders of collect_robotwin.py')
    ap.add_argument('--recheck-dir', type=Path, required=True)
    ap.add_argument('--subtasks', default=','.join(TASKS), help='post-training sub-tasks (default: %(default)s)')
    if command == 'recheck':
        ap.add_argument('--keys', type=Path, help='judge only these root / branch keys')
        ap.add_argument('--gpu-part', type=int); ap.add_argument('--parts', type=int, default=1)
    else:
        ap.add_argument('--sft-run', type=Path, required=True, help='SFT run of the policy (resolved_config.json)')
        ap.add_argument('--out', type=Path, required=True)
        ap.add_argument('--epochs', type=int, default=2)
        ap.add_argument('--dry-run', action='store_true', help='format check while the recheck still runs: skip roots not judged yet')
    a = ap.parse_args(argv)
    COLLECTIONS, REC, NORM = a.collection, a.recheck_dir, a.recheck_dir / 'normalized'
    TASKS = tuple(t for t in a.subtasks.split(',') if t)
    R.OUT = REC
    if command == 'recheck':
        recheck_main(a)
    else:
        SFT_RUN = a.sft_run
        build_main(a)


if __name__ == '__main__':
    main()
