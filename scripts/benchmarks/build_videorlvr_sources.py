"""Symbolic sources of the VideoRLVR FlowFree and Sokoban puzzles, parsed from VideoRLVR-Data.

    python scripts/benchmarks/build_videorlvr_sources.py --upstream data/upstream/VideoRLVR-Data \
        --code third_party/VideoRLVR --out outputs/videorlvr_sources

FlowFree boards are read from the input image and the solution video with the evaluator of the VideoRLVR code
(https://github.com/luka-group/VideoRLVR, --code); Sokoban puzzles come from the upstream metadata and must replay.
Writes <out>/{flowfree,sokoban}/{train,test}/<VideoRLVR id>.json (the VideoRLVR split and id in upstream_split and
upstream_id), an audit of every puzzle and a summary with the duplicate layouts. The released sources of data/VideoRLVR
are built from these: FlowFree with the single-stroke walk recovered from the drawing order of the solution video
(src/videcide/tasks/flowfree_task.py, recover_order), Sokoban with the render theme and prompt of the training videos;
the released data numbers the puzzles of every set (sample ids <game>_<set>_<index>) and keeps upstream_split and
upstream_id.
"""
import argparse, csv, hashlib, json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from videcide.tasks.flowfree_task import extract_source, load_upstream_evaluator
from videcide.tasks.sokoban_task import replay as replay_sokoban

UP = CODE = None   # set from --upstream and --code
META = {('flowfree', 'train'): 'train/metadata_flowfree_only.csv', ('flowfree', 'test'): 'test/metadata_flowfree.csv',
        ('sokoban', 'train'): 'train/metadata_sokoban_only.csv', ('sokoban', 'test'): 'test/metadata_sokoban.csv'}
_EV = None

def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def flowfree(split, row, out):
    global _EV
    _EV = _EV or load_upstream_evaluator(CODE)
    uid = Path(row['video']).stem
    try:
        src = extract_source(UP / split / row['input_image'], UP / split / row['video'], row['prompt'], _EV)
    except Exception as exc:
        return {'task': 'flowfree', 'split': split, 'upstream_id': uid, 'status': 'rejected', 'error': f'{type(exc).__name__}: {str(exc)[:160]}'}
    eps = [sorted(tuple(c) for c in g) for g in src['endpoints']]
    lay = sha([src['grid_size'], sorted(eps)])
    src.update(upstream_split=split, upstream_id=uid,
               upstream_input_image=row['input_image'], upstream_solution_video=row['video'], layout_sha256=lay)
    (out / 'flowfree' / split / f'{uid}.json').write_text(json.dumps(src))
    return {'task': 'flowfree', 'split': split, 'upstream_id': uid, 'status': 'ok', 'layout_sha256': lay,
            'size': int(src['grid_size']), 'max_endpoints': max(len(g) for g in src['endpoints'])}

def sokoban(split, row, out):
    uid = Path(row['video']).stem
    meta = UP / split / 'sokoban' / 'metadata' / f'{uid}.json'
    src = json.loads(meta.read_text())
    _, _, check = replay_sokoban(src)
    if not check['success']:
        return {'task': 'sokoban', 'split': split, 'upstream_id': uid, 'status': 'rejected', 'error': str(check)[:160]}
    lay = sha([src['grid'], src['player_start'], sorted(map(tuple, src['boxes_start'])), sorted(map(tuple, src['targets']))])
    src.update(upstream_split=split, upstream_id=uid,
               upstream_metadata=str(meta.relative_to(UP)), symbolic_check=check, layout_sha256=lay)
    (out / 'sokoban' / split / f'{uid}.json').write_text(json.dumps(src))
    return {'task': 'sokoban', 'split': split, 'upstream_id': uid, 'status': 'ok', 'layout_sha256': lay,
            'size': f"{src['grid_rows']}x{src['grid_cols']}", 'boxes': int(src['num_boxes']), 'moves': int(src['num_moves'])}

def one(job):
    task, split, row, out = job
    return (flowfree if task == 'flowfree' else sokoban)(split, row, out)

def main():
    global UP, CODE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--upstream', type=Path, default=Path('data/upstream/VideoRLVR-Data'), help='download of VideoRLVR-Data')
    ap.add_argument('--code', type=Path, default=Path('third_party/VideoRLVR'), help='checkout of the VideoRLVR code')
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--workers', type=int, default=48)
    args = ap.parse_args()
    UP, CODE = args.upstream, args.code
    jobs = []
    for (task, split), meta in META.items():
        (args.out / task / split).mkdir(parents=True, exist_ok=True)
        base = UP / meta
        for row in csv.DictReader(open(base)):
            if task == 'sokoban' and split == 'test':
                row = {k: (f'sokoban/{v}' if k in ('video', 'input_image') else v) for k, v in row.items()}
            jobs.append((task, split, row, args.out))
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(one, jobs, chunksize=8))
    (args.out / 'audit_flowfree_sokoban.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in res))
    summ = {}
    for task in ('flowfree', 'sokoban'):
        ok = [r for r in res if r['task'] == task and r['status'] == 'ok']
        tr = [r['layout_sha256'] for r in ok if r['split'] == 'train']
        summ[task] = {'status': dict(Counter(f"{r['split']}:{r['status']}" for r in res if r['task'] == task)),
                      'train_unique_layouts': len(set(tr)), 'train_duplicate_rows': len(tr) - len(set(tr)),
                      'test_layouts_also_in_train': sum(r['split'] == 'test' and r['layout_sha256'] in set(tr) for r in ok),
                      'sizes': {s: dict(Counter(r['size'] for r in ok if r['split'] == s)) for s in ('train', 'test')},
                      'rejection_reasons': dict(Counter(r['error'].split(':')[0] for r in res if r['task'] == task and r['status'] != 'ok'))}
        if task == 'flowfree': summ[task]['max_endpoints'] = dict(Counter(r['max_endpoints'] for r in ok))
        if task == 'sokoban': summ[task]['boxes'] = dict(Counter(r['boxes'] for r in ok)); summ[task]['moves_max'] = max(r['moves'] for r in ok)
    (args.out / 'summary_flowfree_sokoban.json').write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1))

if __name__ == '__main__':
    main()
