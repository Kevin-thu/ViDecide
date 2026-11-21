"""Score VBVR-MultiStep evaluation videos with the strict rule checkers.

    python scripts/eval/score_vbvr.py EVAL_DIR           # strict rules -> EVAL_DIR[/shard*]/strict.json
    python scripts/eval/score_vbvr.py optimal EVAL_DIR   # + shortest solution -> EVAL_DIR/optimality.json
EVAL_DIR is the output of scripts/eval/eval_causal.py (results.json, or shard*/results.json for sharded runs). The
paper reports strict + shortest (the "optimal" counts). The sub-command ``rollouts`` scores the rollouts of
scripts/posttrain/collect_vbvr.py and selects their fork points.
"""
import argparse
import collections
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from videcide.paths import data_path, manifest_path
from videcide.tasks import vbvr_checkers as V

ROWS = {json.loads(l)["sample_id"]: json.loads(l) for m in ("train", "val", "test")
        if manifest_path("vbvr", m).exists() for l in open(manifest_path("vbvr", m))}
TOL = 0.5   # path-length tolerance (cells) of the shortest-solution check of TSP Reward Collection and Ordinal Number Sequence


def one(item):
    sid, mp4 = item
    try:
        return sid, V.check(ROWS[sid], V.read(mp4))
    except Exception as e:
        return sid, {"success": False, "violation": None, "error": repr(e), "goal_reached": False, "eos_detected": False}


def strict_main(d):
    """Strict verdicts of the videos listed in d/results.json, written to d/strict.json."""
    res = json.load(open(d / "results.json"))["results"]
    with ProcessPoolExecutor(48) as p:
        out = dict(p.map(one, [(r["sample_id"], r["mp4"]) for r in res]))
    summ = collections.defaultdict(lambda: {"n": 0, "success": 0, "violation": 0, "incomplete": 0, "error": 0,
                                            "no_eos": 0, "kinds": collections.Counter()})
    for sid, m in out.items():
        s = summ[ROWS[sid]["task"].replace("vbvr_", "")]; s["n"] += 1
        if m.get("error"): s["error"] += 1
        elif m["success"]: s["success"] += 1
        elif m["violation"]: s["violation"] += 1; s["kinds"][m["violation"]] += 1
        else: s["incomplete"] += 1
        s["no_eos"] += not m.get("eos_detected")
    tot = {k: sum(s[k] for s in summ.values()) for k in ("n", "success", "violation", "incomplete", "error", "no_eos")}
    json.dump({"summary": summ, "total": tot, "videos": out}, open(d / "strict.json", "w"), indent=1, default=str)
    for t, s in sorted(summ.items()):
        print(f"{t:8s} n={s['n']:3d} success={s['success']:3d} violation={s['violation']:3d} incomplete={s['incomplete']:3d} "
              f"no_eos={s['no_eos']:3d} kinds={dict(s['kinds'])}")
    print("total", tot)


def judge(args):
    sid, mp4 = args
    row = ROWS[sid]; task = row['task'].replace('vbvr_', ''); meta = V.puzzle(row)
    ref = V.read(data_path(row['native_video']))
    out = {}
    if task in ('snake', 'tsp', 'ordinal'):
        out['model'] = V.CHECK[task](V.read(mp4), meta, ref)
        out['reference'] = V.CHECK[task](ref, meta, ref)
    return sid, task, out


def optimal(task, v, extra, meta):
    """Whether a strict success also uses the fewest moves (or the shortest path) where the prompt asks for it."""
    if task == 'sokoban':
        return True
    if task == 'hanoi':
        return v['moves'] <= v['optimal_moves']
    if task == 'sliding':
        return len(v['changes']) - 1 <= meta['solution_length']
    m, r = extra['model'], extra['reference']
    assert m['success'] == v['success'], (task, 'checker drift')
    if task == 'snake':
        return m['moves'] <= r['moves']
    return m['path_cells'] <= r['path_cells'] + TOL


def optimal_main():
    for d in map(Path, sys.argv[1:]):
        verdict, mp4 = {}, {}
        for f in sorted(d.glob('strict.json')) + sorted(d.glob('shard*/strict.json')):
            verdict.update(json.loads(f.read_text())['videos'])
        for f in sorted(d.glob('results.json')) + sorted(d.glob('shard*/results.json')):
            for r in json.loads(f.read_text())['results']:
                mp4[r['sample_id']] = r['mp4']
        jobs = [(s, mp4[s]) for s, v in verdict.items() if v['success'] and s.split('_')[1] in ('snake', 'tsp', 'ordinal')]
        with Pool(24) as pool:
            extra = {sid: e for sid, _, e in pool.map(judge, jobs, chunksize=2)}
        per = defaultdict(Counter); rows = {}
        for s, v in verdict.items():
            task = s.split('_')[1]; ok = bool(v['success'])
            opt = ok and optimal(task, v, extra.get(s), V.puzzle(ROWS[s]))
            per[task]['n'] += 1; per[task]['strict'] += ok; per[task]['optimal'] += opt
            rows[s] = {'task': task, 'strict': ok, 'optimal': opt, **({k: extra[s][k] for k in extra[s]} if s in extra else {})}
        summary = {t: dict(c) for t, c in sorted(per.items())}
        summary['total'] = {k: sum(c[k] for c in per.values()) for k in ('n', 'strict', 'optimal')}
        (d / 'optimality.json').write_text(json.dumps({'summary': summary, 'tolerance_cells': TOL, 'videos': rows}, default=str))
        print(d.name, json.dumps({t: f"{c['strict']}->{c['optimal']}" for t, c in summary.items()}), flush=True)


def rollouts_write(path, value):
    tmp = Path(str(path) + '.tmp'); tmp.write_text(json.dumps(value, default=str)); os.replace(tmp, path)


def rollouts_one(job):
    out, stage, key = job
    from videcide.training.vbvr_fork import select_fork
    p = Path(out) / stage / 'records' / (key + '.json'); rec = json.loads(p.read_text())
    try:
        frames = V.read(rec['mp4']); m = V.check(rec['row'], frames)
    except Exception as e:
        frames, m = None, {'success': False, 'violation': None, 'error': repr(e)}
    rec['success'] = bool(m['success']); rec['metrics'] = {k: v for k, v in m.items() if k != 'changes'}
    rollouts_write(p, rec)
    if stage == 'roots' and not rec['success']:
        task = rec['task'].replace('vbvr_', '')
        sel = select_fork(frames, task, m) if frames is not None else {'status': 'error', 'error': m.get('error')}
        rollouts_write(Path(out) / 'forks' / (key + '.json'), {**rec, 'selection': sel})
    return rec['success']


def rollouts_main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--stage', required=True)
    ap.add_argument('--keys', required=True); ap.add_argument('--workers', type=int, default=16); a = ap.parse_args()
    keys = [k for k in Path(a.keys).read_text().split('\n') if k]
    with ProcessPoolExecutor(a.workers) as ex:
        ok = list(ex.map(rollouts_one, [(a.out, a.stage, k) for k in keys], chunksize=4))
    print(f'scored {len(ok)} {a.stage}: {sum(ok)} successes', flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ('optimal', 'rollouts'):
        {'optimal': optimal_main, 'rollouts': rollouts_main}[sys.argv.pop(1)]()
    else:
        root = Path(sys.argv[1])
        for d in ([root] if (root / "results.json").exists() else sorted(root.glob("shard*"))):
            strict_main(d)
