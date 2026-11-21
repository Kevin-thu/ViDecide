"""RoboTwin-Plan verifier: learned state reader + rule checker + appearance classifier (Sec. B of the supplement).

Score the videos of an evaluation directory written by scripts/eval/eval_causal.py or eval_bidirectional.py:
    python scripts/robotwin/verifier.py --eval-dir outputs/eval/robotwin_cft --out outputs/eval/robotwin_cft/verifier
    python scripts/robotwin/verifier.py --eval-dir DIR --out OUT --manifest data/RoboTwin-Plan/ood_blocks_world_2x2.jsonl
--manifest is the evaluated set (default: the test set). OUT/SUMMARY.json holds the strict success rate overall and per
sub-task, and OUT/checker/combined/<sample id>.json the verdict of every video. Only the first 20 s (320 frames) of a
video count, and every model is read up to its own EOS frame. --part/--parts split the scoring over GPUs.

Needs the DINOv3 ViT-L/16 backbone in checkpoints/dinov3-vitl16 and the released verifier weights in
checkpoints/videcide/robotwin_verifier/ (state_reader.pt, appearance/head.npz).

The sub-command ``rollouts`` scores post-training rollouts and selects their fork points
(called by scripts/posttrain/collect_robotwin.py; it also needs checkpoints/dinov2-base and
checkpoints/videcide/robotwin_verifier/appearance_bank_dinov2.pt).
"""
import argparse
import json
import os
import subprocess
import sys
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np

HERE = Path(__file__).resolve()

sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[2] / 'src'))

import local_quality_binary as L
import scene

TEST = scene.manifest_path('robotwin', 'test')
VERIFIER = Path('checkpoints/videcide/robotwin_verifier')
ART = VERIFIER / 'appearance'          # head.npz of the appearance classifier
READER = VERIFIER / 'state_reader.pt'    # DINOv3 + transformer decoder state reader
GUARD = .03                            # GPU appearance scores this close to the threshold are recomputed on the CPU path
OUT = None


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def normalize(result):
    threshold = result.get('eos_frame'); threshold = 193 if threshold is None else threshold
    removed = [v for v in result['violations'] if v['type'] == 'illegal_move' and 'before' in v and v['before'] == v.get('after') and v['frame'] >= threshold]
    vs = [v for v in result['violations'] if v not in removed]
    return dict(result, violations=vs, violation=vs[0]['type'] if vs else None, success=not vs, terminal_noop_removed=removed)


def init():
    import torch
    L.init()
    torch.set_num_threads(8)
    L.write = save


def init_threads():
    init()
    import torch
    torch.set_num_threads(int(os.environ.get('RECHECK_THREADS', 8)))


def one(job):
    """Appearance verdict of a collected video combined with its collection-time rule verdict (job['rules'])."""
    L.video_call((job, OUT))
    result = json.loads((OUT / 'videos' / f'{job["key"]}.json').read_text())
    assert result['video'] == job['video'] and result['version'] == L.VERSION
    combined = L.combine_with_rules(job['rules'], result)
    save(OUT / 'combined' / f'{job["key"]}.json', combined)
    return dict(key=job['key'], success=combined['success'], appearance_pass=result['success'])


def gpu_init():
    import torch
    torch.set_num_threads(2); cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    return L.load_backbone().to(device='cuda', dtype=torch.float32).eval()


def gpu_infer(model, video, head):
    import torch
    frames, eos = L.read_video(video)
    tokens = []
    for start in range(0, len(frames), 16):
        x = torch.from_numpy(np.stack([cv2.resize(f, (384, 384), interpolation=cv2.INTER_AREA)
                                       for f in frames[start:start + 16]])).permute(0, 3, 1, 2).float() / 255
        x = (x - torch.tensor([.485, .456, .406])[None, :, None, None]) / torch.tensor([.229, .224, .225])[None, :, None, None]
        with torch.inference_mode():
            h = model(pixel_values=x.cuda()).last_hidden_state[:, -576:].float().cpu()
            h = torch.nn.functional.normalize(h, dim=-1).numpy()
        tokens.append(h.astype(np.float16).astype(np.float32))
    result = L.infer_tokens(np.concatenate(tokens), head)
    result.update(version=L.VERSION, frames_scored=len(frames), eos=eos, no_post_eos_frames_used=True, appearance_only=True)
    return result


def score_records(out, part=0, parts=1, device='cuda'):
    import state_checker as C
    import state_reader as S
    import torch
    out = Path(out)
    mine = [json.loads(f.read_text()) for f in sorted(out.glob('records/*.json'))]
    mine = [r for r in mine if zlib.crc32(r['key'].encode()) % parts == part]
    jf = out / f'checker_jobs_part{part}.json'
    save(jf, [dict(key=r['key'], video=r['video'], row=r['row'], eos=r['eos_frame']) for r in mine])
    if not all((out / 'checker/reads' / f"{r['key']}.json").exists() for r in mine):
        S.infer(SimpleNamespace(device=device, videos=jf, reader=READER, out=out / 'checker/reads', shard=0, shards=1,
                                strip_eos_overlay=False))
    if device == 'cuda':
        torch.cuda.empty_cache()
        model = gpu_init()
    head = np.load(ART / 'head.npz')
    for r in mine:
        key = r['key']; dest = out / 'checker/combined' / f'{key}.json'
        if dest.exists():
            continue
        app = gpu_infer(model, r['video'], head) if device == 'cuda' else None
        if app is None or abs(app['score'] - app['threshold']) < GUARD:
            app = L.check_video(r['video'], ART)
        reads = out / 'checker/reads' / f'{key}.json'
        raw = C.check(r['row'], r['video'], reads)
        T = len(json.loads(reads.read_text()))
        noop = [v for v in raw['violations'] if v['type'] == 'illegal_move' and 'before' in v and v['before'] == v.get('after') and v['frame'] >= T]
        vs = [v for v in raw['violations'] if v not in noop]
        comb = L.combine_with_rules(dict(raw, violations=vs, violation=vs[0]['type'] if vs else None, success=not vs), app)
        save(dest, {**comb, 'key': key, 'sample_id': r['sample_id'], 'video': r['video'], 'eos_frame': r['eos_frame'],
                    'checker_eos_frame': comb.get('eos_frame'), 'raw_rules': raw, 'removed_terminal_self_transitions': noop})


def eos_of(video):
    return scene.find_eos_frame(scene.read_visible(video))


def import_eval_dir(eval_dir, out, manifest):
    rows = {r['sample_id']: r for r in scene.read_manifest(manifest)}
    res = {}
    for f in sorted(Path(eval_dir).rglob('results.json')):
        for r in json.loads(f.read_text())['results']:
            res[r['sample_id']] = r
    sids = sorted(res)
    with ProcessPoolExecutor(16) as pool:
        eos = list(pool.map(eos_of, [res[s]['mp4'] for s in sids], chunksize=4))
    for sid, e in zip(sids, eos):
        save(out / 'records' / f'{sid}.json', dict(key=sid, sample_id=sid, row=rows[sid], video=res[sid]['mp4'], eos_frame=e))


def summary(out):
    tab = defaultdict(lambda: [0, 0]); noeos = 0
    for f in sorted(out.glob('records/*.json')):
        r = json.loads(f.read_text())
        v = json.loads((out / 'checker/combined' / f.name).read_text()); ok = bool(v['success'])
        noeos += v['eos_frame'] is None
        for k in (r['row']['subtask'], 'ALL'):
            tab[k][0] += ok; tab[k][1] += 1
    res = dict(success=tab['ALL'][0], n=tab['ALL'][1], no_eos=noeos,
               by_subtask={k: dict(success=v[0], n=v[1]) for k, v in sorted(tab.items()) if k != 'ALL'})
    save(out / 'SUMMARY.json', res); print(json.dumps(res), flush=True)


def rollouts_write(path, value):
    tmp = Path(str(path) + '.tmp'); tmp.write_text(json.dumps(value, default=str)); os.replace(tmp, path)


def make_job(args):
    key, mp4, row = args
    return {'key': f'c_{key}', 'video': mp4, 'row': row, 'eos': scene.find_eos_frame(scene.read_visible(mp4)),
            'colours': scene.Geometry(row).info['colours']}


def judge(args):
    cv2.setNumThreads(1)
    import state_checker as C
    j, chk = args; k = j['key']; o = Path(chk)
    try:
        return k, C.check(j['row'], j['video'], o / 'reads' / f'{k}.json', appearance_path=o / 'appearance' / f'{k}.npy')
    except Exception:
        import traceback
        return k, {'success': False, 'violation': 'error', 'error': traceback.format_exc()[-800:], 'violations': []}


def rollouts_main():
    """Score the rollouts <out>/<stage>/records/<key>.json listed in --keys and select the fork point of failed roots."""
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--stage', required=True, choices=['roots', 'branches'])
    ap.add_argument('--keys', required=True, type=Path); ap.add_argument('--workers', type=int, default=32); a = ap.parse_args()
    from videcide.training.robotwin_fork import select_fork
    full = scene.rows_by_id()
    keys = [k for k in a.keys.read_text().split('\n') if k]
    recs = {k: json.loads((a.out / a.stage / 'records' / f'{k}.json').read_text()) for k in keys}
    chk = a.out / 'checker'; chk.mkdir(exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        jobs = list(ex.map(make_job, [(k, r['mp4'], full[r['sample_id']]) for k, r in recs.items()], chunksize=8))
    jf = chk / f'jobs.{a.stage}.{os.getpid()}.json'; jf.write_text(json.dumps(jobs))
    py = sys.executable
    for script, extra in (('appearance_scores.py', ['--videos', jf, '--out', chk / 'appearance']),
                          ('state_reader.py', ['infer', '--videos', jf, '--reader', READER, '--out', chk / 'reads'])):
        code = subprocess.call([py, '-u', str(Path(__file__).resolve().parent / script), *map(str, extra)],
                               stdout=open(chk / f'{script[:-3]}.log', 'a'), stderr=subprocess.STDOUT)
        if code:
            sys.exit(f'{script} failed (exit {code})')
    with ProcessPoolExecutor(a.workers) as ex:
        res = dict(ex.map(judge, [(j, str(chk)) for j in jobs], chunksize=4))
    ok = 0
    for k, r in recs.items():
        m = res['c_' + k]; r['success'] = bool(m['success']); r['metrics'] = m; ok += r['success']
        rollouts_write(a.out / a.stage / 'records' / f'{k}.json', r)
        if a.stage == 'roots' and not r['success']:
            info = json.loads(scene.data_path(r['source_path']).read_text())['info']['info']
            ap = chk / 'appearance' / f'c_{k}.npy'
            exc = (np.clip(np.load(ap).astype(np.float32) - scene.APPEAR_PATCH_MAX, 0, None).reshape(-1, 576).sum(1) if ap.exists() else None)
            sel = select_fork(m, info, appearance_excess=exc) if m.get('violation') != 'error' else {'status': 'error', 'error': m.get('error')}
            rollouts_write(a.out / 'forks' / f'{k}.json', {**r, 'selection': sel})
    jf.unlink()
    print(f'scored {len(recs)} {a.stage}: {ok} successes', flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'rollouts':
        sys.argv.pop(1)
        rollouts_main()
    else:
        ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        ap.add_argument('--eval-dir', type=Path, required=True, help='output of scripts/eval/eval_causal.py (results.json)')
        ap.add_argument('--out', type=Path, required=True)
        ap.add_argument('--manifest', type=Path, default=TEST, help='evaluated set (default: %(default)s)')
        ap.add_argument('--device', default='cuda')
        ap.add_argument('--part', type=int, default=0); ap.add_argument('--parts', type=int, default=1)
        ap.add_argument('--reader', type=Path, default=READER, help='state reader weights')
        ap.add_argument('--appearance', type=Path, default=ART, help='folder with the appearance head (head.npz)')
        a = ap.parse_args()
        READER, ART = a.reader, a.appearance
        if not (a.out / 'records').exists():
            import_eval_dir(a.eval_dir, a.out, a.manifest)
        score_records(a.out, a.part, a.parts, a.device)
        if all((a.out / 'checker/combined' / f.name).exists() for f in a.out.glob('records/*.json')):
            summary(a.out)
