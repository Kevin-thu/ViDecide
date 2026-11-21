"""Render RoboTwin-Plan episodes with RoboTwin 2.0 on one or more GPUs.

    python scripts/robotwin/render_episodes.py --task blocksworld_tri --seeds 0:2000 --gpus 0,1
    python scripts/robotwin/render_episodes.py --task blocksworld_tri --seeds 0:2000 --gpus 0 -- --exact-moves 4 --blocks 3

Every worker runs scripts/robotwin/collect_videos.py with the Python of the RoboTwin environment ($ROBOTWIN_PYTHON,
see scripts/setup/setup_robotwin.sh) on its own part of the seed range. An episode is kept if the expert plan
succeeds in the simulator; it is written to <out>/<episode task>/<seed>/ (video.mp4, state.json, meta.json), by
default data/RoboTwin-Plan/episodes/. Then gate the episodes and write manifest rows with build_manifest.py.

The settings below are those of the released episodes: 384 x 384 ray-traced frames from a fixed camera (vertical field
of view 36.8 degrees, at (0, 0.44, 1.20) looking at (0, -0.185, 0.76)), one frame per 90 simulation steps, the goal
model at (0, 0.04), and a 12 cm lift of the gripper before every grasp in Tower of Hanoi. Episodes longer than
--max-frames are dropped (193 frames for the training tasks, 185 for the 2x2 layouts, no limit for the longer plans).
Arguments after `--` go to collect_videos.py, e.g. --exact-moves N and --blocks N (plan length and block count, used
to balance the Blocks World triangle episodes), --sort-initial and --require-stacked (Color Sorting start states).
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import scene

PY = os.environ.get('ROBOTWIN_PYTHON', 'python')   # Python of the RoboTwin 2.0 environment
CAMERA = ['--size', '384', '--fovy', '36.8', '--cam-pos', '0,0.44,1.20', '--cam-target', '0,-0.185,0.76']
SAVE_FREQ = '90'
MODEL_XY = '0.0,0.04'
LIFT_BEFORE_GRASP = {'hanoi3d': '0.12', 'hanoi_tri': '0.12', 'hanoi_grid': '0.12', 'hanoi_tri_ood': '0.12'}
MAX_FRAMES = {'blocksworld_grid': 185, 'hanoi_grid': 185, 'blocksworld_tri_ood': 0, 'hanoi_tri_ood': 0}   # others: 193


def worker_command(a, seeds: str, extra: list[str]) -> list[str]:
    folder = scene.TASKS[a.task][0]
    max_frames = a.max_frames if a.max_frames is not None else MAX_FRAMES.get(a.task, 193)
    return [PY, str(HERE / 'collect_videos.py'), '--task', a.task, '--folder', folder, '--phase', 'both',
            '--seeds', seeds, '--save-freq', SAVE_FREQ, *CAMERA, '--max-frames', str(max_frames),
            '--out', str(a.out.resolve()), *extra]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', required=True, help='RoboTwin task of scripts/robotwin/planning_tasks.py')
    ap.add_argument('--seeds', required=True, help='start:stop range of simulator seeds')
    ap.add_argument('--gpus', default='0', help='comma-separated GPUs')
    ap.add_argument('--workers', type=int, default=3, help='render workers per GPU (more than 3 slowed the driver down)')
    ap.add_argument('--out', type=Path, default=None, help='episode root (default: data/RoboTwin-Plan/episodes)')
    ap.add_argument('--max-frames', type=int, default=None, help='drop longer episodes (0: no limit)')
    ap.add_argument('--logs', type=Path, default=Path('outputs/robotwin_render'), help='one log per worker')
    argv = sys.argv[1:]
    extra = argv[argv.index('--') + 1:] if '--' in argv else []
    a = ap.parse_args(argv[:argv.index('--')] if '--' in argv else argv)
    if a.task not in scene.TASKS:
        ap.error(f'unknown task {a.task}; one of {", ".join(scene.TASKS)}')
    a.out = a.out or scene.DATA / 'RoboTwin-Plan' / 'episodes'
    lo, hi = map(int, a.seeds.split(':'))
    slots = [(g, w) for g in a.gpus.split(',') for w in range(a.workers)]
    a.logs.mkdir(parents=True, exist_ok=True)
    procs = []
    for i, (gpu, w) in enumerate(slots):
        begin, end = lo + (hi - lo) * i // len(slots), lo + (hi - lo) * (i + 1) // len(slots)
        if begin == end:
            continue
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'PYTHONUNBUFFERED': '1', 'ROBOTWIN_MODEL_XY': MODEL_XY,
               'ROBOTWIN_LIFT_BEFORE_GRASP': LIFT_BEFORE_GRASP.get(a.task, '0')}
        env.setdefault('VK_ICD_FILENAMES', '/etc/vulkan/icd.d/nvidia_icd.json')   # NVIDIA Vulkan driver for SAPIEN
        log = open(a.logs / f'{a.task}_{begin}_{end}.log', 'a')
        procs.append(subprocess.Popen(worker_command(a, f'{begin}:{end}', extra), env=env, stdout=log,
                                      stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    print(f'{a.task}: {len(procs)} workers, exit codes {codes}; episodes in {a.out / scene.TASKS[a.task][0]}', flush=True)
    if any(codes):
        sys.exit(1)


if __name__ == '__main__':
    main()
