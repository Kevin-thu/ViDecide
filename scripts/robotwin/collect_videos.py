# Adapted from RoboTwin 2.0 (https://github.com/RoboTwin-Platform/RoboTwin, scripts/collect_data.py), MIT License,
# Copyright (c) 2025 Tianxing Chen. This file keeps the MIT License (see NOTICE), not the S-Lab License.
"""Plan, simulate and render RoboTwin-Plan episodes of one task with RoboTwin 2.0 (run with the RoboTwin Python).

For every seed, the task (scripts/robotwin/planning_tasks.py) draws a puzzle and its shortest plan, the expert executes
the plan with RoboTwin's grasp and place primitives, and an extra camera records the episode. Successful episodes are
written to <out>/<folder>/<seed>/: video.mp4 (16 fps, with eight hold frames and the red EOS border on the last
four), state.json (block poses, gripper openings and end-effector positions of every frame) and meta.json (camera,
render settings and the symbolic puzzle: places, block colours and sizes, start, goal and plan). Normally started by
render_episodes.py, which sets the options of the released episodes.

--phase plan keeps only the planned joint paths of successful seeds (<out>/plans/), and --phase render replays them
with the camera; the default, both, does both in one pass.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from videcide.tasks.video import add_eos_border, pad_to_vae_length

END_HOLD = 8
# RoboTwin 2.0 checkout (scripts/setup/setup_robotwin.sh); RoboTwin loads its assets relative to its root, so the
# script runs from there and resolves relative paths of its arguments against the directory it was started in.
CALLER_CWD = Path.cwd()
ROOT = Path(os.environ.get("ROBOTWIN_ROOT", Path(__file__).resolve().parents[2] / "third_party" / "RoboTwin")).resolve()
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

if "--planner" in sys.argv and sys.argv[sys.argv.index("--planner") + 1].startswith("mplib"):
    import types
    for name in ("curobo", "curobo.types", "curobo.types.math", "curobo.types.robot", "curobo.wrap", "curobo.wrap.reacher",
                 "curobo.wrap.reacher.motion_gen", "curobo.util"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["curobo.types.math"].Pose = object
    sys.modules["curobo.types.robot"].JointState = object
    for n in ("MotionGen", "MotionGenConfig", "MotionGenPlanConfig", "PoseCostMetric"):
        setattr(sys.modules["curobo.wrap.reacher.motion_gen"], n, object)
    sys.modules["curobo.util"].logger = types.SimpleNamespace(setup_logger=lambda **k: None)

import sapien.core as sapien
import yaml
from envs._GLOBAL_CONFIGS import CONFIGS_PATH
from envs.utils import Actor


def task_args(task: str, config: str, save_freq: int, planner: str | None = None) -> dict:
    args = yaml.safe_load(open(Path(CONFIGS_PATH) / f"{config}.yml"))
    args["task_name"] = task
    emb = yaml.safe_load(open(Path(CONFIGS_PATH) / "_embodiment_config.yml"))
    robot_file = emb[args["embodiment"][0]]["file_path"]
    args["left_robot_file"] = args["right_robot_file"] = robot_file
    args["dual_arm_embodied"] = True
    cfg = yaml.safe_load(open(Path(robot_file) / "config.yml"))
    if planner:
        cfg["planner"] = planner
    args["left_embodiment_config"] = args["right_embodiment_config"] = cfg
    args["embodiment_name"] = str(args["embodiment"][0])
    args["task_config"] = config
    args["save_freq"] = save_freq
    args["render_freq"] = 0
    args["need_plan"] = True
    args["save_data"] = True
    args["save_path"] = str(Path(tempfile.gettempdir()) / "robotwin_unused")   # RoboTwin's own data saving is not used
    return args


def look_at(pos, target):
    f = np.asarray(target, float) - np.asarray(pos, float); f /= np.linalg.norm(f)
    left = np.cross([0, 0, 1], f); left /= np.linalg.norm(left)
    up = np.cross(f, left)
    m = np.eye(4); m[:3, :3] = np.stack([f, left, up], axis=1); m[:3, 3] = pos
    return sapien.Pose(m)


class Recorder:

    def __init__(self, env, size: int, fovy: float, pos, target, every: int):
        self.env, self.every, self.steps = env, every, 0
        scene = env.scene
        step = scene.step

        def counted_step(*a, **k):
            out = step(*a, **k)
            self.steps += 1
            if self.steps % self.every == 0:
                self()
            return out
        scene.step = counted_step
        self.cam = env.scene.add_camera(name="video_cam", width=size, height=size, fovy=np.deg2rad(fovy), near=0.05, far=50)
        self.cam.entity.set_pose(look_at(pos, target))
        self.actors = {k: v for k, v in vars(env).items() if isinstance(v, Actor) and k not in ("table", "wall")}
        self.frames, self.states = [], []

    def __call__(self):
        env = self.env
        env._update_render()
        self.cam.take_picture()
        self.frames.append((self.cam.get_picture("Color")[..., :3] * 255).clip(0, 255).astype(np.uint8))
        st = {"actors": {k: np.concatenate([a.get_pose().p, a.get_pose().q]).round(5).tolist() for k, a in self.actors.items()},
              "gripper": [round(float(env.robot.get_left_gripper_val()), 4), round(float(env.robot.get_right_gripper_val()), 4)],
              "ee": [np.asarray(env.get_arm_pose("left"))[:3].round(4).tolist(), np.asarray(env.get_arm_pose("right"))[:3].round(4).tolist()]}
        self.states.append(st)


def write_mp4(path: Path, frames, fps: float):
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)],
                         stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait()!=0:raise RuntimeError(f'ffmpeg failed writing {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, help="task of planning_tasks.py")
    ap.add_argument("--folder", help="output folder name of the episodes (default: the task name)")
    ap.add_argument("--config", default="demo_clean", help="RoboTwin task config (clean table)")
    ap.add_argument("--phase", choices=("plan", "render", "both"), default="both",
                    help="plan: run the expert planner without rendering, keep successful seeds + joint paths; "
                         "render: replay saved plans with the video camera; both: single pass (renders failures too)")
    ap.add_argument("--seeds", default="0:10", help="start:stop range of seeds (plan/both); render takes all planned seeds in range")
    ap.add_argument("--max-success", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--planner", default=None, help="override the embodiment planner, e.g. mplib_RRT (CPU only)")
    ap.add_argument("--save-freq", type=int, default=90, help="sim steps (1/250 s) per captured frame")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--fovy", type=float, default=36.8)
    ap.add_argument("--cam-pos", default="0,0.44,1.20")
    ap.add_argument("--cam-target", default="0,-0.185,0.76")
    ap.add_argument("--shader", default="rt", help="rt (ray tracing, RoboTwin default) or default (rasterisation)")
    ap.add_argument("--spp", type=int, default=4, help="ray-tracing samples per pixel (RoboTwin default 32; 4 + OIDN looks the same at 384)")
    ap.add_argument("--denoiser", default="oidn", help="oidn (RoboTwin default) or none")
    ap.add_argument('--exact-moves', type=int, help='only puzzles whose shortest plan has this many moves')
    ap.add_argument('--blocks', type=int, help='number of blocks')
    ap.add_argument('--sort-initial', choices=['all_off_home', 'partial'],
                    help='Color Sorting: no block / some blocks start on their own pad')
    ap.add_argument('--max-frames', type=int, default=0, help='drop episodes with more native frames (0: no limit)')
    ap.add_argument('--require-stacked', action='store_true', help='Color Sorting: the start has a stack of two blocks')
    ap.add_argument('--seed-list', type=Path, help='file with one seed per line; overrides --seeds')
    a = ap.parse_args()
    a.out = a.out if a.out.is_absolute() else CALLER_CWD / a.out
    if a.seed_list is not None and not a.seed_list.is_absolute():
        a.seed_list = CALLER_CWD / a.seed_list
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from planning_tasks import TASKS as PLANNING
    env = PLANNING[a.task](); base = a.task
    if a.blocks:env.N_BLOCKS=(a.blocks,)
    if a.require_stacked:env.REQUIRE_STACKED=True
    if a.exact_moves:env.EXACT_MOVES=a.exact_moves
    if a.sort_initial:
        env.SORT_INITIAL=a.sort_initial
        if a.sort_initial=='partial':
            env.MAX_MOVES=(2,6);env.BALANCED=False
    args = task_args(base, a.config, a.save_freq, a.planner)
    lo, hi = map(int, a.seeds.split(":"))
    out = a.out / (a.folder or a.task); out.mkdir(parents=True, exist_ok=True)
    plans = a.out / "plans" / (a.folder or a.task)
    if a.phase != "both":
        plans.mkdir(parents=True, exist_ok=True)
    pos, tgt = [float(x) for x in a.cam_pos.split(",")], [float(x) for x in a.cam_target.split(",")]
    if a.seed_list:
        seeds = [int(x) for x in a.seed_list.read_text().split()]
    elif a.phase == "render":
        seeds = sorted(int(f.stem) for f in plans.glob("*.pkl") if lo <= int(f.stem) < hi)
    else:
        seeds = list(range(lo, hi))
    ok = 0
    for seed in seeds:
        if a.max_success and ok >= a.max_success:
            break
        if a.phase != "plan" and (out / f"{seed:06d}" / "video.mp4").exists():
            continue
        if a.phase == "plan" and (plans / f"{seed:06d}.pkl").exists():
            ok += 1; continue
        t0 = time.time(); rec = None
        try:
            if a.phase == "render":
                traj = pickle.load(open(plans / f"{seed:06d}.pkl", "rb"))
                env.setup_demo(now_ep_num=ok, seed=seed, **{**args, "need_plan": False, "save_data": True})
                env.set_path_lst({"need_plan": False, **traj})
            else:
                env.setup_demo(now_ep_num=ok, seed=seed, **{**args, "save_data": a.phase == "both"})
            env._take_picture = lambda: None
            if a.phase != "plan":
                sapien.render.set_camera_shader_dir(a.shader)
                if a.shader == "rt":
                    sapien.render.set_ray_tracing_samples_per_pixel(a.spp)
                    sapien.render.set_ray_tracing_denoiser(a.denoiser if a.denoiser != "none" else "none")
                rec = Recorder(env, a.size, a.fovy, pos, tgt, a.save_freq)
                rec()
            info = env.play_once()
            good = bool(env.plan_success and env.check_success())
            if good and a.phase != "render" and hasattr(env, "planned_joints_legal"):
                good = bool(env.planned_joints_legal()[0])
            if good and a.phase == "plan":
                with open(plans / f"{seed:06d}.pkl", "wb") as f:
                    pickle.dump({"left_joint_path": env.left_joint_path, "right_joint_path": env.right_joint_path,
                                 "info": ({**json.loads(json.dumps(info, default=str)), **({"model_xy": list(__import__("planning_tasks").MODEL_XY)} if a.task in PLANNING else {})} if info else None)}, f)
                ok += 1
            elif good:
                d = out / f"{seed:06d}"; d.mkdir(exist_ok=True)
                video = add_eos_border(pad_to_vae_length(np.stack(rec.frames + [rec.frames[-1]] * END_HOLD)))
                if a.max_frames and len(video)>a.max_frames:
                    print(f'{a.task} seed {seed} excluded: {len(video)}>{a.max_frames} native frames',flush=True)
                    continue
                write_mp4(d / "video.mp4", video, 16)
                json.dump(rec.states, open(d / "state.json", "w"))
                json.dump({"task": a.task, "seed": seed, "frames": len(video), "motion_frames": len(rec.frames),
                           "fps": 16, "robot_hz": 250 / a.save_freq, "speedup": round(16 / (250 / a.save_freq), 2),
                           "render": {"shader": a.shader, "spp": a.spp, "denoiser": a.denoiser},
                           "camera": {"size": a.size, "fovy": a.fovy, "pos": pos, "target": tgt},
                           "info": ({**json.loads(json.dumps(info, default=str)), **({"model_xy": list(__import__("planning_tasks").MODEL_XY)} if a.task in PLANNING else {})} if info else None)},
                          open(d / "meta.json", "w"), indent=1)
                ok += 1
            nf = len(rec.frames) if rec else 0
            why = '' if good else f" plan_success={getattr(env, 'plan_success', None)} observed={getattr(env, 'observed_state', lambda: None)() if hasattr(env, 'observed_state') else None} goal={getattr(env, 'goal_state', None)}"
            print(f"{a.task} {a.phase} seed {seed} {'OK' if good else 'fail'} frames={nf} {time.time() - t0:.1f}s{why}", flush=True)
        except Exception as e:
            print(f"{a.task} {a.phase} seed {seed} error {type(e).__name__}: {str(e)[:120]}", flush=True)
        finally:
            try:
                env.close_env()
            except Exception:
                pass
    print(f"done {ok} episodes", flush=True)


if __name__ == "__main__":
    main()
