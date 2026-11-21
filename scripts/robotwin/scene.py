"""Scene geometry, video reading and shared rules of RoboTwin-Plan (used by the verifier and the data scripts).

RoboTwin-Plan lives in data/RoboTwin-Plan/: one manifest per set (train, val, test and the three out-of-distribution
sets) and the episodes in episodes/<episode task>/<seed>/ (video.mp4, state.json with the simulator state of every
frame, meta.json with the camera and the symbolic puzzle). TASKS lists the RoboTwin task (environment) of every
episode folder with its sub-task label and prompt.
"""
from __future__ import annotations
import functools
import sys
from pathlib import Path
import itertools
import json

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from videcide.paths import DATA_ROOT, data_path, manifest_path, read_manifest
from videcide.tasks.video_parser import find_eos_frame


DATA = DATA_ROOT

# Manifests of RoboTwin-Plan: the training splits and the out-of-distribution sets of the paper.
SETS = ("train", "val", "test", "ood_blocks_world_2x2", "ood_tower_of_hanoi_2x2", "ood_longer_plans")

_BLOCKS_WORLD = ("Two robot arms move one top block at a time between the {} grey places on the table until the stacks "
                 "match the small model at the front of the table.")
_HANOI = ("Two robot arms move all blocks onto the green pad, one top block at a time, never placing a larger block on a "
          "smaller one.")
_SORT = "Two robot arms move one top block at a time until every block stands alone on the pad of its own colour; "

# RoboTwin task (scripts/robotwin/planning_tasks.py) -> (episode folder, manifest task, sub-task label, prompt).
# The label gives the layout and the range of plan lengths; the last four tasks form the out-of-distribution sets.
TASKS = {
    "blocksworld": ("blocks_world_row", "robotwin_blocksworld", "blocksworld/row_2-4", _BLOCKS_WORLD.format("three")),
    "blocksworld_tri": ("blocks_world_triangle", "robotwin_blocksworld", "blocksworld/triangle_2-6",
                        _BLOCKS_WORLD.format("three")),
    "hanoi3d": ("tower_of_hanoi_row", "robotwin_hanoi", "hanoi/row_2-4", _HANOI),
    "hanoi_tri": ("tower_of_hanoi_triangle", "robotwin_hanoi", "hanoi/triangle_2-5", _HANOI),
    "sort_blocked": ("color_sorting_buffer", "robotwin_sort", "sort/buffer_3", _SORT + "the grey pad is a free buffer."),
    "sort_nobuffer": ("color_sorting_no_buffer", "robotwin_sort", "sort/nobuffer_4",
                      _SORT + "there is no free pad, so a blocking block can only be put on top of another block."),
    "blocksworld_grid": ("blocks_world_2x2", "robotwin_blocksworld", "blocksworld/grid_2-6", _BLOCKS_WORLD.format("four")),
    "hanoi_grid": ("tower_of_hanoi_2x2", "robotwin_hanoi", "hanoi/grid_2-5", _HANOI),
    "blocksworld_tri_ood": ("blocks_world_triangle_longer", "robotwin_blocksworld", "blocksworld/triangle_7-8",
                            _BLOCKS_WORLD.format("three")),
    "hanoi_tri_ood": ("tower_of_hanoi_triangle_longer", "robotwin_hanoi", "hanoi/triangle_6-7", _HANOI),
}
EPISODE_TASK = {folder: task for task, (folder, *_) in TASKS.items()}

# The evaluation ignores everything after 20 s (320 frames at 16 fps), as the generation does.
VISIBLE_FRAMES = 320


def manifest_rows(names=SETS) -> list[dict]:
    """Rows of the given RoboTwin-Plan manifests (default: all of them that exist)."""
    paths = [manifest_path("robotwin", name) for name in names]
    return [row for path in paths if path.exists() for row in read_manifest(path)]


@functools.lru_cache(maxsize=1)
def rows_by_id() -> dict[str, dict]:
    """Every RoboTwin-Plan row of every manifest, by sample id."""
    return {row["sample_id"]: row for row in manifest_rows()}


HUES = {"red": (0, 8, 172, 180), "yellow": (22, 36), "green": (50, 70), "cyan": (84, 96), "blue": (112, 128), "magenta": (142, 160)}


MODEL_XYZ = (0.0, 0.14, 0.76)


def read(path):
    cap, out = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            return np.stack(out)
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))


def read_visible(path):
    """The frames of a video that the evaluation looks at (the first VISIBLE_FRAMES)."""
    return read(path)[:VISIBLE_FRAMES]


def project(p, cam):
    pos, tgt = np.array(cam["pos"], float), np.array(cam["target"], float)
    f = tgt - pos; f /= np.linalg.norm(f)
    left = np.cross([0, 0, 1], f); left /= np.linalg.norm(left)
    up = np.cross(f, left)
    d = np.asarray(p, float) - pos
    size = cam["size"]; foc = (size / 2) / np.tan(np.deg2rad(cam["fovy"]) / 2)
    return size / 2 - foc * (d @ left) / (d @ f), size / 2 - foc * (d @ up) / (d @ f)


def hue_mask(hsv, colour):
    r = HUES[colour]; h = hsv[..., 0]
    m = (h >= r[0]) & (h <= r[1])
    if len(r) == 4:
        m |= (h >= r[2]) & (h <= r[3])
    return m


class Geometry:

    def __init__(self, row):
        meta = json.loads(data_path(row['source_path']).read_text())
        self.info = info = meta['info']['info']; self.cam = cam = meta['camera']
        st0 = json.loads(data_path(row['state_path']).read_text())[0]['actors']
        sizes = [float(x) for x in info['sizes']]; n = len(sizes); size = cam['size']
        self.sizes = sizes
        self.floor = floor = min(st0[f'block{b + 1}'][2] - sizes[b] for b in range(n))
        self.cells, self.cell_area, self.cell_sil = {}, {}, {}
        for b in range(n):
            others = [2 * sizes[j] for j in range(n) if j != b]
            heights = sorted({round(sum(c), 4) for k in range(len(others) + 1) for c in itertools.combinations(others, k)})
            for s_, xy in enumerate(info['slots']):
                for z in heights:
                    h = sizes[b]; zz = floor + z
                    self.cells[(b, s_, z)] = np.array(project((xy[0], xy[1] + h, zz), cam))
                    corners = np.array([project((xy[0] + dx * h, xy[1] + dy * h, zz + dz * 2 * h), cam)
                                        for dx in (-1, 1) for dy in (-1, 1) for dz in (0, 1)], np.float32)
                    hull = cv2.convexHull(corners)
                    self.cell_area[(b, s_, z)] = float(cv2.contourArea(hull))
                    sil = np.zeros((size, size), np.uint8); cv2.fillConvexPoly(sil, hull.astype(np.int32), 1)
                    self.cell_sil[(b, s_, z)] = sil.astype(bool)
        h = float(np.median(sizes))
        c0 = np.array(project((info['slots'][0][0], info['slots'][0][1], floor), cam))
        c1 = np.array(project((info['slots'][0][0] + 2 * h, info['slots'][0][1], floor), cam))
        self.edge_px = float(np.linalg.norm(c1 - c0))
        self.cube_area = float(np.median([a for (b, s_, z), a in self.cell_area.items() if z == 0]))
        self.pad_quads = []
        for xy in info['slots']:
            q = np.array([project((xy[0] + dx, xy[1] + dy, floor), cam) for dx, dy in
                          [(-.05, -.05), (-.05, .05), (.05, .05), (.05, -.05)]], np.int32)
            m = np.zeros((size, size), np.uint8); cv2.fillConvexPoly(m, q, 1); self.pad_quads.append(m.astype(bool))
        self.card = None
        if info['mode'] == 'blocksworld':
            mxy = meta['info'].get('model_xy') or info.get('model_xy') or MODEL_XYZ[:2]
            dy = 0.155 if 'triangle' in row['subtask'] else 0.14 if 'grid' in row['subtask'] else None
            m = 0.011; pts = []
            for i in range(len(info['slots'])):
                x, y = (mxy[0] + 0.5 * info['slots'][i][0], mxy[1] + 0.5 * (info['slots'][i][1] + dy)) if dy is not None \
                    else (mxy[0] + (i - 1) * 0.05, mxy[1])
                for dx in (-0.02, 0.02):
                    for dy in (-0.02, 0.02):
                        for z in (0.76, 0.76 + 2 * m * n + 0.004):
                            pts.append(project((x + dx, y + dy, z), cam))
            pts = np.array(pts); self.card = (*(pts.min(0) - 3), *(pts.max(0) + 3))

    def in_card(self, x, y):
        return self.card is not None and self.card[0] <= x <= self.card[2] and self.card[1] <= y <= self.card[3]

    def cell_of(self, base, block=None):
        best, d = None, 1e9
        for k, p in self.cells.items():
            if block is not None and k[0] != block:
                continue
            dd = float(np.linalg.norm(base - p))
            if dd < d:
                best, d = k, dd
        return (best[1], best[2]), d / self.edge_px


def arrangements(n, slots):
    for order in itertools.permutations(range(n)):
        for cuts in itertools.combinations_with_replacement(range(n + 1), slots - 1):
            bounds = (0, *cuts, n)
            yield tuple(tuple(order[bounds[i]:bounds[i + 1]]) for i in range(slots))


def legal_transition(before, after, sizes=None):
    for src, stack in enumerate(before):
        if not stack:
            continue
        block = stack[-1]
        for dst in range(len(before)):
            if src == dst:
                continue
            candidate = [list(s) for s in before]
            candidate[src].pop()
            candidate[dst].append(block)
            if tuple(tuple(s) for s in candidate) == after:
                return sizes is None or not before[dst] or sizes[before[dst][-1]] >= sizes[block]
    return False


PERSIST = 3


MIN_REST, OVERLAP_MIN_FRAMES = 3, 20


APPEAR_PATCH_MAX = 0.472


SEVERITY_MAX = 2.0


def first_persistent(flags, k=PERSIST):
    run = 0
    for t, f in enumerate(flags):
        run = run + 1 if f else 0
        if run >= k:
            return t - k + 1
    return None


PREPROCESS_VERSION = 'eos_border_replaced_by_last_clean_ring'   # cache tag of remove_eos_overlay


def remove_eos_overlay(frames):
    a = np.asarray(frames)
    if a.ndim != 4 or not len(a):
        raise ValueError('Need the complete video prefix, starting at frame zero')
    w = max(3, round(10 * min(a.shape[1:3]) / 384))
    ring = np.ones(a.shape[1:3], dtype=bool)
    ring[w:-w, w:-w] = False
    pixels = a.astype(np.float32)
    other = np.maximum(pixels[..., 1], pixels[..., 2])
    red = ((pixels[..., 0] >= 60) & (pixels[..., 0] >= 1.5 * other)
           & (pixels[..., 0] - other >= 30))
    close = np.linalg.norm(pixels - np.array([230, 30, 30]), axis=-1) <= 75
    mask = (red | close) & ring
    if mask[0].any():
        raise ValueError('Initial reserved ring already contains EOS; no clean reference')
    out = a.copy()
    reference = a[0][ring].copy()
    for t in range(len(a)):
        if mask[t].any():
            out[t][ring] = reference
        else:
            reference = a[t][ring].copy()
    return out


def cache_contract(out, contract):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'INPUT_CONTRACT.json'
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise ValueError(f'Cache contract changed: {path}; use a new directory')
    else:
        if any(p.is_file() for p in out.iterdir()):
            raise ValueError(f'Existing cache has no input contract: {out}')
        path.write_text(json.dumps(contract, sort_keys=True, indent=2) + '\n')
