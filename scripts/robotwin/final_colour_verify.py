"""Reads the final block arrangement of a RoboTwin-Plan video from the block colours alone (no learned model).

The three frames before the EOS frame are compared with the rendered silhouettes of every possible arrangement; the
read is certain if all three agree with a clear margin. Post-training uses it as an extra check that a successful
rollout really ends in the goal arrangement.
"""
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene

EDGE_PX = 16
MARGIN = 0.03
OFFSETS = (1, 3, 5)
VERSION = 'final_colour_reader'


def arrangements(nb, ns):
    out = set()
    for assign in itertools.product(range(ns), repeat=nb):
        per = [[b for b in range(nb) if assign[b] == s] for s in range(ns)]
        for orders in itertools.product(*[itertools.permutations(p) for p in per]):
            out.add(tuple(tuple(o) for o in orders))
    return sorted(out)


class Reader:
    def __init__(self, row):
        self.geo = g = scene.Geometry(row); info = g.info
        self.cols = info['colours']; self.nb = len(self.cols); self.ns = len(info['slots'])
        size = g.cam['size']; self.size = size
        self.pad_aware = info.get('mode') == 'sorting'
        self.pad_background = None
        if self.pad_aware:
            self.pad_background = np.full((size, size), -1, np.int8)
            for xy, colour in zip(info['slots'], info['pads']):
                if colour not in self.cols:
                    continue
                quad = np.array([scene.project((xy[0] + dx, xy[1] + dy, g.floor - .001), g.cam)
                                 for dx, dy in [(-.045, -.045), (-.045, .045), (.045, .045), (.045, -.045)]], np.int32)
                cv2.fillConvexPoly(self.pad_background, quad, self.cols.index(colour))
        ignore = np.zeros((size, size), bool)
        if g.card is not None:
            x0, y0, x1, y1 = [int(v) for v in g.card]; ignore[max(0, y0):y1, max(0, x0):x1] = True
        if not self.pad_aware:
            for q in g.pad_quads:
                ignore |= q
        ignore[:EDGE_PX] = ignore[-EDGE_PX:] = True; ignore[:, :EDGE_PX] = ignore[:, -EDGE_PX:] = True
        self.ignore = ignore
        self.arrs = arrangements(self.nb, self.ns)
        self.region = np.zeros((size, size), bool)
        for sil in g.cell_sil.values():
            self.region |= sil
        if self.pad_aware:
            self.region |= self.pad_background >= 0
        self.region &= ~ignore
        self.pred = {a: self.paint(a) for a in self.arrs}

    def sil(self, b, s_, below):
        return self.geo.cell_sil[(b, s_, round(sum(2 * self.geo.sizes[j] for j in below), 4))]

    def paint(self, arr):
        lab = self.pad_background.copy() if self.pad_aware else np.full((self.size, self.size), -1, np.int8)
        items = []
        for s_, stack in enumerate(arr):
            base = self.geo.cells[(stack[0], s_, 0.0)][1] if stack else 0.0
            for i, b in enumerate(stack):
                items.append((base, i, b, s_, stack[:i]))
        for _, _, b, s_, below in sorted(items):
            lab[self.sil(b, s_, below)] = b
        return lab

    def observe(self, frame):
        f = frame if frame.shape[0] == self.size else cv2.resize(frame, (self.size, self.size))
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV); sat = (hsv[..., 1] > 120) & (hsv[..., 2] > 70)
        lab = np.full((self.size, self.size), -1, np.int8)
        for b, c in enumerate(self.cols):
            lab[scene.hue_mask(hsv, c) & sat] = b
        return lab

    def read(self, frame):
        obs = self.observe(frame); r = self.region; scores = []
        for a in self.arrs:
            p = self.pred[a]; inter = union = 0
            for b in range(self.nb):
                pb, ob = (p == b) & r, (obs == b) & r
                inter += (pb & ob).sum(); union += (pb | ob).sum()
            scores.append(inter / max(union, 1))
        order = np.argsort(scores)[::-1]
        return self.arrs[order[0]], float(scores[order[0]]), float(scores[order[0]] - scores[order[1]])


def read_final(row, video_path, reader=None):
    frames = scene.read_visible(video_path); eos = scene.find_eos_frame(frames)
    end = eos if eos is not None else len(frames)
    rd = reader or Reader(row); reads = []
    for o in OFFSETS:
        t = end - o
        if t >= 0:
            a, sc, m = rd.read(frames[t]); reads.append({'frame': t, 'state': [list(s) for s in a], 'iou': round(sc, 3), 'margin': round(m, 3)})
    states = {json.dumps(x['state']) for x in reads}
    certain = len(states) == 1 and all(x['margin'] >= MARGIN for x in reads)
    return {'final': reads[0]['state'] if reads else None, 'certain': certain, 'reads': reads, 'eos': eos,
            'checker_version': VERSION, 'pad_aware': getattr(rd, 'pad_aware', False)}
