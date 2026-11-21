# Built on RoboTwin 2.0 (https://github.com/RoboTwin-Platform/RoboTwin, envs/_base_task.py), MIT License,
# Copyright (c) 2025 Tianxing Chen. This file keeps the MIT License (see NOTICE), not the S-Lab License.
"""RoboTwin-Plan tasks for RoboTwin 2.0, built from its grasp and place primitives (Sec. A of the supplement).

Blocks stand in stacks on a few places of the table. One move takes the top block of a place and puts it on another
place (on the table or on its top block). The symbolic plan is the shortest move sequence (breadth-first search), and
the expert executes it with RoboTwin's planner, so every episode shows a correct, shortest solution. The goal is
visible in the first frame.

  Blocks World     3 or 4 coloured blocks on three places in a row or a triangle; the goal arrangement is shown by a
                   small model at the front of the table.
  Tower of Hanoi   3 blocks of different sizes on three pads; all must end on the green pad, never a larger block on a
                   smaller one.
  Color Sorting    blocks on a 2x2 layout of coloured pads; each block must end alone on the pad of its colour, with a
                   free grey buffer pad for three blocks or without one for four blocks.

Tasks (TASKS): blocksworld / blocksworld_tri (Blocks World, row / triangle), hanoi3d / hanoi_tri (Tower of Hanoi, row /
triangle), sort_blocked / sort_nobuffer (Color Sorting with / without buffer), and the out-of-distribution tasks
blocksworld_grid / hanoi_grid (2x2 layout) and blocksworld_tri_ood / hanoi_tri_ood (longer plans).
Success (check_success) = every block on its goal place and level within tolerance, both grippers open.
"""
from __future__ import annotations

from collections import deque

import os
import numpy as np
import sapien

from envs._base_task import Base_Task
from envs.utils import ArmTag, create_box

PALETTE = {"red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1),
           "yellow": (1, 1, 0), "magenta": (1, 0, 1), "cyan": (0, 1, 1)}
GREY, PAD_GREEN = (0.6, 0.6, 0.6), (0.1, 0.75, 0.2)
ROW = [(-0.12, -0.18), (0.0, -0.18), (0.12, -0.18)]
TRI = [(-0.07, -0.21), (0.07, -0.21), (0.0, -0.10)]
GRID = [(-0.07, -0.20), (0.07, -0.20), (-0.07, -0.08), (0.07, -0.08)]
MODEL_DY = {id(TRI): 0.155, id(GRID): 0.14}
MODEL_XY = tuple(float(v) for v in os.environ.get('ROBOTWIN_MODEL_XY', '0.0,0.04').split(','))   # goal model on the table
PAD_H = 0.002
SIZES4 = [0.032, 0.028, 0.024, 0.020]


def moves(state, sizes=None):
    for s, src in enumerate(state):
        if not src:
            continue
        b = src[-1]
        for d, dst in enumerate(state):
            if d == s:
                continue
            if sizes is not None and dst and sizes[dst[-1]] < sizes[b]:
                continue
            new = [list(x) for x in state]
            new[s].pop(); new[d].append(b)
            yield (s, d), tuple(tuple(x) for x in new)


def bfs(start, goal_fn, sizes=None, max_depth=10):
    q, seen = deque([(start, [])]), {start}
    while q:
        st, path = q.popleft()
        if goal_fn(st):
            return path
        if len(path) >= max_depth:
            continue
        for mv, nxt in moves(st, sizes):
            if nxt not in seen:
                seen.add(nxt); q.append((nxt, path + [mv]))
    return None


def random_stacks(rng, blocks, n_slots, max_h):
    while True:
        st = [[] for _ in range(n_slots)]
        for b in rng.permutation(blocks):
            st[rng.integers(n_slots)].append(int(b))
        if max(len(s) for s in st) <= max_h:
            return tuple(tuple(s) for s in st)


class StackPlanning(Base_Task):
    SORT_N = 3
    BALANCED = False
    GRASP_HIGH_ONTO_STACK = os.environ.get('ROBOTWIN_GRASP_FIX', '0') == '1'
    GRASP_ONTO_STACK_DIS = float(os.environ.get('ROBOTWIN_GRASP_FIX_GD', '0.015'))
    ONTO_STACK_OPPOSITE_ARM = os.environ.get('ROBOTWIN_ONTO_STACK_ARM', '') == 'opposite'
    PLACE_ONTO_DIS = float(os.environ.get('ROBOTWIN_PLACE_ONTO_DIS', '0.0'))
    RETREAT_UP_FIRST = float(os.environ.get('ROBOTWIN_RETREAT_UP_FIRST', '0'))
    LIFT_BEFORE_GRASP = float(os.environ.get('ROBOTWIN_LIFT_BEFORE_GRASP', '0'))
    SLOT_MARKERS = True
    SAFE_TRANSIT = False
    ROW_SLOTS = ROW
    MODE = "blocksworld"
    N_BLOCKS = (3, 4)
    MAX_MOVES = (2, 6)
    SIZE_RULE = False

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def _episode(self):
        rng = np.random.default_rng(np.random.randint(1 << 30))
        names = list(PALETTE)
        want = getattr(self, 'EXACT_MOVES', None)
        if want is None and self.BALANCED:
            want = int(rng.integers(self.MAX_MOVES[0], self.MAX_MOVES[1] + 1))
        while True:
            if self.MODE == "blocksworld":
                n = int(rng.choice(self.N_BLOCKS)); self.slots = self.ROW_SLOTS; ns = len(self.slots)
                colours = [names[i] for i in rng.permutation(len(names))[:n]]
                start = random_stacks(rng, range(n), ns, 3); goal = random_stacks(rng, range(n), ns, 3)
                sizes = [0.025] * n; pads = [None] * ns
                plan = bfs(start, lambda s: s == goal, max_depth=self.MAX_MOVES[1])
            elif self.MODE == "hanoi":
                n = int(rng.choice(self.N_BLOCKS)) if self.N_BLOCKS != (3, 4) else 3; self.slots = self.ROW_SLOTS
                colours = [names[i] for i in rng.permutation(len(names))[:n]]
                sizes = [0.032, 0.026, 0.020] if n == 3 else SIZES4
                ns = len(self.slots); st = [[] for _ in range(ns)]
                for b in range(n):
                    st[rng.integers(ns)].append(b)
                start = tuple(tuple(s) for s in st)
                target = int(rng.integers(ns)); goal = tuple(tuple(range(n)) if i == target else () for i in range(ns))
                pads = ["green" if i == target else "grey" for i in range(ns)]
                plan = bfs(start, lambda s: s == goal, sizes=sizes, max_depth=self.MAX_MOVES[1])
            else:
                n = self.SORT_N; self.slots = GRID
                colours = [names[i] for i in rng.permutation(len(names))[:n]]
                sizes = ([0.032, 0.026, 0.020] if n == 3 else SIZES4) if self.SIZE_RULE else [0.025] * n
                order = rng.permutation(4)
                pads = [colours[i] if i < n else "grey" for i in order]
                home = {b: pads.index(colours[b]) for b in range(n)}
                while True:
                    st = random_stacks(rng, range(n), 4, 2)
                    if getattr(self,'REQUIRE_STACKED',False) and max(map(len,st))<2:continue
                    if self.SIZE_RULE and any(list(x) != sorted(x) for x in st):continue
                    on_home=sum(any(b in stack and slot==home[b] for slot,stack in enumerate(st)) for b in range(n))
                    initial=getattr(self,'SORT_INITIAL','all_off_home')
                    if (initial=='all_off_home' and on_home==0) or (initial=='partial' and 0<on_home<n):
                        break
                start = st
                goal = tuple(tuple(b for b in range(n) if home[b] == i) for i in range(4))
                plan = bfs(start, lambda s: s == goal, sizes=sizes if self.SIZE_RULE else None, max_depth=self.MAX_MOVES[1])
            if plan is not None and self.MAX_MOVES[0] <= len(plan) <= self.MAX_MOVES[1] and (want is None or len(plan) == want):
                break
        self.colours, self.block_sizes, self.pads = colours, sizes, pads
        self.start_state, self.goal_state, self.plan = start, goal, plan

    def load_actors(self):
        self._episode()
        z0 = 0.741 + self.table_z_bias
        self.pad_actors = []
        for i, p in enumerate(self.pads):
            if p is None:
                continue
            col = GREY if p == "grey" else PAD_GREEN if p == "green" else PALETTE[p]
            self.pad_actors.append(create_box(scene=self, pose=sapien.Pose([*self.slots[i], z0 - 0.001 + PAD_H], [1, 0, 0, 0]),
                                              half_size=(0.045, 0.045, PAD_H), color=col, is_static=True, name=f"pad{i}"))
        if self.MODE == "blocksworld" and self.SLOT_MARKERS:
            for i, (x, y) in enumerate(self.slots):
                create_box(scene=self, pose=sapien.Pose([x, y, z0 - 0.001 - PAD_H + 0.0005], [1, 0, 0, 0]),
                           half_size=(0.045, 0.045, PAD_H), color=GREY, is_static=True, name=f"slot{i}")
        self.blocks = [None] * len(self.colours)
        for i, stack in enumerate(self.start_state):
            z = z0 + (2 * PAD_H if self.pads[i] is not None else 0)
            for b in stack:
                h = self.block_sizes[b]
                yaw = np.random.uniform(-0.15, 0.15)
                self.blocks[b] = create_box(scene=self, pose=sapien.Pose([*self.slots[i], z + h + 0.001],
                                                                          [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]),
                                            half_size=(h, h, h), color=PALETTE[self.colours[b]], name="box")
                z += 2 * h
        for k, b in enumerate(self.blocks):
            setattr(self, f"block{k + 1}", b)
        for x, y in self.slots:
            self.prohibited_area.append([x - 0.06, y - 0.06, x + 0.06, y + 0.06])
        if self.MODE == "blocksworld":
            m = 0.011
            for i, stack in enumerate(self.goal_state):
                x, y = MODEL_XY[0] + (i - 1) * 0.05, MODEL_XY[1]
                if id(self.slots) in MODEL_DY:
                    x = MODEL_XY[0] + 0.5 * self.slots[i][0]; y = MODEL_XY[1] + 0.5 * (self.slots[i][1] + MODEL_DY[id(self.slots)])
                create_box(scene=self, pose=sapien.Pose([x, y, z0 + 0.001], [1, 0, 0, 0]), half_size=(0.02, 0.02, 0.001),
                           color=GREY, is_static=True, name=f"goal_pad{i}")
                for lvl, b in enumerate(stack):
                    create_box(scene=self, pose=sapien.Pose([x, y, z0 + 0.002 + m * (2 * lvl + 1)], [1, 0, 0, 0]),
                               half_size=(m, m, m), color=PALETTE[self.colours[b]], is_static=True, name=f"goal_{b}")
            self.prohibited_area.append([MODEL_XY[0] - 0.1, MODEL_XY[1] - 0.06, MODEL_XY[0] + 0.1, MODEL_XY[1] + 0.06])

    def _optional_move(self, action):
        ok_before = self.plan_success
        self.move(action())
        if self.plan_success:
            return True
        self.plan_success = ok_before
        return False

    def play_once(self):
        self.last_gripper = None
        state = [list(s) for s in self.start_state]
        z0 = 0.74 + self.table_z_bias + (2 * PAD_H)
        for s, d in self.plan:
            b = state[s][-1]; blk = self.blocks[b]
            (x_src, _), (x_dst, y_dst) = self.slots[s], self.slots[d]
            side = x_src if abs(x_src) > 1e-6 else x_dst
            if self.ONTO_STACK_OPPOSITE_ARM and state[d] and abs(x_dst) > 1e-6 and abs(x_src) < 1e-6:
                side = -x_dst
            arm = ArmTag("left" if side < 0 else "right")
            gd = 0.015 if len(state[s]) > 1 else (self.GRASP_ONTO_STACK_DIS if (self.GRASP_HIGH_ONTO_STACK and state[d]) else 0.0)
            if self.last_gripper is not None and self.last_gripper != arm:
                if self.SAFE_TRANSIT:
                    self._optional_move(lambda: self.move_by_displacement(arm_tag=arm.opposite, z=0.06))
                self.move(self.back_to_origin(arm_tag=arm.opposite))
            if self.LIFT_BEFORE_GRASP:
                self._optional_move(lambda: self.move_by_displacement(arm_tag=arm, z=self.LIFT_BEFORE_GRASP))
            self.move(self.grasp_actor(blk, arm_tag=arm, pre_grasp_dis=0.09, grasp_dis=gd))
            lift = 0.10
            if self.SAFE_TRANSIT:
                lo_, hi_ = sorted((s, d))
                span = [d] if self.slots is TRI else range(lo_, hi_ + 1)
                h_between = max([sum(2 * self.block_sizes[x] for x in state[i]) for i in span if i != s] + [0.0])
                h_src = sum(2 * self.block_sizes[x] for x in state[s][:-1])
                lift = max(0.10, h_between - h_src + 0.05)
            if lift > 0.10 and self._optional_move(lambda: self.move_by_displacement(arm_tag=arm, z=lift)):
                pass
            else:
                self.move(self.move_by_displacement(arm_tag=arm, z=0.10))
            if state[d]:
                top = self.blocks[state[d][-1]]
                self.move(self.place_actor(blk, target_pose=top.get_functional_point(1), arm_tag=arm, functional_point_id=0,
                                           pre_dis=0.05, dis=self.PLACE_ONTO_DIS, pre_dis_axis="fp"))
            else:
                self.move(self.place_actor(blk, target_pose=[x_dst, y_dst, z0, 0, 1, 0, 0], arm_tag=arm, functional_point_id=0,
                                           pre_dis=0.09, dis=0.02, constrain="align"))
            if self.RETREAT_UP_FIRST and state[d]:
                self._optional_move(lambda: self.move_by_displacement(arm_tag=arm, z=self.RETREAT_UP_FIRST))
            retreat = lambda z: self.move_by_displacement(arm_tag=arm, z=z, move_axis="arm")
            if not self.SAFE_TRANSIT:
                self.move(retreat(0.07))
            elif not self._optional_move(lambda: retreat(0.07)):
                self.move(retreat(0.04))
            state[s].pop(); state[d].append(b)
            self.last_gripper = arm
        self.info["info"] = {"mode": self.MODE, "colours": self.colours, "sizes": self.block_sizes, "pads": self.pads,
                             "slots": self.slots, "start": self.start_state, "goal": self.goal_state,
                             "plan": [list(m) for m in self.plan], **({"size_rule": True} if self.SIZE_RULE else {})}
        return self.info

    def observed_state(self):
        z0 = 0.74 + self.table_z_bias
        st = [[] for _ in self.slots]
        for b, blk in enumerate(self.blocks):
            p = blk.get_pose().p
            i = int(np.argmin([np.hypot(p[0] - x, p[1] - y) for x, y in self.slots]))
            if np.hypot(p[0] - self.slots[i][0], p[1] - self.slots[i][1]) > 0.035:
                return None
            st[i].append((p[2], b))
        return tuple(tuple(b for _, b in sorted(s)) for s in st)

    def check_success(self):
        if self.observed_state() != self.goal_state or not self.is_left_gripper_open() or not self.is_right_gripper_open():
            return False
        floor = 0.741 + self.table_z_bias
        for slot, stack in enumerate(self.goal_state):
            height = floor + (2 * PAD_H if self.pads[slot] is not None else 0)
            for block in stack:
                pose = self.blocks[block].get_pose()
                half = self.block_sizes[block]
                if abs(float(pose.p[2]) - (height + half + .001)) > .015:
                    return False
                w, x, y, z = pose.q
                if 1 - 2 * (x * x + y * y) < .9:
                    return False
                height += 2 * half
        return True


class blocksworld(StackPlanning):
    MODE = "blocksworld"; MAX_MOVES = (2, 4)


class hanoi3d(StackPlanning):
    MODE = "hanoi"; MAX_MOVES = (2, 4)


class sort_blocked(StackPlanning):
    MODE = "sorting"; MAX_MOVES = (3, 4)


class blocksworld_tri(StackPlanning):
    ROW_SLOTS = TRI; SAFE_TRANSIT = True; BALANCED = True; MODE = "blocksworld"; N_BLOCKS = (3, 4); MAX_MOVES = (2, 6)


class hanoi_tri(StackPlanning):
    ROW_SLOTS = TRI; SAFE_TRANSIT = True; BALANCED = True; MODE = "hanoi"; N_BLOCKS = (3,); MAX_MOVES = (2, 5)


class sort_nobuffer(StackPlanning):
    SAFE_TRANSIT = True; BALANCED = True; MODE = "sorting"; SORT_N = 4; MAX_MOVES = (5, 6)


class blocksworld_grid(StackPlanning):
    ROW_SLOTS = GRID; SAFE_TRANSIT = True; BALANCED = True; MODE = "blocksworld"; N_BLOCKS = (3, 4); MAX_MOVES = (2, 6)


class hanoi_grid(StackPlanning):
    ROW_SLOTS = GRID; SAFE_TRANSIT = True; BALANCED = True; MODE = "hanoi"; N_BLOCKS = (3,); MAX_MOVES = (2, 5)


class blocksworld_tri_ood(blocksworld_tri):
    MAX_MOVES = (7, 8)


class hanoi_tri_ood(hanoi_tri):
    MAX_MOVES = (6, 7)


TASKS = {"blocksworld": blocksworld, "blocksworld_tri": blocksworld_tri, "hanoi3d": hanoi3d, "hanoi_tri": hanoi_tri,
         "sort_blocked": sort_blocked, "sort_nobuffer": sort_nobuffer,
         "blocksworld_grid": blocksworld_grid, "hanoi_grid": hanoi_grid,                 # out of distribution: 2x2 layout
         "blocksworld_tri_ood": blocksworld_tri_ood, "hanoi_tri_ood": hanoi_tri_ood}     # out of distribution: longer plans
