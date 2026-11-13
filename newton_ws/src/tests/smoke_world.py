"""Prova di base della fisica: caduta libera e hover.

Da lanciare nel container:
    cd /workspace/newton
    uv run python /workspace/project_workspace/src/tests/smoke_world.py
"""

import os
import sys

# Rende importabili i package di src/ (drone_model, scene)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from drone_model.params import DroneParams
from scene.test_world import World, WorldConfig

START_Z = 2.0  # abbastanza alto da non toccare il pavimento nei test
T_END = 0.5  # s


def run(label, action, expected):
    p = DroneParams()
    w = World(p, WorldConfig(start_position=(0.0, 0.0, START_Z)))
    n = int(round(T_END / w.cfg.dt))
    for _ in range(n):
        w.step(action)
    z = w.position()[2]
    print(f"{label}: z dopo {T_END} s = {z:.3f} m (atteso circa {expected:.3f} m)")


p = DroneParams()
g = p.gravity

# 1) Caduta libera: spinta nulla (azione -1). z = z0 - g t^2 / 2
run("caduta libera", [-1.0, 0.0, 0.0, 0.0], START_Z - 0.5 * g * T_END**2)

# 2) Hover: spinta = peso, cioè (a+1)/2 * 1.8 * W = W  ->  a = 2/1.8 - 1
a_hover = 2.0 / p.thrust_to_weight - 1.0
run(f"hover (azione {a_hover:.3f})", [a_hover, 0.0, 0.0, 0.0], START_Z)
