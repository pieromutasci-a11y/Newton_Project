"""Visualizza il World con il viewer di Newton (finestra OpenGL).

Da lanciare nel container (serve il display X11, già configurato in run.sh):
    cd /workspace/newton
    uv run python /workspace/project_workspace/src/tests/view_world.py --mode hover
    uv run python /workspace/project_workspace/src/tests/view_world.py --mode fall

Altri viewer di Newton (utili se non c'è un display): --viewer rerun | usd | viser | null
"""

import os
import sys

# Rende importabili i package di src/ (drone_model, scene)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import newton.examples

from drone_model.params import DroneParams
from scene.test_world import World, WorldConfig


class Example:
    """Forma richiesta da newton.examples.run: __init__(viewer, args), step(), render()."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.params = DroneParams()
        self.world = World(self.params, WorldConfig(start_position=(0.0, 0.0, args.start_height)))

        # Azioni costanti per la prova: [spinta, momento x, momento y, momento z]
        if args.mode == "fall":
            self.action = [-1.0, 0.0, 0.0, 0.0]  # spinta nulla: caduta libera
        else:
            a_hover = 2.0 / self.params.thrust_to_weight - 1.0  # spinta = peso
            self.action = [a_hover, 0.0, 0.0, 0.0]

        self.sim_time = 0.0
        self.viewer.set_model(self.world.model)

    def step(self):
        self.world.step(self.action)
        self.sim_time += self.world.cfg.dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.world.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--mode", choices=["hover", "fall"], default="hover", help="Azione costante da applicare.")
    parser.add_argument("--start-height", type=float, default=1.0, help="Quota iniziale del drone [m].")
    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
