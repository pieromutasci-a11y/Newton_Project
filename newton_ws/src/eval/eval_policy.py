"""Valutazione con GUI del controllore di posizione addestrato.

La rete (best.pt o last.pt) comanda i droni, i riferimenti sono pallini nello spazio 3D (una linea li
collega al drone corrispondente). Stessa fisica e stesso env del training, ma senza gradienti.

Uso (nel container, serve il display X11 di run.sh):
    python /workspace/project_workspace/src/eval/eval_policy.py --run apg_obsfix_h32
    python /workspace/project_workspace/src/eval/eval_policy.py --run apg_obsfix_h32 --ckpt last --num_drones 6

Senza display: --viewer null (stampa solo le metriche).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import newton
import newton.examples
import torch
import warp as wp

from controllers.pos_controller.pos_controller_env import PosControllerEnv, yaw_from_quat
from controllers.pos_controller.pos_controller_mlp import PosControllerMLP
from controllers.pos_controller.target_handler import TargetHandler
from drone_model.params import DroneParams
from eval.plots import save_eval_plots
from scene.train_world import TrainWorld, TrainWorldConfig

DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
TARGET_COLOR = (1.0, 0.85, 0.1)  # giallo
TARGET_RADIUS = 0.07  # m
LINK_COLOR = (0.6, 0.6, 0.6)
YAW_ARROW_LEN = 0.5  # m
DRONE_YAW_COLOR = (0.2, 1.0, 1.0)  # ciano: yaw del drone (il riferimento è giallo, come il suo pallino)


class Example:
    """Forma richiesta da newton.examples.run: __init__(viewer, args), step(), render()."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        dev = "cuda:0"

        # Mondo con l'aspetto da quadricottero; horizon=1 perché non serve la finestra di backprop
        self.world = TrainWorld(
            DroneParams(), TrainWorldConfig(num_worlds=args.num_drones, horizon=1, visual=True)
        )
        # Coda di riferimenti (stessa logica del training: si avanza quando il drone arriva e ci resta 1.2 s).
        # Nessun reset in valutazione: la sequenza viene percorsa una volta sola e poi la valutazione si chiude.
        targets = TargetHandler(args.num_drones, dev, seed=args.seed, n_waypoints=args.n_waypoints)
        self.env = PosControllerEnv(self.world, targets, seed=args.seed)
        self.finished = False
        self.args = args
        self.params = DroneParams()
        self.plot_drone = args.plot_drone
        self.rec = []  # una riga per passo di controllo del drone `plot_drone` (colonne: vedi eval/plots.py)
        self.pace = args.pace
        self.t_start = time.perf_counter()

        # Rete addestrata
        ckpt_path = os.path.join(args.data_dir, "policy", args.run, f"{args.ckpt}.pt")
        ckpt = torch.load(ckpt_path, map_location=dev)
        self.policy = PosControllerMLP().to(dev)
        self.policy.load_state_dict(ckpt["policy"])  # include le statistiche di normalizzazione
        self.policy.eval()
        print(f"Policy: {ckpt_path} (iterazione {ckpt['iteration']})")

        self.obs = self.env.observations()
        self.sim_time = 0.0
        self.steps = 0
        self.dist_sum = torch.zeros((), device=dev)

        # Stato "da disegnare": la posa viene copiata dalla simulazione a ogni frame
        self.draw_state = self.world.model.state()
        self.draw_q = wp.to_torch(self.draw_state.body_q, requires_grad=False)
        self.draw_q.copy_(self.world.q)

        n = args.num_drones
        self.target_colors = wp.array([wp.vec3(*TARGET_COLOR)] * n, dtype=wp.vec3)

        self.link_colors = wp.array([wp.vec3(*LINK_COLOR)] * n, dtype=wp.vec3)

        self.target_xf = torch.zeros(n, 7, device=dev)
        self.target_xf[:, 6] = 1.0
        self.viewer.set_model(self.world.model)
        self.viewer.set_camera(wp.vec3(4.0, -4.0, 2.5), pitch=-20.0, yaw=135.0)

    @torch.no_grad()
    def step(self):
        actions = self.policy(self.obs)
        self.env.step(actions)
        e = self.env.errors()
        # Distanza dal riferimento attivo durante il passo (media sui droni)
        self.dist_sum += e.pos_err.norm(dim=-1).mean()
        # Registrazione del drone `plot_drone`: posa, riferimento (quello attivo nel passo), velocità, azioni
        d = self.plot_drone
        pos, tgt = self.world.pos[d], self.env.targets.step_target_pos[d]
        self.rec.append(
            torch.cat(
                [
                    torch.tensor([self.sim_time + self.world.ctrl_dt], device=pos.device),
                    pos, tgt,
                    yaw_from_quat(self.world.q[d : d + 1, 3:7]),
                    self.env.targets.step_target_yaw[d : d + 1],
                    e.lin_vel_b[d], e.ang_vel_b[d], actions[d],
                ]
            )
        )
        # Stacca dal grafo (horizon=1) SENZA resettare i mondi: env.end_window() resetterebbe i droni
        self.world.detach()
        self.env.prev_actions = self.env.prev_actions.detach()
        self.obs = self.env.observations()

        self.sim_time += self.world.ctrl_dt
        self.steps += 1
        if self.pace:  # un passo di controllo = ctrl_dt secondi veri: se la simulazione è in anticipo sull'orologio, aspetta
            ahead = self.t_start + self.sim_time - time.perf_counter()
            if ahead > 0:
                time.sleep(ahead)
        if bool(self.env.sequence_done().all()):
            print(f"\nSequenza di {self.env.targets.n_waypoints} riferimenti completata a t = {self.sim_time:.1f} s.")
            self.finished = True
        elif self.sim_time >= self.args.max_s:
            print(f"\nTempo massimo ({self.args.max_s:.0f} s) raggiunto prima di finire la sequenza: valutazione chiusa.")
            self.finished = True
        if self.steps % round(1.0 / self.world.ctrl_dt) == 0:  # ogni secondo simulato
            print(f"t = {self.sim_time:6.1f} s | distanza media dal riferimento {float(self.dist_sum) / round(1.0 / self.world.ctrl_dt):.3f} m | riferimento {int(self.env.targets.wp_idx.max()) + 1}/{self.env.targets.n_waypoints}")
            self.dist_sum.zero_()

    def render(self):
        self.draw_q.copy_(self.world.q)
        self.target_xf[:, :3] = self.env.targets.target_pos  # posa (x, y, z, quaternione identità) di ogni sfera
        targets_t = self.env.targets.target_pos
        targets = wp.from_torch(targets_t.contiguous(), dtype=wp.vec3)
        drones = wp.from_torch(self.world.pos.contiguous(), dtype=wp.vec3)

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.draw_state)
        # Riferimenti: sfere solide, come il bersaglio dell'esempio diffsim_drone di Newton
        target_xf = wp.from_torch(self.target_xf, dtype=wp.transform)
        self.viewer.log_shapes("/targets", newton.GeoType.SPHERE, (TARGET_RADIUS,), target_xf, self.target_colors)
        self.viewer.log_lines("/target_links", drones, targets, LINK_COLOR)  # drone -> riferimento
        # Proiezioni verticali sul pavimento (aiutano a leggere altezza e posizione): drone e riferimento
        pd, pt = self.world.pos.contiguous().clone(), self.env.targets.target_pos.contiguous().clone()
        pd[:, 2], pt[:, 2] = 0.004, 0.004
        floor_d, floor_t = wp.from_torch(pd, dtype=wp.vec3), wp.from_torch(pt, dtype=wp.vec3)
        self.viewer.log_lines("/drone_drop", drones, floor_d, (0.8, 0.8, 0.9))
        self.viewer.log_lines("/target_drop", targets, floor_t, TARGET_COLOR)
        # Frecce di yaw (orizzontali, partono dal centro): giallo = yaw di riferimento, ciano = yaw del drone
        self.viewer.log_arrows("/target_yaw", targets, self._yaw_tip(targets_t, self.env.targets.target_yaw), TARGET_COLOR)
        self.viewer.log_arrows(
            "/drone_yaw", drones, self._yaw_tip(self.world.pos, yaw_from_quat(self.world.q[:, 3:7])), DRONE_YAW_COLOR
        )
        self.viewer.end_frame()

    def save_plots(self):
        """Salva grafici e riepilogo in data/results/<run>/ (la cartella dei log di quella policy)."""
        if not self.args.plots or len(self.rec) < 2:
            return
        out_dir = os.path.join(self.args.data_dir, "results", self.args.run)
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.join(out_dir, f"eval_{self.args.ckpt}")
        rec = torch.stack(self.rec).cpu().numpy()
        title = f"{self.args.run} ({self.args.ckpt}.pt) - valutazione, drone {self.plot_drone}, {self.world.ctrl_dt * 1000:.0f} ms/passo"
        print("\n" + save_eval_plots(rec, self.params, base + ".png", base + "_summary.txt", title))
        print(f"\nGrafici e riepilogo salvati in {out_dir}/ (eval_{self.args.ckpt}.png, eval_{self.args.ckpt}_summary.txt)")

    @staticmethod
    def _yaw_tip(pos: torch.Tensor, yaw: torch.Tensor):
        """Punta di una freccia orizzontale lunga YAW_ARROW_LEN che parte da pos e punta verso l'angolo di yaw."""
        d = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=-1)
        return wp.from_torch((pos + YAW_ARROW_LEN * d).contiguous(), dtype=wp.vec3)


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--run", required=True, help="nome della run in data/policy/")
    parser.add_argument("--ckpt", choices=["best", "last"], default="best")
    parser.add_argument("--num_drones", type=int, default=1)
    parser.add_argument("--n_waypoints", type=int, default=5, help="riferimenti da raggiungere prima di chiudere")
    parser.add_argument("--max_s", type=float, default=120.0, help="tempo massimo di simulazione [s], per non restare bloccati")
    parser.add_argument("--pace", action=argparse.BooleanOptionalAction, default=True, help="simula a tempo reale")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--plot_drone", type=int, default=0, help="quale drone registrare per i grafici")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True, help="salva i grafici alla fine")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)
    # Ciclo di valutazione proprio (invece di newton.examples.run) per poter chiudere a sequenza completata
    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()
    while viewer.is_running() and not example.finished:
        if viewer.should_step():  # rispetta la pausa del viewer
            example.step()
        example.render()
    viewer.close()
    example.save_plots()  # a fine sequenza, alla chiusura della finestra o a fine --num-frames
