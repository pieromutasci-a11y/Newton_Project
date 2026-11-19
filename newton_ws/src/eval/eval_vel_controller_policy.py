"""Valutazione con GUI del controllore di velocità addestrato.

La rete (best.pt o last.pt) comanda il drone con una sequenza di 3 riferimenti di velocità (vx, vy, vz, wz nel sistema
del corpo, stesso campo e stesse modalità del training); finito il terzo la valutazione si chiude e salva i grafici.
Stessa fisica e stesso env del training (50 Hz), ma senza gradienti e senza reset.

Frecce, tutte con l'origine nel CENTRO del drone:
  - velocità lineare, in 3D, 0.6 m di freccia per 1 m/s (convenzione di Isaac Lab: verde = comando, blu = reale);
  - velocità angolare di yaw (wz) come VETTORE lungo l'asse di rotazione (asse z del corpo; regola della mano destra:
    wz > 0 punta verso l'alto del drone), 0.3 m per 1 rad/s. Colori diversi, perché sta sulla stessa retta della freccia
    di vz: giallo = comando, ciano = reale (come le frecce di yaw della valutazione di posizione).

Uso (nel container, serve il display X11 di run.sh):
    python /workspace/project_workspace/src/eval/eval_vel_controller_policy.py --run <nome>
    python /workspace/project_workspace/src/eval/eval_vel_controller_policy.py --run <nome> --ckpt last --n_refs 5

Riferimento scelto a mano (nel sistema del corpo): con --vx --vy --vz --wz il riferimento è FISSO per tutta la prova
(gli assi non indicati valgono 0) e la valutazione dura --duration secondi (default 10). Senza nessuno dei quattro,
i riferimenti sono casuali come sopra:
    python /workspace/project_workspace/src/eval/eval_vel_controller_policy.py --run <nome> --vx 0.5 --vy -0.3
    python /workspace/project_workspace/src/eval/eval_vel_controller_policy.py --run <nome> --vz 0.8 --wz 1.0 --duration 15

Senza display: --viewer null (stampa solo le metriche).
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import newton.examples
import torch
import warp as wp

from controllers.math_utils import quat_rotate
from controllers.noise import NoiseConfig
from controllers.vel_controller.ref_handler import RefHandler
from controllers.vel_controller.vel_controller_env import VelControllerEnv
from controllers.vel_controller.vel_controller_mlp import VelControllerMLP
from drone_model.params import DroneParams
from eval.plots_vel_controller import save_vel_eval_plots
from scene.train_world import TrainWorld, TrainWorldConfig

DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "vel_controller"))
REF_COLOR = (0.2, 0.9, 0.2)  # verde: comando (riferimento), velocità lineare
REAL_COLOR = (0.2, 0.5, 1.0)  # blu: velocità lineare reale
REF_ANG_COLOR = (1.0, 0.85, 0.1)  # giallo: comando di velocità angolare (wz)
REAL_ANG_COLOR = (0.2, 1.0, 1.0)  # ciano: velocità angolare reale
ARROW_PER_MS = 0.6  # m di freccia per 1 m/s
ARROW_PER_RADS = 0.3  # m di freccia per 1 rad/s (velocità angolare di yaw, lungo l'asse z del corpo)
MIN_ARROW = 1e-3  # m: sotto questa lunghezza la freccia non si disegna


class Example:
    """Forma richiesta da newton.examples.run: __init__(viewer, args), step(), render()."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        dev = "cuda:0"
        self.dev = dev
        self.args = args
        self.params = DroneParams()
        # Seme diverso a ogni lancio: riferimenti e posizione di partenza sono sempre nuovi.
        # Per ripetere una prova basta passare lo stesso seme con --seed N (viene stampato qui sotto e nel titolo dei grafici).
        self.seed = args.seed if args.seed is not None else int(time.time() * 1000) % (2**31)
        print(f"Seed: {self.seed} (per ripetere gli stessi riferimenti: --seed {self.seed})")

        # Mondo con l'aspetto da quadricottero, SENZA pavimento (il drone si muove nello spazio libero), controllo a 50 Hz
        # (decimation 2); horizon=1: niente finestra di backprop
        self.world = TrainWorld(
            self.params, TrainWorldConfig(num_worlds=args.num_drones, horizon=1, decimation=2, visual=True, ground=False)
        )
        # Riferimenti più lunghi che in training (là 1-2.5 s): si vede il drone raggiungere la velocità e mantenerla
        refs = RefHandler(
            args.num_drones, self.world.ctrl_dt, dev, seed=self.seed, hold_time_range_s=(args.hold_min, args.hold_max)
        )
        # Il drone parte in aria, vicino all'origine: la sequenza di riferimenti lo sposta di qualche metro
        self.env = VelControllerEnv(
            self.world, refs, seed=self.seed, spawn_xy_range=(-0.5, 0.5), spawn_z_range=(1.5, 2.5),
            noise=NoiseConfig(att=math.radians(args.noise_att_deg), lin_vel=args.noise_lin_vel, ang_vel=args.noise_ang_vel),
        )
        # Riferimento scelto a mano da terminale: fisso per tutta la prova (gli assi non indicati a 0)
        given = [args.vx, args.vy, args.vz, args.wz]
        self.fixed = any(v is not None for v in given)
        if self.fixed:
            fixed_ref = [0.0 if v is None else float(v) for v in given]
            refs.set_constant(fixed_ref)
            beyond = [n for n, v, r in zip(("vx", "vy", "vz", "wz"), fixed_ref, refs.vel_range.tolist()) if abs(v) > r]
            if beyond:
                print(f"ATTENZIONE: {', '.join(beyond)} fuori dal campo del training (±{refs.vel_range.tolist()}): la rete non ha visto questi valori")
        self.finished = False
        self.plot_drone = args.plot_drone
        self.rec = []  # una riga per passo di controllo del drone `plot_drone` (colonne: eval/plots_vel_controller.py)
        self.pace = args.pace
        self.t_start = time.perf_counter()
        self.shown_refs = torch.zeros(args.num_drones, dtype=torch.long, device=dev)

        # Rete addestrata
        ckpt_path = os.path.join(args.data_dir, "policy", args.run, f"{args.ckpt}.pt")
        ckpt = torch.load(ckpt_path, map_location=dev)
        self.policy = VelControllerMLP().to(dev)
        self.policy.load_state_dict(ckpt["policy"])  # include le statistiche di normalizzazione
        self.policy.eval()
        print(f"Policy: {ckpt_path} (iterazione {ckpt['iteration']})")
        print("Frecce dal centro del drone. Velocità lineare: verde = riferimento, blu = reale. Velocità angolare (lungo z del corpo): giallo = riferimento, ciano = reale")

        self.obs = self.env.observations()
        self.sim_time = 0.0
        self.steps = 0

        # Stato "da disegnare": la posa viene copiata dalla simulazione a ogni frame
        self.draw_state = self.world.model.state()
        self.draw_q = wp.to_torch(self.draw_state.body_q, requires_grad=False)
        self.draw_q.copy_(self.world.q)

        self.viewer.set_model(self.world.model)
        self.viewer.set_camera(wp.vec3(5.0, -5.0, 3.5), pitch=-20.0, yaw=135.0)

    def _announce_new_refs(self):
        """Stampa ogni riferimento quando comincia (il drone `plot_drone`)."""
        d = self.plot_drone
        k = int(self.env.refs.n_samples[d])
        if self.fixed:
            if int(self.shown_refs[d]) == 0:
                r = self.env.refs.ref[d]
                print(
                    f"Riferimento FISSO per {self.args.duration:.0f} s: "
                    f"vx {float(r[0]):+.2f}  vy {float(r[1]):+.2f}  vz {float(r[2]):+.2f} m/s,  wz {float(r[3]):+.2f} rad/s"
                )
        elif k > int(self.shown_refs[d]) and k <= self.args.n_refs:
            r = self.env.refs.ref[d]
            hold = self.env.refs._hold_left[d] * self.world.ctrl_dt
            print(
                f"Riferimento {k}/{self.args.n_refs} (modalità {int(self.env.refs.mode[d])}): "
                f"vx {float(r[0]):+.2f}  vy {float(r[1]):+.2f}  vz {float(r[2]):+.2f} m/s,  wz {float(r[3]):+.2f} rad/s  "
                f"per {float(hold):.1f} s"
            )
        self.shown_refs[:] = self.env.refs.n_samples

    @torch.no_grad()
    def step(self):
        if self.steps == 0:
            self._announce_new_refs()
        actions = self.policy(self.obs)
        self.env.step(actions)
        e = self.env.errors()  # rispetto al riferimento attivo durante il passo
        # Registrazione del drone `plot_drone`: velocità nel corpo, riferimento, inclinazione, azioni
        d = self.plot_drone
        tilt = torch.rad2deg(torch.acos(torch.clamp(-e.gravity_b[d, 2], -1.0, 1.0)))
        self.rec.append(
            torch.cat(
                [
                    torch.tensor([self.sim_time + self.world.ctrl_dt], device=self.dev),
                    e.lin_vel_b[d], e.ang_vel_b[d],
                    self.env.refs.step_ref[d],
                    tilt.reshape(1), actions[d],
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

        # Fine: con riferimento fisso, dopo --duration secondi; altrimenti quando tutti i droni hanno finito il riferimento
        # numero n_refs (è stato campionato il successivo)
        if self.fixed and self.sim_time >= self.args.duration:
            print(f"\nRiferimento fisso mantenuto per {self.sim_time:.1f} s: valutazione chiusa.")
            self.finished = True
        elif not self.fixed and bool((self.env.refs.n_samples > self.args.n_refs).all()):
            print(f"\n{self.args.n_refs} riferimenti completati a t = {self.sim_time:.1f} s.")
            self.finished = True
        elif self.sim_time >= self.args.max_s:
            print(f"\nTempo massimo ({self.args.max_s:.0f} s) raggiunto: valutazione chiusa.")
            self.finished = True
        else:
            self._announce_new_refs()

    @staticmethod
    def _arrow(start, vec, color, viewer, name):
        """Disegna una freccia da `start` (N,3) lunga `vec` (N,3), nascosta se tutte sono quasi nulle."""
        hidden = bool((vec.norm(dim=-1) < MIN_ARROW).all())
        starts = wp.from_torch(start.contiguous(), dtype=wp.vec3)
        ends = wp.from_torch((start + vec).contiguous(), dtype=wp.vec3)
        viewer.log_arrows(name, starts, ends, color, hidden=hidden)

    @torch.no_grad()
    def render(self):
        self.draw_q.copy_(self.world.q)
        pos = self.world.pos
        quat = self.world.q[:, 3:7]
        ref = self.env.refs.ref  # (N, 4) nel corpo
        n = pos.shape[0]

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.draw_state)

        # Tutte le frecce partono dal centro del drone (pos).
        # Velocità lineare: il riferimento è nel corpo, si ruota nel mondo; la velocità reale è già nel mondo.
        self._arrow(pos, ARROW_PER_MS * quat_rotate(quat, ref[:, :3]), REF_COLOR, self.viewer, "/ref_lin_vel")
        self._arrow(pos, ARROW_PER_MS * self.world.qd[:, :3], REAL_COLOR, self.viewer, "/real_lin_vel")

        # Velocità angolare di yaw: vettore lungo l'asse z del corpo (asse di rotazione), con segno: wz > 0 punta in alto
        # nel drone, wz < 0 in basso. Colori diversi da quelli lineari: sta sulla stessa retta della freccia di vz.
        z_body = quat_rotate(quat, torch.tensor([[0.0, 0.0, 1.0]], device=self.dev).expand(n, 3))
        wz_real = self.env.errors().ang_vel_b[:, 2:3]
        self._arrow(pos, ARROW_PER_RADS * ref[:, 3:4] * z_body, REF_ANG_COLOR, self.viewer, "/ref_yaw_rate")
        self._arrow(pos, ARROW_PER_RADS * wz_real * z_body, REAL_ANG_COLOR, self.viewer, "/real_yaw_rate")
        self.viewer.end_frame()

    def save_plots(self):
        """Salva grafici e riepilogo in data/vel_controller/results/<run>/ (la cartella dei log di quella policy)."""
        if not self.args.plots or len(self.rec) < 2:
            return
        out_dir = os.path.join(self.args.data_dir, "results", self.args.run)
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.join(out_dir, f"eval_{self.args.ckpt}")
        rec = torch.stack(self.rec).cpu().numpy()
        title = f"{self.args.run} ({self.args.ckpt}.pt) - valutazione velocità, drone {self.plot_drone}, seed {self.seed}, {self.world.ctrl_dt * 1000:.0f} ms/passo"
        print("\n" + save_vel_eval_plots(rec, self.params, base + ".png", base + "_summary.txt", title))
        print(f"\nGrafici e riepilogo salvati in {out_dir}/ (eval_{self.args.ckpt}.png, eval_{self.args.ckpt}_summary.txt)")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--run", required=True, help="nome della run in data/vel_controller/policy/")
    parser.add_argument("--ckpt", choices=["best", "last"], default="best")
    parser.add_argument("--num_drones", type=int, default=1)
    parser.add_argument("--vx", type=float, default=None, help="riferimento fisso di vx [m/s] nel corpo (se indichi uno dei quattro, niente riferimenti casuali)")
    parser.add_argument("--vy", type=float, default=None, help="riferimento fisso di vy [m/s]")
    parser.add_argument("--vz", type=float, default=None, help="riferimento fisso di vz [m/s]")
    parser.add_argument("--wz", type=float, default=None, help="riferimento fisso di wz [rad/s]")
    parser.add_argument("--duration", type=float, default=10.0, help="durata della prova con riferimento fisso [s]")
    parser.add_argument("--n_refs", type=int, default=3, help="riferimenti di velocità da percorrere prima di chiudere")
    parser.add_argument("--hold_min", type=float, default=3.0, help="durata minima di ogni riferimento [s] (training: 1.0)")
    parser.add_argument("--hold_max", type=float, default=5.0, help="durata massima di ogni riferimento [s] (training: 2.5)")
    parser.add_argument("--max_s", type=float, default=40.0, help="tempo massimo di simulazione [s], per non restare bloccati")
    parser.add_argument("--pace", action=argparse.BooleanOptionalAction, default=True, help="simula a tempo reale")
    parser.add_argument("--noise_att_deg", type=float, default=0.0, help="rumore bianco sull'assetto misurato, roll/pitch/yaw [gradi]")
    parser.add_argument("--noise_lin_vel", type=float, default=0.0, help="rumore bianco sulla velocità lineare misurata [m/s]")
    parser.add_argument("--noise_ang_vel", type=float, default=0.0, help="rumore bianco sulla velocità angolare misurata [rad/s]")
    parser.add_argument("--seed", type=int, default=None, help="seme dei riferimenti (default: casuale a ogni lancio)")
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
