"""Ambiente del controllore di velocità (anello interno della cascata).

Ingresso della rete: riferimenti di velocità (vx, vy, vz, wz) nel sistema del corpo. Uscita: 4 azioni in [-1, 1]
= spinta e momenti (drone_model/actuators.py). Come in Isaac Lab (vel_controller_env): 17 osservazioni,
controllo a 50 Hz, nessuna posizione né yaw assoluti.

Mette insieme il mondo fisico (scene/train_world.py, da creare con decimation=2: 50 Hz) e i riferimenti
(ref_handler.py). Si occupa di: applicare le azioni, calcolare gli errori, costruire le osservazioni, e tutta la gestione
di terminazioni, spawn e reset. Il costo NON sta qui: si calcola in train/cost_function_vel_controller.py a partire
dagli errori di questo env. Non conosce la rete né l'ottimizzatore, e non vede Warp: parla solo con tensori PyTorch.

Uso tipico (un'iterazione di training = una finestra di `horizon` passi):

    world = TrainWorld(params, TrainWorldConfig(num_worlds=N, decimation=2, horizon=H))
    env = VelControllerEnv(world)
    obs = env.observations()
    for t in range(horizon):
        actions = policy(obs)
        prev_actions = env.prev_actions       # azione del passo precedente (da leggere PRIMA di step)
        obs = env.step(actions)
        errors = env.errors()                 # per il costo
    loss.backward(); optimizer.step()
    env.end_window()                          # stacca dal grafo e resetta i mondi terminati
"""

import math
from dataclasses import dataclass

import torch

from scene.train_world import TrainWorld

from ..math_utils import quat_rotate_inverse
from ..noise import NoiseConfig, measure_state
from .ref_handler import RefHandler


def vel_observations(
    q: torch.Tensor, qd: torch.Tensor, gravity_dir: torch.Tensor, prev_actions: torch.Tensor, ref: torch.Tensor
) -> torch.Tensor:
    """(N, 17) osservazioni della rete di velocità: velocità lineare nel corpo (3), velocità angolare nel corpo (3),
    gravità proiettata nel corpo (3), azione precedente della rete di velocità (4) e riferimento (vx, vy, vz, wz) (4).

    Funzione unica: la usano sia VelControllerEnv (training della velocità) sia il controllore di posizione quando
    comanda la rete di velocità congelata (cascata), così le due osservazioni non possono divergere.
    """
    quat = q[:, 3:7]
    return torch.cat(
        [
            quat_rotate_inverse(quat, qd[:, :3]),
            quat_rotate_inverse(quat, qd[:, 3:6]),
            quat_rotate_inverse(quat, gravity_dir),
            prev_actions,
            ref,
        ],
        dim=-1,
    )


@dataclass
class VelErrors:
    """Errori rispetto al riferimento, calcolati una volta sola. Sono la base di osservazioni e costo."""

    vel_err: torch.Tensor  # (N, 4) riferimento - velocità, (vx, vy, vz, wz) nel sistema del corpo
    lin_vel_b: torch.Tensor  # (N, 3) velocità lineare nel sistema del corpo
    ang_vel_b: torch.Tensor  # (N, 3) velocità angolare nel sistema del corpo
    gravity_b: torch.Tensor  # (N, 3) gravità proiettata nel corpo: (0, 0, -1) se il drone è dritto


class VelControllerEnv:
    # Osservazioni (17): velocità lineare nel corpo (3) + velocità angolare nel corpo (3) + gravità proiettata (3)
    #   + azione precedente (4) + riferimento di velocità (4: vx, vy, vz, wz)
    # Devono coincidere con input_dim e output_dim di mlp_config.yaml.
    obs_dim = 17
    # Azioni (4), normalizzate in [-1, 1]: spinta, momento x, momento y, momento z (vedi drone_model/actuators.py)
    action_dim = 4

    def __init__(
        self,
        world: TrainWorld,
        refs: RefHandler | None = None,
        *,
        seed: int = 0,
        spawn_xy_range: tuple[float, float] = (-1.0, 1.0),  # m, posizione di partenza dopo un reset
        spawn_z_range: tuple[float, float] = (0.5, 1.5),  # m
        max_tilt_deg: float = 65.0,  # inclinazione massima prima del reset, come in Isaac Lab
        max_episode_s: float = 10.0,  # durata massima di un episodio, come in Isaac Lab (episode_length_s)
        max_distance: float = 20.0,  # m, oltre questa distanza dall'origine il drone è perso e si resetta
        max_ang_vel: float = 30.0,  # rad/s, oltre questa velocità angolare il drone sta per divergere: reset
        noise: NoiseConfig | None = None,  # rumore sul feedback letto dalla rete (default: nessuno); il costo usa lo stato vero
    ):
        self.world = world
        self.n = world.cfg.num_worlds
        self.dt = world.ctrl_dt  # intervallo di controllo [s] (0.02 con decimation 2: 50 Hz)
        self.device = world.q.device

        # Riferimenti: se non ne viene passato uno, si usa quello di default (campo e permanenze in ref_handler.py)
        self.refs = refs or RefHandler(self.n, self.dt, self.device, seed=seed)

        self.spawn_xy_range = spawn_xy_range
        self.spawn_z_range = spawn_z_range
        self._cos_max_tilt = math.cos(math.radians(max_tilt_deg))
        self._max_steps = int(max_episode_s / self.dt)
        self._max_distance = max_distance
        self._max_ang_vel = max_ang_vel

        # Generatore casuale (seme diverso dai riferimenti) per le posizioni di partenza
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(seed + 1)
        self.noise = noise or NoiseConfig()
        self._gen_noise = torch.Generator(device=self.device)
        self._gen_noise.manual_seed(seed + 2)

        self._gravity_dir = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.n, 3)

        # Azione applicata al passo precedente (serve alle osservazioni e al costo sulla variazione delle azioni)
        self.prev_actions = torch.zeros(self.n, self.action_dim, device=self.device)
        self._steps_alive = torch.zeros(self.n, dtype=torch.long, device=self.device)

        # Prima inizializzazione: tutti i mondi
        self._reset_worlds(torch.arange(self.n, device=self.device))

    # ------------------------------------------------------------------ passo
    def step(self, actions: torch.Tensor) -> torch.Tensor:
        """Applica le azioni (N, 4), avanza di un passo e restituisce le osservazioni (N, obs_dim).

        Dopo la chiamata: lo stato nuovo è in self.world.q / self.world.qd, e self.errors() dà gli errori
        rispetto al riferimento attivo durante il passo. L'azione precedente va letta PRIMA della chiamata
        (self.prev_actions viene aggiornata con `actions`).
        """
        self.world.step(actions)
        self.refs.step()  # eventuale nuovo riferimento: lo vedrà la rete al passo successivo
        self.prev_actions = actions
        self._steps_alive = self._steps_alive + 1
        return self.observations()

    # ------------------------------------------------------------------ errori e osservazioni
    def _compute_errors(self, ref: torch.Tensor) -> VelErrors:
        q, qd = self.world.q, self.world.qd
        quat = q[:, 3:7]
        # Convenzione verificata: qd = (velocità lineare | velocità angolare), nel sistema del mondo
        lin_vel_b = quat_rotate_inverse(quat, qd[:, :3])
        ang_vel_b = quat_rotate_inverse(quat, qd[:, 3:6])
        vel = torch.cat([lin_vel_b, ang_vel_b[:, 2:3]], dim=-1)  # (vx, vy, vz, wz): gli assi del riferimento
        return VelErrors(
            vel_err=ref - vel,
            lin_vel_b=lin_vel_b,
            ang_vel_b=ang_vel_b,
            gravity_b=quat_rotate_inverse(quat, self._gravity_dir),
        )

    def errors(self) -> VelErrors:
        """Errori dopo l'ultimo passo, rispetto al riferimento che era attivo DURANTE quel passo (per il costo)."""
        return self._compute_errors(self.refs.step_ref)

    def observations(self) -> torch.Tensor:
        """(N, obs_dim) osservazioni per la rete, rispetto al riferimento attuale. Con il rumore la rete legge lo stato
        MISURATO (velocità e assetto con rumore); il riferimento e la sua azione precedente non hanno rumore."""
        q, qd = measure_state(self.world.q, self.world.qd, self.noise, self._gen_noise)
        self.last_measured = (q, qd)  # per i grafici di valutazione
        return vel_observations(q, qd, self._gravity_dir, self.prev_actions, self.refs.ref)

    # ------------------------------------------------------------------ terminazioni e reset
    def _uniform(self, low: float, high: float, shape) -> torch.Tensor:
        return low + (high - low) * torch.rand(shape, device=self.device, generator=self._gen)

    def _reset_worlds(self, ids: torch.Tensor) -> None:
        """Riporta i droni `ids` a una posizione casuale, dritti e fermi, con nuovi riferimenti."""
        k = ids.numel()
        spawn = torch.stack(
            [
                self._uniform(*self.spawn_xy_range, (k,)),
                self._uniform(*self.spawn_xy_range, (k,)),
                self._uniform(*self.spawn_z_range, (k,)),
            ],
            dim=-1,
        )
        self.world.reset(ids, spawn)
        self.refs.reset(ids)
        self._steps_alive[ids] = 0
        self.prev_actions = self.prev_actions.clone()
        self.prev_actions[ids] = 0.0

    def done(self) -> torch.Tensor:
        """(N,) bool: mondi da resettare: inclinazione eccessiva, episodio troppo lungo, drone perso,
        rotazione troppo veloce o stato esploso."""
        with torch.no_grad():
            q, qd = self.world.q, self.world.qd
            cos_tilt = -quat_rotate_inverse(q[:, 3:7], self._gravity_dir)[:, 2]
            too_tilted = cos_tilt < self._cos_max_tilt
            timeout = self._steps_alive >= self._max_steps
            # Stato non finito (NaN/inf: con i confronti sopra darebbe False e il mondo non verrebbe mai resettato)
            exploded = ~torch.isfinite(q).all(dim=-1) | ~torch.isfinite(qd).all(dim=-1)
            lost = q[:, :3].abs().amax(dim=-1) > self._max_distance
            spinning = qd[:, 3:6].norm(dim=-1) > self._max_ang_vel
            return too_tilted | timeout | exploded | lost | spinning

    def end_window(self) -> None:
        """Da chiamare a fine finestra: stacca dal grafo e resetta i mondi terminati."""
        finished = self.done().nonzero(as_tuple=False).squeeze(-1)
        self.world.detach()
        self.prev_actions = self.prev_actions.detach()
        if finished.numel() > 0:
            self._reset_worlds(finished)
