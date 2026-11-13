"""Ambiente del controllore di posizione.

Mette insieme il mondo fisico (scene/train_world.py) e i riferimenti (target_handler.py). Si occupa di:
applicare le azioni, calcolare gli errori, costruire le osservazioni per la rete, e tutta la gestione
di terminazioni, spawn e reset. Il costo NON sta qui: si calcola in train/cost_function.py a partire
dagli errori di questo env.

Non conosce la rete né l'ottimizzatore, e non vede Warp: parla solo con tensori PyTorch.

Uso tipico (un'iterazione di training = una finestra di `horizon` passi):

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

from .target_handler import TargetHandler


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Ruota il vettore v (nel mondo) nel sistema del corpo. q: (N, 4) come (x, y, z, w), v: (N, 3)."""
    q_vec, q_w = q[:, :3], q[:, 3:4]
    a = v * (2.0 * q_w**2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (q_vec * v).sum(dim=-1, keepdim=True) * 2.0
    return a - b + c


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Angolo di imbardata (rotazione attorno a z del mondo) da un quaternione (x, y, z, w)."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@dataclass
class Errors:
    """Errori rispetto al riferimento, calcolati una volta sola. Sono la base di osservazioni e costo."""

    pos_err: torch.Tensor  # (N, 3) riferimento - posizione, nel sistema del mondo [m]
    yaw_err: torch.Tensor  # (N,) yaw di riferimento - yaw del drone [rad]
    lin_vel_b: torch.Tensor  # (N, 3) velocità lineare nel sistema del corpo (errore rispetto a velocità nulla)
    ang_vel_b: torch.Tensor  # (N, 3) velocità angolare nel sistema del corpo
    gravity_b: torch.Tensor  # (N, 3) gravità proiettata nel corpo: (0, 0, -1) se il drone è dritto


class PosControllerEnv:
    # Osservazioni (18): errore di posizione nel sistema del corpo (3) + sin/cos dell'errore di yaw (2)
    #   + velocità lineare nel corpo (3) + velocità angolare nel corpo (3)
    #   + gravità proiettata nel corpo (3) + azione precedente (4)
    # Devono coincidere con input_dim e output_dim di mlp_config.yaml.
    obs_dim = 18
    # Azioni (4), normalizzate in [-1, 1]: spinta, momento x, momento y, momento z (vedi drone_model/actuators.py)
    action_dim = 4

    def __init__(
        self,
        world: TrainWorld,
        targets: TargetHandler | None = None,
        *,
        seed: int = 0,
        spawn_xy_range: tuple[float, float] = (-1.0, 1.0),  # m, posizione di partenza dopo un reset
        spawn_z_range: tuple[float, float] = (0.5, 1.5),  # m
        max_tilt_deg: float = 65.0,  # inclinazione massima prima del reset, come in Isaac Lab
        max_episode_s: float = 15.0,  # durata massima di un episodio [s]
        max_distance: float = 20.0,  # m, oltre questa distanza dall'origine il drone è perso e si resetta
        # Avanzamento della coda di riferimenti (valori di Isaac Lab): si passa al successivo quando il drone
        # è entro queste soglie di posizione e yaw da almeno `hold_time_s`
        reach_threshold: float = 0.15,  # m
        reach_yaw_threshold: float = 0.20,  # rad
        hold_time_s: float = 1.2,  # s
    ):
        self.world = world
        self.n = world.cfg.num_worlds
        self.dt = world.ctrl_dt  # intervallo di controllo [s] (25 Hz con decimation 4)
        self.device = world.q.device

        # Riferimenti: se non ne viene passato uno, si usa quello di default (intervalli in target_handler.py)
        self.targets = targets or TargetHandler(self.n, self.device, seed=seed)

        self.spawn_xy_range = spawn_xy_range
        self.spawn_z_range = spawn_z_range
        self._cos_max_tilt = math.cos(math.radians(max_tilt_deg))
        self._max_steps = int(max_episode_s / self.dt)
        self._max_distance = max_distance
        self._reach_threshold = reach_threshold
        self._reach_yaw_threshold = reach_yaw_threshold
        self._hold_time_s = hold_time_s

        # Generatore casuale (seme diverso dai riferimenti) per le posizioni di partenza
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(seed + 1)

        self._gravity_dir = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.n, 3)

        # Azione applicata al passo precedente (serve alle osservazioni e al costo sulla variazione delle azioni)
        self.prev_actions = torch.zeros(self.n, self.action_dim, device=self.device)
        self._steps_alive = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self._hold_timer = torch.zeros(self.n, device=self.device)  # s trascorsi sul riferimento attivo, entro le soglie

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
        self.targets.step()  # memorizza il riferimento attivo durante il passo (per il costo)
        self._update_waypoints()  # eventuale avanzamento al riferimento successivo: lo vedrà la rete al passo dopo
        self.prev_actions = actions
        self._steps_alive = self._steps_alive + 1
        return self.observations()

    @torch.no_grad()
    def _update_waypoints(self) -> None:
        """Se il drone è rimasto entro le soglie sul riferimento per `hold_time_s`, passa al successivo."""
        e = self.errors()
        yaw_abs = torch.atan2(torch.sin(e.yaw_err), torch.cos(e.yaw_err)).abs()
        reached = (e.pos_err.norm(dim=-1) < self._reach_threshold) & (yaw_abs < self._reach_yaw_threshold)
        self._hold_timer = torch.where(reached, self._hold_timer + self.dt, torch.zeros_like(self._hold_timer))
        advance = ((self._hold_timer >= self._hold_time_s) & ~self.targets.is_last).nonzero(as_tuple=False).squeeze(-1)
        if advance.numel() > 0:
            self.targets.advance(advance)
            self._hold_timer[advance] = 0.0

    def sequence_done(self) -> torch.Tensor:
        """(N,) bool: il drone è sull'ULTIMO riferimento della coda e ci è rimasto per `hold_time_s`."""
        return self.targets.is_last & (self._hold_timer >= self._hold_time_s)

    # ------------------------------------------------------------------ errori e osservazioni
    def _compute_errors(self, target_pos: torch.Tensor, target_yaw: torch.Tensor) -> Errors:
        q, qd = self.world.q, self.world.qd
        pos, quat = q[:, :3], q[:, 3:7]
        # Convenzione verificata: qd = (velocità lineare | velocità angolare), nel sistema del mondo
        return Errors(
            pos_err=target_pos - pos,
            yaw_err=target_yaw - yaw_from_quat(quat),
            lin_vel_b=quat_rotate_inverse(quat, qd[:, :3]),
            ang_vel_b=quat_rotate_inverse(quat, qd[:, 3:6]),
            gravity_b=quat_rotate_inverse(quat, self._gravity_dir),
        )

    def errors(self) -> Errors:
        """Errori dopo l'ultimo passo, rispetto al riferimento che era attivo DURANTE quel passo (per il costo)."""
        return self._compute_errors(self.targets.step_target_pos, self.targets.step_target_yaw)

    def observations(self) -> torch.Tensor:
        """(N, obs_dim) osservazioni per la rete, rispetto al riferimento attuale."""
        e = self._compute_errors(self.targets.target_pos, self.targets.target_yaw)
        # Errore di posizione nel sistema del CORPO (non del mondo): la rete deve sapere in che direzione del
        # drone si trova il bersaglio, cioè verso quale asse inclinarsi. Con l'errore nel mondo servirebbe
        # l'orientamento assoluto del drone per ruotarlo (in Isaac Lab c'era sin/cos dell'angolo di yaw assoluto).
        pos_err_b = quat_rotate_inverse(self.world.q[:, 3:7], e.pos_err)
        return torch.cat(
            [
                pos_err_b,
                torch.sin(e.yaw_err).unsqueeze(-1),
                torch.cos(e.yaw_err).unsqueeze(-1),
                e.lin_vel_b,
                e.ang_vel_b,
                e.gravity_b,
                self.prev_actions,
            ],
            dim=-1,
        )

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
        self.targets.reset(ids)
        self._steps_alive[ids] = 0
        self._hold_timer[ids] = 0.0
        self.prev_actions = self.prev_actions.clone()
        self.prev_actions[ids] = 0.0

    def done(self) -> torch.Tensor:
        """(N,) bool: mondi da resettare: inclinazione eccessiva, episodio troppo lungo, drone perso o stato esploso."""
        with torch.no_grad():
            quat = self.world.q[:, 3:7]
            cos_tilt = -quat_rotate_inverse(quat, self._gravity_dir)[:, 2]
            too_tilted = cos_tilt < self._cos_max_tilt
            timeout = self._steps_alive >= self._max_steps
            # Stato non finito (NaN/inf: con i confronti sopra darebbe False e il mondo non verrebbe mai resettato)
            # oppure drone lontanissimo dall'area di lavoro
            q, qd = self.world.q, self.world.qd
            exploded = ~torch.isfinite(q).all(dim=-1) | ~torch.isfinite(qd).all(dim=-1)
            lost = q[:, :3].abs().amax(dim=-1) > self._max_distance
            return too_tilted | timeout | exploded | lost

    def end_window(self) -> None:
        """Da chiamare a fine finestra: stacca dal grafo e resetta i mondi terminati."""
        finished = self.done().nonzero(as_tuple=False).squeeze(-1)
        self.world.detach()
        self.prev_actions = self.prev_actions.detach()
        if finished.numel() > 0:
            self._reset_worlds(finished)
