"""Ambiente del controllore di posizione.

Controllo a CASCATA, come in Isaac Lab. Le azioni di questo env sono i 4 riferimenti di velocità (vx, vy, vz, wz)
normalizzati in [-1, 1]; a ogni passo (25 Hz) l'env li moltiplica per la scala e li passa alla rete di velocità
`low_level` (congelata, 50 Hz), che li trasforma in spinta e momenti: per ogni passo di posizione la rete di velocità
agisce 2 volte, con 2 passi di fisica ciascuna.

Mette insieme il mondo fisico (scene/train_world.py, da creare con decimation=2: 50 Hz), la rete di velocità e i
riferimenti di posizione (target_handler.py). Si occupa di: far agire la cascata, calcolare gli errori, costruire le
osservazioni per la rete di posizione, e tutta la gestione di terminazioni, spawn e reset. Il costo NON sta qui: si
calcola in train/cost_function_pos_controller.py a partire dagli errori di questo env.

Non conosce l'ottimizzatore e non vede Warp: parla solo con tensori PyTorch. La rete di velocità gli arriva come una
semplice funzione (vel_obs (N,17) -> azioni (N,4)), di solito PosControllerMLP.vel_policy.

Uso tipico (un'iterazione di training = una finestra di `horizon` passi di posizione):

    world = TrainWorld(params, TrainWorldConfig(num_worlds=N, decimation=2, horizon=2 * horizon))
    env = PosControllerEnv(world, policy.vel_policy)
    obs = env.observations()
    for t in range(horizon):
        actions = policy(obs)                 # riferimenti di velocità normalizzati
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

from ..math_utils import quat_mul, quat_rotate_inverse, yaw_from_quat  # noqa: F401
from ..noise import NoiseConfig, measure_state  # noqa: F401 (NoiseConfig riesportata: la usano train, eval e test) (riesportate: i test le importano da qui)
from ..vel_controller.vel_controller_env import vel_observations
from .target_handler import TargetHandler

LOW_LEVEL_DT = 0.02  # s: la rete di velocità è addestrata a 50 Hz (fisica a 100 Hz con decimation 2)


@dataclass
class Errors:
    """Errori rispetto al riferimento, calcolati una volta sola. Sono la base di osservazioni e costo."""

    pos_err: torch.Tensor  # (N, 3) riferimento - posizione, nel sistema del mondo [m]
    yaw_err: torch.Tensor  # (N,) yaw di riferimento - yaw del drone [rad]
    lin_vel_b: torch.Tensor  # (N, 3) velocità lineare nel sistema del corpo
    ang_vel_b: torch.Tensor  # (N, 3) velocità angolare nel sistema del corpo
    gravity_b: torch.Tensor  # (N, 3) gravità proiettata nel corpo: (0, 0, -1) se il drone è dritto
    # (N, 3) velocità del riferimento nel sistema del corpo: nulla per i waypoint, quella della curva per le traiettorie.
    # None = nulla (il costo la tratta come zero).
    ref_vel_b: torch.Tensor | None = None


class PosControllerEnv:
    # Osservazioni (18 + 5 * (window_points - 1) = 33): errore di posizione nel sistema del corpo (3) + sin/cos
    #   dell'errore di yaw (2) + velocità lineare nel corpo (3) + velocità angolare nel corpo (3)
    #   + gravità proiettata nel corpo (3) + azione precedente (4)
    #   + anteprima: per ognuno dei window_points - 1 punti futuri, incremento di posizione rispetto al punto prima
    #     nel sistema del corpo (3) + sin/cos dell'incremento di yaw (2). Tutti nulli (sin 0, cos 1) per i waypoint.
    # Devono coincidere con input_dim e output_dim di mlp_config.yaml (self.obs_dim, dipende dalla finestra).
    base_obs_dim = 18
    # Azioni (4), normalizzate in [-1, 1]: riferimenti di velocità vx, vy, vz, wz nel sistema del corpo; l'env li
    # moltiplica per vel_scale (m/s e rad/s) e li dà alla rete di velocità, che produce spinta e momenti.
    action_dim = 4

    def __init__(
        self,
        world: TrainWorld,
        low_level,
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
        control_dt: float = 0.04,  # s, intervallo di controllo della rete di posizione (25 Hz, come Isaac Lab)
        # Scala dei riferimenti di velocità: stesso campo su cui è stata addestrata la rete di velocità
        # (Isaac Lab: target_lin_vel_xy_scale = target_lin_vel_z_scale = 1.0 m/s, target_yaw_vel_scale = 1.5 rad/s)
        vel_scale: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.5),
        noise: NoiseConfig | None = None,  # rumore sul feedback (default: nessuno)
    ):
        self.world = world
        self.low_level = low_level  # rete di velocità (congelata): osservazioni (N,17) -> azioni (N,4)
        self.n = world.cfg.num_worlds
        self.device = world.q.device
        assert abs(world.ctrl_dt - LOW_LEVEL_DT) < 1e-9, (
            f"il mondo deve avere un passo di controllo di {LOW_LEVEL_DT} s (50 Hz, decimation=2), non {world.ctrl_dt} s: "
            "è la frequenza a cui è stata addestrata la rete di velocità"
        )
        # Passi della rete di velocità per ogni passo di posizione (2: 50 Hz contro 25 Hz)
        self.n_sub = round(control_dt / world.ctrl_dt)
        assert abs(self.n_sub * world.ctrl_dt - control_dt) < 1e-9, "control_dt deve essere un multiplo di 0.02 s"
        self.dt = control_dt  # intervallo di controllo della posizione [s]
        self.vel_scale = torch.tensor(vel_scale, device=self.device)

        # Riferimenti: se non ne viene passato uno, si usa quello di default (intervalli in target_handler.py)
        self.targets = targets or TargetHandler(self.n, self.device, seed=seed, dt=control_dt)
        self.obs_dim = self.base_obs_dim + 5 * (self.targets.window_points - 1)

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
        self.noise = noise or NoiseConfig()
        self._gen_noise = torch.Generator(device=self.device)
        self._gen_noise.manual_seed(seed + 2)

        self._gravity_dir = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.n, 3)

        # Azione (riferimenti di velocità normalizzati) del passo precedente: serve alle osservazioni e al costo
        self.prev_actions = torch.zeros(self.n, self.action_dim, device=self.device)
        # Ultima azione della rete di velocità (spinta e momenti): serve alle sue osservazioni (e ai grafici)
        self.low_prev_actions = torch.zeros(self.n, 4, device=self.device)
        self.v_ref = torch.zeros(self.n, 4, device=self.device)  # ultimi riferimenti di velocità [m/s, rad/s] (per i log)
        self._steps_alive = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self._hold_timer = torch.zeros(self.n, device=self.device)  # s trascorsi sul riferimento attivo, entro le soglie

        # Prima inizializzazione: tutti i mondi
        self._reset_worlds(torch.arange(self.n, device=self.device))

    # ------------------------------------------------------------------ passo
    def step(self, actions: torch.Tensor) -> torch.Tensor:
        """Applica i riferimenti di velocità (N, 4) normalizzati, avanza di un passo (0.04 s) e restituisce le
        osservazioni (N, obs_dim).

        Per ogni passo la rete di velocità agisce `n_sub` volte (2): ogni volta legge lo stato, il riferimento e la sua
        azione precedente, produce spinta e momenti, e il mondo avanza di 0.02 s (2 passi di fisica) tenendoli costanti.

        Dopo la chiamata: lo stato nuovo è in self.world.q / self.world.qd, e self.errors() dà gli errori
        rispetto al riferimento attivo durante il passo. L'azione precedente va letta PRIMA della chiamata
        (self.prev_actions viene aggiornata con `actions`).
        """
        self.v_ref = torch.clamp(actions, -1.0, 1.0) * self.vel_scale  # (N, 4) m/s e rad/s, nel corpo
        for _ in range(self.n_sub):
            qm, qdm = self._measure() if self.noise.to_vel_net else (self.world.q, self.world.qd)
            vel_obs = vel_observations(qm, qdm, self._gravity_dir, self.low_prev_actions, self.v_ref)
            wrench = self.low_level(vel_obs)  # spinta e momenti normalizzati, in [-1, 1]
            self.world.step(wrench)
            self.low_prev_actions = wrench
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
        """(N,) bool: il drone è sull'ULTIMO riferimento della coda e ci è rimasto per `hold_time_s`
        (mai vero per le traiettorie: durano finché dura l'episodio)."""
        return self.targets.is_last & ~self.targets.is_traj & (self._hold_timer >= self._hold_time_s)

    # ------------------------------------------------------------------ errori e osservazioni
    def _measure(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Stato MISURATO (q (N,7), qd (N,6)): quello vero più rumore bianco (NoiseConfig), nuova estrazione a ogni chiamata."""
        return measure_state(self.world.q, self.world.qd, self.noise, self._gen_noise)

    def _compute_errors(self, target_pos: torch.Tensor, target_yaw: torch.Tensor, target_vel: torch.Tensor, measured=None) -> Errors:
        q, qd = measured if measured is not None else (self.world.q, self.world.qd)
        pos, quat = q[:, :3], q[:, 3:7]
        # Convenzione verificata: qd = (velocità lineare | velocità angolare), nel sistema del mondo
        return Errors(
            pos_err=target_pos - pos,
            yaw_err=target_yaw - yaw_from_quat(quat),
            lin_vel_b=quat_rotate_inverse(quat, qd[:, :3]),
            ang_vel_b=quat_rotate_inverse(quat, qd[:, 3:6]),
            gravity_b=quat_rotate_inverse(quat, self._gravity_dir),
            ref_vel_b=quat_rotate_inverse(quat, target_vel),
        )

    def errors(self) -> Errors:
        """Errori dopo l'ultimo passo, rispetto al riferimento che era attivo DURANTE quel passo (per il costo)."""
        t = self.targets
        return self._compute_errors(t.step_target_pos, t.step_target_yaw, t.step_target_vel)

    def observations(self) -> torch.Tensor:
        """(N, obs_dim) osservazioni per la rete, rispetto al riferimento attuale."""
        t = self.targets
        measured = self._measure()  # feedback con rumore (se configurato); il costo usa invece errors(), lo stato vero
        self.last_measured = measured  # (q, qd) letti dalla rete di posizione: per i grafici di valutazione
        e = self._compute_errors(t.target_pos, t.target_yaw, t.target_vel, measured)
        # Errore di posizione nel sistema del CORPO (non del mondo): la rete deve sapere in che direzione del
        # drone si trova il bersaglio, cioè verso quale asse inclinarsi. Con l'errore nel mondo servirebbe
        # l'orientamento assoluto del drone per ruotarlo (in Isaac Lab c'era sin/cos dell'angolo di yaw assoluto).
        quat = measured[0][:, 3:7]
        pos_err_b = quat_rotate_inverse(quat, e.pos_err)
        # Anteprima: incrementi tra i punti futuri del riferimento (nel corpo, come l'errore di posizione)
        dpos, dyaw = t.window()  # (N, H-1, 3), (N, H-1)
        h = dpos.shape[1]
        dpos_b = quat_rotate_inverse(quat.repeat_interleave(h, dim=0), dpos.reshape(-1, 3)).reshape(self.n, h, 3)
        # un blocco da 5 per punto futuro: incremento di posizione (3) + sin/cos dell'incremento di yaw (2)
        preview = torch.cat([dpos_b, torch.sin(dyaw).unsqueeze(-1), torch.cos(dyaw).unsqueeze(-1)], dim=-1).reshape(self.n, h * 5)
        return torch.cat(
            [
                pos_err_b,
                torch.sin(e.yaw_err).unsqueeze(-1),
                torch.cos(e.yaw_err).unsqueeze(-1),
                e.lin_vel_b,
                e.ang_vel_b,
                e.gravity_b,
                self.prev_actions,
                preview,
            ],
            dim=-1,
        )

    # ------------------------------------------------------------------ terminazioni e reset
    def _uniform(self, low: float, high: float, shape) -> torch.Tensor:
        return low + (high - low) * torch.rand(shape, device=self.device, generator=self._gen)

    def _reset_worlds(self, ids: torch.Tensor) -> None:
        """Riporta i droni `ids` a una posizione casuale, dritti e fermi, con nuovi riferimenti."""
        k = ids.numel()
        self.targets.reset(ids)  # prima i riferimenti: le traiettorie decidono dove nasce il drone
        spawn = torch.stack(
            [
                self._uniform(*self.spawn_xy_range, (k,)),
                self._uniform(*self.spawn_xy_range, (k,)),
                self._uniform(*self.spawn_z_range, (k,)),
            ],
            dim=-1,
        )
        # Traiettoria: il drone nasce vicino al punto di partenza della curva (entro 0.3 m), fermo
        near = self.targets.target_pos[ids] + self._uniform(-0.3, 0.3, (k, 3))
        near[:, 2] = near[:, 2].clamp(min=0.3)
        spawn = torch.where(self.targets.is_traj[ids][:, None], near, spawn)
        self.world.reset(ids, spawn)
        self._steps_alive[ids] = 0
        self._hold_timer[ids] = 0.0
        self.prev_actions = self.prev_actions.clone()
        self.prev_actions[ids] = 0.0
        self.low_prev_actions = self.low_prev_actions.clone()
        self.low_prev_actions[ids] = 0.0

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
        self.low_prev_actions = self.low_prev_actions.detach()
        if finished.numel() > 0:
            self._reset_worlds(finished)
