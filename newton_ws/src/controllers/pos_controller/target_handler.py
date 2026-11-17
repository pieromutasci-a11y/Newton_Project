"""Generazione dei riferimenti (x, y, z, yaw) nel sistema del mondo.

Ogni mondo, a inizio episodio, è di uno di due tipi (estratto a caso con probabilità `traj_prob`):

  * WAYPOINT (come in Isaac Lab): una CODA di `n_waypoints` riferimenti casuali fermi, con un riferimento attivo
    (il primo). Si passa al successivo con advance(), chiamato dall'env quando il drone ha raggiunto il riferimento
    e ci è rimasto per il tempo richiesto. Dopo l'ultimo il riferimento resta lo stesso.
  * TRAIETTORIA: una curva continua nel tempo, p(t) (retta, cerchio, ellisse, otto, elica). Il riferimento attivo è
    il punto della curva all'istante t, che scorre da solo (nessun avanzamento per raggiungimento).

In entrambi i casi l'env legge la stessa interfaccia: riferimento attivo (posizione, yaw), la sua velocità
(nulla per i waypoint) e la FINESTRA di anteprima: i punti a t, t+Δ, t+2Δ, ... come incrementi tra punti consecutivi
(nulli per i waypoint: "bersaglio fermo"). Così una traiettoria è, per la rete, una coda di waypoint molto fitti.

Si occupa SOLO dei riferimenti: non conosce il mondo fisico, né i reset, né la rete, né quando si avanza.
I riferimenti non dipendono dalla policy: niente di quello che sta qui è nel grafo di autograd.
"""

import math

import torch

# Tipi di traiettoria
LINE, CIRCLE, ELLIPSE, EIGHT, HELIX = range(5)
KIND_NAMES = {"line": LINE, "circle": CIRCLE, "ellipse": ELLIPSE, "eight": EIGHT, "helix": HELIX}


def _wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


class TargetHandler:
    def __init__(
        self,
        num_worlds: int,
        device,
        *,
        seed: int = 0,
        n_waypoints: int = 3,  # riferimenti per episodio (mondi waypoint)
        xy_range: tuple[float, float] = (-1.5, 1.5),  # m, x e y uniformi in questo intervallo
        z_range: tuple[float, float] = (0.5, 1.5),  # m
        traj_prob: float = 0.0,  # probabilità che un episodio sia una traiettoria (0: solo waypoint, come prima)
        traj_speed_range: tuple[float, float] = (0.1, 0.5),  # m/s, picco della velocità della curva
        dt: float = 0.04,  # s, passo di controllo della rete di posizione (avanza il tempo delle traiettorie)
        window_points: int = 4,  # punti della finestra: l'attuale + (window_points - 1) futuri
        window_dt: float = 0.5,  # s tra due punti consecutivi della finestra
    ):
        # il riferimento di yaw dei waypoint è uniforme in [-pi, pi]
        self.n = num_worlds
        self.device = device
        self.n_waypoints = n_waypoints
        self.xy_range = xy_range
        self.z_range = z_range
        self.traj_prob = traj_prob
        self.traj_speed_range = traj_speed_range
        self.dt = dt
        self.window_points = window_points
        self.window_dt = window_dt

        # Generatore casuale con seme: esperimenti ripetibili
        self._gen = torch.Generator(device=device)
        self._gen.manual_seed(seed)

        # Code di riferimenti (N, K, ...) e indice del riferimento attivo di ogni mondo (mondi waypoint)
        self.wp_pos = torch.zeros(num_worlds, n_waypoints, 3, device=device)
        self.wp_yaw = torch.zeros(num_worlds, n_waypoints, device=device)
        self.wp_idx = torch.zeros(num_worlds, dtype=torch.long, device=device)

        # Traiettorie (mondi traiettoria): tempo e parametri della curva. Vedi _sample_traj.
        self.is_traj = torch.zeros(num_worlds, dtype=torch.bool, device=device)
        self.t = torch.zeros(num_worlds, device=device)
        z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
        self.tr = {
            "kind": torch.zeros(num_worlds, dtype=torch.long, device=device),
            "center": z(num_worlds, 3),
            "amp": z(num_worlds, 3),
            "omega": z(num_worlds, 3),
            "phase": z(num_worlds, 3),
            "psi": z(num_worlds),  # rotazione della curva nel piano xy
            "yaw_mode": torch.zeros(num_worlds, dtype=torch.long, device=device),  # 0 fisso, 1 nel verso di marcia, 2 ruota
            "yaw0": z(num_worlds),
            "yaw_rate": z(num_worlds),
        }

        # Riferimento attivo (nel sistema del mondo)
        self.target_pos = torch.zeros(num_worlds, 3, device=device)  # (N, 3) x, y, z [m]
        self.target_yaw = torch.zeros(num_worlds, device=device)  # (N,) yaw [rad]
        self.target_vel = torch.zeros(num_worlds, 3, device=device)  # (N, 3) velocità del riferimento [m/s]

        self.sample(torch.arange(num_worlds, device=device))

        # Riferimento che era attivo DURANTE l'ultimo passo. Serve al costo: a fine passo il riferimento
        # di un waypoint può già essere avanzato, ma l'errore di quel passo va calcolato con il vecchio.
        # Per le traiettorie è invece il punto della curva all'istante dello stato appena raggiunto.
        self.step_target_pos = self.target_pos
        self.step_target_yaw = self.target_yaw
        self.step_target_vel = self.target_vel

    @property
    def is_last(self) -> torch.Tensor:
        """(N,) bool: il riferimento attivo è l'ultimo della coda (le traiettorie non avanzano: sempre True)."""
        return (self.wp_idx >= self.n_waypoints - 1) | self.is_traj

    def _uniform(self, low, high, shape) -> torch.Tensor:
        return low + (high - low) * torch.rand(shape, device=self.device, generator=self._gen)

    # ------------------------------------------------------------------ curve
    def _sample_traj(self, k: int, kind=None, speed_range=None) -> dict:
        """Parametri di `k` traiettorie. Posizione nel piano locale (poi ruotato di psi):

            x' = ax sin(wx t + px),  y' = ay sin(wy t + py),  z = cz + az sin(wz t + pz)

        retta: ay = 0; cerchio: ax = ay, py = px + 90°; ellisse: ax != ay; otto: wy = 2 wx; elica: cerchio con z lenta.
        Il picco della norma della velocità vale `speed` (campionata in speed_range); la curva sta dentro xy_range e z_range.
        """
        u = self._uniform
        two_pi = 2.0 * math.pi
        if kind is None:
            kind = torch.randint(0, 5, (k,), device=self.device, generator=self._gen)
        else:
            kind = torch.as_tensor(kind, device=self.device).expand(k).clone()
        is_line, is_eight, is_helix = kind == LINE, kind == EIGHT, kind == HELIX

        r1, r2 = u(0.3, 0.9, (k,)), u(0.3, 0.9, (k,))
        ax = torch.where(is_line, u(0.4, 1.2, (k,)), r1)
        ay = torch.where(is_line, torch.zeros_like(r1), torch.where((kind == ELLIPSE) | is_eight, r2, r1))
        az = torch.where(is_helix, u(0.15, 0.4, (k,)), torch.zeros_like(r1))
        amp = torch.stack([ax, ay, az], dim=-1)

        ratio = torch.stack(  # rapporto tra le frequenze degli assi (x : y : z)
            [torch.ones_like(r1), torch.where(is_eight, 2.0 * torch.ones_like(r1), torch.ones_like(r1)), torch.where(is_helix, 0.25 * torch.ones_like(r1), torch.zeros_like(r1))],
            dim=-1,
        )
        lo, hi = speed_range or self.traj_speed_range
        speed = u(lo, hi, (k,))
        sign = torch.where(torch.rand(k, device=self.device, generator=self._gen) < 0.5, -1.0, 1.0)
        # Frequenza base tale che il picco della NORMA della velocità sia `speed` (limite superiore esatto per ogni tipo)
        axy = torch.maximum(ax, ay)
        peak = torch.where(is_eight, torch.sqrt(ax**2 + (2.0 * ay) ** 2), torch.sqrt(axy**2 + (az * ratio[:, 2]) ** 2))
        base = speed / peak.clamp(min=1e-6)
        omega = sign[:, None] * base[:, None] * ratio

        px = u(0.0, two_pi, (k,))
        py = torch.where(is_eight, 2.0 * px, px + math.pi / 2)
        phase = torch.stack([px, py, u(0.0, two_pi, (k,))], dim=-1)

        half = self.xy_range[1]
        reach = torch.hypot(ax, ay)  # raggio della circonferenza che contiene la curva
        cxy = u(-1.0, 1.0, (k, 2)) * (half - reach).clamp(min=0.0)[:, None]
        cz = u(self.z_range[0], self.z_range[1], (k,))
        cz = torch.minimum(torch.maximum(cz, self.z_range[0] + az), self.z_range[1] - az)
        center = torch.cat([cxy, cz[:, None]], dim=-1)

        # Yaw: fisso, nel verso di marcia (solo se la velocità non si annulla mai: non per la retta) o in rotazione lenta
        draw = u(0.0, 1.0, (k,))
        yaw_mode = torch.where(draw < 0.35, torch.ones_like(kind), torch.where(draw < 0.65, 2 * torch.ones_like(kind), torch.zeros_like(kind)))
        yaw_mode = torch.where((yaw_mode == 1) & is_line, torch.zeros_like(yaw_mode), yaw_mode)
        yaw_rate = torch.where(torch.rand(k, device=self.device, generator=self._gen) < 0.5, -1.0, 1.0) * u(0.1, 0.4, (k,))
        return {
            "kind": kind,
            "center": center,
            "amp": amp,
            "omega": omega,
            "phase": phase,
            "psi": u(0.0, two_pi, (k,)),
            "yaw_mode": yaw_mode,
            "yaw0": u(-math.pi, math.pi, (k,)),
            "yaw_rate": yaw_rate,
        }

    @torch.no_grad()
    def _curve(self, t: torch.Tensor):
        """Posizione (N,3), velocità (N,3) e yaw (N,) della curva di ogni mondo all'istante t (N,)."""
        p = self.tr
        th = p["omega"] * t[:, None] + p["phase"]
        s, c = torch.sin(th), torch.cos(th)
        lx, ly = p["amp"][:, 0] * s[:, 0], p["amp"][:, 1] * s[:, 1]
        vlx = p["amp"][:, 0] * p["omega"][:, 0] * c[:, 0]
        vly = p["amp"][:, 1] * p["omega"][:, 1] * c[:, 1]
        cp, sp = torch.cos(p["psi"]), torch.sin(p["psi"])
        pos = torch.stack(
            [p["center"][:, 0] + cp * lx - sp * ly, p["center"][:, 1] + sp * lx + cp * ly, p["center"][:, 2] + p["amp"][:, 2] * s[:, 2]],
            dim=-1,
        )
        vel = torch.stack([cp * vlx - sp * vly, sp * vlx + cp * vly, p["amp"][:, 2] * p["omega"][:, 2] * c[:, 2]], dim=-1)
        heading = torch.atan2(vel[:, 1], vel[:, 0])
        yaw = torch.where(
            p["yaw_mode"] == 1, heading, torch.where(p["yaw_mode"] == 2, _wrap_to_pi(p["yaw0"] + p["yaw_rate"] * t), p["yaw0"])
        )
        return pos, vel, yaw

    # ------------------------------------------------------------------ riferimento attivo
    def _refresh(self) -> None:
        """Il riferimento attivo è l'elemento wp_idx della coda (waypoint) o il punto della curva all'istante t
        (traiettoria). Nuovi tensori: mai in place, vedi sample."""
        ar = torch.arange(self.n, device=self.device)
        tp, tv, ty = self._curve(self.t)
        m = self.is_traj
        self.target_pos = torch.where(m[:, None], tp, self.wp_pos[ar, self.wp_idx])
        self.target_yaw = torch.where(m, ty, self.wp_yaw[ar, self.wp_idx])
        self.target_vel = torch.where(m[:, None], tv, torch.zeros_like(tv))

    def sample(self, ids: torch.Tensor) -> None:
        """Nuovi riferimenti (x, y, z, yaw) per i mondi `ids`: coda di waypoint, oppure traiettoria con probabilità
        `traj_prob`; il primo riferimento diventa quello attivo.

        Si creano nuovi tensori (non si modifica in place): quelli vecchi possono essere già
        stati usati nel grafo di autograd dai passi precedenti della finestra.
        """
        k, m = ids.numel(), self.n_waypoints
        new_pos = torch.stack(
            [
                self._uniform(*self.xy_range, (k, m)),
                self._uniform(*self.xy_range, (k, m)),
                self._uniform(*self.z_range, (k, m)),
            ],
            dim=-1,
        )
        new_yaw = self._uniform(-math.pi, math.pi, (k, m))
        wp_pos, wp_yaw, wp_idx = self.wp_pos.clone(), self.wp_yaw.clone(), self.wp_idx.clone()
        wp_pos[ids], wp_yaw[ids], wp_idx[ids] = new_pos, new_yaw, 0
        self.wp_pos, self.wp_yaw, self.wp_idx = wp_pos, wp_yaw, wp_idx
        self._set_traj(ids, self._uniform(0.0, 1.0, (k,)) < self.traj_prob, self._sample_traj(k))
        self._refresh()

    def _set_traj(self, ids: torch.Tensor, is_traj: torch.Tensor, params: dict) -> None:
        """Scrive i parametri delle traiettorie per i mondi `ids` e azzera il loro tempo (tensori nuovi)."""
        new = {}
        for name, value in self.tr.items():
            v = value.clone()
            v[ids] = params[name]
            new[name] = v
        self.tr = new
        it, t = self.is_traj.clone(), self.t.clone()
        it[ids], t[ids] = is_traj, 0.0
        self.is_traj, self.t = it, t

    def set_constant(self, pos, yaw) -> None:
        """Imposta un riferimento FISSO (x, y, z, yaw) in tutta la coda di ogni mondo (valutazione con riferimento
        scelto a mano). pos: (3,) uguale per tutti oppure (N, 3); yaw: scalare [rad] oppure (N,). Il riferimento non
        avanza né scade; un reset successivo ricampiona a caso quel mondo."""
        pos = torch.as_tensor(pos, dtype=torch.float32, device=self.device).expand(self.n, 3)
        yaw = torch.as_tensor(yaw, dtype=torch.float32, device=self.device).expand(self.n)
        self.wp_pos = pos[:, None, :].repeat(1, self.n_waypoints, 1)
        self.wp_yaw = yaw[:, None].repeat(1, self.n_waypoints)
        self.wp_idx = torch.zeros_like(self.wp_idx)
        self.is_traj = torch.zeros_like(self.is_traj)
        self._refresh()
        self.step_target_pos, self.step_target_yaw, self.step_target_vel = self.target_pos, self.target_yaw, self.target_vel

    def set_trajectory(self, kind, speed: float | None = None) -> None:
        """Tutti i mondi seguono una traiettoria del tipo `kind` (nome di KIND_NAMES o intero), con picco di velocità
        `speed` [m/s] (default: casuale in traj_speed_range). Valutazione con traiettoria scelta a mano."""
        kind = KIND_NAMES[kind] if isinstance(kind, str) else int(kind)
        ids = torch.arange(self.n, device=self.device)
        params = self._sample_traj(self.n, kind=kind, speed_range=None if speed is None else (speed, speed))
        self._set_traj(ids, torch.ones(self.n, dtype=torch.bool, device=self.device), params)
        self._refresh()
        self.step_target_pos, self.step_target_yaw, self.step_target_vel = self.target_pos, self.target_yaw, self.target_vel

    def reset(self, ids: torch.Tensor) -> None:
        """Nuovi riferimenti per i mondi appena resettati."""
        self.sample(ids)
        self.step_target_pos, self.step_target_yaw, self.step_target_vel = self.target_pos, self.target_yaw, self.target_vel

    def step(self) -> None:
        """Da chiamare dopo ogni passo di simulazione. Waypoint: memorizza il riferimento attivo durante il passo.
        Traiettoria: avanza il tempo di `dt`; il riferimento per il costo è il punto della curva all'istante nuovo."""
        old = (self.target_pos, self.target_yaw, self.target_vel)
        self.t = self.t + self.dt
        self._refresh()
        m = self.is_traj
        self.step_target_pos = torch.where(m[:, None], self.target_pos, old[0])
        self.step_target_yaw = torch.where(m, self.target_yaw, old[1])
        self.step_target_vel = torch.where(m[:, None], self.target_vel, old[2])

    def advance(self, ids: torch.Tensor) -> None:
        """Passa al riferimento successivo nei mondi `ids` (che non siano già sull'ultimo)."""
        wp_idx = self.wp_idx.clone()
        wp_idx[ids] = torch.clamp(wp_idx[ids] + 1, max=self.n_waypoints - 1)
        self.wp_idx = wp_idx
        self._refresh()

    # ------------------------------------------------------------------ finestra di anteprima
    @torch.no_grad()
    def window(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Anteprima: incrementi tra punti consecutivi a t, t+Δ, ..., t+(H-1)Δ. Ritorna (dpos (N, H-1, 3) nel mondo
        [m], dyaw (N, H-1) [rad]). Nulli per i mondi waypoint (bersaglio fermo)."""
        h = self.window_points - 1
        if h <= 0:
            return torch.zeros(self.n, 0, 3, device=self.device), torch.zeros(self.n, 0, device=self.device)
        pts, yaws = [], []
        for k in range(h + 1):
            p, _, y = self._curve(self.t + k * self.window_dt)
            pts.append(p)
            yaws.append(y)
        dpos = torch.stack(pts[1:], dim=1) - torch.stack(pts[:-1], dim=1)
        dyaw = _wrap_to_pi(torch.stack(yaws[1:], dim=1) - torch.stack(yaws[:-1], dim=1))
        m = self.is_traj
        return dpos * m[:, None, None], dyaw * m[:, None]
