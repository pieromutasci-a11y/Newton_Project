"""Mondo per il training: N droni in N mondi paralleli, differenziabile, con interfaccia PyTorch.

Regola di progetto: Warp resta DENTRO questa classe. Da fuori (env, training) si vedono solo
tensori PyTorch. Qui NON ci sono bersagli, costo né rete: è solo la fisica.

Differenziabilità (backpropagation attraverso la fisica):
  - ogni passo di simulazione è una torch.autograd.Function (_SimStep). Nel forward registra
    i kernel di Warp in un wp.Tape; nel backward ripercorre quel tape all'indietro.
  - la simulazione procede per FINESTRE di `horizon` passi di CONTROLLO (BPTT troncata): per ogni passo
    di fisica si conserva uno stato diverso (serve al backward). Alla fine della finestra si chiama
    detach(), che stacca lo stato dal grafo e riparte dall'indice 0.

Frequenze, come in Isaac Lab: la fisica gira a 1/dt = 100 Hz; ogni passo di controllo = `decimation`
passi di fisica (4 -> controllo a 25 Hz) con la stessa azione tenuta costante (zero-order hold).

Senza pavimento né contatti (come nel progetto Isaac Lab): dinamica liscia, quindi gradienti
affidabili. Solver: SolverSemiImplicit (differenziabile, come nell'esempio diffsim_drone).
"""

from dataclasses import dataclass

import newton
import torch
import warp as wp

from drone_model.actuators import launch_apply_wrench
from drone_model.builder import add_drone
from drone_model.params import DroneParams


@dataclass
class TrainWorldConfig:
    num_worlds: int = 1024  # droni (uno per mondo) simulati in parallelo
    dt: float = 0.01  # s, passo di FISICA (100 Hz, come in Isaac Lab)
    decimation: int = 4  # passi di fisica per ogni passo di controllo (4 -> rete a 25 Hz, come in Isaac Lab)
    horizon: int = 32  # passi di CONTROLLO per finestra di backpropagation (troncamento)
    visual: bool = False  # aspetto da quadricottero (solo per la visualizzazione; nel training resta False)


class _SimStep(torch.autograd.Function):
    """Un passo di simulazione come operazione di PyTorch, con backward calcolato da Warp."""

    @staticmethod
    def forward(ctx, world, q_in, qd_in, actions):
        # q_in e qd_in non vengono letti: servono solo a collegare questo passo al precedente
        # nel grafo di autograd (i dati sono già negli stati di Warp, scritti dal passo prima).
        t, d = world.tick, world.cfg.decimation
        a_wp = world._actions[t]

        # Le azioni di PyTorch vengono copiate nel buffer di Warp di questo passo di controllo
        wp.to_torch(a_wp, requires_grad=False).copy_(actions)

        # `decimation` passi di fisica con la stessa azione (zero-order hold), in un unico tape
        tape = wp.Tape()
        with tape:
            for k in range(d):
                s_in, s_out = world._states[t * d + k], world._states[t * d + k + 1]
                s_in.clear_forces()
                launch_apply_wrench(a_wp, world._body_ids, s_in.body_q, s_in.body_f, world.params)
                world._solver.step(s_in, s_out, world._control, None, world.sim_dt)

        ctx.world, ctx.tape, ctx.t = world, tape, t
        # Lo stato di uscita è una vista (senza copia) della memoria di Warp
        s_last = world._states[(t + 1) * d]
        return (
            wp.to_torch(s_last.body_q, requires_grad=False),
            wp.to_torch(s_last.body_qd, requires_grad=False),
        )

    @staticmethod
    def backward(ctx, grad_q_out, grad_qd_out):
        world, tape, t = ctx.world, ctx.tape, ctx.t
        d = world.cfg.decimation
        chunk = world._states[t * d : (t + 1) * d + 1]  # stati toccati da questo passo di controllo
        s_first, s_last = chunk[0], chunk[-1]
        a_wp = world._actions[t]

        # 1. I gradienti che arrivano da PyTorch diventano gli "aggiunti" dello stato di uscita
        zeros_q = torch.zeros_like(wp.to_torch(s_last.body_q, requires_grad=False))
        zeros_qd = torch.zeros_like(wp.to_torch(s_last.body_qd, requires_grad=False))
        wp.to_torch(s_last.body_q.grad, requires_grad=False).copy_(zeros_q if grad_q_out is None else grad_q_out)
        wp.to_torch(s_last.body_qd.grad, requires_grad=False).copy_(zeros_qd if grad_qd_out is None else grad_qd_out)

        # 2. Backward di Warp: ripercorre al contrario i kernel di tutti i passi di fisica
        tape.backward()

        # 3. Si leggono i gradienti rispetto a stato di ingresso e azioni (copiandoli: i buffer vengono azzerati).
        #    Il gradiente delle azioni è la somma sui `decimation` passi in cui l'azione è stata tenuta.
        grad_q_in = wp.to_torch(s_first.body_q.grad, requires_grad=False).clone()
        grad_qd_in = wp.to_torch(s_first.body_qd.grad, requires_grad=False).clone()
        grad_actions = wp.to_torch(a_wp.grad, requires_grad=False).clone()

        # 4. Azzeramento degli aggiunti (anche degli stati intermedi), perché i buffer vengono riusati
        tape.zero()
        for st in chunk:
            st.body_q.grad.zero_()
            st.body_qd.grad.zero_()
        a_wp.grad.zero_()

        # Un valore di ritorno per ogni argomento del forward: (world, q_in, qd_in, actions)
        return None, grad_q_in, grad_qd_in, grad_actions


class TrainWorld:
    def __init__(self, params: DroneParams | None = None, cfg: TrainWorldConfig | None = None):
        self.params = params or DroneParams()
        self.cfg = cfg or TrainWorldConfig()
        self.sim_dt = self.cfg.dt  # passo di fisica
        self.ctrl_dt = self.cfg.dt * self.cfg.decimation  # passo di controllo (intervallo tra due azioni)
        n = self.cfg.num_worlds
        horizon = self.cfg.horizon

        # --- 1. Scena: un mondo con un drone, replicato N volte (mondi separati, senza interazione) ---
        gravity = wp.vec3(0.0, 0.0, -self.params.gravity)
        single = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=gravity)
        self.handle = add_drone(single, self.params, label="drone", position=(0.0, 0.0, 0.0), visual=self.cfg.visual)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=gravity)
        builder.replicate(single, copy_count=n)
        if self.cfg.visual:
            builder.add_ground_plane()  # solo grafico: il solver del training non usa contatti

        # requires_grad=True: il modello e gli stati hanno anche gli array dei gradienti
        self.model = builder.finalize(requires_grad=True)
        assert self.model.body_count == n, "ci aspettiamo un corpo per mondo"

        # --- 2. Solver (senza contatti) e buffer ---
        self._solver = newton.solvers.SolverSemiImplicit(self.model)
        self._control = self.model.control()

        # Uno stato per ogni passo di FISICA della finestra (+1 per quello iniziale): servono al backward
        self._states = [self.model.state() for _ in range(horizon * self.cfg.decimation + 1)]
        # Azioni di ogni passo di controllo, con gradiente
        self._actions = [wp.zeros((n, 4), dtype=float, requires_grad=True) for _ in range(horizon)]
        # Il corpo del mondo w è il corpo w (un solo corpo per mondo, copie in ordine)
        self._body_ids = wp.array(list(range(n)), dtype=int)

        # Posizione nella finestra e stato corrente come tensori (viste sulla memoria di Warp)
        self.tick = 0
        self._set_current_to_initial()

    # ------------------------------------------------------------------ stato (tensori torch)
    @property
    def q(self) -> torch.Tensor:
        """(N, 7): posizione (x, y, z) e quaternione (x, y, z, w) di ogni drone, nel sistema del mondo."""
        return self._q

    @property
    def qd(self) -> torch.Tensor:
        """(N, 6): velocità spaziale di ogni drone (vedi test per l'ordine lineare/angolare)."""
        return self._qd

    @property
    def pos(self) -> torch.Tensor:
        return self._q[:, :3]

    # ------------------------------------------------------------------ simulazione
    def step(self, actions: torch.Tensor) -> None:
        """Avanza di un passo di controllo (`decimation` passi di fisica, azione costante). actions: (N, 4) in [-1, 1] = [spinta, momento x, momento y, momento z].

        Differenziabile: il gradiente passa dalle azioni agli stati successivi.
        """
        if self.tick >= self.cfg.horizon:
            raise RuntimeError("Finestra piena: chiamare detach() prima di proseguire.")
        self._q, self._qd = _SimStep.apply(self, self._q, self._qd, actions)
        self.tick += 1

    def detach(self) -> None:
        """Fine finestra (troncamento): lo stato corrente diventa lo stato iniziale, staccato dal grafo."""
        if self.tick > 0:
            last = self.tick * self.cfg.decimation
            wp.copy(self._states[0].body_q, self._states[last].body_q)
            wp.copy(self._states[0].body_qd, self._states[last].body_qd)
        self.tick = 0
        self._set_current_to_initial()

    def reset(self, ids: torch.Tensor, positions: torch.Tensor) -> None:
        """Riporta i droni `ids` alle posizioni date, dritti e fermi. Solo a inizio finestra (tick = 0)."""
        assert self.tick == 0, "reset() solo dopo detach() (a inizio finestra)"
        q0 = wp.to_torch(self._states[0].body_q, requires_grad=False)
        qd0 = wp.to_torch(self._states[0].body_qd, requires_grad=False)
        q0[ids, :3] = positions
        q0[ids, 3:] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=q0.device)  # orientamento identità (x, y, z, w)
        qd0[ids] = 0.0

    # ------------------------------------------------------------------ interni
    def _set_current_to_initial(self) -> None:
        """Lo stato corrente è la vista sullo stato 0 (foglia del grafo, senza gradiente)."""
        self._q = wp.to_torch(self._states[0].body_q, requires_grad=False)
        self._qd = wp.to_torch(self._states[0].body_qd, requires_grad=False)
