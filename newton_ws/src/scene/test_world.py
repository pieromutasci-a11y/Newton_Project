"""Ambiente di simulazione: un mondo fisico con pavimento, gravità e un drone.

Per ora un solo drone in un solo mondo. Il multi-drone e i mondi paralleli si aggiungono
dopo (builder.replicate), una volta verificata la fisica di base.

Qui NON ci sono bersagli, reward né policy: World è solo la fisica.
"""

from dataclasses import dataclass

import newton
import numpy as np
import warp as wp

from drone_model.actuators import launch_apply_wrench
from drone_model.builder import add_drone
from drone_model.params import DroneParams


@dataclass
class WorldConfig:
    dt: float = 0.01  # s, passo di controllo (100 Hz, come in Isaac Lab)
    substeps: int = 1  # sotto-passi di fisica per ogni dt (più alti = contatti più stabili)
    ground: bool = True  # pavimento
    start_position: tuple[float, float, float] = (0.0, 0.0, 1.0)  # m, posizione iniziale del drone
    xpbd_iterations: int = 5  # iterazioni del solver XPBD


class World:
    def __init__(self, params: DroneParams | None = None, cfg: WorldConfig | None = None):
        self.params = params or DroneParams()
        self.cfg = cfg or WorldConfig()
        self.sim_dt = self.cfg.dt / self.cfg.substeps

        # --- 1. Costruzione della scena (CPU, una volta sola) ---
        # Asse z verso l'alto; la gravità è -g lungo z (come negli esempi di Newton)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=wp.vec3(0.0, 0.0, -self.params.gravity))
        if self.cfg.ground:
            builder.add_ground_plane()
        self.handle = add_drone(
            builder,
            self.params,
            label="drone_0",
            position=self.cfg.start_position,
            visual=True,
        )

        # --- 2. finalize: la descrizione diventa un Model con i dati sulla GPU ---
        self.model = builder.finalize()

        # --- 3. Solver, stati e contatti ---
        self.solver = newton.solvers.SolverXPBD(self.model, iterations=self.cfg.xpbd_iterations)
        self.state_0 = self.model.state()  # stato corrente
        self.state_1 = self.model.state()  # stato successivo (doppio buffer)
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        # Copia dello stato iniziale, usata da reset()
        self._initial_state = self.model.state()

        # --- 4. Buffer per l'attuazione ---
        # Indice del corpo del drone e azioni (1 drone x 4 azioni), in memoria GPU
        self.body_ids = wp.array([self.handle.body], dtype=int)
        self.actions = wp.zeros((1, 4), dtype=float)

    def step(self, actions=None) -> None:
        """Avanza di un passo dt.

        actions: 4 numeri in [-1, 1] = [spinta, momento x, momento y, momento z].
        Se None si riusano le azioni dell'ultimo passo.
        """
        if actions is not None:
            self.actions.assign(np.asarray(actions, dtype=np.float32).reshape(1, 4))

        for _ in range(self.cfg.substeps):
            # Azzera le forze del passo precedente, poi applica spinta e momenti al corpo
            self.state_0.clear_forces()
            launch_apply_wrench(
                self.actions, self.body_ids, self.state_0.body_q, self.state_0.body_f, self.params
            )

            # Contatti (pavimento) e integrazione della dinamica
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # Lo stato nuovo diventa quello corrente
            self.state_0, self.state_1 = self.state_1, self.state_0

    def reset(self) -> None:
        """Riporta il drone alla posizione iniziale, fermo."""
        self.state_0.assign(self._initial_state)
        self.actions.zero_()

    # --- Lettura dello stato (copia su CPU: solo per debug e test, non nel training) ---
    def position(self) -> np.ndarray:
        """Posizione (x, y, z) del drone [m]."""
        return self.state_0.body_q.numpy()[self.handle.body, :3]

    def pose(self) -> np.ndarray:
        """Posizione e orientamento: 7 numeri (x, y, z, quaternione)."""
        return self.state_0.body_q.numpy()[self.handle.body]
