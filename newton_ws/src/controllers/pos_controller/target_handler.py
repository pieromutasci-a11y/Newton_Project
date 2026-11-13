"""Generazione dei riferimenti (x, y, z, yaw) nel sistema del mondo, come in Isaac Lab.

Ogni mondo ha una CODA di `n_waypoints` riferimenti casuali, estratta a inizio episodio, e un
riferimento attivo (il primo). Si passa al successivo con advance(), chiamato dall'env quando il drone
ha raggiunto il riferimento e ci è rimasto per il tempo richiesto. Dopo l'ultimo il riferimento resta lo stesso.

Si occupa SOLO dei riferimenti: non conosce il mondo fisico, né i reset, né la rete, né quando si avanza.
"""

import math

import torch


class TargetHandler:
    def __init__(
        self,
        num_worlds: int,
        device,
        *,
        seed: int = 0,
        n_waypoints: int = 3,  # riferimenti per episodio
        xy_range: tuple[float, float] = (-1.5, 1.5),  # m, x e y uniformi in questo intervallo
        z_range: tuple[float, float] = (0.5, 1.5),  # m
    ):
        # il riferimento di yaw è uniforme in [-pi, pi]
        self.n = num_worlds
        self.device = device
        self.n_waypoints = n_waypoints
        self.xy_range = xy_range
        self.z_range = z_range

        # Generatore casuale con seme: esperimenti ripetibili
        self._gen = torch.Generator(device=device)
        self._gen.manual_seed(seed)

        # Code di riferimenti (N, K, ...) e indice del riferimento attivo di ogni mondo
        self.wp_pos = torch.zeros(num_worlds, n_waypoints, 3, device=device)
        self.wp_yaw = torch.zeros(num_worlds, n_waypoints, device=device)
        self.wp_idx = torch.zeros(num_worlds, dtype=torch.long, device=device)

        # Riferimento attivo (nel sistema del mondo)
        self.target_pos = torch.zeros(num_worlds, 3, device=device)  # (N, 3) x, y, z [m]
        self.target_yaw = torch.zeros(num_worlds, device=device)  # (N,) yaw [rad]

        self.sample(torch.arange(num_worlds, device=device))

        # Riferimento che era attivo DURANTE l'ultimo passo. Serve al costo: a fine passo il riferimento
        # può già essere avanzato, ma l'errore di quel passo va calcolato con il vecchio.
        self.step_target_pos = self.target_pos
        self.step_target_yaw = self.target_yaw

    @property
    def is_last(self) -> torch.Tensor:
        """(N,) bool: il riferimento attivo è l'ultimo della coda."""
        return self.wp_idx >= self.n_waypoints - 1

    def _uniform(self, low: float, high: float, shape) -> torch.Tensor:
        return low + (high - low) * torch.rand(shape, device=self.device, generator=self._gen)

    def _refresh(self) -> None:
        """Il riferimento attivo è l'elemento wp_idx della coda (nuovi tensori: mai in place, vedi sample)."""
        ar = torch.arange(self.n, device=self.device)
        self.target_pos = self.wp_pos[ar, self.wp_idx]
        self.target_yaw = self.wp_yaw[ar, self.wp_idx]

    def sample(self, ids: torch.Tensor) -> None:
        """Nuova coda di riferimenti (x, y, z, yaw) per i mondi `ids`; il primo diventa quello attivo.

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
        self._refresh()

    def reset(self, ids: torch.Tensor) -> None:
        """Nuova coda per i mondi appena resettati."""
        self.sample(ids)
        self.step_target_pos, self.step_target_yaw = self.target_pos, self.target_yaw

    def step(self) -> None:
        """Da chiamare dopo ogni passo di simulazione: memorizza il riferimento attivo durante quel passo."""
        self.step_target_pos, self.step_target_yaw = self.target_pos, self.target_yaw

    def advance(self, ids: torch.Tensor) -> None:
        """Passa al riferimento successivo nei mondi `ids` (che non siano già sull'ultimo)."""
        wp_idx = self.wp_idx.clone()
        wp_idx[ids] = torch.clamp(wp_idx[ids] + 1, max=self.n_waypoints - 1)
        self.wp_idx = wp_idx
        self._refresh()
