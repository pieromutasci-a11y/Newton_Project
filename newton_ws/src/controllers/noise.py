"""Rumore bianco sul FEEDBACK (stato misurato), comune ai controllori di posizione e di velocità."""

from dataclasses import dataclass

import torch

from .math_utils import quat_mul


@dataclass
class NoiseConfig:
    """Rumore bianco gaussiano sul feedback, indipendente a ogni lettura. Tutto a zero = misure esatte.

    Si applica solo a ciò che la rete vede: il costo, le terminazioni e i grafici usano lo stato vero.
    """

    pos: float = 0.0  # m, deviazione standard sulla posizione (x, y, z)
    att: float = 0.0  # rad, deviazione standard della rotazione d'assetto (roll, pitch e yaw, vettore di rotazione)
    lin_vel: float = 0.0  # m/s, sulla velocità lineare (nel mondo; il rumore è isotropo)
    ang_vel: float = 0.0  # rad/s, sulla velocità angolare
    to_vel_net: bool = True  # (solo cascata) se False la rete di velocità legge lo stato vero

    def any(self) -> bool:
        return self.pos > 0 or self.att > 0 or self.lin_vel > 0 or self.ang_vel > 0


def measure_state(q: torch.Tensor, qd: torch.Tensor, noise: NoiseConfig, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Stato MISURATO (q (N,7), qd (N,6)): quello vero più rumore bianco. Una nuova estrazione a ogni chiamata.
    Il rumore è additivo (il gradiente passa invariato); sull'assetto è una piccola rotazione casuale."""
    if not noise.any():
        return q, qd
    n, dev = q.shape[0], q.device
    randn = lambda: torch.randn(n, 3, device=dev, generator=gen)  # noqa: E731
    pos, quat = q[:, :3] + noise.pos * randn(), q[:, 3:7]
    if noise.att > 0:
        dq = torch.cat([0.5 * noise.att * randn(), torch.ones(n, 1, device=dev)], dim=-1)
        quat = quat_mul(quat, dq / dq.norm(dim=-1, keepdim=True))
    lin, ang = qd[:, :3] + noise.lin_vel * randn(), qd[:, 3:6] + noise.ang_vel * randn()
    return torch.cat([pos, quat], dim=-1), torch.cat([lin, ang], dim=-1)
