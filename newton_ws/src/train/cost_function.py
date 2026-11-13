"""Funzione costo del controllore di posizione, ispirata alla reward di Isaac Lab (pos_controller_env.py).

Il costo è il contrario di una reward: più è basso, meglio è. Per il differentiable learning ogni
termine deve essere DERIVABILE (anche in zero), perché il gradiente attraversa la fisica fino ai pesi:
  - niente penalità a gradino (come i -50 di "morte" per tilt o fuori dai limiti) né bonus "target raggiunto":
    hanno gradiente nullo quasi ovunque. Le terminazioni restano nell'env (reset), non nel costo;
  - niente norme, valori assoluti o acos che in zero hanno derivata infinita o NaN (si usa una radice
    "smussata" e 1 - cos(tilt) al posto dell'angolo);
  - niente "alive bonus": è una costante, il suo gradiente è zero.

Gli errori arrivano da env.errors() (una sola implementazione, la stessa delle osservazioni).
I pesi sono valori di PARTENZA da tarare: i rapporti tra i termini di posizione e yaw sono quelli di
Isaac Lab (25 : 35 : 8 : 8), gli altri sono stime.
"""

import math
from dataclasses import dataclass

import torch

from controllers.pos_controller.pos_controller_env import Errors
from drone_model.params import DroneParams


@dataclass
class CostWeights:
    # --- Posizione: termine ad ampio raggio + termine di precisione (come in Isaac Lab) ---
    pos_approach: float = 1.0  # peso di 1 - exp(-beta * distanza); Isaac Lab: 25
    pos_approach_beta: float = 0.5
    pos_prec: float = 1.4  # Isaac Lab: 35 (rapporto 35/25)
    pos_prec_beta: float = 8.0

    # --- Yaw: errore e precisione ---
    yaw: float = 0.32  # peso di |errore di yaw| [rad]; Isaac Lab: -8 (rapporto 8/25)
    yaw_prec: float = 0.32  # peso di 1 - exp(-beta * |errore|); Isaac Lab: 8
    yaw_prec_beta: float = 7.0

    # --- Velocità ---
    ang_vel_xy: float = 0.01  # roll/pitch^2; Isaac Lab: 0, ma qui si comandano i momenti, quindi serve (0.002 -> 0.01: meno tremolio)
    ang_vel_z: float = 0.002  # yaw^2, più pesante vicino al target (come in Isaac Lab)
    lin_vel: float = 0.1  # smorzamento generale / limite di velocità (0.01 -> 0.1: a 3 m/s costava solo 0.1 per passo)
    lin_vel_brake: float = 0.5  # frenata vicino al target (in Isaac Lab solo per la modalità uniciclo)
    brake_dist: float = 0.5  # m, entro questa distanza la frenata si attiva (come in Isaac Lab)

    # --- Assetto ---
    upright: float = 0.5  # peso di 1 - cos(tilt): tiene il drone dritto (0.1 -> 0.5)
    tilt: float = 10.0  # penalità crescente OLTRE tilt_free_deg (sostituisce la "morte" a gradino)
    tilt_free_deg: float = 25.0  # inclinazione senza penalità aggiuntiva (Isaac Lab, controllore di velocità: 35)

    # --- Azioni ---
    action_rate: float = 0.1  # (azione - azione precedente)^2: azioni lisce; Isaac Lab: -25 (in proporzione ai pesi di posizione sarebbe 1.0)
    effort: float = 0.01  # distanza dall'azione di hover: penalizza i comandi aggressivi; Isaac Lab: -8


def make_hover_action(params: DroneParams, device) -> torch.Tensor:
    """Azione di hover: spinta pari al peso, momenti nulli. Con thrust_to_weight = 1.8 è (0.111, 0, 0, 0)."""
    return torch.tensor([2.0 / params.thrust_to_weight - 1.0, 0.0, 0.0, 0.0], device=device)


def _smooth_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Norma euclidea (N, k) -> (N,), derivabile anche in zero (la norma vera ha derivata NaN in zero)."""
    return torch.sqrt(x.pow(2).sum(dim=-1) + eps * eps)


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def cost_function(
    errors: Errors,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    hover_action: torch.Tensor,
    w: CostWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Costo di un passo per ogni mondo.

    errors       : env.errors() dopo il passo
    actions      : (N, 4) azioni appena applicate
    prev_actions : (N, 4) azioni del passo precedente (env.prev_actions letto PRIMA di env.step)
    hover_action : (4,) vedi make_hover_action
    Ritorna (costo totale (N,), dizionario dei singoli termini (N,) per i log).
    """
    w = w or CostWeights()
    e = errors

    dist = _smooth_norm(e.pos_err)  # distanza dal bersaglio [m]
    yaw_abs = torch.sqrt(_wrap_to_pi(e.yaw_err).pow(2) + 1e-12)  # |errore di yaw| in [0, pi]
    lin_vel_sq = e.lin_vel_b.pow(2).sum(dim=-1)

    one_minus_cos_tilt = 1.0 + e.gravity_b[:, 2]  # 0 se dritto (gravity_b z = -1), 1 se orizzontale
    tilt_excess = torch.relu(one_minus_cos_tilt - (1.0 - math.cos(math.radians(w.tilt_free_deg))))

    # Frenata: 1 sul bersaglio, 0 oltre brake_dist
    brake_weight = torch.clamp(1.0 - dist / w.brake_dist, min=0.0, max=1.0)
    # La velocità angolare di yaw pesa di più vicino al bersaglio (come in Isaac Lab)
    wz_proximity = 1.0 + 9.0 * torch.exp(-2.0 * dist)

    terms = {
        "pos_approach": w.pos_approach * (1.0 - torch.exp(-w.pos_approach_beta * dist)),
        "pos_prec": w.pos_prec * (1.0 - torch.exp(-w.pos_prec_beta * dist)),
        "yaw": w.yaw * yaw_abs,
        "yaw_prec": w.yaw_prec * (1.0 - torch.exp(-w.yaw_prec_beta * yaw_abs)),
        "ang_vel_xy": w.ang_vel_xy * e.ang_vel_b[:, :2].pow(2).sum(dim=-1),
        "ang_vel_z": w.ang_vel_z * wz_proximity * e.ang_vel_b[:, 2].pow(2),
        "lin_vel": w.lin_vel * lin_vel_sq,
        "lin_vel_brake": w.lin_vel_brake * brake_weight * lin_vel_sq,
        "upright": w.upright * one_minus_cos_tilt,
        "tilt": w.tilt * tilt_excess.pow(2),
        "action_rate": w.action_rate * (actions - prev_actions).pow(2).sum(dim=-1),
        "effort": w.effort * (actions - hover_action).pow(2).sum(dim=-1),
    }
    total = torch.stack(list(terms.values())).sum(dim=0)
    return total, terms
