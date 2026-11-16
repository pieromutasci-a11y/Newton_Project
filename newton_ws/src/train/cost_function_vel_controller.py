"""Funzione costo del controllore di velocità, ispirata alla reward di Isaac Lab (vel_controller_env._get_rewards).

Il costo è il contrario di una reward: più è basso, meglio è. Per il differentiable learning ogni termine deve essere
DERIVABILE (anche in zero), perché il gradiente attraversa la fisica fino ai pesi:
  - niente penalità a gradino (la "morte" di -10 di Isaac Lab per tilt eccessivo): ha gradiente nullo quasi ovunque.
    Le terminazioni restano nell'env (reset); al suo posto un termine di tilt liscio e più forte;
  - niente `acos`: la sua derivata è infinita a angolo zero e, moltiplicata per lo zero di un relu, darebbe NaN.
    Si usa 1 - cos(tilt) = 1 + (gravità nel corpo)_z;
  - nessuna norma o valore assoluto: tutti i termini sono quadratici.

Termini (stessi di Isaac Lab):
  - errore di velocità, normalizzato asse per asse sul campo dei riferimenti (vx, vy, vz, wz): è il compito.
    Due termini con pesi indipendenti: lineare (vx, vy, vz) e angolare (wz);
  - variazione delle azioni, con peso CRESCENTE durante il training (Isaac Lab: curriculum 0.01 -> 0.1 in 40000 passi):
    prima la rete impara a seguire il riferimento, poi a farlo con dolcezza;
  - velocità angolare di roll e pitch non voluta;
  - tilt oltre una soglia libera (35° come in Isaac Lab).
In più, rispetto a Isaac Lab: un piccolo termine `upright` e un tilt più pesante (vedi CostWeights).

Gli errori arrivano da env.errors() (una sola implementazione, la stessa delle osservazioni).
I pesi sono Isaac Lab divisi per il peso dell'errore di velocità (7), come per il controllore di posizione:
sono valori di PARTENZA da tarare.
"""

import math
from dataclasses import dataclass

import torch

from controllers.vel_controller.vel_controller_env import VelErrors


@dataclass
class CostWeights:
    # --- Compito: seguire il riferimento di velocità, con un termine per la parte lineare e uno per quella angolare ---
    vel_error_lin: float = 3.0  # peso di sum_{vx,vy,vz} ((v - v_rif) / campo)^2; Isaac Lab: 7 per tutti gli assi (qui diviso per 7 = 1.0, poi alzato)
    vel_error_ang: float = 1.5  # peso di ((wz - wz_rif) / campo)^2 (velocità di yaw)

    # --- Azioni: variazione tra due passi, peso crescente (curriculum di Isaac Lab) ---
    action_rate_start: float = 0.5  # Isaac Lab: 0.01 (0.0014 = 0.01/7 era troppo poco: gli impulsi dei comandi al cambio di riferimento costavano ~niente)
    action_rate_end: float = 0.9  # Isaac Lab: 0.1 (idem: prima 0.014)
    action_rate_ramp_iterations: int = 1500  # iterazioni di training per passare da start a end (Isaac Lab: 40000 passi)

    # --- Assetto e rotazioni ---
    ang_vel_xy: float = 0.03  # roll/pitch^2; Isaac Lab: 0.05 / 7 = 0.007 (circa il doppio: picchi di 6-9 rad/s al cambio di riferimento)
    upright: float = 0.05  # peso di 1 - cos(tilt): tiene il drone dritto (non c'è in Isaac Lab)
    tilt: float = 5.0  # penalità crescente OLTRE tilt_free_deg (Isaac Lab: 2/7 sull'angolo in rad, ma con la "morte" -10)
    tilt_free_deg: float = 25.0  # inclinazione senza penalità aggiuntiva (Isaac Lab: 35; abbassata per ridurre le inclinazioni al cambio di riferimento)


def action_rate_weight(w: CostWeights, iteration: int) -> float:
    """Peso della variazione delle azioni all'iterazione `iteration`: sale linearmente da start a end."""
    progress = min(1.0, iteration / max(1, w.action_rate_ramp_iterations))
    return w.action_rate_start + progress * (w.action_rate_end - w.action_rate_start)


def cost_function(
    errors: VelErrors,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    vel_range: torch.Tensor,
    w: CostWeights | None = None,
    action_rate_scale: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Costo di un passo per ogni mondo.

    errors            : env.errors() dopo il passo
    actions           : (N, 4) azioni appena applicate
    prev_actions      : (N, 4) azioni del passo precedente (env.prev_actions letto PRIMA di env.step)
    vel_range         : (4,) campo dei riferimenti (env.refs.vel_range), per normalizzare gli errori
    action_rate_scale : peso della variazione delle azioni a questa iterazione (action_rate_weight);
                        se None si usa il valore finale (action_rate_end)
    Ritorna (costo totale (N,), dizionario dei singoli termini (N,) per i log).
    """
    w = w or CostWeights()
    e = errors
    action_rate_w = w.action_rate_end if action_rate_scale is None else action_rate_scale

    err_norm = e.vel_err / vel_range  # (N, 4): ogni asse in unità del proprio campo
    one_minus_cos_tilt = 1.0 + e.gravity_b[:, 2]  # 0 se dritto (gravity_b z = -1), 1 se orizzontale, 2 se capovolto
    tilt_excess = torch.relu(one_minus_cos_tilt - (1.0 - math.cos(math.radians(w.tilt_free_deg))))

    terms = {
        "vel_error_lin": w.vel_error_lin * err_norm[:, :3].pow(2).sum(dim=-1),
        "vel_error_ang": w.vel_error_ang * err_norm[:, 3].pow(2),
        "action_rate": action_rate_w * (actions - prev_actions).pow(2).sum(dim=-1),
        "ang_vel_xy": w.ang_vel_xy * e.ang_vel_b[:, :2].pow(2).sum(dim=-1),
        "upright": w.upright * one_minus_cos_tilt,
        "tilt": w.tilt * tilt_excess.pow(2),
    }
    total = torch.stack(list(terms.values())).sum(dim=0)
    return total, terms
