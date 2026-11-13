"""Parametri fisici del drone (DJI Tello). Tutte le grandezze sono in unità SI.

I valori di massa, inerzia, thrust-to-weight e scala dei momenti sono gli stessi
usati nel progetto Isaac Lab (drone_physics.py e *_env_cfg.py), così i risultati
sono confrontabili.
"""

from dataclasses import dataclass


@dataclass
class DroneParams:
    # --- Dinamica del corpo rigido (valori del Tello, come in Isaac Lab) ---
    mass: float = 0.087  # kg
    # Tensore d'inerzia diagonale rispetto al centro di massa [kg*m^2]
    ixx: float = 1.04e-4
    iyy: float = 1.11e-4
    izz: float = 2.07e-4
    gravity: float = 9.81  # m/s^2, serve solo per calcolare il peso

    # --- Geometria di collisione ---
    # SEMI-estensioni (half-extent) di una scatola che approssima il corpo.
    # Valori indicativi, da affinare: servono per le collisioni tra droni, non alla dinamica
    # (massa e inerzia sono già fissate sopra e non dipendono da queste dimensioni).
    half_x: float = 0.05  # m
    half_y: float = 0.05  # m
    half_z: float = 0.02  # m

    # --- Conversione azioni normalizzate [-1, 1] -> fisica (come in Isaac Lab) ---
    thrust_to_weight: float = 1.8  # spinta massima / peso
    # Scala dei momenti (roll, pitch, yaw) corrispondente ad azione = 1 [N*m]
    moment_scale_x: float = 0.021
    moment_scale_y: float = 0.021
    moment_scale_z: float = 0.0085

    @property
    def weight(self) -> float:
        """Peso del drone [N]."""
        return self.mass * self.gravity

    @property
    def max_thrust(self) -> float:
        """Spinta massima [N] (azione di spinta = +1). La minima è 0 (azione = -1)."""
        return self.thrust_to_weight * self.weight
