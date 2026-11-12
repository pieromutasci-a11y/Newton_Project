"""Attuatori del drone (propeller).

Per ora contiene solo la struttura dati letta dai kernel.
Il kernel che calcola forze e coppie verrà aggiunto qui.
"""

import warp as wp


@wp.struct
class Propeller:
    body: int                  # indice del corpo a cui è attaccato (nel Model)
    pos: wp.vec3               # posizione nel sistema di riferimento del corpo [m]
    dir: wp.vec3               # direzione della spinta nel sistema del corpo
    max_thrust: float          # spinta massima [N]
    max_torque: float          # coppia di reazione massima [N*m]
    turning_direction: float   # +1 / -1: verso di rotazione
