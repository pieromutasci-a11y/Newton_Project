"""Attuazione del drone: dalle azioni della policy a forza e coppia sul corpo.

Stesso schema di Isaac Lab (_pre_physics_step + _apply_action): la policy produce
4 azioni in [-1, 1] = [spinta, momento x, momento y, momento z], e queste diventano
UNA spinta lungo l'asse z del corpo e UN momento nel sistema del corpo, applicati
direttamente al corpo rigido. Non si modellano i singoli motori.

    spinta  = max_thrust * (a0 + 1) / 2      -> in [0, max_thrust]
    momento = moment_scale * (a1, a2, a3)    -> in [-scale, +scale] per asse
"""

import warp as wp


@wp.kernel
def apply_wrench(
    actions: wp.array2d[float],
    body_ids: wp.array[int],
    max_thrust: float,
    moment_scale: wp.vec3,
    body_q: wp.array[wp.transform],
    body_f: wp.array[wp.spatial_vector],
):
    """Un thread per drone.

    actions      : (n_droni, 4) azioni della policy in [-1, 1]
    body_ids     : indice del corpo di ciascun drone (da DroneHandle.body)
    max_thrust   : spinta massima [N]
    moment_scale : momento massimo per asse [N*m]
    body_q       : posizione + orientamento di tutti i corpi
    body_f       : forze esterne sui corpi (uscita)
    """
    i = wp.tid()
    body = body_ids[i]

    # Le azioni fuori da [-1, 1] vengono limitate (come il clamp di Isaac Lab)
    a0 = wp.clamp(actions[i, 0], -1.0, 1.0)
    a1 = wp.clamp(actions[i, 1], -1.0, 1.0)
    a2 = wp.clamp(actions[i, 2], -1.0, 1.0)
    a3 = wp.clamp(actions[i, 3], -1.0, 1.0)

    # Spinta e momento nel sistema del CORPO: la spinta spinge lungo +z del corpo
    thrust_body = wp.vec3(0.0, 0.0, max_thrust * (a0 + 1.0) * 0.5)
    moment_body = wp.vec3(
        moment_scale[0] * a1,
        moment_scale[1] * a2,
        moment_scale[2] * a3,
    )

    # body_f è espresso nel sistema del MONDO: si ruotano i vettori con l'orientamento del corpo
    # (come fa l'esempio diffsim_drone con wp.transform_vector)
    tf = body_q[body]
    force = wp.transform_vector(tf, thrust_body)
    torque = wp.transform_vector(tf, moment_body)

    # atomic_add: se in futuro si sommano altri contributi (drag, vento) sullo stesso corpo,
    # si accumulano senza sovrascriversi. Prima di ogni passo si azzera con state.clear_forces().
    # L'ordine (forza, coppia) del spatial_vector è quello dell'esempio diffsim_drone.
    wp.atomic_add(body_f, body, wp.spatial_vector(force, torque))


def launch_apply_wrench(actions, body_ids, body_q, body_f, params):
    """Lancia il kernel per tutti i droni. `params` è una DroneParams."""
    wp.launch(
        apply_wrench,
        dim=actions.shape[0],
        inputs=(
            actions,
            body_ids,
            params.max_thrust,
            wp.vec3(params.moment_scale_x, params.moment_scale_y, params.moment_scale_z),
            body_q,
        ),
        outputs=(body_f,),
    )
