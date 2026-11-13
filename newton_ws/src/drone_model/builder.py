"""Costruzione fisica del drone dentro un newton.ModelBuilder.

Questo codice gira sulla CPU, una volta sola, prima di builder.finalize().
Non contiene kernel.

Come in Isaac Lab, il drone è UN SOLO corpo rigido: spinta e momenti verranno
applicati direttamente al corpo (vedi actuators.py), senza modellare i singoli motori.
"""

import math
from dataclasses import dataclass

import newton
import warp as wp

from .params import DroneParams


@dataclass
class DroneHandle:
    """Indici necessari per ritrovare il drone nel Model e nello State."""

    body: int  # indice del corpo rigido (posizione in state.body_q, state.body_qd, state.body_f)


def _quat_y(angle: float):
    return wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle)


def _quat_z(angle: float):
    return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)


def _add_visual_shapes(builder: newton.ModelBuilder, body: int, color, s: float) -> None:
    """Aspetto da quadricottero a X: scocca, 4 bracci, 4 motori con elica. Il davanti è l'asse +x del corpo.

    Solo grafica: collisioni disattivate, nessun effetto su massa, inerzia o dinamica.
    s = fattore di scala grafico (il Tello è ~10 cm: a scala 1 è un puntino in una scena di qualche metro).
    """
    cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, has_particle_collision=False)
    dark, grey, rear = (0.12, 0.12, 0.14), (0.25, 0.26, 0.30), (0.70, 0.72, 0.78)
    arm = 0.060 * s  # distanza centro-motore sul piano

    # Scocca (ellissoide schiacciato) + calotta colorata sopra + "camera" chiara davanti (indica lo yaw)
    builder.add_shape_ellipsoid(body, rx=0.036 * s, ry=0.024 * s, rz=0.011 * s, cfg=cfg, color=dark)
    builder.add_shape_ellipsoid(
        body, xform=wp.transform(p=wp.vec3(-0.004 * s, 0.0, 0.007 * s)), rx=0.022 * s, ry=0.017 * s, rz=0.008 * s,
        cfg=cfg, color=color,
    )
    builder.add_shape_sphere(
        body, xform=wp.transform(p=wp.vec3(0.034 * s, 0.0, 0.0)), radius=0.009 * s, cfg=cfg, color=(0.95, 0.95, 1.0)
    )

    for k in range(4):
        ang = math.radians(45.0 + 90.0 * k)  # motori a +-45 gradi dall'asse x
        cx, cy = arm * math.cos(ang), arm * math.sin(ang)
        front = k in (0, 3)  # le due eliche davanti (x > 0) hanno il colore del drone, quelle dietro grigio chiaro

        # Braccio: capsula (lungo z di default) ruotata in orizzontale verso il motore
        builder.add_shape_capsule(
            body,
            xform=wp.transform(p=wp.vec3(0.5 * cx, 0.5 * cy, 0.0), q=wp.mul(_quat_z(ang), _quat_y(math.pi / 2.0))),
            radius=0.0045 * s, half_height=0.5 * arm - 0.0045 * s, cfg=cfg, color=grey,
        )
        # Motore (cilindro verticale) ed elica (disco sottile)
        builder.add_shape_cylinder(
            body, xform=wp.transform(p=wp.vec3(cx, cy, 0.004 * s)), radius=0.0085 * s, half_height=0.008 * s,
            cfg=cfg, color=dark,
        )
        builder.add_shape_cylinder(
            body, xform=wp.transform(p=wp.vec3(cx, cy, 0.0135 * s)), radius=0.030 * s, half_height=0.0007 * s,
            cfg=cfg, color=color if front else rear,
        )


def add_drone(
    builder: newton.ModelBuilder,
    params: DroneParams,
    label: str,
    collision_group: int = 1,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    visual: bool = False,
    visual_scale: float = 3.0,
    color: tuple[float, float, float] = (0.20, 0.55, 1.00),
) -> DroneHandle:
    """Aggiunge un drone al builder e ne restituisce l'handle.

    Il builder viene creato e chiuso (finalize) da chi chiama (scene/world.py).
    collision_group: ATTENZIONE, in Newton 0 = collisioni DISATTIVATE per la forma, 1 = gruppo
    di default (collide con pavimento e altri droni). Con più mondi paralleli, mondi diversi
    sono già separati da builder.replicate(), quindi basta lasciare 1.
    position: posizione iniziale del drone nel mondo [m], orientamento identità (dritto).
    visual: aggiunge l'aspetto da quadricottero (solo grafica, per le finestre di visualizzazione).
    Con visual=True la scatola di collisione diventa invisibile; con False (training) resta quella.
    visual_scale, color: scala grafica e colore dell'aspetto.
    """
    # Corpo libero (6 gradi di libertà) con massa e inerzia imposte direttamente.
    # lock_inertia=True: le forme aggiunte dopo NON modificano massa e inerzia,
    # che restano esattamente quelle del Tello (come l'override fatto in Isaac Lab).
    inertia = wp.mat33(
        params.ixx, 0.0, 0.0,
        0.0, params.iyy, 0.0,
        0.0, 0.0, params.izz,
    )
    body = builder.add_body(
        xform=wp.transform(p=wp.vec3(*position), q=wp.quat_identity()),
        mass=params.mass,
        inertia=inertia,
        lock_inertia=True,
        label=label,
    )

    # Forma di collisione: una scatola che approssima il corpo (hx, hy, hz sono semi-estensioni).
    # Serve per le collisioni tra droni e con l'ambiente, non influisce su massa e inerzia.
    cfg = newton.ModelBuilder.ShapeConfig(collision_group=collision_group, is_visible=not visual)
    builder.add_shape_box(
        body,
        hx=params.half_x,
        hy=params.half_y,
        hz=params.half_z,
        cfg=cfg,
    )

    if visual:
        _add_visual_shapes(builder, body, color, visual_scale)

    return DroneHandle(body=body)
