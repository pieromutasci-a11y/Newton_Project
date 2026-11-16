"""Prove della funzione costo del controllore di velocità.

    python /workspace/project_workspace/src/tests/test_vel_cost.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from controllers.vel_controller.vel_controller_env import VelControllerEnv, VelErrors
from controllers.vel_controller.vel_controller_mlp import VelControllerMLP
from drone_model.params import DroneParams
from scene.train_world import TrainWorld, TrainWorldConfig
from train.cost_function_vel_controller import CostWeights, action_rate_weight, cost_function

dev = "cuda:0"
W = CostWeights()
RANGE = torch.tensor([1.0, 1.0, 1.0, 1.5], device=dev)
ok = lambda c: "OK" if c else "ERRORE"  # noqa: E731


def make(vel_err=(0, 0, 0, 0), ang_vel=(0, 0, 0), tilt_deg=0.0, grad=False):
    """Errori di un solo drone. gravity_b = (sin a, 0, -cos a) per un'inclinazione a."""
    a = math.radians(tilt_deg)
    t = lambda x: torch.tensor([x], dtype=torch.float32, device=dev, requires_grad=grad)  # noqa: E731
    return VelErrors(
        vel_err=t(list(vel_err)),
        lin_vel_b=torch.zeros(1, 3, device=dev),
        ang_vel_b=t(list(ang_vel)),
        gravity_b=t([math.sin(a), 0.0, -math.cos(a)]),
    )


zero_a = torch.zeros(1, 4, device=dev)


def cost(**kw):
    e = make(**{k: v for k, v in kw.items() if k in ("vel_err", "ang_vel", "tilt_deg")})
    total, terms = cost_function(e, kw.get("a", zero_a), kw.get("prev", zero_a), RANGE, W, kw.get("scale"))
    return float(total), {k: float(v) for k, v in terms.items()}


# ---------------------------------------------------------------- 1. valori attesi
print("== 1. Valori del costo in situazioni note")
c0, _ = cost()
print(f"  fermo, dritto, errore nullo: costo = {c0:.6f} (atteso 0): {ok(abs(c0) < 1e-9)}")
_, t = cost(vel_err=(1.0, 0, 0, 0))
print(f"  errore di 1 m/s su vx: vel_error_lin = {t['vel_error_lin']:.3f} (atteso {W.vel_error_lin}), vel_error_ang = {t['vel_error_ang']:.3f} (atteso 0): {ok(abs(t['vel_error_lin'] - W.vel_error_lin) < 1e-6 and t['vel_error_ang'] == 0)}")
_, t = cost(vel_err=(0, 0, 0, 1.5))
print(f"  errore di 1.5 rad/s su wz (= il suo campo): vel_error_ang = {t['vel_error_ang']:.3f} (atteso {W.vel_error_ang}, normalizzato), vel_error_lin = {t['vel_error_lin']:.3f} (atteso 0): {ok(abs(t['vel_error_ang'] - W.vel_error_ang) < 1e-6 and t['vel_error_lin'] == 0)}")
_, t = cost(vel_err=(1.0, 1.0, 1.0, 1.5))
print(f"  errore pari al campo su tutti gli assi: lin {t['vel_error_lin']:.3f} (atteso {3 * W.vel_error_lin}) + ang {t['vel_error_ang']:.3f} (atteso {W.vel_error_ang}): {ok(abs(t['vel_error_lin'] - 3 * W.vel_error_lin) < 1e-6 and abs(t['vel_error_ang'] - W.vel_error_ang) < 1e-6)}")
_, t = cost(ang_vel=(5.0, 0, 0))
print(f"  velocità angolare 5 rad/s su x: ang_vel_xy = {t['ang_vel_xy']:.4f} (atteso {W.ang_vel_xy * 25:.4f})")
_, t = cost(ang_vel=(0, 0, 5.0))
print(f"  velocità angolare 5 rad/s su z (yaw): ang_vel_xy = {t['ang_vel_xy']:.4f} (atteso 0: lo yaw è nell'errore di velocità)")
TH = W.tilt_free_deg  # soglia libera del tilt [gradi]
print(f"  inclinazione  |  upright    tilt   (soglia libera {TH:.0f} gradi)")
for deg in (0, 20, TH, 45, 65, 90, 180):
    _, t = cost(tilt_deg=deg)
    print(f"    {deg:4.0f} gradi |  {t['upright']:.4f}  {t['tilt']:.4f}")
_, t_at = cost(tilt_deg=TH)
_, t_below = cost(tilt_deg=TH - 5)
_, t_above = cost(tilt_deg=TH + 10)
print(f"  tilt nullo sotto la soglia ({TH - 5:.0f} gradi), ~0 sulla soglia ({TH:.0f}), positivo sopra ({TH + 10:.0f}): {ok(t_below['tilt'] == 0 and t_at['tilt'] < 1e-6 and t_above['tilt'] > 0)}")
da = torch.zeros(1, 4, device=dev); da[0, 1] = 1.0
_, tA = cost(a=da, scale=action_rate_weight(W, 0))
_, tB = cost(a=da, scale=action_rate_weight(W, 10**6))
print(f"  variazione di 1 su una azione: action_rate = {tA['action_rate']:.4f} all'inizio, {tB['action_rate']:.4f} a fine rampa")
print("  peso della variazione delle azioni alle iterazioni 0, 750, 1500, 3000:", [round(action_rate_weight(W, i), 5) for i in (0, 750, 1500, 3000)], "(da 0.0014 a 0.014)")

# ---------------------------------------------------------------- 2. gradienti finiti nei punti critici
print("\n== 2. Gradienti finiti nei punti critici")
cases = [
    ("tutto a zero (riferimento raggiunto, dritto, fermo)", dict()),
    (f"inclinazione esattamente sulla soglia ({W.tilt_free_deg:.0f} gradi)", dict(tilt_deg=float(W.tilt_free_deg))),
    ("drone capovolto (180 gradi)", dict(tilt_deg=180.0)),
    ("inclinazione di 90 gradi", dict(tilt_deg=90.0)),
]
for name, kw in cases:
    e = make(grad=True, **kw)
    a = zero_a.clone().requires_grad_(True)
    total, _ = cost_function(e, a, zero_a, RANGE, W)
    total.sum().backward()
    g = torch.cat([x.grad.flatten() for x in (e.vel_err, e.ang_vel_b, e.gravity_b, a)])
    print(f"  {name:55s} gradienti finiti: {ok(bool(torch.isfinite(g).all()))}")

# ---------------------------------------------------------------- 3. direzione del gradiente
print("\n== 3. Il gradiente punta nella direzione giusta")
for sgn in (+1, -1):
    e = make(vel_err=(0.5 * sgn, 0, 0, 0), grad=True)
    total, _ = cost_function(e, zero_a, zero_a, RANGE, W)
    total.sum().backward()
    print(f"  d costo / d (errore vx) con errore {0.5 * sgn:+.1f} m/s = {float(e.vel_err.grad[0, 0]):+.3f} (atteso stesso segno dell'errore)")
e = make(tilt_deg=60.0, grad=True)
total, _ = cost_function(e, zero_a, zero_a, RANGE, W)
total.sum().backward()
print(f"  d costo / d (gravity_b z) a 60 gradi = {float(e.gravity_b.grad[0, 2]):+.3f} (atteso > 0: più orizzontale = costo più alto): {ok(float(e.gravity_b.grad[0, 2]) > 0)}")

# ---------------------------------------------------------------- 4. finestra reale con la rete
print("\n== 4. Costo reale: rete iniziale, 4096 mondi, finestra di 32 passi (decimation 2)")
N, H = 4096, 32
world = TrainWorld(DroneParams(), TrainWorldConfig(num_worlds=N, horizon=H, decimation=2))
env = VelControllerEnv(world, seed=2)
policy = VelControllerMLP().to(dev)
obs = env.observations()
loss = 0.0
acc = {}
for _ in range(H):
    a = policy(obs)
    prev = env.prev_actions
    obs = env.step(a)
    total, terms = cost_function(env.errors(), a, prev, env.refs.vel_range, W, action_rate_weight(W, 0))
    loss = loss + total.mean()
    for k, v in terms.items():
        acc[k] = acc.get(k, 0.0) + float(v.detach().mean()) / H
loss = loss / H
loss.backward()
g = torch.cat([p.grad.flatten() for p in policy.parameters()])
print(f"  loss (media sulla finestra) = {float(loss.detach()):.4f} | gradienti sui pesi finiti: {ok(bool(torch.isfinite(g).all()))} | norma = {float(g.norm()):.2f}")
print("  contributo medio di ogni termine per passo:")
for k, v in acc.items():
    print(f"    {k:14s} {v:8.4f}")
