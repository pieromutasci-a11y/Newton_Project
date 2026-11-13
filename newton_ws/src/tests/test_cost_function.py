"""Prove della funzione costo: valori di riferimento, monotonia, gradienti finiti nei punti critici.

Da lanciare nel container:
    python /workspace/project_workspace/src/tests/test_cost_function.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from controllers.pos_controller.pos_controller_env import Errors, PosControllerEnv
from controllers.pos_controller.pos_controller_mlp import PosControllerMLP
from drone_model.params import DroneParams
from scene.train_world import TrainWorld, TrainWorldConfig
from train.cost_function import CostWeights, cost_function, make_hover_action

dev = "cuda:0"
params = DroneParams()
hover = make_hover_action(params, dev)
print("azione di hover:", [round(float(v), 4) for v in hover])


def make_errors(n=1, dist=0.0, yaw=0.0, tilt_deg=0.0, ang=0.0, lin=0.0, requires_grad=False):
    """Errori sintetici: bersaglio a distanza `dist` lungo x, drone inclinato di `tilt_deg` attorno a x."""
    gz = -math.cos(math.radians(tilt_deg))
    e = Errors(
        pos_err=torch.tensor([[dist, 0.0, 0.0]], device=dev).repeat(n, 1),
        yaw_err=torch.full((n,), yaw, device=dev),
        lin_vel_b=torch.tensor([[lin, 0.0, 0.0]], device=dev).repeat(n, 1),
        ang_vel_b=torch.tensor([[ang, 0.0, 0.0]], device=dev).repeat(n, 1),
        gravity_b=torch.tensor([[0.0, math.sin(math.radians(tilt_deg)), gz]], device=dev).repeat(n, 1),
    )
    if requires_grad:
        for name in ("pos_err", "yaw_err", "lin_vel_b", "ang_vel_b", "gravity_b"):
            setattr(e, name, getattr(e, name).clone().requires_grad_(True))
    return e


a_hover = hover.unsqueeze(0)

# ---------------------------------------------------------------- 1. valori di riferimento
print("\n== 1. Costo in situazioni note (azione di hover, azione uguale alla precedente)")
cases = {
    "sul bersaglio, dritto, fermo": make_errors(),
    "a 0.1 m": make_errors(dist=0.1),
    "a 1 m": make_errors(dist=1.0),
    "a 3 m": make_errors(dist=3.0),
    "errore di yaw 90 gradi": make_errors(yaw=math.pi / 2),
    "inclinato di 20 gradi (sotto la soglia libera)": make_errors(tilt_deg=20),
    "inclinato di 60 gradi": make_errors(tilt_deg=60),
    "velocità angolare 10 rad/s": make_errors(ang=10.0),
    "a 0.2 m con velocità 1 m/s (frenata)": make_errors(dist=0.2, lin=1.0),
    "a 2 m con velocità 1 m/s": make_errors(dist=2.0, lin=1.0),
}
for name, e in cases.items():
    total, terms = cost_function(e, a_hover, a_hover, hover)
    print(f"  {name:48s} costo = {float(total):8.4f}")

zero_total, _ = cost_function(cases["sul bersaglio, dritto, fermo"], a_hover, a_hover, hover)
print("  costo sul bersaglio quasi zero:", float(zero_total) < 1e-2)

# Monotonia nella distanza
d = torch.linspace(0.0, 3.0, 31)
vals = [float(cost_function(make_errors(dist=float(x)), a_hover, a_hover, hover)[0]) for x in d]
print("  costo crescente con la distanza:", all(b >= a for a, b in zip(vals, vals[1:])))

# ---------------------------------------------------------------- 2. gradienti nei punti critici
print("\n== 2. Gradienti finiti nei punti critici (dove norma, valore assoluto e acos darebbero NaN)")
for label, kwargs in {
    "tutto a zero (bersaglio raggiunto, dritto, fermo)": dict(),
    "inclinazione esattamente sulla soglia (35 gradi)": dict(tilt_deg=35.0),
    "errore di yaw esattamente 180 gradi": dict(yaw=math.pi),
    "drone capovolto (180 gradi)": dict(tilt_deg=180.0),
}.items():
    e = make_errors(requires_grad=True, **kwargs)
    act = a_hover.clone().requires_grad_(True)
    total, _ = cost_function(e, act, a_hover, hover)
    total.sum().backward()
    grads = [e.pos_err.grad, e.yaw_err.grad, e.lin_vel_b.grad, e.ang_vel_b.grad, e.gravity_b.grad, act.grad]
    ok = all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
    print(f"  {label:52s} gradienti finiti: {ok}")

# ---------------------------------------------------------------- 3. segno dei gradienti
print("\n== 3. Il gradiente punta nella direzione giusta")
e = make_errors(dist=1.0, requires_grad=True)
cost_function(e, a_hover, a_hover, hover)[0].sum().backward()
print(f"  d costo / d (errore di posizione x) a 1 m = {float(e.pos_err.grad[0, 0]):+.4f}  (atteso > 0: errore più grande = costo più alto)")
e = make_errors(tilt_deg=60.0, requires_grad=True)
cost_function(e, a_hover, a_hover, hover)[0].sum().backward()
print(f"  d costo / d (gravity_b z) a 60 gradi     = {float(e.gravity_b.grad[0, 2]):+.4f}  (atteso > 0: più orizzontale = costo più alto)")

# ---------------------------------------------------------------- 4. costo reale dall'env, con la rete iniziale
print("\n== 4. Costo reale: rete iniziale, 4096 mondi, finestra di 32 passi")
N, H = 4096, 32
world = TrainWorld(params, TrainWorldConfig(num_worlds=N, horizon=H))
env = PosControllerEnv(world, seed=3)
policy = PosControllerMLP().to(dev)
obs = env.observations()
sums = None
loss = 0.0
for _ in range(H):
    actions = policy(obs)
    prev_actions = env.prev_actions
    obs = env.step(actions)
    total, terms = cost_function(env.errors(), actions, prev_actions, hover)
    loss = loss + total.mean()
    sums = {k: v.detach().mean() / H for k, v in terms.items()} if sums is None else {
        k: sums[k] + v.detach().mean() / H for k, v in terms.items()
    }
loss = loss / H
loss.backward()
print(f"  loss (media sulla finestra) = {float(loss.detach()):.4f}")
print("  contributo medio di ogni termine al costo per passo:")
for k, v in sorted(sums.items(), key=lambda kv: -float(kv[1])):
    print(f"    {k:14s} {float(v):8.4f}")
grads = [p.grad for p in policy.parameters()]
print("  gradienti sui pesi della rete finiti:", all(torch.isfinite(g).all() for g in grads))
