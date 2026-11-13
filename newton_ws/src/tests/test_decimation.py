"""Controllo a 25 Hz sopra la fisica a 100 Hz (decimation=4): equivalenza e gradiente.

    python /workspace/project_workspace/src/tests/test_decimation.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from drone_model.params import DroneParams
from scene.train_world import TrainWorld, TrainWorldConfig

dev = "cuda:0"
params = DroneParams()
ids4 = torch.arange(4, device=dev)


def reset_all(w, z=1.0):
    n = w.cfg.num_worlds
    pos = torch.zeros(n, 3, device=dev)
    pos[:, 2] = z
    w.detach()
    w.reset(torch.arange(n, device=dev), pos)


print("== 1. Un passo di controllo (decimation=4) = 4 passi di fisica con la stessa azione")
w4 = TrainWorld(params, TrainWorldConfig(num_worlds=4, horizon=5, decimation=4))
w1 = TrainWorld(params, TrainWorldConfig(num_worlds=4, horizon=20, decimation=1))
reset_all(w4), reset_all(w1)
torch.manual_seed(0)
acts = 0.3 * torch.randn(5, 4, 4, device=dev)
for t in range(5):
    w4.step(acts[t])
    for _ in range(4):
        w1.step(acts[t])
dq = (w4.q - w1.q).abs().max().item()
dqd = (w4.qd - w1.qd).abs().max().item()
print(f"differenza massima: posa {dq:.2e}, velocità {dqd:.2e}   ->", "OK" if max(dq, dqd) < 1e-5 else "ERRORE")
print(f"ctrl_dt = {w4.ctrl_dt} s (sim_dt = {w4.sim_dt} s)")

print("\n== 2. Gradiente con decimation=4: autograd contro differenza finita")
H, N = 8, 4
w = TrainWorld(params, TrainWorldConfig(num_worlds=N, horizon=H, decimation=4))
target = torch.tensor([0.5, -0.3, 1.4], device=dev)
base = torch.tensor([0.11, 0.05, -0.03, 0.02], device=dev).repeat(H, N, 1)
base = base + 0.02 * torch.randn_like(base)


def rollout_loss(actions):
    reset_all(w)
    for t in range(H):
        w.step(actions[t])
    return ((w.pos - target) ** 2).sum()


a = base.clone().requires_grad_(True)
rollout_loss(a).backward()
g = a.grad.clone()
ok = True
for (t, n, k) in [(0, 0, 0), (0, 1, 1), (3, 2, 2), (5, 3, 3), (7, 0, 0), (2, 1, 0)]:
    eps = 1e-2
    ap, am = base.clone(), base.clone()
    ap[t, n, k] += eps
    am[t, n, k] -= eps
    with torch.no_grad():
        fd = (rollout_loss(ap) - rollout_loss(am)).item() / (2 * eps)
    rel = abs(fd - g[t, n, k].item()) / max(abs(fd), 1e-6)
    ok &= rel < 0.05
    print(f"  a[t={t}, mondo={n}, k={k}]: autograd {g[t, n, k].item():+.5f} | diff. finita {fd:+.5f} | err. rel. {rel:.1e}")
print("->", "OK" if ok else "ERRORE")
