"""Prove di TrainWorld: convenzioni dello stato, gradiente attraverso la fisica, tempi.

Da lanciare nel container:
    python /workspace/project_workspace/src/tests/test_train_world.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import warp as wp

from drone_model.params import DroneParams
from scene.train_world import TrainWorld, TrainWorldConfig

dev = "cuda:0"
params = DroneParams()
g = params.gravity


def reset_all(w, z=1.0):
    n = w.cfg.num_worlds
    ids = torch.arange(n, device=dev)
    pos = torch.zeros(n, 3, device=dev)
    pos[:, 2] = z
    w.detach()
    w.reset(ids, pos)


# ---------------------------------------------------------------- 1. convenzioni dello stato
print("== 1. Convenzioni dello stato (caduta libera, 20 passi, spinta nulla)")
w = TrainWorld(params, TrainWorldConfig(decimation=1, num_worlds=4, horizon=30))
reset_all(w, z=1.0)
a_fall = torch.tensor([-1.0, 0.0, 0.0, 0.0], device=dev).repeat(4, 1)
for _ in range(20):
    w.step(a_fall)
print("q  (x y z | qx qy qz qw) mondo 0:", [round(float(v), 4) for v in w.q[0]])
print("qd (6 componenti)        mondo 0:", [round(float(v), 4) for v in w.qd[0]])
n = 20
print(f"atteso: z = {1.0 - g * 0.01**2 * n * (n + 1) / 2:.4f},  velocità verticale = {-g * 0.01 * n:.4f}")

# ---------------------------------------------------------------- 2. gradiente: torch vs differenza finita
print("\n== 2. Gradiente attraverso la fisica: autograd di PyTorch contro differenza finita")
H, N = 30, 4
w = TrainWorld(params, TrainWorldConfig(decimation=1, num_worlds=N, horizon=H))
target = torch.tensor([0.5, -0.3, 1.4], device=dev)
base = torch.tensor([0.11, 0.05, -0.03, 0.02], device=dev).repeat(H, N, 1)
base = base + 0.02 * torch.randn_like(base)


def rollout_loss(actions):
    reset_all(w, z=1.0)
    for t in range(H):
        w.step(actions[t])
    return ((w.pos - target) ** 2).sum()


acts = base.clone().requires_grad_(True)
loss = rollout_loss(acts)
loss.backward()
g_auto = acts.grad.clone()
print("costo =", float(loss), "| nan nel gradiente:", bool(torch.isnan(g_auto).any()))

eps = 1e-2
print("passo mondo asse   autograd        diff.finita     err.rel.")
for step, world, axis in [(0, 0, 0), (0, 1, 2), (10, 2, 1), (20, 3, 3), (20, 0, 2)]:
    bp, bm = base.clone(), base.clone()
    bp[step, world, axis] += eps
    bm[step, world, axis] -= eps
    with torch.no_grad():
        fd = (float(rollout_loss(bp)) - float(rollout_loss(bm))) / (2 * eps)
    ga = float(g_auto[step, world, axis])
    print(f"{step:4d} {world:5d} {axis:4d}   {ga:+.5e}   {fd:+.5e}   {abs(ga - fd) / max(abs(fd), 1e-8):.1e}")

# Indipendenza dei mondi: cambiare le azioni del mondo 1 non deve cambiare il gradiente del mondo 0
print("\nIndipendenza dei mondi: gradiente del mondo 0 dipende dalle azioni del mondo 1?")
acts2 = base.clone()
acts2[:, 1, :] += 0.3
acts2.requires_grad_(True)
rollout_loss(acts2).backward()
print("  differenza massima sul mondo 0:", float((acts2.grad[:, 0] - g_auto[:, 0]).abs().max()))

# ---------------------------------------------------------------- 3. due finestre con troncamento
print("\n== 3. Due finestre consecutive con detach()")
reset_all(w, z=1.0)
a = torch.full((N, 4), 0.0, device=dev, requires_grad=True)
for t in range(H):
    w.step(a)
l1 = (w.pos[:, 2] - 1.4).pow(2).sum()
l1.backward()
w.detach()
for t in range(H):
    w.step(a)
l2 = (w.pos[:, 2] - 1.4).pow(2).sum()
l2.backward()
print("ok: z dopo 2 finestre (mondo 0) =", float(w.pos[0, 2]), "| grad azioni finito:", bool(torch.isfinite(a.grad).all()))

# ---------------------------------------------------------------- 4. tempi
print("\n== 4. Tempi (forward + backward di una finestra)")
for n_worlds, horizon in [(1024, 32), (4096, 32)]:
    wt = TrainWorld(params, TrainWorldConfig(decimation=1, num_worlds=n_worlds, horizon=horizon))
    acts_t = torch.zeros(horizon, n_worlds, 4, device=dev, requires_grad=True)

    def one_iter():
        reset_all(wt, z=1.0)
        for t in range(horizon):
            wt.step(acts_t[t])
        ((wt.pos - target) ** 2).sum().backward()

    one_iter()  # riscaldamento (compilazione dei kernel)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        one_iter()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 5
    steps = n_worlds * horizon
    print(f"N={n_worlds:5d}, H={horizon}: {dt * 1000:7.1f} ms/iterazione  ->  {steps / dt:,.0f} passi-drone/s")
    free, total = torch.cuda.mem_get_info()
    print(f"        memoria GPU usata (totale processo): {(total - free) / 2**20:.0f} MiB")
    del wt
