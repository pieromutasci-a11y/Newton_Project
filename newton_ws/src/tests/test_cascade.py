"""Prove della cascata: rete di posizione -> riferimenti di velocità -> rete di velocità (congelata) -> fisica.

    python /workspace/project_workspace/src/tests/test_cascade.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from controllers.pos_controller.pos_controller_env import PosControllerEnv
from controllers.pos_controller.pos_controller_mlp import PosControllerMLP
from drone_model.params import DroneParams
from scene.train_world import TrainWorld, TrainWorldConfig

dev = "cuda:0"
ok = lambda c: "OK" if c else "ERRORE"  # noqa: E731
VEL_CKPT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "vel_controller", "policy", "apg_vel_v7", "best.pt")
assert os.path.isfile(VEL_CKPT), f"serve il checkpoint della rete di velocità: {VEL_CKPT}"

N, H = 8, 32
world = TrainWorld(DroneParams(), TrainWorldConfig(num_worlds=N, horizon=2 * H, decimation=2))
policy = PosControllerMLP(vel_ckpt=VEL_CKPT).to(dev)
env = PosControllerEnv(world, policy.vel_policy, seed=1)

# ---------------------------------------------------------------- 1. la rete di velocità è congelata
print("== 1. La rete di velocità è congelata")
print("  parametri della velocità senza gradiente:", ok(not any(p.requires_grad for p in policy.vel_policy.parameters())))
print("  parametri della posizione con gradiente:", ok(all(p.requires_grad for p in policy.net.parameters())))
policy.train()
print("  dopo policy.train() la rete di velocità è ancora in eval():", ok(not policy.vel_policy.training and policy.training))
stats = [policy.vel_policy.obs_mean.clone(), policy.vel_policy.obs_var.clone(), policy.vel_policy.obs_count.clone()]
x = torch.randn(256, 33, device=dev)
for _ in range(5):
    policy(x)
    env.step(torch.zeros(N, 4, device=dev))
env.end_window()
now = [policy.vel_policy.obs_mean, policy.vel_policy.obs_var, policy.vel_policy.obs_count]
print("  statistiche di normalizzazione della velocità invariate dopo passi in training:", ok(all(torch.equal(a, b) for a, b in zip(stats, now))))
print("  statistiche della POSIZIONE invece si aggiornano:", ok(float(policy.obs_count) > 1.0))

# ---------------------------------------------------------------- 2. checkpoint auto-contenuto
print("\n== 2. Il checkpoint della posizione contiene anche la rete di velocità")
sd = policy.state_dict()
print("  chiavi della velocità nello state_dict:", ok(any(k.startswith("vel_policy.") for k in sd)))
fresh = PosControllerMLP().to(dev)  # senza checkpoint della velocità: pesi casuali
vobs = torch.randn(64, 17, device=dev)
before = (fresh.vel_policy.eval()(vobs) - policy.vel_policy(vobs)).abs().max().item()
fresh.load_state_dict(sd)
after = (fresh.vel_policy.eval()(vobs) - policy.vel_policy(vobs)).abs().max().item()
print(f"  rete di velocità diversa prima del caricamento ({before:.3f}), identica dopo ({after:.1e}):", ok(before > 1e-3 and after < 1e-6))
x = torch.randn(64, 33, device=dev)
print("  la posizione dà le stesse uscite dopo il caricamento:", ok(torch.equal(policy.eval()(x), fresh.eval()(x))))
policy.train()


# ---------------------------------------------------------------- 3. inseguimento di un comando di velocità
def start_still(z=2.0):
    """Droni fermi e dritti in (0, 0, z), stato pulito, senza riferimenti che scadono."""
    world.detach()
    world.reset(torch.arange(N, device=dev), torch.tensor([0.0, 0.0, z], device=dev).expand(N, 3))
    env.prev_actions = torch.zeros(N, 4, device=dev)
    env.low_prev_actions = torch.zeros(N, 4, device=dev)


print("\n== 3. Comando di velocità costante (a = (0.5, 0, 0.3, 0.4) -> 0.5 m/s, 0.3 m/s, 0.6 rad/s): la cascata lo insegue")
start_still()
a = torch.tensor([0.5, 0.0, 0.3, 0.4], device=dev).expand(N, 4)
with torch.no_grad():
    for _ in range(H):  # 1.28 s
        env.step(a)
e = env.errors()
v = torch.cat([e.lin_vel_b, e.ang_vel_b[:, 2:3]], dim=-1)[0]
want = (a[0] * env.vel_scale)
print("  comando (vx, vy, vz, wz):", [round(float(x), 3) for x in want])
print("  velocità reale dopo 1.28 s:", [round(float(x), 3) for x in v])
print("  errore massimo sull'asse:", f"{float((v - want).abs().max()):.3f}", "(atteso < 0.15):", ok(float((v - want).abs().max()) < 0.15))
start_still()
with torch.no_grad():
    for _ in range(H):
        env.step(torch.zeros(N, 4, device=dev))
print(f"  comando nullo = hover: velocità finale {float(torch.cat([env.errors().lin_vel_b, env.errors().ang_vel_b], -1).abs().max()):.4f}, quota {float(world.pos[0, 2]):.3f} m (partenza 2.0)")

# ---------------------------------------------------------------- 4. gradiente attraverso la rete di velocità e la fisica
print("\n== 4. Gradiente della posizione finale rispetto al comando: autograd contro differenza finita")
K = 25  # 1 s


def final_x(a):
    start_still()
    for _ in range(K):
        env.step(a)
    return world.pos[:, 0]  # (N,)


a0 = torch.tensor([0.3, 0.1, -0.2, 0.2], device=dev).expand(N, 4).clone()
for axis, name in enumerate(["vx", "vy", "vz", "wz"]):
    a = a0.clone().requires_grad_(True)
    final_x(a).sum().backward()
    g = float(a.grad[:, axis].sum())
    eps = 1e-2
    with torch.no_grad():
        ap, am = a0.clone(), a0.clone()
        ap[:, axis] += eps
        am[:, axis] -= eps
        xp = final_x(ap).clone()  # clone: il risultato è una vista sulla memoria di Warp, che il rollout dopo sovrascrive
        xm = final_x(am).clone()
        fd = float((xp - xm).sum()) / (2 * eps)
    rel = abs(g - fd) / max(abs(fd), 1e-3)
    print(f"  d x_finale / d comando {name}: autograd {g:+.4f} | diff. finita {fd:+.4f} | err. rel. {rel:.1e}  {ok(rel < 0.05 or abs(g - fd) < 2e-3)}")
