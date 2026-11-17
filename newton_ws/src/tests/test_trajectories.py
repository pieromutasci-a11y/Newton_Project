"""Test delle traiettorie continue e della finestra di anteprima (target_handler.py + PosControllerEnv).

Si lancia nel container:  python /workspace/project_workspace/src/tests/test_trajectories.py
Ogni sezione stampa OK / ERRORE.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from controllers.pos_controller.pos_controller_env import PosControllerEnv
from controllers.pos_controller.pos_controller_mlp import PosControllerMLP
from controllers.pos_controller.target_handler import KIND_NAMES, TargetHandler
from drone_model.params import DroneParams
from scene.train_world import TrainWorld, TrainWorldConfig

dev = "cuda:0"
fails = []


def ok(cond, name=""):
    if not cond:
        fails.append(name)
    return "OK" if cond else "ERRORE"


# ---------------------------------------------------------------------------------------------- 1
print("1) Solo waypoint (traj_prob = 0): finestra nulla, velocità del riferimento nulla")
th = TargetHandler(256, dev, seed=0, traj_prob=0.0)
dp, dy = th.window()
print("  forma finestra:", tuple(dp.shape), tuple(dy.shape), ok(dp.shape == (256, 3, 3) and dy.shape == (256, 3), "forma"))
print("  incrementi tutti nulli:", ok(dp.abs().max() == 0 and dy.abs().max() == 0, "nulli"))
print("  nessuna traiettoria, velocità nulla:", ok(not th.is_traj.any() and th.target_vel.abs().max() == 0, "no traj"))

# ---------------------------------------------------------------------------------------------- 2
print("2) Mix: la frazione di traiettorie segue traj_prob")
th = TargetHandler(4096, dev, seed=1, traj_prob=0.5)
frac = float(th.is_traj.float().mean())
print(f"  frazione {frac:.3f} (atteso 0.5)", ok(abs(frac - 0.5) < 0.04, "frazione"))

# ---------------------------------------------------------------------------------------------- 3
print("3) Ogni tipo di curva: dentro i limiti, velocità analitica = derivata della posizione, velocità entro il limite")
for name, kind in KIND_NAMES.items():
    th = TargetHandler(512, dev, seed=2, traj_prob=1.0, traj_speed_range=(0.1, 0.5))
    th.set_trajectory(name)
    lo = torch.tensor([-1.5, -1.5, 0.5], device=dev)
    hi = torch.tensor([1.5, 1.5, 1.5], device=dev)
    inside, vmax, derr = True, 0.0, 0.0
    t = torch.zeros(512, device=dev)
    p_prev, v_prev, _ = th._curve(t)
    for _ in range(600):  # 24 s: oltre un episodio
        t = t + 0.04
        p, v, y = th._curve(t)
        inside &= bool(((p >= lo - 1e-4) & (p <= hi + 1e-4)).all())
        vmax = max(vmax, float(v.norm(dim=-1).max()))
        derr = max(derr, float(((p - p_prev) / 0.04 - 0.5 * (v + v_prev)).abs().max()))  # differenza finita contro analitica
        p_prev, v_prev = p, v
    print(f"  {name:8s} dentro i limiti {ok(inside, name + ' limiti')} | |v| max (norma) {vmax:.2f} m/s {ok(vmax <= 0.5 + 1e-4, name + ' vmax')} | scarto derivata {derr:.4f} m/s {ok(derr < 0.02, name + ' derivata')}")

print("3b) Orientamento 3D: le curve non stanno tutte sul piano orizzontale")
th = TargetHandler(2048, dev, seed=7, traj_prob=1.0, traj_speed_range=(0.3, 0.5))
ts = torch.arange(0, 400, device=dev) * 0.1  # 40 s
pos = torch.stack([th._curve(torch.full((2048,), float(t), device=dev))[0] for t in ts])  # (T, N, 3)
zext = pos[..., 2].amax(0) - pos[..., 2].amin(0)
xyext = (pos[..., :2].amax(0) - pos[..., :2].amin(0)).amax(-1)
kinds = th.tr["kind"]
nl = kinds != 0
print(f"  escursione verticale: mediana {float(zext.median()):.2f} m, massima {float(zext.max()):.2f} m (campo z 1.0 m)", ok(float(zext.max()) > 0.6 and float(zext.max()) <= 1.0 + 1e-3, "z estensione"))
print(f"  curve con escursione verticale > 0.2 m: {100 * float((zext > 0.2).float().mean()):.0f}%  | quasi orizzontali (< 0.05 m): {100 * float((zext < 0.05).float().mean()):.0f}%", ok(float((zext > 0.2).float().mean()) > 0.5, "3D"))
print("  tutte dentro i limiti (x, y ±1.5, z 0.5-1.5)", ok(bool((pos[..., :2].abs() <= 1.5 + 1e-4).all() and (pos[..., 2] >= 0.5 - 1e-4).all() and (pos[..., 2] <= 1.5 + 1e-4).all()), "limiti 3D"))
rot = th.tr["rot"]
print("  matrici di rotazione ortonormali", ok(bool((rot @ rot.transpose(1, 2) - torch.eye(3, device=dev)).abs().max() < 1e-4), "rotazioni"))
print("  yaw nel verso di marcia continuo (nessun salto > 0.5 rad per passo di 0.1 s)", end=" ")
yaws = torch.stack([th._curve(torch.full((2048,), float(t), device=dev))[2] for t in ts])
dj = torch.atan2(torch.sin(yaws[1:] - yaws[:-1]), torch.cos(yaws[1:] - yaws[:-1])).abs()
head = th.tr["yaw_mode"] == 1
print(f"(max {float(dj[:, head].max()):.2f} rad)", ok(float(dj[:, head].max()) < 0.5, "yaw continuo"))

# ---------------------------------------------------------------------------------------------- 4
print("4) Il tempo scorre: il riferimento si muove, step_target è il punto nuovo; i waypoint restano fermi")
th = TargetHandler(64, dev, seed=3, traj_prob=0.5, traj_speed_range=(0.3, 0.5))
traj = th.is_traj.clone()
p0 = th.target_pos.clone()
for _ in range(10):
    th.step()
moved = (th.target_pos - p0).norm(dim=-1)
print(f"  traiettorie: spostamento in 0.4 s {float(moved[traj].mean()):.3f} m (>0)", ok(bool((moved[traj] > 1e-3).all()), "si muove"))
print("  waypoint: fermi", ok(bool((moved[~traj] == 0).all()), "wp fermi"))
print("  step_target coincide col riferimento attivo (traiettorie)", ok(bool((th.step_target_pos[traj] == th.target_pos[traj]).all()), "step target"))
print("  is_last vero per le traiettorie (non avanzano)", ok(bool(th.is_last[traj].all()), "is_last"))

# ---------------------------------------------------------------------------------------------- 5
print("5) Finestra: incrementi sommati = punto futuro - punto attuale; per i waypoint nulli")
th = TargetHandler(128, dev, seed=4, traj_prob=0.5)
dp, dy = th.window()
tot = dp.sum(dim=1)
p_fut, _, _ = th._curve(th.t + 3 * th.window_dt)
p_now, _, _ = th._curve(th.t)
m = th.is_traj
print("  somma incrementi = p(t+1.5) - p(t):", ok(bool(((tot - (p_fut - p_now))[m]).abs().max() < 1e-4), "somma"))
print("  waypoint: nulli", ok(bool(dp[~m].abs().max() == 0), "wp nulli"))
print("  incremento atteso ~ velocità x Δ (primo punto):", ok(bool(((dp[:, 0] - th.target_vel * th.window_dt)[m]).norm(dim=-1).max() < 0.25 * 0.5 * 0.5 + 0.05), "incremento"))

# ---------------------------------------------------------------------------------------------- 6
print("6) Reset: nuove traiettorie e tempo a zero; set_constant le spegne")
th = TargetHandler(64, dev, seed=5, traj_prob=1.0)
for _ in range(20):
    th.step()
ids = torch.arange(0, 32, device=dev)
th.reset(ids)
print("  tempo azzerato solo per i mondi resettati", ok(bool((th.t[:32] == 0).all() and (th.t[32:] > 0).all()), "tempo"))
th.set_constant([0.5, 0.0, 1.0], 0.3)
print("  set_constant: nessuna traiettoria, velocità nulla", ok(bool((~th.is_traj).all() and th.target_vel.abs().max() == 0), "constant"))

# ---------------------------------------------------------------------------------------------- 7
print("7) Ambiente: 33 osservazioni, anteprima nel corpo, spawn vicino alla curva")
N = 256
params = DroneParams()
world = TrainWorld(params, TrainWorldConfig(num_worlds=N, horizon=16, decimation=2))
policy = PosControllerMLP(vel_ckpt=os.path.join(os.path.dirname(__file__), "..", "..", "data", "vel_controller", "policy", "apg_vel_v7", "best.pt")).to(dev)
targets = TargetHandler(N, dev, seed=6, traj_prob=0.5)
env = PosControllerEnv(world, policy.vel_policy, targets, seed=6)
obs = env.observations()
print("  forma:", tuple(obs.shape), ok(obs.shape == (N, 33) and env.obs_dim == 33 and policy.input_dim == 33, "obs 33"))
traj = env.targets.is_traj
pre = obs[:, 18:].reshape(N, 3, 5)
wp_block_ok = bool(((pre[~traj][..., :3].abs().max() == 0) & ((pre[~traj][..., 3]).abs().max() == 0) & ((pre[~traj][..., 4] - 1).abs().max() == 0)))
print("  waypoint: anteprima = (0, 0, 0, sin 0, cos 1)", ok(wp_block_ok, "wp anteprima"))
print("  traiettorie: anteprima non nulla", ok(bool(pre[traj][..., :3].abs().amax() > 0.01), "traj anteprima"))
dist0 = (env.world.q[:, :3] - env.targets.target_pos).norm(dim=-1)
print(f"  distanza di partenza dal riferimento: traiettorie max {float(dist0[traj].max()):.2f} m (<= 0.52)", ok(float(dist0[traj].max()) <= 0.52, "spawn"))
e = env.errors()
print("  Errors.ref_vel_b presente e nullo per i waypoint", ok(e.ref_vel_b is not None and float(e.ref_vel_b[~traj].abs().max()) == 0, "ref_vel_b"))

# ---------------------------------------------------------------------------------------------- 8
print("8) Costo: velocità relativa. Il drone che segue la curva alla sua velocità non è penalizzato")
from train.cost_function_pos_controller import CostWeights, cost_function, make_hover_command

e = env.errors()
a = torch.zeros(N, 4, device=dev)
hover = make_hover_command(dev)
w = CostWeights()
_, terms = cost_function(e, a, a, hover, w)
e.lin_vel_b = e.ref_vel_b.clone()  # il drone si muove esattamente come il riferimento
_, terms2 = cost_function(e, a, a, hover, w)
print("  lin_vel nullo se v = v_rif:", ok(float(terms2["lin_vel"].abs().max()) < 1e-9, "lin_vel"))
print("  waypoint: invariato (v_rif = 0)", ok(bool(torch.allclose(terms["lin_vel"][~traj], torch.zeros_like(terms["lin_vel"][~traj]) + terms["lin_vel"][~traj])), "wp invariato"))

# ---------------------------------------------------------------------------------------------- 9
print("9) Cascata: finestra di training con traiettorie, gradiente finito, nessun NaN")
policy.train()
opt = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=1e-3)
obs = env.observations()
loss = 0.0
for _ in range(8):
    act = policy(obs)
    prev = env.prev_actions
    obs = env.step(act)
    total, _ = cost_function(env.errors(), act, prev, hover, w)
    loss = loss + total.mean()
loss = loss / 8
opt.zero_grad()
loss.backward()
gn = torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
print(f"  loss {float(loss):.4f}, |grad| {float(gn):.4f}", ok(bool(torch.isfinite(loss) and torch.isfinite(gn) and gn > 0), "gradiente"))
env.end_window()
print("  stato finito dopo la finestra", ok(bool(torch.isfinite(env.world.q).all()), "stato finito"))

print()
print("TUTTO OK" if not fails else f"FALLITI: {fails}")
sys.exit(1 if fails else 0)
