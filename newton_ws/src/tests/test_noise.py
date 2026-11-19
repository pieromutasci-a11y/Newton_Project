"""Prove del rumore bianco sul feedback: zero = identico a prima, statistiche corrette, il costo usa lo stato vero,
il gradiente passa, la rete di velocità legge o no il rumore.

    python /workspace/project_workspace/src/tests/test_noise.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from controllers.math_utils import quat_mul, quat_rotate_inverse
from controllers.pos_controller.pos_controller_env import NoiseConfig, PosControllerEnv
from controllers.pos_controller.pos_controller_mlp import PosControllerMLP
from drone_model.params import DroneParams
from scene.train_world import TrainWorld, TrainWorldConfig
from train.cost_function_pos_controller import CostWeights, cost_function, make_hover_command

dev = "cuda:0"
fails = []


def ok(cond, name):
    if not cond:
        fails.append(name)
    return "OK" if cond else "ERRORE"


VEL_CKPT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "vel_controller", "policy", "apg_vel_v7", "best.pt")
VEL_CKPT = VEL_CKPT if os.path.isfile(VEL_CKPT) else None
N, H = 4096, 8
params = DroneParams()


def make_env(noise=None, seed=3):
    world = TrainWorld(params, TrainWorldConfig(num_worlds=N, horizon=2 * H, decimation=2))
    policy = PosControllerMLP(vel_ckpt=VEL_CKPT).to(dev)
    torch.manual_seed(0)
    return PosControllerEnv(world, policy.vel_policy, seed=seed, noise=noise), policy


print("1) Rumore zero: osservazioni identiche allo stato vero")
env, _ = make_env()
o1 = env.observations()
o2 = env.observations()
print("  due letture uguali", ok(torch.equal(o1, o2), "zero deterministico"))
q, qd = env._measure()
print("  stato misurato = stato vero", ok(q is env.world.q and qd is env.world.qd, "misura identità"))

print("2) Statistiche del rumore sullo stato misurato")
sp, sa, sv, sw = 0.02, math.radians(1.5), 0.05, 0.04
env, _ = make_env(NoiseConfig(sp, sa, sv, sw))
q, qd = env.world.q, env.world.qd
qm, qdm = env._measure()
dp, dv, dw = qm[:, :3] - q[:, :3], qdm[:, :3] - qd[:, :3], qdm[:, 3:6] - qd[:, 3:6]
print(f"  posizione: std {float(dp.std()):.4f} (atteso {sp}), media {float(dp.mean()):+.4f}", ok(abs(float(dp.std()) - sp) < 0.1 * sp and abs(float(dp.mean())) < 0.1 * sp, "pos std"))
print(f"  vel lineare: std {float(dv.std()):.4f} (atteso {sv})", ok(abs(float(dv.std()) - sv) < 0.1 * sv, "vel std"))
print(f"  vel angolare: std {float(dw.std()):.4f} (atteso {sw})", ok(abs(float(dw.std()) - sw) < 0.1 * sw, "ang vel std"))
qn = qm[:, 3:7]
print("  quaternione misurato di norma 1", ok(bool(torch.allclose(qn.norm(dim=-1), torch.ones(N, device=dev), atol=1e-5)), "norma quaternione"))
# angolo di rotazione tra assetto vero e misurato: rotazione relativa = q_vero^-1 * q_misurato
q_true = q[:, 3:7]
conj = torch.cat([-q_true[:, :3], q_true[:, 3:4]], dim=-1)
rel = quat_mul(conj, qn)
ang = 2.0 * torch.atan2(rel[:, :3].norm(dim=-1), rel[:, 3].abs())
# per un vettore di rotazione con componenti N(0, sa^2) la norma ha valore quadratico medio sa*sqrt(3)
rms = float(ang.pow(2).mean().sqrt())
print(f"  angolo d'errore d'assetto RMS {math.degrees(rms):.3f}° (atteso {math.degrees(sa * math.sqrt(3)):.3f}°)", ok(abs(rms - sa * math.sqrt(3)) < 0.1 * sa * math.sqrt(3), "assetto rms"))
print("  due estrazioni diverse", ok(not torch.equal(env._measure()[0], qm), "rumore bianco"))

print("3) Il costo e gli errori veri non vedono il rumore; le osservazioni sì")
env0, _ = make_env()
envn, _ = make_env(NoiseConfig(sp, sa, sv, sw))
e0, en = env0.errors(), envn.errors()
print("  errors() uguali con e senza rumore", ok(torch.equal(e0.pos_err, en.pos_err) and torch.equal(e0.lin_vel_b, en.lin_vel_b) and torch.equal(e0.gravity_b, en.gravity_b), "errors veri"))
o0, on = env0.observations(), envn.observations()
d = (o0 - on).abs()
print(f"  osservazioni diverse: scarto medio {float(d.mean()):.4f}", ok(float(d.mean()) > 1e-4, "osservazioni rumorose"))
print(f"  posizione nel corpo: scarto RMS {float((o0[:, :3] - on[:, :3]).pow(2).mean().sqrt()):.4f} m (>= rumore di posizione {sp})", ok(float((o0[:, :3] - on[:, :3]).pow(2).mean().sqrt()) >= 0.9 * sp, "pos nel corpo"))

print("4) to_vel_net: la rete di velocità legge lo stato misurato oppure quello vero")


def first_vel_obs(noise):
    """Osservazioni (N,17) che la rete di velocità legge al primo passo (stato iniziale, azione nulla)."""
    e, _ = make_env(noise)
    got = []
    e.low_level = lambda obs: (got.append(obs.clone()), torch.zeros(obs.shape[0], 4, device=dev))[1]
    e.step(torch.zeros(N, 4, device=dev))
    return got[0]


clean = first_vel_obs(None)
noisy_on = first_vel_obs(NoiseConfig(sp, sa, sv, sw, to_vel_net=True))
noisy_off = first_vel_obs(NoiseConfig(sp, sa, sv, sw, to_vel_net=False))
print(f"  to_vel_net=True: scarto medio dalle letture senza rumore {float((noisy_on[:, :9] - clean[:, :9]).abs().mean()):.4f}", ok(float((noisy_on[:, :9] - clean[:, :9]).abs().mean()) > 1e-3, "to_vel_net acceso"))
print(f"  to_vel_net=False: identiche a quelle senza rumore", ok(torch.equal(noisy_off, clean), "to_vel_net spento"))

print("5) Gradiente finito attraverso le osservazioni rumorose")
envn, policy = make_env(NoiseConfig(sp, sa, sv, sw))
policy.train()
hover, w = make_hover_command(dev), CostWeights()
obs = envn.observations()
loss = 0.0
for _ in range(H):
    act = policy(obs)
    prev = envn.prev_actions
    obs = envn.step(act)
    total, _ = cost_function(envn.errors(), act, prev, hover, w)
    loss = loss + total.mean()
loss = loss / H
loss.backward()
gn = torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
print(f"  loss {float(loss.detach()):.4f}, |grad| {float(gn):.4f}", ok(bool(torch.isfinite(loss) and torch.isfinite(gn) and gn > 0), "gradiente rumoroso"))

print()
print("TUTTO OK" if not fails else f"FALLITI: {fails}")
sys.exit(1 if fails else 0)
