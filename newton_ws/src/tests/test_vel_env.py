"""Prove del controllore di velocità: riferimenti, osservazioni, sistema del corpo, reset, gradiente.

    python /workspace/project_workspace/src/tests/test_vel_env.py
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from controllers.math_utils import quat_rotate, quat_rotate_inverse
from controllers.vel_controller.ref_handler import RefHandler
from controllers.vel_controller.vel_controller_mlp import VelControllerMLP
from controllers.vel_controller.vel_controller_env import VelControllerEnv
from drone_model.params import DroneParams
from scene.train_world import TrainWorld, TrainWorldConfig

dev = "cuda:0"
ok = lambda c: "OK" if c else "ERRORE"  # noqa: E731

# ---------------------------------------------------------------- 1. riferimenti: campo, modalità, permanenze
print("== 1. Riferimenti di velocità (20000 campioni, dt = 0.02 s)")
rh = RefHandler(20000, 0.02, dev, seed=3)
rng = rh.vel_range
ratio = (rh.ref / rng).abs()
print(f"  campo (vx, vy, vz, wz) = {[float(x) for x in rng]}: massimo |rif|/campo = {float(ratio.max()):.3f} (atteso <= 1): {ok(ratio.max() <= 1.0 + 1e-6)}")
frac = torch.bincount(rh.mode, minlength=5).float() / rh.n
print("  frazione delle 5 modalità:", [round(float(f), 3) for f in frac], "(attese 0.35 0.15 0.15 0.30 0.05)")
nonzero = (rh.ref != 0).sum(dim=-1)
m = rh.mode
print(f"  modalità 2 (un asse): righe con esattamente 1 valore non nullo: {ok(bool((nonzero[m == 2] == 1).all()))}")
print(f"  modalità 1 (sottoinsieme): almeno 1 valore non nullo: {ok(bool((nonzero[m == 1] >= 1).all()))}")
print(f"  modalità 3 (ampiezza minima): |rif|/campo >= 0.6 su tutti gli assi: {ok(bool((ratio[m == 3] >= 0.6 - 1e-6).all()))}")
print(f"  modalità 4 (hover): |rif|/campo <= 0.15: {ok(bool((ratio[m == 4] <= 0.15 + 1e-6).all()))} | zero esatto: {float((nonzero[m == 4] == 0).float().mean()):.2f} (atteso 0.50)")
hold_s = rh._hold_left.float() * rh.dt
print(f"  tempo di permanenza: min {hold_s.min():.2f} s  max {hold_s.max():.2f} s  (atteso 1.0-2.5 s)")

# ---------------------------------------------------------------- 2. più riferimenti per episodio
print("\n== 2. Riferimenti per episodio (10 s = 500 passi a 50 Hz)")
rh = RefHandler(512, 0.02, dev, seed=4)
changes = torch.zeros(512, device=dev)
prev = rh.ref.clone()
for _ in range(500):
    rh.step()
    changes += (rh.ref != prev).any(dim=-1).float()
    prev = rh.ref.clone()
per_ep = changes + 1  # + il riferimento iniziale
print(f"  riferimenti per episodio: min {int(per_ep.min())}  media {float(per_ep.mean()):.1f}  max {int(per_ep.max())}  (attesi circa 5-11)")

# ---------------------------------------------------------------- 3. osservazioni allo spawn
N = 4096
world = TrainWorld(DroneParams(), TrainWorldConfig(num_worlds=N, horizon=32, decimation=2))
env = VelControllerEnv(world, seed=1)
print(f"\n== 3. Osservazioni allo spawn (drone dritto e fermo); ctrl_dt = {world.ctrl_dt} s")
obs = env.observations()
print("  shape:", tuple(obs.shape), "(atteso (N, 17))", ok(obs.shape == (N, 17) and env.obs_dim == 17))
print("  velocità lineare e angolare nel corpo (mondo 0):", [round(float(v), 3) for v in obs[0, :6]], "(atteso tutto 0)")
print("  gravità proiettata (mondo 0):", [round(float(v), 3) for v in obs[0, 6:9]], "(atteso 0 0 -1)")
print("  azione precedente nulla:", ok(bool((obs[:, 9:13] == 0).all())), "| riferimento = quello del handler:", ok(bool(torch.equal(obs[:, 13:17], env.refs.ref))))
e = env.errors()
print("  errore = riferimento - (vx, vy, vz, wz):", ok(bool(torch.allclose(e.vel_err, env.refs.step_ref - torch.cat([e.lin_vel_b, e.ang_vel_b[:, 2:3]], -1)))))

# ---------------------------------------------------------------- 4. riferimento scaduto
print("\n== 4. Riferimento scaduto: errors() (costo) usa il vecchio, observations() usa il nuovo")
env.refs._hold_left[:] = 1  # tutti i riferimenti scadono al prossimo passo
old = env.refs.ref.clone()
obs = env.step(torch.zeros(N, 4, device=dev))
e = env.errors()
vel = torch.cat([e.lin_vel_b, e.ang_vel_b[:, 2:3]], -1).detach()
print("  errors() calcolato con il riferimento VECCHIO:", ok(bool(torch.allclose(e.vel_err.detach(), old - vel, atol=1e-5))))
print("  observations() contiene il riferimento NUOVO:", ok(bool(torch.equal(obs[:, 13:17], env.refs.ref))), "| riferimento cambiato:", bool((env.refs.ref != old).any()))

# ---------------------------------------------------------------- 5. sistema del corpo
print("\n== 5. Sistema del corpo: drone ruotato di 90 gradi in yaw")
env.end_window()  # a inizio finestra: si può scrivere sullo stato
s = math.sin(math.pi / 4)
world.q[:, 3:] = torch.tensor([0.0, 0.0, s, s], device=dev)  # yaw 90 gradi (x, y, z, w)
world.qd[:, :3] = torch.tensor([1.0, 0.0, 0.0], device=dev)  # velocità +1 m/s lungo x del MONDO
world.qd[:, 3:6] = torch.tensor([0.0, 0.0, 0.5], device=dev)  # velocità angolare 0.5 rad/s attorno a z
obs = env.observations()
print("  velocità lineare nel corpo (mondo 0):", [round(float(v), 3) for v in obs[0, :3]], "(atteso 0 -1 0: l'asse x del mondo è l'asse -y del corpo)")
print("  velocità angolare nel corpo (mondo 0):", [round(float(v), 3) for v in obs[0, 3:6]], "(atteso 0 0 0.5)")

# ---------------------------------------------------------------- 6. terminazioni e reset
print("\n== 6. Terminazioni e reset")
env.end_window()
env.end_window()
sq = math.sin(math.pi / 4)
world.q[0, 3:] = torch.tensor([sq, 0.0, 0.0, sq], device=dev)  # 90 gradi di roll: troppo inclinato
world.q[1, 0] = float("nan")  # stato esploso
world.q[2, 1] = 500.0  # drone perso
world.qd[3, 3:6] = torch.tensor([0.0, 40.0, 0.0], device=dev)  # rotazione troppo veloce
env._steps_alive[4] = 10**6  # episodio finito
d = env.done()
print("  done() per inclinazione, NaN, drone perso, rotazione veloce, timeout (mondi 0-4):", [bool(x) for x in d[:5]], "| altri mondi flaggati:", int(d[5:].sum()))
env.end_window()
print("  dopo end_window: stato finito:", ok(bool(torch.isfinite(world.q).all() and torch.isfinite(world.qd).all())),
      "| nessun mondo da resettare:", ok(not bool(env.done().any())), "| passi vissuti mondo 4:", int(env._steps_alive[4]))

# ---------------------------------------------------------------- 7. un giro completo con gradiente
print("\n== 7. Una iterazione: rete -> env -> fisica (decimation 2) -> errore di velocità -> backward")
policy = torch.nn.Sequential(torch.nn.Linear(env.obs_dim, 64), torch.nn.ELU(), torch.nn.Linear(64, env.action_dim), torch.nn.Tanh()).to(dev)


def one_iteration():
    obs = env.observations()
    loss = 0.0
    for _ in range(world.cfg.horizon):
        a = policy(obs)
        obs = env.step(a)
        loss = loss + env.errors().vel_err.pow(2).sum(-1).mean()
    loss = loss / world.cfg.horizon
    policy.zero_grad()
    loss.backward()
    env.end_window()
    return loss


loss = one_iteration()
g = torch.cat([p.grad.flatten() for p in policy.parameters()])
print(f"  loss = {float(loss.detach()):.4f} | gradienti presenti e finiti: {ok(bool(torch.isfinite(g).all() and g.abs().sum() > 0))} | norma = {float(g.norm()):.3f}")
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(5):
    one_iteration()
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / 5
print(f"  tempo per iterazione (N={N}, H={world.cfg.horizon}): {dt * 1000:.1f} ms  ->  {N * world.cfg.horizon / dt:,.0f} passi-drone/s")

# ---------------------------------------------------------------- 8. rete e normalizzazione delle osservazioni
print("\n== 8. Rete del controllore di velocità e normalizzazione delle osservazioni")
net = VelControllerMLP().to(dev)
print(f"  dimensioni di mlp_config.yaml ({net.input_dim} -> {net.output_dim}) coerenti con l'env ({env.obs_dim} -> {env.action_dim}):", ok(net.input_dim == env.obs_dim and net.output_dim == env.action_dim))
x = torch.randn(4096, net.input_dim, device=dev) * 3.0 + 2.0
net.train()
out = net(x).detach()
print(f"  uscite in [-1, 1]: {ok(bool(out.abs().max() <= 1.0))} | all'inizio quasi nulle (|a| max {float(out.abs().max()):.3f}): vicino all'hover")
for _ in range(20):
    net(x)
print(f"  dopo 20 batch in training: media stimata {float(net.obs_mean.mean()):.3f} (vera 2), dev. std stimata {float(net.obs_var.sqrt().mean()):.3f} (vera 3)")
before = net.obs_mean.clone(), net.obs_var.clone(), net.obs_count.clone()
net.act(x)
print("  act() (valutazione): le statistiche NON cambiano:", ok(all(torch.equal(a, b) for a, b in zip(before, (net.obs_mean, net.obs_var, net.obs_count)))))
bad = x.clone(); bad[3, 5] = float("nan"); bad[7, 0] = float("inf")
net.train(); net(bad)
print("  un batch con NaN/inf non rovina le statistiche:", ok(bool(torch.isfinite(net.obs_mean).all() and torch.isfinite(net.obs_var).all())))
net2 = VelControllerMLP().to(dev)
net2.load_state_dict(net.state_dict())
net.eval(); net2.eval()
print("  le statistiche stanno nello state_dict (checkpoint): ", ok(bool(torch.equal(net(x), net2(x)))), "| stesse uscite dopo il caricamento")

# Uso congelato (come sotto il controllore di posizione): niente aggiornamenti, ma il gradiente arriva agli ingressi
net.eval().requires_grad_(False)
stats = net.obs_mean.clone()
xi = x[:8].clone().requires_grad_(True)
net(xi).pow(2).sum().backward()
print(f"  congelata: parametri senza gradiente: {ok(all(p.grad is None for p in net.parameters()))} | gradiente sugli ingressi finito e non nullo: {ok(bool(torch.isfinite(xi.grad).all() and xi.grad.abs().sum() > 0))} | statistiche invariate: {ok(bool(torch.equal(stats, net.obs_mean)))}")

# ---------------------------------------------------------------- 9. rotazione corpo -> mondo e contatore dei riferimenti
print("\n== 9. quat_rotate (corpo -> mondo) e contatore dei riferimenti campionati")
qq = torch.randn(1000, 4, device=dev); qq = qq / qq.norm(dim=-1, keepdim=True)
vv = torch.randn(1000, 3, device=dev)
print("  quat_rotate(q, quat_rotate_inverse(q, v)) = v:", ok(bool(torch.allclose(quat_rotate(qq, quat_rotate_inverse(qq, vv)), vv, atol=1e-5))))
s90 = math.sin(math.pi / 4)
q90 = torch.tensor([[0.0, 0.0, s90, s90]], device=dev)
print("  yaw 90 gradi: l'asse x del corpo (1,0,0) nel mondo è", [round(float(x), 3) for x in quat_rotate(q90, torch.tensor([[1.0, 0.0, 0.0]], device=dev))[0]], "(atteso 0 1 0)")
rh = RefHandler(8, 0.02, dev, seed=5)
print("  riferimenti campionati all'inizio: tutti 1:", ok(bool((rh.n_samples == 1).all())))
for _ in range(500):
    rh.step()
print(f"  dopo 10 s: da {int(rh.n_samples.min())} a {int(rh.n_samples.max())} riferimenti (attesi 5-11)")
rh.reset(torch.tensor([0, 1], device=dev))
print("  dopo reset dei mondi 0 e 1: contatore a 1 solo lì:", ok(bool((rh.n_samples[:2] == 1).all() and (rh.n_samples[2:] > 1).all())))

# ---------------------------------------------------------------- 10. riferimento imposto a mano (valutazione)
print("\n== 10. RefHandler.set_constant: riferimento fisso che non scade mai")
rh = RefHandler(4, 0.02, dev, seed=6)
rh.set_constant([0.5, -0.3, 0.0, 1.0])
for _ in range(3000):  # 60 s: molto più della permanenza massima di 2.5 s
    rh.step()
want = torch.tensor([0.5, -0.3, 0.0, 1.0], device=dev)
print("  dopo 60 s il riferimento è ancora quello imposto, in tutti i mondi:", ok(bool((rh.ref == want).all() and (rh.step_ref == want).all())), "| contatore a 1:", ok(bool((rh.n_samples == 1).all())))
rh.reset(torch.tensor([0], device=dev))
print("  un reset ricampiona a caso quel mondo:", ok(bool((rh.ref[0] != want).any() and (rh.ref[1:] == want).all())))
