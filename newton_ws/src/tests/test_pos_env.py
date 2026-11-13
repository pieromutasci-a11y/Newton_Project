"""Prove del controllore di posizione: riferimenti, osservazioni, errori, reset, e una iterazione completa.

Da lanciare nel container:
    python /workspace/project_workspace/src/tests/test_pos_env.py
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from controllers.pos_controller.pos_controller_env import PosControllerEnv, quat_rotate_inverse
from controllers.pos_controller.pos_controller_mlp import PosControllerMLP
from drone_model.params import DroneParams
from scene.train_world import TrainWorld, TrainWorldConfig

dev = "cuda:0"

N, H = 4096, 32
world = TrainWorld(DroneParams(), TrainWorldConfig(num_worlds=N, horizon=H))
env = PosControllerEnv(world, seed=1)

# ---------------------------------------------------------------- 1. riferimenti casuali
print("== 1. Riferimenti casuali (N =", N, ")")
tp, ty = env.targets.target_pos, env.targets.target_yaw
for i, name in enumerate("xyz"):
    print(f"  target {name}: min {tp[:, i].min():+.3f}  max {tp[:, i].max():+.3f}  media {tp[:, i].mean():+.3f}")
print(f"  target yaw: min {ty.min():+.3f}  max {ty.max():+.3f}  (atteso in [{-math.pi:.3f}, {math.pi:.3f}])")
print(f"  coda di riferimenti: {tuple(env.targets.wp_pos.shape)} (atteso (N, 3, 3)) | indice attivo: tutti 0: {bool((env.targets.wp_idx == 0).all())}")
print(f"  attivo = primo della coda: {bool(torch.equal(env.targets.target_pos, env.targets.wp_pos[:, 0]))}")
sp = world.q[:, :3]
print(f"  spawn z: min {sp[:, 2].min():.3f}  max {sp[:, 2].max():.3f}")

# ---------------------------------------------------------------- 2. osservazioni e errori allo spawn
print("\n== 2. Osservazioni ed errori allo spawn (drone dritto e fermo)")
obs = env.observations()
print("  shape:", tuple(obs.shape), "(atteso (N,", env.obs_dim, "))")
e = env.errors()
print("  gravità proiettata (mondo 0):", [round(float(v), 3) for v in e.gravity_b[0]], "(atteso 0 0 -1)")
print(
    "  velocità lineare e angolare nel corpo (mondo 0):",
    [round(float(v), 3) for v in torch.cat([e.lin_vel_b[0], e.ang_vel_b[0]])],
    "(atteso tutto 0)",
)
print("  errore di posizione = riferimento - posizione:", bool(torch.allclose(e.pos_err, env.targets.target_pos - world.pos)))

# ---------------------------------------------------------------- 3. frame di qd con yaw = 90 gradi
print("\n== 3. Sistema di riferimento di qd: drone ruotato di 90 gradi in yaw, momento di rollio")
w2 = TrainWorld(DroneParams(), TrainWorldConfig(num_worlds=2, horizon=10))
ids = torch.arange(2, device=dev)
w2.reset(ids, torch.tensor([[0.0, 0.0, 1.0]] * 2, device=dev))
s = math.sin(math.pi / 4)
w2.q[:, 3:] = torch.tensor([0.0, 0.0, s, s], device=dev)  # yaw 90 gradi (x, y, z, w)
a = torch.tensor([0.11, 1.0, 0.0, 0.0], device=dev).repeat(2, 1)  # momento attorno a x del CORPO
for _ in range(10):
    w2.step(a)
print("  qd angolare (mondo 0):", [round(float(v), 4) for v in w2.qd[0, 3:]], "(atteso componente 4 != 0: nel mondo)")
del w2

# ---------------------------------------------------------------- 4. iterazione completa (senza ottimizzatore)
print("\n== 4. Una iterazione completa: rete -> env -> fisica -> errori -> backward")
policy = PosControllerMLP().to(dev)
assert policy.input_dim == env.obs_dim and policy.output_dim == env.action_dim, "dimensioni di mlp_config.yaml diverse da quelle dell'env"
print("  dimensioni rete/env coerenti:", policy.input_dim, "->", policy.output_dim)


def temp_cost(env, actions, prev_actions):
    """Costo PROVVISORIO, solo per questa prova (il costo vero andrà in train/cost_function.py)."""
    err = env.errors()
    return err.pos_err.pow(2).sum(-1) + 0.01 * (actions - prev_actions).pow(2).sum(-1)


def one_iteration():
    obs = env.observations()
    loss = 0.0
    for _ in range(H):
        actions = policy(obs)
        prev_actions = env.prev_actions
        obs = env.step(actions)
        loss = loss + temp_cost(env, actions, prev_actions).mean()
    loss = loss / H
    loss.backward()
    return loss


loss = one_iteration()
grads = [p.grad for p in policy.parameters()]
print(f"  loss iniziale = {float(loss.detach()):.4f}")
print("  gradienti presenti:", all(g is not None for g in grads), "| finiti:", all(torch.isfinite(g).all() for g in grads))
print(f"  norma del gradiente = {torch.sqrt(sum((g**2).sum() for g in grads)):.4f}")
env.end_window()
policy.zero_grad()

one_iteration()  # riscaldamento
env.end_window()
policy.zero_grad()
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    one_iteration()
    env.end_window()
    policy.zero_grad()
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / 5
print(f"  tempo per iterazione (N={N}, H={H}): {dt * 1000:.1f} ms  ->  {N * H / dt:,.0f} passi-drone/s")

# ---------------------------------------------------------------- 5. reset dei mondi falliti
print("\n== 5. Reset di un mondo con drone ribaltato")
env.end_window()  # stato pulito a inizio finestra
s45 = math.sin(math.pi / 4)
world.q[0, 3:] = torch.tensor([s45, 0.0, 0.0, s45], device=dev)  # rotazione di 90 gradi attorno a x: drone di lato
print("  done() prima del reset, mondo 0:", bool(env.done()[0]), "| altri mondi flaggati:", int(env.done()[1:].sum()))
env.end_window()
print("  dopo end_window: orientamento mondo 0 =", [round(float(v), 3) for v in world.q[0, 3:]], "(atteso 0 0 0 1)")
print("  done() dopo, mondo 0:", bool(env.done()[0]), "| passi vissuti:", int(env._steps_alive[0]))

# ---------------------------------------------------------------- 6. il costo usa il riferimento attivo durante il passo
print("\n== 6. Avanzamento: arrivo + permanenza -> riferimento successivo; errors() (costo) usa il vecchio, observations() il nuovo")
env.end_window()
env.targets.wp_yaw[:] = 0.0  # yaw di riferimento 0 = yaw dello spawn: così le soglie di arrivo dipendono solo dalla posizione
env.targets._refresh()
ids_all = torch.arange(N, device=dev)
world.reset(ids_all, env.targets.target_pos)  # droni esattamente sul riferimento, dritti e fermi
env._hold_timer.zero_()
hover = torch.tensor([0.111, 0.0, 0.0, 0.0], device=dev).repeat(N, 1)
old_target = env.targets.target_pos.clone()
n_hold = int(round(env._hold_time_s / env.dt))
steps = 0
with torch.no_grad():
    for _ in range(world.cfg.horizon):  # una sola finestra
        obs = env.step(hover)
        steps += 1
        if int(env.targets.wp_idx.max()) > 0:  # primo passo in cui scatta l'avanzamento
            break
    e = env.errors()
    pos = world.pos.detach()
print(f"  avanzamento al passo {steps} ({steps * env.dt:.2f} s; atteso {n_hold}-{n_hold + 1}): {n_hold <= steps <= n_hold + 1}")
print(f"  indice attivo dopo: {int(env.targets.wp_idx.min())}..{int(env.targets.wp_idx.max())} (atteso 1 per tutti)")
print("  errors() calcolato con il riferimento VECCHIO:", bool(torch.allclose(e.pos_err, old_target - pos, atol=1e-5)))
print(
    "  observations() calcolato con il riferimento NUOVO:",
    bool(torch.allclose(obs[:, :3], quat_rotate_inverse(world.q[:, 3:7].detach(), env.targets.target_pos - pos), atol=1e-5)),
    "| riferimento cambiato:",
    bool((env.targets.target_pos != old_target).all()),
)
# Sull'ultimo riferimento non si avanza più
env.targets.wp_idx[:] = env.targets.n_waypoints - 1
env.targets._refresh()
env._hold_timer[:] = 10.0
env.step(hover)
print("  sull'ultimo riferimento l'indice non supera l'ultimo:", bool((env.targets.wp_idx == env.targets.n_waypoints - 1).all()))

# ---------------------------------------------------------------- 7. l'errore di posizione nelle osservazioni è nel sistema del corpo
print("\n== 7. Errore di posizione nel sistema del corpo (drone ruotato di 90 gradi in yaw)")
env.end_window()  # a inizio finestra: si può scrivere sullo stato
s45 = math.sin(math.pi / 4)
world.q[:, 3:] = torch.tensor([0.0, 0.0, s45, s45], device=dev)  # yaw 90 gradi per tutti i droni
env.targets.target_pos = world.q[:, :3].detach().clone() + torch.tensor([1.0, 0.0, 0.0], device=dev)  # bersaglio a +1 m sull'asse x del MONDO
obs = env.observations()
print("  errore nel mondo (1, 0, 0) -> nel corpo (mondo 0):", [round(float(v), 3) for v in obs[0, :3]],
      "(atteso 0 -1 0: con yaw 90 gradi l'asse x del mondo è l'asse -y del corpo)")

# ---------------------------------------------------------------- 8. normalizzazione delle osservazioni
print("\n== 8. Normalizzazione delle osservazioni (media e varianza mobili)")
env.end_window()
net = PosControllerMLP().to(dev)
net.train()
for _ in range(20):
    x = torch.randn(N, env.obs_dim, device=dev) * 3.0 + 2.0  # osservazioni con media 2 e deviazione standard 3
    net(x)
print(f"  dopo 20 batch in training: media stimata {float(net.obs_mean.mean()):.3f} (vera 2), "
      f"dev. std stimata {float(net.obs_var.sqrt().mean()):.3f} (vera 3)")
count_before = float(net.obs_count)
net.act(torch.randn(N, env.obs_dim, device=dev) * 10.0)
print("  in valutazione (act) le statistiche NON cambiano:", float(net.obs_count) == count_before)
sd = net.state_dict()
print("  statistiche nello state_dict (quindi nei checkpoint):", all(k in sd for k in ("obs_mean", "obs_var", "obs_count")))

# ---------------------------------------------------------------- 9. robustezza a stati non finiti
print("\n== 9. Un drone con stato NaN non avvelena il normalizzatore e viene resettato")
net = PosControllerMLP().to(dev)
net.train()
batch = torch.randn(256, net.input_dim, device=dev)
batch[3, 5] = float("nan")
batch[7, 0] = float("inf")
net(batch)
print("  statistiche finite dopo un batch con NaN/inf:", bool(torch.isfinite(net.obs_mean).all() and torch.isfinite(net.obs_var).all()))
env.end_window()
world.q[5, 0] = float("nan")
world.q[6, 1] = 500.0
d = env.done()
print("  done() segnala NaN (mondo 5) e drone perso (mondo 6):", bool(d[5]), bool(d[6]), "| altri mondi:", int(d[7:].sum()))
env.end_window()
print("  dopo end_window lo stato è finito:", bool(torch.isfinite(world.q).all()))
