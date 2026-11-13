"""Controllo del gradiente: autograd di Warp (wp.Tape) contro differenza finita.

Rollout di H passi con SemiImplicit, azioni per passo (array con requires_grad),
costo = distanza^2 della posizione finale dal bersaglio.
"""
import sys

sys.path.insert(0, "/workspace/project_workspace/src")

import newton
import numpy as np
import warp as wp

from drone_model.actuators import launch_apply_wrench
from drone_model.builder import add_drone
from drone_model.params import DroneParams

H = 50
DT = 0.01
params = DroneParams()
target = wp.vec3(0.5, -0.3, 1.4)

builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=wp.vec3(0.0, 0.0, -params.gravity))
handle = add_drone(builder, params, "drone_0", position=(0.0, 0.0, 1.0))
model = builder.finalize(requires_grad=True)
solver = newton.solvers.SolverSemiImplicit(model)

states = [model.state() for _ in range(H + 1)]
control = model.control()
body_ids = wp.array([handle.body], dtype=int)
actions = [wp.zeros((1, 4), dtype=float, requires_grad=True) for _ in range(H)]
loss = wp.zeros(1, dtype=float, requires_grad=True)


@wp.kernel
def final_cost(body_q: wp.array[wp.transform], body: wp.array[int], target: wp.vec3, loss: wp.array[float]):
    p = wp.transform_get_translation(body_q[body[0]])
    wp.atomic_add(loss, 0, wp.length_sq(p - target))


def forward():
    loss.zero_()
    for i in range(H):
        states[i].clear_forces()
        launch_apply_wrench(actions[i], body_ids, states[i].body_q, states[i].body_f, params)
        solver.step(states[i], states[i + 1], control, None, DT)
    wp.launch(final_cost, dim=1, inputs=(states[H].body_q, body_ids, target), outputs=(loss,))


def set_actions(base):
    for i in range(H):
        actions[i].assign(base[i].reshape(1, 4))


rng = np.random.default_rng(0)
base = np.tile(np.array([0.11, 0.05, -0.03, 0.02], dtype=np.float32), (H, 1))
base += 0.02 * rng.standard_normal(base.shape).astype(np.float32)

# --- gradiente con wp.Tape ---
set_actions(base)
tape = wp.Tape()
with tape:
    forward()
tape.backward(loss=loss)
cost0 = float(loss.numpy()[0])
grad_tape = np.stack([actions[i].grad.numpy().reshape(4) for i in range(H)])
print("costo =", cost0)
print("norma gradiente (tape) =", np.linalg.norm(grad_tape))
print("nan nel gradiente:", bool(np.isnan(grad_tape).any()))


# --- differenza finita (centrale) su alcune componenti ---
def cost_of(b):
    tape.zero()
    set_actions(b)
    forward()
    return float(loss.numpy()[0])


eps = 1e-2
print("\npasso  asse   tape            diff.finita     errore rel.")
for step in (0, 10, 25, 49):
    for axis in range(4):
        bp = base.copy(); bp[step, axis] += eps
        bm = base.copy(); bm[step, axis] -= eps
        fd = (cost_of(bp) - cost_of(bm)) / (2 * eps)
        g = grad_tape[step, axis]
        rel = abs(g - fd) / max(abs(fd), 1e-8)
        print(f"{step:4d}   {axis}   {g:+.6e}   {fd:+.6e}   {rel:.2e}")
