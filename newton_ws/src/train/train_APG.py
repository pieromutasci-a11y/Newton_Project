"""Training del controllore di posizione con analytic policy gradient (APG / BPTT).

Niente PPO, niente critic: il gradiente del costo passa attraverso la fisica differenziabile
(scene/train_world.py) fino ai pesi della rete. Ogni iterazione di training è una FINESTRA di
`horizon` passi (backpropagation troncata): alla fine lo stato viene staccato dal grafo e i mondi
terminati vengono resettati dall'env.

Uso (nel container):
    python /workspace/project_workspace/src/train/train_APG.py
    python /workspace/project_workspace/src/train/train_APG.py --num_envs 8192 --iterations 5000 --lr 3e-4

Output (cartella newton_ws/data):
    data/policy/<nome>/    checkpoint della policy: last.pt (ultimo) e best.pt (miglior costo medio mobile)
    data/results/<nome>/   log per tensorboard:  tensorboard --logdir /workspace/project_workspace/data/results
"""

import argparse
import dataclasses
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from controllers.pos_controller.pos_controller_env import PosControllerEnv
from controllers.pos_controller.pos_controller_mlp import PosControllerMLP
from drone_model.params import DroneParams
from scene.train_world import TrainWorld, TrainWorldConfig
from train.cost_function import CostWeights, cost_function, make_hover_action

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # tensorboard non installato: si logga solo a console
    SummaryWriter = None

# newton_ws/data (la cartella src/ sta in newton_ws/src)
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))


def parse_args():
    p = argparse.ArgumentParser(description="Training APG del controllore di posizione")
    p.add_argument("--num_envs", type=int, default=4096, help="mondi (droni) in parallelo")
    p.add_argument("--horizon", type=int, default=32, help="passi per finestra di backpropagation")
    p.add_argument("--iterations", type=int, default=2000, help="numero di iterazioni (finestre)")
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate di Adam")
    p.add_argument("--grad_clip", type=float, default=1.0, help="norma massima del gradiente")
    p.add_argument("--episode_s", type=float, default=15.0, help="durata massima di un episodio [s]")
    p.add_argument(
        "--cost", action="append", default=[], metavar="NOME=VALORE",
        help="sovrascrive un peso di CostWeights (ripetibile), es. --cost action_rate=0.5 --cost lin_vel=0.05",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_interval", type=int, default=20, help="ogni quante iterazioni stampare e loggare")
    p.add_argument("--ckpt_interval", type=int, default=200, help="ogni quante iterazioni salvare i checkpoint")
    p.add_argument("--run_name", type=str, default=None, help="nome della run (default: <data-ora>_apg)")
    p.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="cartella con le sottocartelle policy/ e results/")
    return p.parse_args()


def main():
    args = parse_args()
    dev = "cuda:0"
    torch.manual_seed(args.seed)

    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S_apg")
    ckpt_dir = os.path.join(args.data_dir, "policy", run_name)  # checkpoint della policy
    log_dir = os.path.join(args.data_dir, "results", run_name)  # log di tensorboard
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir) if SummaryWriter is not None else None

    # --- Mondo, ambiente, rete, ottimizzatore ---
    params = DroneParams()
    world = TrainWorld(params, TrainWorldConfig(num_worlds=args.num_envs, horizon=args.horizon))
    env = PosControllerEnv(world, seed=args.seed, max_episode_s=args.episode_s)
    policy = PosControllerMLP().to(dev)
    assert policy.input_dim == env.obs_dim and policy.output_dim == env.action_dim, (
        f"mlp_config.yaml ({policy.input_dim} -> {policy.output_dim}) non coincide con "
        f"l'env ({env.obs_dim} -> {env.action_dim})"
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    weights = CostWeights()
    valid = {f.name: f.type for f in dataclasses.fields(CostWeights)}
    for item in args.cost:
        name, _, value = item.partition("=")
        if name not in valid or not value:
            raise SystemExit(f"--cost {item!r} non valido. Nomi disponibili: {', '.join(valid)}")
        weights = dataclasses.replace(weights, **{name: float(value)})
    if args.cost:
        print("Pesi del costo modificati:", ", ".join(args.cost))
    hover = make_hover_action(params, dev)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Run: {run_name} | policy -> {ckpt_dir} | log -> {log_dir}")
    print(f"Mondi: {args.num_envs} | orizzonte: {args.horizon} passi | rete: {n_params:,} parametri | lr {args.lr}")

    # Medie mobili tenute su GPU (nessuna sincronizzazione ad ogni iterazione)
    ema_loss = None
    best_ema = float("inf")
    skipped = 0  # iterazioni saltate per gradiente non finito
    acc = {}  # accumulatori per i log (tensori su GPU)
    n_acc = 0
    t_log = time.perf_counter()

    for it in range(1, args.iterations + 1):
        # ------------------------------------------------------------ una finestra: forward
        obs = env.observations()
        loss = 0.0
        for _ in range(args.horizon):
            actions = policy(obs)
            prev_actions = env.prev_actions  # va letta PRIMA di step
            obs = env.step(actions)
            total, terms = cost_function(env.errors(), actions, prev_actions, hover, weights)
            loss = loss + total.mean()
            for k, v in terms.items():
                acc[f"cost/{k}"] = acc.get(f"cost/{k}", 0.0) + v.detach().mean() / args.horizon
        loss = loss / args.horizon

        # Metriche a fine finestra (senza gradiente)
        with torch.no_grad():
            e = env.errors()
            acc["metric/dist_m"] = acc.get("metric/dist_m", 0.0) + torch.sqrt(e.pos_err.pow(2).sum(-1)).mean()
            acc["metric/yaw_err_deg"] = acc.get("metric/yaw_err_deg", 0.0) + torch.rad2deg(
                torch.atan2(torch.sin(e.yaw_err), torch.cos(e.yaw_err)).abs()
            ).mean()
            acc["metric/tilt_deg"] = acc.get("metric/tilt_deg", 0.0) + torch.rad2deg(
                torch.acos(torch.clamp(-e.gravity_b[:, 2], -1.0, 1.0))
            ).mean()
            acc["metric/waypoint_idx"] = acc.get("metric/waypoint_idx", 0.0) + env.targets.wp_idx.float().mean()
            acc["metric/reset_frac"] = acc.get("metric/reset_frac", 0.0) + env.done().float().mean()

        # ------------------------------------------------------------ backward e aggiornamento dei pesi
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        if torch.isfinite(grad_norm):
            optimizer.step()
        else:  # gradiente esploso o NaN: si salta l'aggiornamento invece di rovinare i pesi
            skipped += 1

        env.end_window()  # stacca dal grafo e resetta i mondi terminati

        loss_d = loss.detach()
        ema_loss = loss_d if ema_loss is None else 0.98 * ema_loss + 0.02 * loss_d
        acc["loss"] = acc.get("loss", 0.0) + loss_d
        acc["grad_norm"] = acc.get("grad_norm", 0.0) + grad_norm.detach()
        n_acc += 1

        # ------------------------------------------------------------ log
        if it % args.log_interval == 0:
            torch.cuda.synchronize()
            now = time.perf_counter()
            dt_iter = (now - t_log) / n_acc
            means = {k: float(v) / n_acc for k, v in acc.items()}
            fps = args.num_envs * args.horizon / dt_iter
            print(
                f"[{it:5d}/{args.iterations}] loss {means['loss']:.4f} | dist {means['metric/dist_m']:.3f} m | "
                f"yaw {means['metric/yaw_err_deg']:.1f}° | tilt {means['metric/tilt_deg']:.1f}° | "
                f"wp {means['metric/waypoint_idx']:.2f} | reset {means['metric/reset_frac']:.3f} | |grad| {means['grad_norm']:.3f} | "
                f"{dt_iter * 1000:.0f} ms/it ({fps / 1e6:.2f} M passi/s)" + (f" | saltate {skipped}" if skipped else "")
            )
            if writer is not None:
                for k, v in means.items():
                    writer.add_scalar(k if "/" in k else f"train/{k}", v, it)
                writer.add_scalar("perf/ms_per_iteration", dt_iter * 1000, it)
            acc, n_acc, t_log = {}, 0, time.perf_counter()

        # ------------------------------------------------------------ checkpoint
        if it % args.ckpt_interval == 0 or it == args.iterations:
            state = {
                "iteration": it,
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema_loss": float(ema_loss),
                "args": vars(args),
            }
            torch.save(state, os.path.join(ckpt_dir, "last.pt"))
            if float(ema_loss) < best_ema:
                best_ema = float(ema_loss)
                torch.save(state, os.path.join(ckpt_dir, "best.pt"))

    if writer is not None:
        writer.close()
    print(f"Fine. Checkpoint in {ckpt_dir} (best: loss media mobile {best_ema:.4f})")


if __name__ == "__main__":
    main()
