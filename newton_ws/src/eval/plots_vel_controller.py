"""Grafici e riepilogo numerico della valutazione del controllore di velocità: riferimento contro velocità reale
(vx, vy, vz, wz), velocità angolare, inclinazione, spinta e momenti generati dalla rete, con i limiti fisici.

Riceve la registrazione di UN drone, una riga per passo di controllo, con colonne COLS (velocità nel sistema del corpo).
"""

import matplotlib

matplotlib.use("Agg")  # nessuna finestra: salva solo i file
import matplotlib.pyplot as plt
import numpy as np

from drone_model.params import DroneParams

# Colonne di ogni riga registrata
COLS = ["t", "vx", "vy", "vz", "wx", "wy", "wz", "rvx", "rvy", "rvz", "rwz", "tilt", "a0", "a1", "a2", "a3"]
IDX = {c: i for i, c in enumerate(COLS)}
AXES = [("vx", "rvx", "vx [m/s]"), ("vy", "rvy", "vy [m/s]"), ("vz", "rvz", "vz [m/s]"), ("wz", "rwz", "wz [rad/s]")]
TAIL_S = 0.5  # finestra finale di ogni riferimento su cui si misura l'errore a regime


def _col(rec, name):
    return rec[:, IDX[name]]


def save_vel_eval_plots(rec: np.ndarray, params: DroneParams, out_png: str, out_txt: str, title: str) -> str:
    t = _col(rec, "t")
    dt = float(np.median(np.diff(t)))
    thrust = params.max_thrust * (_col(rec, "a0") + 1.0) / 2.0  # N
    m_lim = np.array([params.moment_scale_x, params.moment_scale_y, params.moment_scale_z])
    moments = np.stack([_col(rec, f"a{k}") * m_lim[k - 1] for k in (1, 2, 3)], axis=1)  # N*m

    ref = np.stack([_col(rec, r) for _, r, _ in AXES], axis=1)  # (T, 4)
    real = np.stack([_col(rec, v) for v, _, _ in AXES], axis=1)
    changed = np.where(np.abs(np.diff(ref, axis=0)).sum(axis=1) > 1e-9)[0] + 1  # indici dove cambia il riferimento
    bounds = np.concatenate([[0], changed, [len(t)]])  # segmenti a riferimento costante

    fig, axes = plt.subplots(4, 2, figsize=(15, 11), sharex=True)
    fig.suptitle(title, fontsize=13)
    for k, (v, r, label) in enumerate(AXES):
        ax = axes[k, 0]
        ax.plot(t, _col(rec, r), "--", color="#2ca02c", lw=1.6, label="riferimento")
        ax.plot(t, _col(rec, v), color="#1f77b4", lw=1.3, label="reale")
        ax.set_ylabel(label)
        ax.legend(loc="upper right", fontsize=8, ncol=2)

    ax = axes[0, 1]
    for n, c in zip("xyz", ("#1f77b4", "#2ca02c", "#d62728")):
        ax.plot(t, _col(rec, f"w{n}"), color=c, lw=1.2, label=f"ω{n}")
    ax.set_ylabel("vel. angolare nel corpo [rad/s]")
    ax.legend(loc="upper right", fontsize=8, ncol=3)

    ax = axes[1, 1]
    ax.plot(t, _col(rec, "tilt"), color="#9467bd", lw=1.3)
    ax.set_ylabel("inclinazione [gradi]")

    ax = axes[2, 1]
    ax.plot(t, thrust, color="#9467bd", lw=1.3, label="spinta")
    ax.axhline(params.weight, color="gray", ls=":", label=f"hover = peso ({params.weight:.2f} N)")
    ax.axhline(params.max_thrust, color="#d62728", ls="--", label=f"massima ({params.max_thrust:.2f} N)")
    ax.axhline(0.0, color="#d62728", ls="--", label="minima (0 N)")
    ax.set_ylabel("spinta [N]")
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    ax = axes[3, 1]
    for k, (n, c) in enumerate(zip("xyz", ("#1f77b4", "#2ca02c", "#d62728"))):
        ax.plot(t, moments[:, k], color=c, lw=1.2, label=f"M{n} (max ±{m_lim[k]:.4f})")
        ax.axhline(m_lim[k], color=c, ls="--", lw=0.8)
        ax.axhline(-m_lim[k], color=c, ls="--", lw=0.8)
    ax.set_ylabel("momenti [N·m]")
    ax.legend(loc="upper right", fontsize=8, ncol=3)

    for ax in axes.ravel():
        ax.grid(alpha=0.3)
        for i in changed:
            ax.axvline(t[i], color="gray", lw=0.6, alpha=0.5)
    axes[3, 0].set_xlabel("tempo [s]  (linee grigie = nuovo riferimento)")
    axes[3, 1].set_xlabel("tempo [s]")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    # --- Riepilogo numerico ---
    err = real - ref
    rms = np.sqrt((err**2).mean(axis=0))
    tail = max(1, int(round(TAIL_S / dt)))
    sat = lambda a: 100.0 * np.mean(np.abs(a) > 0.95)  # noqa: E731  % di passi con comando quasi al limite
    v = real[:, :3]
    w = np.stack([_col(rec, f"w{n}") for n in "xyz"], axis=1)
    lines = [
        title,
        f"passi di controllo registrati: {len(t)} ({t[-1] - t[0]:.1f} s a {1.0 / dt:.0f} Hz), {len(bounds) - 1} riferimenti",
        "",
        "errore di velocità (reale - riferimento), RMS su tutta la prova:",
        "  vx {:.3f} m/s | vy {:.3f} m/s | vz {:.3f} m/s | wz {:.3f} rad/s".format(*rms),
        "",
        f"per ogni riferimento: valore e errore medio assoluto negli ultimi {TAIL_S:.1f} s (a regime)",
    ]
    for k in range(len(bounds) - 1):
        a, b = bounds[k], bounds[k + 1]
        seg = slice(max(a, b - tail), b)
        e_tail = np.abs(err[seg]).mean(axis=0)
        r = ref[a]
        lines.append(
            f"  {k + 1}) rif ({r[0]:+.2f}, {r[1]:+.2f}, {r[2]:+.2f}) m/s, wz {r[3]:+.2f} rad/s "
            f"[{t[a]:.1f}-{t[b - 1]:.1f} s] | errore a regime: {e_tail[0]:.3f}, {e_tail[1]:.3f}, {e_tail[2]:.3f} m/s, {e_tail[3]:.3f} rad/s"
        )
    lines += [
        "",
        "                     min        max     limite",
        f"spinta [N]        {thrust.min():7.3f}   {thrust.max():7.3f}   0 .. {params.max_thrust:.3f}  (hover {params.weight:.3f})",
    ]
    for k, n in enumerate("xyz"):
        lines.append(f"momento {n} [N·m]  {moments[:, k].min():+8.4f}  {moments[:, k].max():+8.4f}   ±{m_lim[k]:.4f}")
    lines += [
        "",
        f"velocità lineare max nel corpo [m/s]: x {np.abs(v[:, 0]).max():.2f}  y {np.abs(v[:, 1]).max():.2f}  z {np.abs(v[:, 2]).max():.2f}",
        f"|ω| max nel corpo [rad/s]:             x {np.abs(w[:, 0]).max():.2f}  y {np.abs(w[:, 1]).max():.2f}  z {np.abs(w[:, 2]).max():.2f}",
        f"inclinazione massima: {_col(rec, 'tilt').max():.1f} gradi",
        "",
        "passi con comando oltre il 95% del limite:  "
        + "  ".join(f"{name} {sat(_col(rec, f'a{k}')):.1f}%" for k, name in enumerate(["spinta", "Mx", "My", "Mz"])),
    ]
    text = "\n".join(lines)
    with open(out_txt, "w") as f:
        f.write(text + "\n")
    return text
