"""Grafici e riepilogo numerico di una valutazione: posizione e yaw reali contro riferimento, velocità,
spinta e momenti generati dalla rete, con i limiti fisici per capire se siamo nei range giusti.

Riceve la registrazione di UN drone, una riga per passo di controllo, con colonne (vedi COLS).
Le righe tutte NaN separano gli episodi (reset): le linee vengono spezzate lì.
"""

import matplotlib

matplotlib.use("Agg")  # nessuna finestra: salva solo i file
import matplotlib.pyplot as plt
import numpy as np

from drone_model.params import DroneParams

# Colonne di ogni riga registrata
COLS = ["t", "x", "y", "z", "xr", "yr", "zr", "yaw", "yaw_r", "vx", "vy", "vz", "wx", "wy", "wz", "a0", "a1", "a2", "a3"]
IDX = {c: i for i, c in enumerate(COLS)}


def _col(rec, name):
    return rec[:, IDX[name]]


def _break_wrap(y_deg):
    """Spezza la linea dove l'angolo salta da +180 a -180 (altrimenti si disegna una riga verticale)."""
    y = y_deg.copy()
    jump = np.abs(np.diff(y)) > 180.0
    y[1:][jump] = np.nan
    return y


def save_eval_plots(rec: np.ndarray, params: DroneParams, out_png: str, out_txt: str, title: str) -> str:
    t = _col(rec, "t")
    thrust = params.max_thrust * (_col(rec, "a0") + 1.0) / 2.0  # N
    m_lim = np.array([params.moment_scale_x, params.moment_scale_y, params.moment_scale_z])
    moments = np.stack([_col(rec, f"a{k}") * m_lim[k - 1] for k in (1, 2, 3)], axis=1)  # N*m

    # Istanti in cui cambia il riferimento (linee verticali sottili)
    ref = np.stack([_col(rec, "xr"), _col(rec, "yr"), _col(rec, "zr")], axis=1)
    changed = np.where(np.nan_to_num(np.abs(np.diff(ref, axis=0)).sum(axis=1)) > 1e-6)[0] + 1

    fig, axes = plt.subplots(4, 2, figsize=(15, 11), sharex=True)
    fig.suptitle(title, fontsize=13)

    def pair(ax, real, refv, name, unit):
        ax.plot(t, refv, "--", color="#d9a400", lw=1.6, label=f"{name} riferimento")
        ax.plot(t, real, color="#1f77b4", lw=1.3, label=f"{name} reale")
        ax.set_ylabel(f"{name} [{unit}]")
        ax.legend(loc="upper right", fontsize=8, ncol=2)

    pair(axes[0, 0], _col(rec, "x"), _col(rec, "xr"), "x", "m")
    pair(axes[1, 0], _col(rec, "y"), _col(rec, "yr"), "y", "m")
    pair(axes[2, 0], _col(rec, "z"), _col(rec, "zr"), "z", "m")
    pair(axes[3, 0], _break_wrap(np.rad2deg(_col(rec, "yaw"))), _break_wrap(np.rad2deg(_col(rec, "yaw_r"))), "yaw", "deg")
    axes[3, 0].set_ylim(-190, 190)

    ax = axes[0, 1]
    for n, c in zip("xyz", ("#1f77b4", "#2ca02c", "#d62728")):
        ax.plot(t, _col(rec, f"v{n}"), color=c, lw=1.2, label=f"v{n}")
    ax.set_ylabel("vel. lineare nel corpo [m/s]")
    ax.legend(loc="upper right", fontsize=8, ncol=3)

    ax = axes[1, 1]
    for n, c in zip("xyz", ("#1f77b4", "#2ca02c", "#d62728")):
        ax.plot(t, _col(rec, f"w{n}"), color=c, lw=1.2, label=f"ω{n}")
    ax.set_ylabel("vel. angolare nel corpo [rad/s]")
    ax.legend(loc="upper right", fontsize=8, ncol=3)

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
        for tc in t[changed]:
            ax.axvline(tc, color="gray", lw=0.6, alpha=0.5)
    axes[3, 0].set_xlabel("tempo [s]  (linee grigie = nuovo riferimento)")
    axes[3, 1].set_xlabel("tempo [s]")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    # --- Riepilogo numerico: i valori da confrontare con i limiti ---
    ok = np.isfinite(t)
    sat = lambda a: 100.0 * np.nanmean(np.abs(a[ok]) > 0.95)  # % di passi con comando quasi al limite
    v = np.stack([_col(rec, f"v{n}") for n in "xyz"], axis=1)
    w = np.stack([_col(rec, f"w{n}") for n in "xyz"], axis=1)
    lines = [
        title,
        f"passi di controllo registrati: {int(ok.sum())} ({np.nanmax(t) - np.nanmin(t):.1f} s)",
        "",
        "                     min        max     limite",
        f"spinta [N]        {np.nanmin(thrust):7.3f}   {np.nanmax(thrust):7.3f}   0 .. {params.max_thrust:.3f}  (hover {params.weight:.3f})",
    ]
    for k, n in enumerate("xyz"):
        lines.append(f"momento {n} [N·m]  {np.nanmin(moments[:, k]):+8.4f}  {np.nanmax(moments[:, k]):+8.4f}   ±{m_lim[k]:.4f}")
    lines += [
        "",
        f"|v lineare| max nel corpo [m/s]:  x {np.nanmax(np.abs(v[:, 0])):.2f}  y {np.nanmax(np.abs(v[:, 1])):.2f}  z {np.nanmax(np.abs(v[:, 2])):.2f}   (norma max {np.nanmax(np.linalg.norm(v, axis=1)):.2f})",
        f"|ω| max nel corpo [rad/s]:       x {np.nanmax(np.abs(w[:, 0])):.2f}  y {np.nanmax(np.abs(w[:, 1])):.2f}  z {np.nanmax(np.abs(w[:, 2])):.2f}",
        "",
        "passi con comando oltre il 95% del limite:  "
        + "  ".join(f"{name} {sat(_col(rec, f'a{k}')):.1f}%" for k, name in enumerate(["spinta", "Mx", "My", "Mz"])),
    ]
    text = "\n".join(lines)
    with open(out_txt, "w") as f:
        f.write(text + "\n")
    return text
