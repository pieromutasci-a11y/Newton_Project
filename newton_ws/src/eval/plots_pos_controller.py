"""Grafici e riepilogo numerico di una valutazione, in TRE figure: (1) posizione e yaw reali contro riferimento,
(2) velocità reali contro i riferimenti di velocità comandati, (3) spinta e momenti con i limiti fisici.
Con il rumore sul feedback si aggiungono due figure: posizione e yaw reali contro misurati, velocità reali contro misurate.

Riceve la registrazione di UN drone, una riga per passo di controllo, con colonne (vedi COLS).
Le righe tutte NaN separano gli episodi (reset): le linee vengono spezzate lì.
"""

import matplotlib

matplotlib.use("Agg")  # nessuna finestra: salva solo i file
import matplotlib.pyplot as plt
import numpy as np

from drone_model.params import DroneParams

# Colonne di ogni riga registrata
# a0..a3 = azioni della rete di VELOCITÀ (spinta e momenti); rvx..rwz = riferimenti di velocità comandati dalla rete di posizione
COLS = ["t", "x", "y", "z", "xr", "yr", "zr", "yaw", "yaw_r", "vx", "vy", "vz", "wx", "wy", "wz", "a0", "a1", "a2", "a3", "rvx", "rvy", "rvz", "rwz"]
IDX = {c: i for i, c in enumerate(COLS)}
# Feedback misurato (stesso istante e stessa cadenza): posizione e yaw misurati, velocità lineare e angolare nel corpo misurate
MCOLS = ["xm", "ym", "zm", "yawm", "vxm", "vym", "vzm", "wxm", "wym", "wzm"]


def _col(rec, name):
    return rec[:, IDX[name]]


def _break_wrap(y_deg):
    """Spezza la linea dove l'angolo salta da +180 a -180 (altrimenti si disegna una riga verticale)."""
    y = y_deg.copy()
    jump = np.abs(np.diff(y)) > 180.0
    y[1:][jump] = np.nan
    return y


def save_eval_plots(rec: np.ndarray, params: DroneParams, out_prefix: str, out_txt: str, title: str, meas: np.ndarray | None = None) -> str:
    """Salva TRE figure, `<out_prefix>_posizione.png`, `<out_prefix>_velocita.png` e `<out_prefix>_spinta_momenti.png`,
    più il riepilogo numerico in `out_txt`. Con `meas` (righe di MCOLS, stessa lunghezza di `rec`) salva anche
    `<out_prefix>_feedback_posizione.png` e `<out_prefix>_feedback_velocita.png` e aggiunge al riepilogo l'errore di misura.
    Ritorna il testo del riepilogo."""
    t = _col(rec, "t")
    thrust = params.max_thrust * (_col(rec, "a0") + 1.0) / 2.0  # N
    m_lim = np.array([params.moment_scale_x, params.moment_scale_y, params.moment_scale_z])
    moments = np.stack([_col(rec, f"a{k}") * m_lim[k - 1] for k in (1, 2, 3)], axis=1)  # N*m

    # Istanti in cui cambia il riferimento (linee verticali sottili)
    ref = np.stack([_col(rec, "xr"), _col(rec, "yr"), _col(rec, "zr")], axis=1)
    changed = np.where(np.nan_to_num(np.abs(np.diff(ref, axis=0)).sum(axis=1)) > 1e-6)[0] + 1

    def new_figure(n_rows: int, fig_title: str):
        fig, axes = plt.subplots(n_rows, 1, figsize=(11, 2.6 * n_rows + 0.8), sharex=True)
        axes = np.atleast_1d(axes)
        fig.suptitle(f"{title}\n{fig_title}", fontsize=12)
        return fig, axes

    def finish(fig, axes, path):
        for ax in axes:
            ax.grid(alpha=0.3)
            for tc in t[changed]:
                ax.axvline(tc, color="gray", lw=0.6, alpha=0.5)
        axes[-1].set_xlabel("tempo [s]  (linee grigie = nuovo riferimento di posizione)")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)

    def pair(ax, real, refv, name, unit, real_label="reale", ref_label="riferimento", ref_color="#d9a400"):
        ax.plot(t, refv, "--", color=ref_color, lw=1.6, label=f"{name} {ref_label}")
        ax.plot(t, real, color="#1f77b4", lw=1.3, label=f"{name} {real_label}")
        ax.set_ylabel(f"{name} [{unit}]")
        ax.legend(loc="upper right", fontsize=8, ncol=2)

    # --- Figura 1: posizione e yaw, reali contro riferimento ---
    fig, axes = new_figure(4, "posizione e yaw: reali contro riferimento")
    pair(axes[0], _col(rec, "x"), _col(rec, "xr"), "x", "m")
    pair(axes[1], _col(rec, "y"), _col(rec, "yr"), "y", "m")
    pair(axes[2], _col(rec, "z"), _col(rec, "zr"), "z", "m")
    pair(axes[3], _break_wrap(np.rad2deg(_col(rec, "yaw"))), _break_wrap(np.rad2deg(_col(rec, "yaw_r"))), "yaw", "deg")
    axes[3].set_ylim(-190, 190)
    finish(fig, axes, out_prefix + "_posizione.png")

    # --- Figura 2: velocità reali contro i riferimenti di velocità comandati dalla rete di posizione ---
    fig, axes = new_figure(5, "velocità nel sistema del corpo: reali contro i riferimenti comandati dalla rete di posizione")
    for ax, (v, r, name, unit) in zip(axes, (("vx", "rvx", "vx", "m/s"), ("vy", "rvy", "vy", "m/s"), ("vz", "rvz", "vz", "m/s"), ("wz", "rwz", "wz", "rad/s"))):
        pair(ax, _col(rec, v), _col(rec, r), name, unit, ref_label="comando", ref_color="#2ca02c")
    ax = axes[4]  # roll e pitch rate: la rete di posizione non li comanda, li gestisce la rete di velocità
    ax.plot(t, _col(rec, "wx"), color="#1f77b4", lw=1.2, label="ωx (roll)")
    ax.plot(t, _col(rec, "wy"), color="#ff7f0e", lw=1.2, label="ωy (pitch)")
    ax.set_ylabel("ωx, ωy [rad/s]")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    finish(fig, axes, out_prefix + "_velocita.png")

    # --- Figura 3: spinta e momenti prodotti dalla rete di velocità, con i limiti fisici ---
    fig, axes = new_figure(4, "spinta e momenti (uscite della rete di velocità) con i limiti fisici")
    ax = axes[0]
    ax.plot(t, thrust, color="#9467bd", lw=1.3, label="spinta")
    ax.axhline(params.weight, color="gray", ls=":", label=f"hover = peso ({params.weight:.2f} N)")
    ax.axhline(params.max_thrust, color="#d62728", ls="--", label=f"massima ({params.max_thrust:.2f} N)")
    ax.axhline(0.0, color="#d62728", ls="--", label="minima (0 N)")
    ax.set_ylabel("spinta [N]")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    for k, (n, c) in enumerate(zip("xyz", ("#1f77b4", "#2ca02c", "#d62728"))):
        ax = axes[k + 1]
        ax.plot(t, moments[:, k], color=c, lw=1.3, label=f"momento {n}")
        ax.axhline(m_lim[k], color="#d62728", ls="--", lw=0.9, label=f"limite ±{m_lim[k]:.4f} N·m")
        ax.axhline(-m_lim[k], color="#d62728", ls="--", lw=0.9)
        ax.set_ylabel(f"M{n} [N·m]")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    finish(fig, axes, out_prefix + "_spinta_momenti.png")

    # --- Figure 4 e 5 (solo con rumore): feedback reale contro misurato ---
    if meas is not None:
        m = {c: meas[:, i] for i, c in enumerate(MCOLS)}

        def real_meas(ax, real, measured, name, unit):
            ax.plot(t, measured, color="#d62728", lw=0.8, alpha=0.8, label=f"{name} misurata")
            ax.plot(t, real, color="#1f77b4", lw=1.4, label=f"{name} reale")
            ax.set_ylabel(f"{name} [{unit}]")
            ax.legend(loc="upper right", fontsize=8, ncol=2)

        fig, axes = new_figure(4, "feedback di posizione e yaw: reali contro misurati (quelli letti dalla rete)")
        for ax, n in zip(axes[:3], "xyz"):
            real_meas(ax, _col(rec, n), m[n + "m"], n, "m")
        real_meas(axes[3], _break_wrap(np.rad2deg(_col(rec, "yaw"))), _break_wrap(np.rad2deg(m["yawm"])), "yaw", "deg")
        axes[3].set_ylim(-190, 190)
        finish(fig, axes, out_prefix + "_feedback_posizione.png")

        fig, axes = new_figure(6, "feedback di velocità nel sistema del corpo: reali contro misurate (quelle lette dalla rete)")
        for ax, (v, name, unit) in zip(axes, (("vx", "vx", "m/s"), ("vy", "vy", "m/s"), ("vz", "vz", "m/s"), ("wx", "ωx", "rad/s"), ("wy", "ωy", "rad/s"), ("wz", "ωz", "rad/s"))):
            real_meas(ax, _col(rec, v), m[v + "m"], name, unit)
        finish(fig, axes, out_prefix + "_feedback_velocita.png")

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
        "riferimenti di velocità comandati dalla rete di posizione (max in modulo, campo ±1, ±1, ±1 m/s e ±1.5 rad/s):",
        "  vx {:.2f}  vy {:.2f}  vz {:.2f} m/s | wz {:.2f} rad/s | oltre il 95% del campo: {}".format(
            *[np.nanmax(np.abs(_col(rec, c))) for c in ("rvx", "rvy", "rvz", "rwz")],
            "  ".join(
                f"{c} {100.0 * np.nanmean(np.abs(_col(rec, 'r' + c)[ok]) > 0.95 * lim):.1f}%"
                for c, lim in (("vx", 1.0), ("vy", 1.0), ("vz", 1.0), ("wz", 1.5))
            ),
        ),
        "",
        "passi con comando della rete di velocità (spinta e momenti) oltre il 95% del limite:  "
        + "  ".join(f"{name} {sat(_col(rec, f'a{k}')):.1f}%" for k, name in enumerate(["spinta", "Mx", "My", "Mz"])),
    ]
    if meas is not None:
        wrap = lambda a: np.arctan2(np.sin(a), np.cos(a))  # noqa: E731
        rms = lambda a: float(np.sqrt(np.nanmean(a[ok] ** 2)))  # noqa: E731
        lines += [
            "",
            "errore di misura (misurato - reale), RMS:",
            "  posizione x {:.4f}  y {:.4f}  z {:.4f} m | yaw {:.2f}°".format(*[rms(m[n + "m"] - _col(rec, n)) for n in "xyz"], np.rad2deg(rms(wrap(m["yawm"] - _col(rec, "yaw"))))),
            "  velocità lineare nel corpo vx {:.4f}  vy {:.4f}  vz {:.4f} m/s".format(*[rms(m[f"v{n}m"] - _col(rec, f"v{n}")) for n in "xyz"]),
            "  velocità angolare nel corpo ωx {:.4f}  ωy {:.4f}  ωz {:.4f} rad/s".format(*[rms(m[f"w{n}m"] - _col(rec, f"w{n}")) for n in "xyz"]),
        ]
    text = "\n".join(lines)
    with open(out_txt, "w") as f:
        f.write(text + "\n")
    return text
