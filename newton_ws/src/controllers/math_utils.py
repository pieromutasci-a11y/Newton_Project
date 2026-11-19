"""Funzioni matematiche comuni ai controllori (posizione e velocità): solo tensori PyTorch."""

import torch


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Ruota il vettore v (nel mondo) nel sistema del corpo. q: (N, 4) come (x, y, z, w), v: (N, 3)."""
    q_vec, q_w = q[:, :3], q[:, 3:4]
    a = v * (2.0 * q_w**2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (q_vec * v).sum(dim=-1, keepdim=True) * 2.0
    return a - b + c


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Ruota il vettore v (nel sistema del corpo) nel mondo. q: (N, 4) come (x, y, z, w), v: (N, 3).
    È l'inverso di quat_rotate_inverse."""
    q_vec, q_w = q[:, :3], q[:, 3:4]
    a = v * (2.0 * q_w**2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (q_vec * v).sum(dim=-1, keepdim=True) * 2.0
    return a + b + c


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Angolo di imbardata (rotazione attorno a z del mondo) da un quaternione (x, y, z, w)."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Prodotto di quaternioni a * b, entrambi (N, 4) come (x, y, z, w): applica prima b, poi a."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dim=-1,
    )
