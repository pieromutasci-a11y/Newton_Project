"""Controllore di posizione: una MLP semplice che trasforma le osservazioni in riferimenti di velocità.

Policy DETERMINISTICA: nessuna distribuzione gaussiana, nessun critic. Viene addestrata
facendo passare il gradiente del costo attraverso il simulatore differenziabile
(analytic policy gradient / BPTT), non con PPO.

Controllo a CASCATA, come in Isaac Lab: l'uscita di questa rete non è spinta e momenti, ma i 4 riferimenti di
velocità (vx, vy, vz, wz) normalizzati in [-1, 1] (l'env li moltiplica per la scala: ±1 m/s e ±1.5 rad/s). A
trasformarli in spinta e momenti ci pensa la rete di VELOCITÀ, già addestrata, che qui sta dentro come `vel_policy`:
CONGELATA (parametri senza gradiente, sempre in eval(), statistiche di normalizzazione fisse) ma attraversata dal
gradiente verso i riferimenti. Salvando il checkpoint della posizione si salva anche la rete di velocità.

Dimensioni, struttura degli strati e normalizzazione: mlp_config.yaml (rete di posizione) e
controllers/vel_controller/mlp_config.yaml (rete di velocità).
Uscita: riferimenti di velocità normalizzati in [-1, 1] = [vx, vy, vz, wz].
"""

import os

import torch
import yaml
from torch import nn

from ..vel_controller.vel_controller_mlp import VelControllerMLP

DEFAULT_MLP_YAML = os.path.join(os.path.dirname(__file__), "mlp_config.yaml")

# Attivazioni selezionabili dal file yaml
_ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "identity": nn.Identity,
}


class PosControllerMLP(nn.Module):
    def __init__(self, mlp_yaml: str = DEFAULT_MLP_YAML, vel_ckpt: str | None = None):
        """vel_ckpt: checkpoint della rete di velocità da caricare (data/vel_controller/policy/<run>/best.pt).
        Se None la rete di velocità resta con pesi casuali: va bene quando poi si carica un checkpoint della posizione,
        che contiene già anche i pesi della rete di velocità."""
        super().__init__()

        with open(mlp_yaml) as f:
            mlp = yaml.safe_load(f)["mlp"]
        self.input_dim = mlp["input_dim"]
        self.output_dim = mlp["output_dim"]
        self.normalize_obs = mlp.get("normalize_obs", False)
        self.obs_clip = mlp.get("obs_clip", 5.0)

        # Statistiche mobili delle osservazioni (come RunningStandardScaler di skrl). Sono buffer, non
        # parametri: l'ottimizzatore non le tocca, ma finiscono nello state_dict e quindi nei checkpoint.
        self.register_buffer("obs_mean", torch.zeros(self.input_dim))
        self.register_buffer("obs_var", torch.ones(self.input_dim))
        self.register_buffer("obs_count", torch.tensor(1e-4))

        # Pila di strati: Linear -> attivazione, ripetuta per ogni strato nascosto
        layers: list[nn.Module] = []
        in_dim = self.input_dim
        for hidden_dim in mlp["hidden_layers"]:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(_ACTIVATIONS[mlp["activation"]]())
            in_dim = hidden_dim

        # Strato di uscita + attivazione finale (tanh: azioni in [-1, 1])
        out_layer = nn.Linear(in_dim, self.output_dim)
        layers.append(out_layer)
        layers.append(_ACTIVATIONS[mlp["output_activation"]]())
        self.net = nn.Sequential(*layers)

        # Inizializzazione dello strato di uscita con pesi piccoli: all'inizio i riferimenti di velocità sono ~0,
        # quindi (con la rete di velocità sotto) il drone parte in hover e il training non comincia da una caduta.
        nn.init.uniform_(out_layer.weight, -1e-2, 1e-2)
        nn.init.zeros_(out_layer.bias)

        # Rete di velocità (anello interno): congelata. Sta qui dentro per due motivi: viaggia nel checkpoint della
        # posizione, e train()/eval() della posizione non possono mai riattivarla (vedi train()).
        self.vel_policy = VelControllerMLP()
        self.vel_policy.requires_grad_(False)
        self.vel_policy.eval()
        if vel_ckpt is not None:
            self.load_vel_policy(vel_ckpt)

    def load_vel_policy(self, ckpt_path: str) -> dict:
        """Carica i pesi (e le statistiche di normalizzazione) della rete di velocità e la congela. Ritorna il
        checkpoint (iterazione, pesi del costo, ... ) per poterne registrare la provenienza."""
        ckpt = torch.load(ckpt_path, map_location=self.obs_mean.device)
        self.vel_policy.load_state_dict(ckpt["policy"])
        self.vel_policy.requires_grad_(False)
        self.vel_policy.eval()
        return ckpt

    def train(self, mode: bool = True):
        """Come nn.Module.train, ma la rete di velocità resta SEMPRE in eval(): congelata, con le statistiche di
        normalizzazione che non si aggiornano mai mentre si addestra la posizione."""
        super().train(mode)
        self.vel_policy.eval()
        return self

    @torch.no_grad()
    def _update_obs_stats(self, obs: torch.Tensor) -> None:
        """Aggiorna media e varianza mobili con un nuovo batch (unione di due insiemi di campioni).

        Le righe con valori non finiti (un drone con lo stato esploso) vengono ignorate: un solo NaN
        renderebbe NaN le statistiche per sempre e quindi tutte le uscite della rete.
        """
        obs = obs[torch.isfinite(obs).all(dim=-1)]
        batch_count = obs.shape[0]
        if batch_count < 2:
            return
        batch_mean = obs.mean(dim=0)
        batch_var = obs.var(dim=0, unbiased=False)
        delta = batch_mean - self.obs_mean
        total = self.obs_count + batch_count
        m2 = (
            self.obs_var * self.obs_count
            + batch_var * batch_count
            + delta.pow(2) * self.obs_count * batch_count / total
        )
        self.obs_mean.copy_(self.obs_mean + delta * batch_count / total)
        self.obs_var.copy_(m2 / total)
        self.obs_count.copy_(total)

    def _normalize(self, obs: torch.Tensor) -> torch.Tensor:
        if self.training:  # le statistiche si aggiornano solo durante il training
            self._update_obs_stats(obs.detach())
        normalized = (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)
        return torch.clamp(normalized, -self.obs_clip, self.obs_clip)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (n_droni, input_dim) -> riferimenti di velocità normalizzati (n_droni, output_dim) in [-1, 1]."""
        if self.normalize_obs:
            obs = self._normalize(obs)
        return self.net(obs)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Come forward ma in valutazione: senza gradiente e senza aggiornare le statistiche."""
        was_training = self.training
        self.eval()
        actions = self.forward(obs)
        self.train(was_training)
        return actions
