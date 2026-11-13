"""Controllore di posizione: una MLP semplice che trasforma le osservazioni in azioni.

Policy DETERMINISTICA: nessuna distribuzione gaussiana, nessun critic. Viene addestrata
facendo passare il gradiente del costo attraverso il simulatore differenziabile
(analytic policy gradient / BPTT), non con PPO.

Dimensioni, struttura degli strati e normalizzazione: mlp_config.yaml.
Uscita: azioni in [-1, 1] = [spinta, momento x, momento y, momento z].
"""

import os

import torch
import yaml
from torch import nn

DEFAULT_MLP_YAML = os.path.join(os.path.dirname(__file__), "mlp_config.yaml")

# Attivazioni selezionabili dal file yaml
_ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "identity": nn.Identity,
}


class PosControllerMLP(nn.Module):
    def __init__(self, mlp_yaml: str = DEFAULT_MLP_YAML):
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

        # Inizializzazione dello strato di uscita con pesi piccoli: all'inizio le azioni sono ~0.
        # Con azione di spinta = 0 la spinta vale 0.5 * max_thrust = 0.9 * peso (thrust_to_weight = 1.8):
        # il drone parte quasi in hover e il training non comincia da una caduta libera.
        nn.init.uniform_(out_layer.weight, -1e-2, 1e-2)
        nn.init.zeros_(out_layer.bias)

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
        """obs: (n_droni, input_dim) -> azioni: (n_droni, output_dim)."""
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
