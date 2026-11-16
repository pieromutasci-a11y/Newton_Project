"""Generazione dei riferimenti di velocità (vx, vy, vz, wz) nel sistema del CORPO, come in Isaac Lab
(vel_controller_env._sample_new_targets), ma senza curriculum.

Ogni mondo ha un riferimento attivo e un tempo di permanenza: quando scade se ne campiona uno nuovo, quindi un
episodio contiene più riferimenti (episodio 10 s, permanenza 1-2.5 s: da 4 a 10). Non c'è un "arrivo" da
riconoscere (come per la posizione): il riferimento cambia solo a tempo.

Le 5 modalità di campionamento sono quelle di Isaac Lab, con le probabilità del suo ultimo livello di curriculum:
  0. tutti gli assi, valori uniformi su tutto il campo
  1. un sottoinsieme casuale di assi (almeno uno); gli altri a zero
  2. un solo asse, scelto a caso; gli altri a zero
  3. tutti gli assi, con ampiezza minima garantita (niente valori troppo piccoli) e segno casuale
  4. hover: tutto a zero (metà dei casi) oppure una piccola correzione casuale

Si occupa SOLO dei riferimenti: non conosce il mondo fisico, né i reset, né la rete.
"""

import torch


class RefHandler:
    def __init__(
        self,
        num_worlds: int,
        dt: float,
        device,
        *,
        seed: int = 0,
        # Campo dei riferimenti (Isaac Lab: max_lin_vel_xy = max_lin_vel_z = 1.0 m/s, max_ang_vel_z = 1.5 rad/s)
        vel_range: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.5),
        # Probabilità delle 5 modalità (Isaac Lab, ultimo livello di curriculum)
        mode_probs: tuple[float, ...] = (0.35, 0.15, 0.15, 0.30, 0.05),
        min_amplitude_frac: float = 0.6,  # modalità 3: ampiezza minima, frazione del campo (Isaac Lab: 0.6 all'ultimo livello)
        hover_frac: float = 0.15,  # modalità 4: ampiezza della piccola correzione, frazione del campo
        hold_time_range_s: tuple[float, float] = (1.0, 2.5),  # s di permanenza su un riferimento (Isaac Lab, ultimo livello)
    ):
        self.n = num_worlds
        self.dt = dt
        self.device = device
        self.vel_range = torch.tensor(vel_range, device=device)  # (4,)
        self.mode_probs = torch.tensor(mode_probs, device=device)
        assert self.mode_probs.numel() == 5, "servono le probabilità delle 5 modalità"
        self.min_amplitude_frac = min_amplitude_frac
        self.hover_frac = hover_frac
        self.hold_time_range_s = hold_time_range_s

        # Generatore casuale con seme: esperimenti ripetibili
        self._gen = torch.Generator(device=device)
        self._gen.manual_seed(seed)

        # Riferimento attivo (N, 4) = (vx, vy, vz, wz) nel sistema del corpo, e modalità con cui è stato campionato
        self.ref = torch.zeros(num_worlds, 4, device=device)
        self.mode = torch.zeros(num_worlds, dtype=torch.long, device=device)
        self._hold_left = torch.zeros(num_worlds, dtype=torch.long, device=device)  # passi rimasti sul riferimento
        # Quanti riferimenti sono stati campionati in ogni mondo dall'ultimo reset (il primo conta 1): serve alla
        # valutazione per fermarsi dopo un numero fisso di riferimenti
        self.n_samples = torch.zeros(num_worlds, dtype=torch.long, device=device)

        self.sample(torch.arange(num_worlds, device=device))

        # Riferimento che era attivo DURANTE l'ultimo passo. Serve al costo: a fine passo il riferimento
        # può già essere stato sostituito, ma l'errore di quel passo va calcolato con il vecchio.
        self.step_ref = self.ref

    # ------------------------------------------------------------------ campionamento
    def _uniform(self, low: float, high: float, shape) -> torch.Tensor:
        return low + (high - low) * torch.rand(shape, device=self.device, generator=self._gen)

    def sample(self, ids: torch.Tensor) -> None:
        """Nuovi riferimenti e nuovo tempo di permanenza per i mondi `ids`.

        Si creano nuovi tensori (non si modifica in place): quelli vecchi possono essere già
        stati usati nel grafo di autograd dai passi precedenti della finestra.
        """
        k = ids.numel()
        rng = self.vel_range
        mode = torch.multinomial(self.mode_probs, k, replacement=True, generator=self._gen)
        u = self._uniform(-1.0, 1.0, (k, 4))  # valori normalizzati in [-1, 1]

        # 0. tutti gli assi
        v0 = u * rng

        # 1. sottoinsieme casuale di assi (almeno uno)
        p_active = self._uniform(0.3, 0.9, (k, 1))
        mask1 = (torch.rand((k, 4), device=self.device, generator=self._gen) < p_active).float()
        empty = (mask1.sum(dim=-1) == 0).nonzero(as_tuple=False).squeeze(-1)
        if empty.numel() > 0:
            forced = torch.randint(0, 4, (empty.numel(),), device=self.device, generator=self._gen)
            mask1[empty, forced] = 1.0
        v1 = u * rng * mask1

        # 2. un solo asse
        axis = torch.randint(0, 4, (k,), device=self.device, generator=self._gen)
        v2 = u * rng * torch.nn.functional.one_hot(axis, 4).float()

        # 3. tutti gli assi con ampiezza minima garantita
        sign = torch.where(torch.rand((k, 4), device=self.device, generator=self._gen) < 0.5, -1.0, 1.0)
        amplitude = self.min_amplitude_frac + (1.0 - self.min_amplitude_frac) * torch.rand(
            (k, 4), device=self.device, generator=self._gen
        )
        v3 = sign * amplitude * rng

        # 4. hover: zero esatto (metà dei casi) oppure piccola correzione
        exact = torch.rand((k, 1), device=self.device, generator=self._gen) < 0.5
        v4 = torch.where(exact, torch.zeros_like(u), u * self.hover_frac * rng)

        new_ref = torch.stack([v0, v1, v2, v3, v4], dim=1)[torch.arange(k, device=self.device), mode]  # (k, 4)
        hold_steps = (self._uniform(*self.hold_time_range_s, (k,)) / self.dt).long().clamp(min=1)

        ref, modes, hold_left = self.ref.clone(), self.mode.clone(), self._hold_left.clone()
        ref[ids], modes[ids], hold_left[ids] = new_ref, mode, hold_steps
        self.ref, self.mode, self._hold_left = ref, modes, hold_left
        n_samples = self.n_samples.clone()
        n_samples[ids] += 1
        self.n_samples = n_samples

    def set_constant(self, ref) -> None:
        """Imposta lo stesso riferimento fisso (vx, vy, vz, wz) per tutti i mondi, senza scadenza (valutazione con
        riferimento scelto a mano). La modalità vale -1 (imposto). I reset successivi ricampionano a caso."""
        r = torch.tensor(ref, dtype=torch.float32, device=self.device).expand(self.n, 4).clone()
        self.ref, self.step_ref = r, r
        self.mode = torch.full((self.n,), -1, dtype=torch.long, device=self.device)
        self._hold_left = torch.full((self.n,), 10**9, dtype=torch.long, device=self.device)
        self.n_samples = torch.ones(self.n, dtype=torch.long, device=self.device)

    def reset(self, ids: torch.Tensor) -> None:
        """Nuovi riferimenti per i mondi appena resettati."""
        n_samples = self.n_samples.clone()
        n_samples[ids] = 0
        self.n_samples = n_samples
        self.sample(ids)
        self.step_ref = self.ref

    def step(self) -> None:
        """Da chiamare dopo ogni passo di controllo: memorizza il riferimento attivo durante quel passo,
        poi conta alla rovescia e rinnova quelli scaduti."""
        self.step_ref = self.ref
        self._hold_left = self._hold_left - 1
        expired = (self._hold_left <= 0).nonzero(as_tuple=False).squeeze(-1)
        if expired.numel() > 0:
            self.sample(expired)
