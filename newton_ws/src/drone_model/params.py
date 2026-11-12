"""Parametri fisici del drone. Tutte le grandezze sono in unità SI."""

import math
from dataclasses import dataclass


@dataclass
class DroneParams:
    # Geometria: le due barre a croce sono scatole. Le dimensioni sono SEMI-estensioni (half-extent).
    arm_length: float = 0.2          # m, semi-lunghezza di ogni barra = distanza dei propeller dal centro
    bar_half_width: float = 0.01     # m
    bar_half_height: float = 0.01    # m
    density: float = 1750.0          # kg/m^3 (fibra di carbonio)

    # Propeller (valori dell'esempio diffsim_drone)
    thrust_coeff: float = 0.109919   # adimensionale
    power_coeff: float = 0.040164    # adimensionale
    prop_diameter: float = 0.2286    # m
    max_rpm: float = 6396.667        # giri/minuto
    air_density: float = 1.225       # kg/m^3 (livello del mare)

    # Limiti del comando normalizzato di ogni motore
    control_min: float = 0.1
    control_max: float = 1.0

    @property
    def max_rps(self) -> float:
        """Giri al secondo alla massima velocità."""
        return self.max_rpm / 60.0

    @property
    def max_thrust(self) -> float:
        """Spinta massima di un propeller [N]: Ct * rho * n^2 * D^4."""
        return self.thrust_coeff * self.air_density * self.max_rps**2 * self.prop_diameter**4

    @property
    def max_torque(self) -> float:
        """Coppia di reazione massima di un propeller [N*m]: Cp * rho * n^2 * D^5 / (2*pi)."""
        return self.power_coeff * self.air_density * self.max_rps**2 * self.prop_diameter**5 / (2.0 * math.pi)
