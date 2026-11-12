from dataclasses import dataclass

import newton
import warp as wp

from .params import DroneParams
from .actuators import Propeller

@dataclass
class DroneHandle:

    body: int



def _make_propeller(
    body: int, 
    pos: tuple[float, float, float], 
    turning_direction: float, 
    params: DroneParams) -> Propeller:
    
    prop = Propeller()
    prop.body = body
    prop.pos = wp.vec3(*pos)
    prop.dir = wp.vec3(0.0, 0.0, 0.1)
    prop.max_thrust = params.max_thrust
    prop.max_torque = params.max_torque
    prop.turning_direction = turning_direction

    return prop

def add_drone(
    builder: newton.ModelBuilder,
    params: DroneParams,
    label: str,
    collision_group: int = 0) -> tuple[DroneHandle, list[Propeller]]:

    