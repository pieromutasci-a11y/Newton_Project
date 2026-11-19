"""Sensibilità al rumore sul feedback di una policy addestrata (senza riaddestrare): 1024 mondi, 15 s, per riferimento
fisso (wp), curva custom e cerchio. Errore dopo 4 s, assetto, e mondi che verrebbero resettati (inclinazione, divergenza).

    python /workspace/project_workspace/src/eval/noise_sensitivity.py [nome_run] [best|last]
"""
import math, os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from controllers.pos_controller.pos_controller_env import NoiseConfig, PosControllerEnv
from controllers.pos_controller.pos_controller_mlp import PosControllerMLP
from controllers.pos_controller.target_handler import TargetHandler
from drone_model.params import DroneParams
from scene.train_world import TrainWorld, TrainWorldConfig
dev="cuda:0"; N=1024
RUN = sys.argv[1] if len(sys.argv) > 1 else "apg_pos_custom_v1"
CK = sys.argv[2] if len(sys.argv) > 2 else "best"
ck=torch.load(os.path.join(os.path.dirname(__file__), "..", "..", "data", "pos_controller", "policy", RUN, f"{CK}.pt"),map_location=dev)
print(f"policy {RUN}/{CK}.pt")
pol=PosControllerMLP().to(dev); pol.load_state_dict(ck["policy"]); pol.eval()

def run(noise, mode, steps=375):  # 15 s
    world=TrainWorld(DroneParams(), TrainWorldConfig(num_worlds=N, horizon=2*steps, decimation=2))
    th=TargetHandler(N,dev,seed=11,dt=0.04)
    env=PosControllerEnv(world, pol.vel_policy, th, seed=11, noise=noise)
    ids=torch.arange(N,device=dev)
    if mode=="custom": th.set_trajectory("custom",speed=0.4,flat_prob=0.0)
    elif mode=="circle": th.set_trajectory("circle",speed=0.4,flat_prob=0.0)
    else: th.set_constant(torch.tensor([0.5,-0.5,1.0],device=dev),0.0)
    # riposiziona: custom/circle vicino alla partenza della curva, wp da pos casuale
    near=th.target_pos+(torch.rand(N,3,device=dev)*2-1)*(0.3 if mode!="wp" else 0.0)
    if mode=="wp": near=torch.stack([torch.rand(N,device=dev)*2-1,torch.rand(N,device=dev)*2-1,0.5+torch.rand(N,device=dev)],-1)
    world.reset(ids, near); env.prev_actions.zero_(); env.low_prev_actions.zero_()
    errs=[]; tilt=[]; lost=torch.zeros(N,dtype=torch.bool,device=dev)
    with torch.no_grad():
        obs=env.observations()
        for i in range(steps):
            obs=env.step(pol(obs))
            e=env.errors(); d=e.pos_err.norm(dim=-1)
            lost|=env.done()&(env._steps_alive<steps-1)
            errs.append(d)
            tilt.append(torch.rad2deg(torch.acos(torch.clamp(-e.gravity_b[:,2],-1,1))))
    E=torch.stack(errs); T=torch.stack(tilt)
    late=E[100:]  # dopo 4 s
    return float(late.mean()), float(late.amax(0).mean()), float(T.mean()), float(T.amax(0).mean()), int(lost.sum())

cases=[("nessuno",None),
 ("pos 1cm",NoiseConfig(0.01,0,0,0)),
 ("att 0.5°",NoiseConfig(0,math.radians(0.5),0,0)),
 ("vel lin 0.05",NoiseConfig(0,0,0.05,0)),
 ("vel ang 0.05",NoiseConfig(0,0,0,0.05)),
 ("tutto (1cm,0.5°,0.05,0.05)",NoiseConfig(0.01,math.radians(0.5),0.05,0.05)),
 ("tutto x3 (3cm,1.5°,0.15,0.15)",NoiseConfig(0.03,math.radians(1.5),0.15,0.15)),
 ("tutto, vel-net pulita",NoiseConfig(0.01,math.radians(0.5),0.05,0.05,to_vel_net=False)),
]
for mode in ("wp","custom","circle"):
    print("==",mode,"| err medio (dopo 4s) [m], err max medio [m], tilt medio [°], tilt max medio [°], resettati")
    for name,nz in cases:
        r=run(nz,mode); print(f"  {name:32s} {r[0]:.4f}  {r[1]:.4f}  {r[2]:5.2f}  {r[3]:5.2f}  {r[4]}")
