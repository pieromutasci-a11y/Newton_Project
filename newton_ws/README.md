# Multi-drone con Newton: stato del progetto

Riassunto di dove siamo arrivati e di cosa vogliamo fare, per riprendere il lavoro (anche su un'altra
macchina o con una nuova sessione di Claude: leggere questo file per primo).

Ultimo aggiornamento: 13/11/2026.

## 1. Obiettivo

Addestrare **più droni (multi-agent)** con **Newton** (NVIDIA Warp) e **confrontare tempi di training e risultati
con il progetto Isaac Lab** esistente (DJI Tello, PPO/IPPO):

- simulazione e training **interamente su GPU**, senza saturare la CPU;
- controllore addestrato con **APG** (*analytic policy gradient*): il gradiente del costo passa attraverso la fisica
  differenziabile fino ai pesi, **senza PPO**, critic né esplorazione;
- **controllo a cascata** come in Isaac Lab: rete di posizione → riferimenti di velocità → rete di velocità → spinta e momenti.

**Dove siamo:** il solo controllore di posizione, che comanda direttamente spinta e momenti (cascata di velocità non ancora fatta),
per un drone singolo in 4096 mondi paralleli. Funziona: raggiunge sequenze di 3 riferimenti con precisione di pochi cm.
Il multi-drone vero e proprio (più droni nello stesso mondo) è il passo finale, non ancora iniziato.

## 2. Ambiente (Docker)

Sull'host (`/home/piero/Scrivania/Newton/`):

```
docker_ws/Dockerfile.newton_examples   # immagine: CUDA 12.4 runtime + uv + clone di Newton + torch (CUDA 13) + tensorboard
docker_ws/build.sh                     # docker build ... -t newton-examples
run.sh                                 # avvia il container (nome newton-lab-Mutasci, --rm, GPU, X11)
exec.sh                                # seconda shell nello stesso container
chown_me.sh                            # sudo chown -R $USER:$USER . (i file creati dal container sono di root)
newton_ws/                             # QUESTO progetto, montato in /workspace/project_workspace
```

Nel container il venv di Newton è il Python di default e `PYTHONPATH` punta a `newton_ws/src`, quindi si lancia
`python /workspace/project_workspace/src/...` da qualunque cartella.

Requisiti (documentazione di Newton): GPU NVIDIA **Turing o più recente** (compute capability 7.5+) e driver **R580+**
per Warp 1.18; macchina di lavoro: RTX A1000 (8 GB), driver 580. Una GPU Maxwell (es. MX130) non è supportata dal setup attuale.
Memoria misurata del training: circa 0.4 GB con 1024 mondi, 0.9 GB con 4096 (orizzonte 32).

```bash
./docker_ws/build.sh        # una volta
./run.sh                    # avvia il container (lasciarlo aperto)
./exec.sh                   # altre shell
```

## 3. Struttura del codice (`src/`)

Regola: **Warp solo dentro `TrainWorld`**; sopra (env, costo, rete, training) solo tensori PyTorch.

```
src/
  drone_model/                        # com'è fatto il drone
    params.py                         # DroneParams: massa 0.087 kg, inerzie, thrust/weight 1.8, scale dei momenti (come Isaac Lab)
    actuators.py                      # UNICO kernel Warp nostro: apply_wrench (azioni -> spinta e momenti sul corpo)
    builder.py                        # add_drone: UN corpo rigido + scatola di collisione; con visual=True aspetto da quadricottero
  scene/
    train_world.py                    # TrainWorld: N mondi, fisica differenziabile, ponte Warp<->PyTorch (_SimStep), decimation (uguale per posizione e velocità)
    test_world.py                     # mondo singolo con pavimento e XPBD (solo per prove e viewer)
  controllers/pos_controller/
    mlp_config.yaml                   # rete: 18 -> [256,128,64] ELU -> 4 tanh, normalizzazione delle osservazioni
    pos_controller_mlp.py             # PosControllerMLP (deterministica, statistiche mobili) + la rete di velocità CONGELATA dentro (vel_policy)
    pos_controller_env.py             # PosControllerEnv: cascata (2 passi di rete di velocità per passo di posizione), errori, osservazioni, riferimenti, reset
    target_handler.py                 # TargetHandler: coda di riferimenti casuali (x, y, z, yaw)
  controllers/vel_controller/         # controllore di velocità (anello interno, 50 Hz): riferimenti (vx, vy, vz, wz) -> spinta e momenti
    mlp_config.yaml                   # rete: 17 -> [256,128,64] ELU -> 4 tanh (separata da quella della posizione)
    vel_controller_mlp.py             # VelControllerMLP (stessa struttura, statistiche mobili, usabile congelata)
    vel_controller_env.py             # VelControllerEnv: 17 osservazioni, errori di velocità, reset (anche per velocità angolare > 30 rad/s)
    ref_handler.py                    # RefHandler: riferimenti di velocità a tempo, 5 modalità di Isaac Lab, senza curriculum
  controllers/math_utils.py           # quat_rotate_inverse, quat_rotate, yaw_from_quat (comuni ai due controllori)
  train/
    cost_function_pos_controller.py   # CostWeights + cost_function (pesi di default = ultimo run, vedi sotto)
    cost_function_vel_controller.py   # costo di velocità: errore normalizzato + azioni (peso crescente) + assetto
    train_APG_pos_controller.py       # training della posizione sopra la rete di velocità congelata (--vel_run): finestre di BPTT, tensorboard, checkpoint
    train_APG_vel_controller.py       # training della velocità (decimation 2: 50 Hz), rampa del peso delle azioni
  eval/
    eval_pos_controller_policy.py     # valutazione con GUI: la rete guida il drone, 5 riferimenti, poi chiude e salva i grafici
    plots_pos_controller.py           # grafici e riepilogo della posizione (posizione, yaw, velocità, spinta, momenti)
    eval_vel_controller_policy.py     # valutazione con GUI della velocità: 3 riferimenti, frecce verde (comando) e blu (reale), poi chiude e salva i grafici
    plots_vel_controller.py           # grafici e riepilogo della velocità (riferimento contro reale, errore a regime, spinta, momenti)
  tests/                              # test_pos_env, test_vel_env, test_cascade, test_cost_function, test_vel_cost, test_decimation, test_train_world, grad_check, ...
data/                                 # una cartella per controllore, ciascuna con policy/ e results/
  pos_controller/
    policy/<run>/{best,last}.pt       # checkpoint della policy
    results/<run>/                    # log tensorboard + grafici di valutazione (eval_best.png, eval_best_summary.txt)
  vel_controller/                     # stessa struttura, per il controllore di velocità
```

## 4. Come funziona

### 4.1 APG contro PPO (in breve)

| PPO (Isaac Lab) | APG (qui) |
|---|---|
| PhysX è una scatola nera: il gradiente si stima da campioni | la fisica è nel grafo: il gradiente è esatto (regola della catena) |
| rete gaussiana + critic + GAE + clipping | rete deterministica con `tanh`, nessun critic |
| reward, anche a gradino (+80 arrivo, −10 morte) | **costo liscio**: niente gradini, norme o `acos` che danno NaN |
| 5 epoche sugli stessi dati | un aggiornamento per finestra |
| sconto γ per i premi lontani | finestra di 32 passi di controllo (BPTT troncata) |

Il gradiente è il prodotto delle derivate della fisica su tutti i passi: è il motivo per cui la fisica deve essere stabile (vedi 6).

### 4.2 Un passo di controllo (cascata)

```
rete di POSIZIONE (PyTorch, 25 Hz) -> 4 riferimenti di velocità normalizzati in [-1,1] -> x scala (1,1,1,1.5): m/s e rad/s, nel corpo
  x2 volte (50 Hz): rete di VELOCITÀ (congelata) legge velocità, gravità, azione precedente e riferimento -> spinta e momenti (N,4)
      x2 passi di fisica a 100 Hz con la stessa azione (zero-order hold), registrati in un wp.Tape:
          clear_forces  ->  apply_wrench [nostro kernel]  ->  solver.step (SolverSemiImplicit, senza contatti)
stato (N,7) e (N,6) visto da PyTorch senza copia (wp.to_torch)
env + costo (PyTorch) -> loss;  backward: tape.backward() ripercorre i kernel al contrario, e il gradiente attraversa
  la rete di velocità congelata (parametri senza gradiente, ma ingressi sì) fino alla rete di posizione
```

- **Frequenze come in Isaac Lab:** fisica 100 Hz (`dt = 0.01`); il mondo ha `decimation = 2` (passo di 0.02 s = 50 Hz, quello della rete di velocità);
  la rete di posizione agisce ogni 2 passi del mondo: **25 Hz** (0.04 s). `--horizon` conta passi di posizione (il mondo ne ha il doppio).
- **Rete di velocità congelata:** si carica con `--vel_run` (es. `apg_vel_v7`) dentro `PosControllerMLP` (`vel_policy`): in `eval()` sempre, statistiche
  di normalizzazione fisse. Il checkpoint della posizione la contiene, quindi la valutazione carica un solo file.
- **Azioni della posizione = riferimenti di velocità** (vx, vy, vz, wz) normalizzati; l'azione precedente nelle osservazioni (4) sono i riferimenti del
  passo prima, non più spinta e momenti. Uscita iniziale ~0: il drone parte in hover. Il termine `effort` del costo ora penalizza |riferimento|² (hover = 0).
- **Drone (rete di velocità → fisica):** un solo corpo rigido. Spinta = `1.8 · peso · (a0+1)/2` lungo z del corpo (0 ... 1.536 N, hover a 0.853 N con a0 = 0.111);
  momenti = `(0.021, 0.021, 0.0085) · a[1:4]`; azioni limitate in [−1, 1]. Stesse formule di Isaac Lab (verificato).
- **Osservazioni (18):** errore di posizione **nel sistema del corpo** (3), sin/cos errore di yaw (2), velocità lineare (3) e angolare (3) nel corpo,
  gravità proiettata (3), azione precedente (4). Normalizzate con media e varianza mobili (come `RunningStandardScaler` di skrl).
- **Riferimenti (come Isaac Lab):** coda di **3 waypoint** per episodio (x, y ∈ ±1.5 m, z ∈ 0.5-1.5 m, yaw ∈ ±π). Si avanza quando il drone sta entro
  **0.15 m e 0.2 rad per 1.2 s**; dopo l'ultimo il riferimento resta. Episodio da **15 s**. Reset anche per inclinazione > 65°, stato non finito o drone oltre 20 m.
- **Training:** `--horizon 32` passi di controllo (1.28 s, 128 passi di fisica) per finestra, 4096 mondi, Adam, lr 1e-3, clip del gradiente a 1.0,
  gli aggiornamenti con gradiente non finito vengono saltati. Circa **170 ms per iterazione, 3000 iterazioni ≈ 8-9 minuti**.

### 4.3 Costo (pesi di default attuali, in `train/cost_function_pos_controller.py`)

Somma di termini per passo (più basso = meglio), tutti derivabili:

| Termine | Peso | Cosa fa |
|---|---|---|
| posizione, ampio raggio | 1.0 · (1 − e^(−0.5 d)) | avvicinarsi |
| posizione, precisione | 1.4 · (1 − e^(−8 d)) | arrivare sul bersaglio |
| yaw | 0.32 · \|err\| e 0.32 · (1 − e^(−7 \|err\|)) | allinearsi |
| velocità lineare | **0.1** · v² (+ 0.5 · v² entro 0.5 m: frenata) | limita la velocità |
| velocità angolare | **0.01** · (ωx² + ωy²); 0.002 su z | smorza rotazioni |
| assetto | **0.5** · (1 − cos tilt); 10 · eccesso² oltre **25°** | limita l'inclinazione |
| azioni | 0.1 · Δazione²; 0.01 · (azione − hover)² | comandi lisci |

I pesi di posizione e yaw sono quelli di Isaac Lab divisi per 25; gli altri sono stime tarate a mano. I valori in grassetto sono quelli cambiati
nell'ultima iterazione. Si possono cambiare da riga di comando con `--cost nome=valore` (ripetibile).

## 5. Comandi

```bash
# training del controllore di posizione (circa 8-9 minuti; lo lancia l'utente; salva in data/pos_controller/).
# Il run_name deve essere nuovo a ogni prova:
docker exec -it newton-lab-Mutasci python /workspace/project_workspace/src/train/train_APG_pos_controller.py \
  --vel_run apg_vel_v7 --num_envs 4096 --horizon 32 --iterations 3000 --lr 1e-3 --grad_clip 1.0 \
  --log_interval 50 --ckpt_interval 200 --run_name <nome>

# training del controllore di velocità (50 Hz, rampa del peso delle azioni; salva in data/vel_controller/):
docker exec -it newton-lab-Mutasci python /workspace/project_workspace/src/train/train_APG_vel_controller.py \
  --num_envs 4096 --horizon 32 --iterations 3000 --lr 1e-3 --grad_clip 1.0 \
  --log_interval 50 --ckpt_interval 200 --run_name <nome>

# valutazione con GUI (chiude da sola dopo 5 riferimenti e salva grafici e riepilogo in data/pos_controller/results/<nome>/)
docker exec -it newton-lab-Mutasci python /workspace/project_workspace/src/eval/eval_pos_controller_policy.py --run <nome> [--ckpt last] [--num_drones 1]

# idem, con riferimento di posizione scelto a mano (x, y, z in metri e yaw in GRADI, nel mondo; fisso per tutta la prova;
# gli assi non indicati restano quelli di partenza; durata --duration, default 10 s)
docker exec -it newton-lab-Mutasci python /workspace/project_workspace/src/eval/eval_pos_controller_policy.py --run <nome> --x 1.0 --y -0.5 [--z 1.2] [--yaw 90] [--duration 10]

# valutazione con GUI del controllore di velocità (3 riferimenti, poi chiude; salva in data/vel_controller/results/<nome>/)
docker exec -it newton-lab-Mutasci python /workspace/project_workspace/src/eval/eval_vel_controller_policy.py --run <nome> [--ckpt last] [--n_refs 3]

# idem, con riferimento di velocità scelto a mano (fisso per tutta la prova; gli assi non indicati valgono 0; durata --duration, default 10 s)
docker exec -it newton-lab-Mutasci python /workspace/project_workspace/src/eval/eval_vel_controller_policy.py --run <nome> --vx 0.5 --vy -0.3 [--vz 0.2] [--wz 1.0] [--duration 10]

# test (secondi)
docker exec newton-lab-Mutasci python /workspace/project_workspace/src/tests/test_pos_env.py
docker exec newton-lab-Mutasci python /workspace/project_workspace/src/tests/test_cost_function.py
docker exec newton-lab-Mutasci python /workspace/project_workspace/src/tests/test_decimation.py
docker exec newton-lab-Mutasci python /workspace/project_workspace/src/tests/test_cascade.py
docker exec newton-lab-Mutasci python /workspace/project_workspace/src/tests/test_vel_env.py
docker exec newton-lab-Mutasci python /workspace/project_workspace/src/tests/test_vel_cost.py

# curve di training (una cartella per controllore)
tensorboard --logdir /workspace/project_workspace/data/pos_controller/results
tensorboard --logdir /workspace/project_workspace/data/vel_controller/results
```

Nel log di training: `dist` (distanza media dal riferimento, un drone fermo vale circa 1.4 m), `wp` (indice medio del riferimento, 0-2),
`|grad|`, `saltate` (aggiornamenti scartati). I file creati dal container sono di root: lanciare `chown_me.sh` prima di modificarli dall'host.

## 6. Risultati e cose imparate

Run in `data/pos_controller/policy/` (tutti con 4096 mondi, orizzonte 32, 3000 iterazioni, 25 Hz, 3 riferimenti, tranne il primo):

| Run | Pesi diversi dal default attuale | Esito |
|---|---|---|
| `apg_obsfix_h32` | schema vecchio (100 Hz, riferimenti a tempo) | **obsoleto**, non confrontabile |
| `apg_25hz_3wp` | `action_rate` 0.1, `lin_vel` 0.01, `upright` 0.1, `ang_vel_xy` 0.002, tilt libero 35° | **divergenza a NaN** poco dopo l'it. 2000; `best.pt` (it. 2000) è buono: dist 0.19 m, **fermo a riposo**; velocità fino a 2.5 m/s, inclinazione 34° |
| `apg_25hz_3wp_ar05` | `action_rate` 0.5 | più preciso (dist 0.13 m) ma **tremolio a riposo**: ciclo limite a 5 Hz sui comandi (±20% di spinta, quota ±2.5 mm) |
| `apg_25hz_3wp_tilt25` | tilt 25°, `upright` 0.5, `ang_vel_xy` 0.01 | meno tremolio, ma **troppo veloce**: fino a 3.2 m/s |
| `apg_25hz_3wp_tilt25_v01` | **= pesi di default attuali** (anche `lin_vel` 0.1) | scelto come il migliore: velocità 1.2-1.9 m/s, inclinazione 15-24°; più lento (dist media di training 0.51 m, tempo a 15 cm 1.0-2.2 s sui salti provati) |

Cose da sapere:

- **Instabilità della rotazione:** con momenti nulli, a velocità angolare alta l'integrazione non conserva l'energia e diverge a NaN
  (misurato: da 60-120 rad/s in pochi secondi, anche da 20 rad/s attorno a z). In training un solo drone in questo stato rende NaN le statistiche di
  normalizzazione e quindi tutta la rete (successo nel run senza protezioni). Il funzionamento normale sta a 7-8 rad/s.
  Protezioni già presenti: il normalizzatore ignora osservazioni non finite; `done()` resetta stati non finiti o a più di 20 m.
- **Sistema troppo ideale:** nessun rumore, disturbo, ritardo né dinamica dei motori (anche in Isaac Lab i motori sono ideali: forza applicata al corpo).
  Un controllore che non oscilla mai non dice nulla sulla robustezza.
- **Limiti di attuazione:** a regime la rete sta sull'hover; nei transitori usa quasi tutto il campo di spinta (0.04-1.5 N) e il momento z
  tocca il massimo (±0.0085 N·m) nel 2-3% dei passi. Da confrontare con i limiti reali del Tello.
- **Comportamento alla partenza:** il costo premia l'arrivo rapido; senza limiti sulla velocità la rete si inclina molto quando cambia il riferimento
  (74° a 3 m con i pesi vecchi). In Isaac Lab è la cascata (riferimenti di velocità limitati a ±1 m/s) a tenerlo basso.
- Il confronto tra run **non è pulito** quando cambiano più pesi insieme; la loss non è confrontabile tra run con pesi diversi.

## 7. Cosa vogliamo fare

Prossimi passi, in ordine:

1. **Misurare il tremolio a riposo** del run migliore (`tilt25_v01`) con la stessa prova usata per `ar05` (variazione delle azioni, ciclo limite):
   finora di questo run sono stati misurati solo velocità, inclinazione e tempo di arrivo.
2. **Robustezza numerica:** reset dei droni con velocità angolare eccessiva (es. > 30 rad/s) prima della divergenza e contatore dei mondi non finiti nel log
   (proposto, non ancora fatto); learning rate decrescente a fine training.
3. **Realismo, uno alla volta** (non ora): rumore sulle osservazioni, disturbi (vento/forze casuali), variazione di massa e spinta, dinamica dei motori
   (filtro del primo ordine), eventuale ritardo. Il rumore va aggiunto anche in valutazione per un confronto onesto.
4. **Controllore di velocità e cascata** (come Isaac Lab):
   - rete di velocità (anello interno, 50 Hz) addestrata con APG: osservazioni 17 (velocità, gravità, azione precedente, riferimento di velocità),
     riferimenti campionati in ±1 m/s (lineari) e ±1.5 rad/s (yaw), con modalità miste e curriculum, costo sull'errore di velocità normalizzato
     + azioni lisce (peso crescente) + tilt;
   - poi congelata; la rete di posizione produce 4 numeri (`tanh`) × scala = riferimenti di velocità, e la catena rete di posizione → rete di velocità → spinta e momenti
     resta differenziabile (gradiente attraverso la rete di velocità: da tenere d'occhio la stabilità).
5. **Multi-drone:** più droni nello stesso mondo (collision group, separazione, formazione), osservazioni dei vicini.
6. **Confronto con Isaac Lab:** tempi di training (qui circa 8-9 minuti per 3000 iterazioni a 4096 mondi) e qualità dei risultati.
7. **Manutenzione:** volume per la cache dei kernel Warp e porta 6006 per tensorboard in `run.sh`; i container girano come root (file di root in `newton_ws`);
   aggiornare la copia su GitHub (`Newton_Project`; non so se il push è aggiornato: la cartella non è un repo git).

## 8. Note di lavoro

- I training lunghi li lancia l'utente: il comando si prepara e si spiega cosa guardare nel log.
- Il progetto Isaac Lab di riferimento è in `/home/piero/Scrivania/Project_Favuzzi/...` (`tasks/direct/pos_controller` e `vel_controller`);
  copia su GitHub: *Position-and-Velocity-Controller-Isaaclab-Djitello*.
- Documentazione di Newton: https://newton-physics.github.io/newton/latest/ (esempio di riferimento `newton/examples/diffsim/example_diffsim_drone.py`).
- Lavori su APG e simulazione differenziabile citati a memoria (da verificare prima di una bibliografia): Wiedemann et al. *Training Efficient Controllers via Analytic Policy Gradient* (ICRA 2023);
  Mora et al. *PODS* (ICML 2021); Xu et al. *SHAC* (ICLR 2022).
