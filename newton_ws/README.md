# Multi-drone con Newton: stato del progetto

Riassunto di dove siamo arrivati e di cosa vogliamo fare, per riprendere il lavoro (anche su un'altra
macchina o con una nuova sessione di Claude: leggere questo file per primo).

Ultimo aggiornamento: 17/11/2026.

## 1. Obiettivo

Addestrare **più droni (multi-agent)** con **Newton** (NVIDIA Warp) e **confrontare tempi di training e risultati
con il progetto Isaac Lab** esistente (DJI Tello, PPO/IPPO):

- simulazione e training **interamente su GPU**, senza saturare la CPU;
- controllori addestrati con **APG** (*analytic policy gradient*): il gradiente del costo passa attraverso la fisica
  differenziabile fino ai pesi della rete, **senza PPO**, critic né esplorazione;
- **controllo a cascata** come in Isaac Lab: rete di posizione → 4 riferimenti di velocità → rete di velocità (congelata) → spinta e momenti.

**Dove siamo:** la cascata è completa per un drone singolo in 4096 mondi paralleli.

1. Controllore di **velocità** addestrato (`apg_vel_v7`, scelto e congelato).
2. Controllore di **posizione** addestrato sopra la rete di velocità congelata (`apg_pos_cascade_v1..v3`).
3. Valutazione con GUI e grafici per entrambi.

**Ramo `traiettorie`** (non ancora addestrato): il controllore di posizione riceve anche una **finestra di anteprima** del riferimento e viene
addestrato su un mix di waypoint fermi e traiettorie continue (vedi 4.5). `main` resta la versione a 18 osservazioni, già addestrata.

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

Requisiti: GPU NVIDIA **Turing o più recente** (compute capability 7.5+) e driver **R580+**; macchina di lavoro: RTX A1000 (8 GB), driver 580.
Una GPU Maxwell (es. MX130) non è supportata da Warp 1.19. Memoria del training di posizione: circa 0.4 GB con 1024 mondi, 0.9 GB con 4096.

```bash
./docker_ws/build.sh        # una volta
./run.sh                    # avvia il container (lasciarlo aperto)
./exec.sh                   # altre shell
```

## 3. Struttura del codice (`src/`)

Regola: **Warp solo dentro `TrainWorld`** (più il kernel `apply_wrench`); sopra (env, costo, rete, training) solo tensori PyTorch.

```
src/
  drone_model/
    params.py                         # DroneParams: massa 0.087 kg, inerzie, thrust/weight 1.8, scale dei momenti (come Isaac Lab)
    actuators.py                      # UNICO kernel Warp nostro: apply_wrench (azioni -> spinta e momenti sul corpo)
    builder.py                        # add_drone: UN corpo rigido + scatola di collisione; con visual=True aspetto da quadricottero
  scene/
    train_world.py                    # TrainWorld: N mondi, fisica differenziabile, ponte Warp<->PyTorch (_SimStep), decimation
    test_world.py                     # mondo singolo con pavimento e XPBD (solo per prove e viewer)
  controllers/
    math_utils.py                     # quat_rotate_inverse, quat_rotate, yaw_from_quat (comuni ai due controllori)
    vel_controller/                   # anello interno, 50 Hz: (vx, vy, vz, wz) -> spinta e momenti
      mlp_config.yaml                 # rete: 17 -> [256,128,64] ELU -> 4 tanh
      vel_controller_mlp.py           # VelControllerMLP (statistiche mobili, usabile congelata)
      vel_controller_env.py           # VelControllerEnv + vel_observations (funzione unica, usata anche dalla cascata)
      ref_handler.py                  # RefHandler: riferimenti di velocità a tempo, 5 modalità, senza curriculum
    pos_controller/                   # anello esterno, 25 Hz: posizione e yaw -> 4 riferimenti di velocità
      mlp_config.yaml                 # rete: 33 -> [256,128,64] ELU -> 4 tanh (separata da quella di velocità)
      pos_controller_mlp.py           # PosControllerMLP + rete di velocità CONGELATA dentro (vel_policy)
      pos_controller_env.py           # PosControllerEnv: cascata (2 passi di rete di velocità per passo di posizione)
      target_handler.py               # TargetHandler: waypoint casuali (x, y, z, yaw) oppure traiettorie continue + finestra di anteprima
  train/
    cost_function_vel_controller.py   # costo di velocità
    cost_function_pos_controller.py   # costo di posizione
    train_APG_vel_controller.py       # training della velocità (50 Hz), rampa del peso delle azioni
    train_APG_pos_controller.py       # training della posizione sopra la velocità congelata (--vel_run)
  eval/
    eval_vel_controller_policy.py     # GUI velocità: 3 riferimenti, frecce comando/reale, poi chiude e salva i grafici
    plots_vel_controller.py           # grafici e riepilogo velocità
    eval_pos_controller_policy.py     # GUI posizione: 5 riferimenti, sfere, frecce di yaw, poi chiude e salva i grafici
    plots_pos_controller.py           # 3 figure: posizione/yaw, velocità, spinta e momenti
  tests/                              # test_pos_env, test_vel_env, test_cascade, test_trajectories, test_cost_function, test_vel_cost, test_decimation, test_train_world, grad_check, ...
data/                                 # una cartella per controllore, ciascuna con policy/ e results/
  vel_controller/{policy,results}/<run>/
  pos_controller/{policy,results}/<run>/
    policy/<run>/{best,last}.pt       # checkpoint (quello di posizione contiene anche la rete di velocità)
    results/<run>/                    # log tensorboard + grafici e riepilogo di valutazione
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

Dove sta Warp: `apply_wrench` è l'unico kernel scritto da noi; i kernel del solver li fornisce Newton. Un `wp.Tape` registra i lanci di un passo
di mondo; `tape.backward()` esegue al contrario i kernel aggiunti (generati in automatico) e `_SimStep` (`torch.autograd.Function`) collega questo
al backward di PyTorch.

### 4.2 Frequenze

- Fisica **100 Hz** (`dt = 0.01`). Il mondo ha `decimation = 2`: passo di 0.02 s = **50 Hz**, quello della rete di velocità.
- La rete di posizione agisce ogni 2 passi del mondo: **25 Hz** (0.04 s). Nel training di posizione `--horizon` conta passi di posizione (il mondo ne ha il doppio).
- Solver: `SolverSemiImplicit` senza contatti. Smorzamento angolare 0.05 (default di Newton). La rotazione libera diverge sopra circa 60 rad/s.

### 4.3 Controllore di velocità (anello interno)

- **Osservazioni (17):** velocità lineare nel corpo (3), velocità angolare nel corpo (3), gravità proiettata (3), azione precedente (4), riferimento (vx, vy, vz, wz) (4).
  Normalizzate con media e varianza mobili (statistiche che ignorano righe non finite).
- **Azioni (4):** spinta e 3 momenti in [−1, 1]. Spinta = `1.8 · peso · (a0+1)/2` lungo z del corpo (0 … 1.536 N, hover a 0.853 N con a0 = 0.111);
  momenti = `(0.021, 0.021, 0.0085) · a[1:4]`. Stesse formule di Isaac Lab.
- **Riferimenti:** `RefHandler`, 5 modalità (probabilità 0.35 / 0.15 / 0.15 / 0.30 / 0.05), campo ±1 m/s lineare, ±1.5 rad/s su wz, permanenza 1.0-2.5 s,
  più riferimenti per episodio, nessun curriculum. Episodio da 10 s.
- **Reset:** inclinazione > 65°, timeout, stato non finito, drone oltre 20 m, |ω| > 30 rad/s.
- **Costo** (`cost_function_vel_controller.py`): errore di velocità lineare (peso 3.0) e angolare (1.0), variazione delle azioni con peso in rampa
  (0.01 → 0.1 nei default, su `--iterations` iterazioni), ω_xy (0.015), assetto (0.05) e tilt oltre 25° (5.0). `best.pt` si sceglie con il costo "monitor"
  calcolato col peso finale delle azioni. I pesi si cambiano con `--cost nome=valore`.
- **Run scelto: `apg_vel_v7`** (3000 iterazioni, `best` = `last`, pulito). Lo hanno addestrato pesi delle azioni più alti (0.5 → 0.9) e ω_xy 0.03:
  controllare i valori salvati nel checkpoint prima di rifarlo.

### 4.4 Controllore di posizione (cascata)

```
rete di POSIZIONE (PyTorch, 25 Hz) -> 4 riferimenti normalizzati in [-1,1] -> x scala (1,1,1,1.5): m/s e rad/s, nel corpo
  x2 volte (50 Hz): rete di VELOCITÀ (congelata) legge velocità, gravità, azione precedente e riferimento -> spinta e momenti (N,4)
      x2 passi di fisica a 100 Hz con la stessa azione (zero-order hold), registrati in un wp.Tape:
          clear_forces  ->  apply_wrench [nostro kernel]  ->  solver.step
stato (N,7) e (N,6) visto da PyTorch senza copia (wp.to_torch)
env + costo (PyTorch) -> loss;  backward: tape.backward() ripercorre i kernel al contrario, e il gradiente attraversa
  la rete di velocità congelata (parametri senza gradiente, ma ingressi sì) fino alla rete di posizione
```

- **Rete di velocità congelata:** si carica con `--vel_run` (es. `apg_vel_v7`) dentro `PosControllerMLP`: sempre in `eval()`, nessun gradiente sui parametri, statistiche fisse.
  Il checkpoint della posizione la contiene: la valutazione carica un solo file. I vecchi checkpoint di posizione (uscita = spinta e momenti) non sono compatibili.
- **Azioni della posizione = riferimenti di velocità**; l'azione precedente nelle osservazioni (4) è il riferimento del passo prima. Uscita iniziale ~0: il drone parte in hover.
  Il termine `effort` del costo penalizza |riferimento|².
- **Osservazioni (33):** errore di posizione **nel sistema del corpo** (3), sin/cos errore di yaw (2), velocità lineare (3) e angolare (3) nel corpo,
  gravità proiettata (3), azione precedente (4) + anteprima del riferimento (15, vedi 4.5).
- **Riferimenti waypoint (come Isaac Lab):** coda di **3 waypoint** per episodio (x, y ∈ ±1.5 m, z ∈ 0.5-1.5 m, yaw ∈ ±π). Si avanza entro **0.15 m e 0.2 rad per 1.2 s**.
  Episodio da **15 s**. Reset anche per inclinazione > 65°, stato non finito o drone oltre 20 m.
- **Training:** `--horizon 32` passi di posizione per finestra, 4096 mondi, Adam, lr 1e-3, `--grad_clip 1.0`; gli aggiornamenti con gradiente non finito sono saltati.
- **Costo** (`cost_function_pos_controller.py`):

| Termine | Peso | Cosa fa |
|---|---|---|
| posizione, ampio raggio | 1.0 · (1 − e^(−0.5 d)) | avvicinarsi |
| posizione, precisione | 1.4 · (1 − e^(−8 d)) | arrivare sul bersaglio |
| yaw | 0.32 · \|err\| e 0.32 · (1 − e^(−7 \|err\|)) | allinearsi |
| velocità lineare | 0.1 · \|v − v_rif\|² (+ 0.5 · \|v − v_rif\|² entro 0.5 m: frenata) | limita la velocità (v_rif = 0 per i waypoint) |
| velocità angolare | 0.01 · (ωx² + ωy²); 0.002 su z | smorza rotazioni |
| assetto | 0.5 · (1 − cos tilt); 10 · eccesso² oltre 25° | limita l'inclinazione |
| azioni | 0.1 · Δazione²; 0.01 · \|riferimento\|² | comandi lisci |

I pesi di posizione e yaw sono quelli di Isaac Lab divisi per 25; gli altri sono stime tarate a mano. Si cambiano con `--cost nome=valore`.
I pesi per la cascata non sono ancora stati ritarati: sono quelli del controllore che comandava direttamente spinta e momenti.

### 4.5 Traiettorie continue e finestra di anteprima (ramo `traiettorie`)

Come in Isaac Lab (`wp_preview_horizon`), la rete vede una **finestra** sul riferimento: l'errore verso il punto attuale più 3 punti futuri come
**incrementi** tra punti consecutivi (nel corpo) con sin/cos del delta di yaw. Una traiettoria è per la rete una coda di waypoint molto fitti.

- **Waypoint fermi:** i punti futuri coincidono con quello attuale, quindi la finestra è nulla (incrementi 0, sin 0, cos 1) e il comportamento è quello di prima
  (nessuna anteprima: evita il ciclo limite attorno al bersaglio visto in Isaac Lab quando si anticipa il waypoint successivo).
- **Traiettorie:** curve nel tempo (`TargetHandler`), valutate a t, t+0.5, t+1.0, t+1.5 s. Tipi: retta, cerchio, ellisse, otto, elica (cerchio con oscillazione lenta lungo la normale).
  **Orientamento:** 30% di curve orizzontali (`--traj_flat_prob`, rotazione solo attorno a z), 70% con orientamento 3D uniforme (piani verticali o inclinati).
  Picco di velocità in 0.1-0.5 m/s (`--traj_speed_min/max`), dentro il campo dei waypoint (x, y ±1.5 m, z 0.5-1.5 m: se l'orientamento non ci sta, si riducono le ampiezze).
  Yaw: fisso, nel verso di marcia (solo per curve con piano quasi orizzontale, entro 25° dalla verticale; non per la retta) o in rotazione lenta (0.1-0.4 rad/s). Il riferimento scorre da solo: nessun avanzamento per raggiungimento, episodio da 15 s;
  il drone parte fermo entro 0.3 m dal punto iniziale.
- **Curve custom:** oltre alle 5 standard, una famiglia casuale: ogni asse locale è una **somma di 3 sinusoidi** con ampiezza, frequenza e fase casuali
  (`l_i(t) = Σ_h a_ih sin(w_ih t + p_ih)`), poi ruotata in 3D come le altre (30% orizzontali / 70% 3D). Curvatura e velocità variano lungo il percorso.
  Ampiezza totale 0.3-0.9 m su x e y, 0-0.4 m su z (ridotta se esce dal campo). Le frequenze sono scalate in modo che il picco della norma della velocità
  (controllato numericamente su 25 s) sia al più quello campionato in 0.1-0.5 m/s, e l'accelerazione resti sotto 0.5 m/s² (maggiorante).
  Yaw: fisso o in rotazione lenta (mai nel verso di marcia: la velocità orizzontale può annullarsi). Le curve standard sono il caso di una sola armonica.
- **Circuiti (solo valutazione, `--traj circuit`):** percorso lungo che avanza nello spazio come un tracciato: avanzamento costante nel mondo (65% del picco di velocità, direzione
  casuale sul piano orizzontale, con `--orient 3d` anche una lieve salita o discesa) più ondulazioni custom più ampie (35%). Attraversa circa 5 m in 20 s e la quota resta nel campo
  (0.5-1.5 m) per tutta la durata della prova. Lo yaw può seguire il verso di marcia (la velocità orizzontale non si annulla mai). Non è nel mix di training.
- **Mix per episodio:** `--traj_prob` (default 0.5): ogni episodio, a ogni reset, è una traiettoria con questa probabilità, altrimenti una coda di 3 waypoint.
  Tra le traiettorie, `--custom_prob` (default 0.5) è la quota di curve custom. Con i default: 50% waypoint, 25% curve standard (5% ciascuna), 25% custom.
  Metriche: `metric/custom_frac`, `metric/dist_custom_m` (nel log `custom`).
- **Costo:** nel termine `lin_vel` e nella frenata si usa la velocità **relativa** a quella del riferimento (per i waypoint è 0, quindi identico a prima).
- **Costo di calcolo:** circa 300 ms per iterazione con 4096 mondi (anche con `--traj_prob 0`: il peso è nelle piccole operazioni sul riferimento e nella cascata).
- **Non ancora fatto:** training vero, curriculum sulla velocità, anteprima anche sui waypoint con attenuazione vicino al bersaglio (`preview_attenuazione_dist` di Isaac Lab).

### 4.6 Rumore sul feedback (ramo `realismo`)

Rumore bianco gaussiano sullo **stato misurato** (`NoiseConfig` in `pos_controller_env.py`): posizione (m), assetto (rotazione casuale attorno a x, y, z, rad),
velocità lineare (m/s) e angolare (rad/s). Indipendente a ogni lettura; vale per la rete di posizione e, di default, anche per quella di velocità
(`--no-noise_vel_net`: la rete di velocità, addestrata senza rumore, legge lo stato vero). Il costo, le terminazioni e i grafici usano lo stato vero.
Argomenti (training ed eval, default 0 = comportamento invariato): `--noise_pos`, `--noise_att_deg`, `--noise_lin_vel`, `--noise_ang_vel`.

Nell'eval con rumore, oltre ai grafici reale-riferimento, si salvano `eval_<ckpt>_feedback_posizione.png` e `eval_<ckpt>_feedback_velocita.png`
(posizione, yaw e velocità reali contro misurati, cioè quelli letti dalla rete) e nel riepilogo l'errore di misura RMS. I grafici reale-riferimento usano sempre lo stato vero.

`src/eval/noise_sensitivity.py [run] [best|last]` misura quanto degrada una policy già addestrata. Risultato su `apg_pos_custom_v1` (1024 mondi, 15 s, un seme):
senza rumore l'errore dopo 4 s è 0.7-4 mm; con 1 cm di rumore sulla posizione o 0.05 m/s sulla velocità lineare l'errore sale a 0.1-2 m con qualche mondo che diverge;
con tutto insieme (1 cm, 0.5°, 0.05 m/s, 0.05 rad/s) la policy diverge (errore medio >10 m). Rumore d'assetto di 0.5° e angolare di 0.05 rad/s sono invece tollerati.
La policy addestrata senza rumore ha quindi un guadagno molto alto sulla posizione e sulla velocità: serve addestrarla con il rumore.

## 5. Comandi

```bash
# training del controllore di velocità (50 Hz; salva in data/vel_controller/; il run_name deve essere nuovo)
docker exec -it newton-lab-Mutasci python /workspace/project_workspace/src/train/train_APG_vel_controller.py \
  --num_envs 4096 --horizon 32 --iterations 3000 --lr 1e-3 --grad_clip 1.0 \
  --log_interval 50 --ckpt_interval 200 --run_name <nome>

# training del controllore di posizione sopra la rete di velocità congelata (salva in data/pos_controller/)
# --traj_prob: frazione di episodi con traiettoria continua (default 0.5; 0 = solo waypoint); --custom_prob: quota di curve custom tra le traiettorie (default 0.5); --traj_speed_min/max: picco di velocità [m/s]
docker exec -it newton-lab-Mutasci python /workspace/project_workspace/src/train/train_APG_pos_controller.py \
  --vel_run apg_vel_v7 [--vel_ckpt best|last] [--traj_prob 0.5] --num_envs 4096 --horizon 32 --iterations 3000 --lr 1e-3 --grad_clip 1.0 \
  --log_interval 50 --ckpt_interval 200 --run_name <nome>

# valutazione della velocità con GUI (3 riferimenti casuali, poi chiude; salva in data/vel_controller/results/<nome>/)
docker exec -it newton-lab-Mutasci python /workspace/project_workspace/src/eval/eval_vel_controller_policy.py --run <nome> [--ckpt last] [--n_refs 3] [--seed N]
# idem, riferimento fisso scelto a mano (assi non indicati = 0; --duration, default 10 s)
docker exec -it newton-lab-Mutasci python /workspace/project_workspace/src/eval/eval_vel_controller_policy.py --run <nome> --vx 0.5 --vy -0.3 [--vz 0.2] [--wz 1.0] [--duration 10]

# valutazione con traiettoria continua (tipo: line, circle, ellipse, eight, helix, custom, circuit; --speed = picco di velocità in m/s; --duration default 20 s)
docker exec -it newton-lab-Mutasci python /workspace/project_workspace/src/eval/eval_pos_controller_policy.py --run <nome> --traj circle --speed 0.4 [--orient flat|3d]   # `random` = tipo standard casuale; dimensioni casuali; flat = piano xy (default), 3d = orientamento casuale

# valutazione della posizione con GUI (5 riferimenti casuali, poi chiude; salva 3 grafici e il riepilogo in data/pos_controller/results/<nome>/)
docker exec -it newton-lab-Mutasci python /workspace/project_workspace/src/eval/eval_pos_controller_policy.py --run <nome> [--ckpt last] [--n_waypoints 5] [--seed N]
# idem, bersaglio fisso (x, y, z in metri e yaw in GRADI; assi non indicati = quelli di partenza; --duration, default 10 s)
docker exec -it newton-lab-Mutasci python /workspace/project_workspace/src/eval/eval_pos_controller_policy.py --run <nome> --x 1.0 --y -0.5 [--z 1.2] [--yaw 90] [--duration 10]

# test (secondi)
for t in test_pos_env test_cost_function test_decimation test_cascade test_trajectories test_vel_env test_vel_cost; do
  docker exec newton-lab-Mutasci python /workspace/project_workspace/src/tests/$t.py; done

# curve di training
tensorboard --logdir /workspace/project_workspace/data/vel_controller/results
tensorboard --logdir /workspace/project_workspace/data/pos_controller/results
```

Nel log di training: `|grad|`, `saltate` (aggiornamenti scartati) e, per la velocità, `vel_err_lin/wz`, `tilt`, `ang_vel_max`, `nonfinite_frac`; per la posizione
`dist` (distanza media dal riferimento, un drone fermo vale circa 1.4 m) e `wp` (indice medio del riferimento, 0-2).
I file creati dal container sono di root: lanciare `chown_me.sh` prima di modificarli dall'host.
Se manca il container: `docker run --rm --gpus all` con i volumi di `run.sh`.

## 6. Cose imparate

- **Collasso del training (velocità v3):** con lr costante e picchi di gradiente (|grad| da 0.1 a 14000) la policy è crollata all'it. ~1900 (ω fino a 190 rad/s, 87% di reset, senza NaN).
  Rimedi: `--grad_clip`, reset per |ω| > 30 rad/s, scelta di `best.pt` col costo monitor. Possibile ulteriore passo: lr decrescente.
- **Instabilità della rotazione:** con momenti nulli a velocità angolare alta l'integrazione diverge a NaN (da 60-120 rad/s in pochi secondi). Un solo drone in questo stato
  rendeva NaN le statistiche di normalizzazione e quindi tutta la rete: ora il normalizzatore ignora righe non finite e `done()` resetta stati non finiti.
- **Pesi del costo:** cambiare più pesi insieme dà risultati non interpretabili (v4 peggiorò). Si cambia un peso alla volta; la loss non è confrontabile tra run con pesi diversi.
  Il peso sulle azioni controlla l'aggressività e il tremolio (valori alti → più oscillazioni a riposo nel primo controllore diretto).
- **Cascata:** il gradiente attraverso la rete di velocità congelata e la fisica coincide con le differenze finite (errore relativo 1e-4 - 2e-3, `test_cascade.py`).
- **Decollo:** in simulazione (senza pavimento) il drone sale da z = 0 a 1 m in circa 1 s senza cedere. Nella realtà ci sono effetto suolo, contatto e limite di spinta
  (1.5 su 1.536 N) e il Tello vero accetta comandi di velocità (`rc`): non è un test di decollo reale.
- **Sistema troppo ideale:** nessun rumore, disturbo, resistenza dell'aria, ritardo né dinamica dei motori (anche in Isaac Lab, versione djitello, la forza è applicata al corpo;
  la versione più recente ha una resistenza dell'aria F = 8e-3·|v_rel|·v_rel e vento). Il drone non oscilla perché non c'è nulla che lo disturbi.
- **Un solo kernel Warp scritto da noi**, `apply_wrench`; il resto è Newton.

## 7. Cosa vogliamo fare

Prossimi passi, in ordine indicativo:

1. **Addestrare sul ramo `traiettorie`** (waypoint + traiettorie, 33 osservazioni) e confrontare con `apg_pos_cascade_v3` (waypoint) e l'errore di inseguimento
   su cerchi, otto, ellissi, eliche. Poi eventuale curriculum sulla velocità e anteprima anche sui waypoint. Se non funziona: `git switch main`.
2. **Realismo, uno alla volta:** resistenza dell'aria lineare e quadratica in `apply_wrench`, disturbi (vento/forze casuali), ritardo e filtro del primo ordine sui motori,
   rumore sulle osservazioni (anche in valutazione), variazione di massa e spinta. Si parte da resistenza + un disturbo.
3. **Multi-drone:** una sola rete con pesi condivisi usata in modo decentralizzato da ogni drone (funziona anche con un drone solo), osservazioni dei vicini a slot
   con flag di presenza e numero variabile, costo di squadra con termine liscio di separazione. Con APG non servono IPPO/MAPPO né critic. Opzione: staccare dal grafo gli stati dei vicini.
4. **Confronto con Isaac Lab:** tempi di training e qualità dei risultati a parità di cascata e di frequenze.
5. **Manutenzione:** volume per la cache dei kernel Warp e porta 6006 per tensorboard in `run.sh`; i container girano come root; sistemare su GitHub:
   rimuovere `data/pos_controller/policy/apg_25hz_3wp/last.pt` (corrotto) e aggiornare il percorso in `.gitignore`.

## 8. Note di lavoro

- I training lunghi li lancia l'utente: si prepara il comando e si spiega cosa guardare nel log.
- Il progetto Isaac Lab di riferimento è in `/home/piero/Scrivania/Project_Favuzzi/...` (`tasks/direct/pos_controller` e `vel_controller`);
  copia su GitHub: *Position-and-Velocity-Controller-Isaaclab-Djitello*.
- Repository di questo progetto: https://github.com/pieromutasci-a11y/Newton_Project (ramo `main`).
- Documentazione di Newton: https://newton-physics.github.io/newton/latest/ (esempio di riferimento `newton/examples/diffsim/example_diffsim_drone.py`).
- Lavori su APG e simulazione differenziabile citati a memoria (da verificare prima di una bibliografia): Wiedemann et al. *Training Efficient Controllers via Analytic Policy Gradient* (ICRA 2023);
  Mora et al. *PODS* (ICML 2021); Xu et al. *SHAC* (ICLR 2022).
