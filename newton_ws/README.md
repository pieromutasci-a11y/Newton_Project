# Multi-drone con Newton: appunti di progetto

Questo file riassume le decisioni prese e lo stato del lavoro, per riprenderlo su un'altra macchina
(anche con una nuova sessione di Claude: leggere questo file per primo).

## 1. Obiettivo

Addestrare **più droni (multi-agent)** con:
- simulazione e training **interamente su GPU**, evitando di saturare la CPU;
- **differentiable learning** (gradienti attraverso la simulazione) per training più rapido e preciso;
- tutto in Python, con kernel **Warp** per la parte fisica e **PyTorch** per la rete.

Scelta: **Newton "puro"** (non Isaac Lab/Isaac Sim con PhysX). Vantaggi: kernel custom su GPU,
differenziabilità, leggerezza, open source. Prezzo: **nessun framework di RL pronto**
(ambienti, osservazioni, reward, reset, domain randomization vanno scritti a mano),
meno documentazione e meno asset, API in evoluzione.
La velocità di training non è garantita a priori: anche PhysX gira su GPU, dipende da caso e solver.
Isaac Lab + Newton esiste ma, nella documentazione di Isaac Lab, è tra le *experimental features*.

## 2. Ambiente (Docker)

Newton gira in un container con GPU NVIDIA. Requisiti dalla documentazione di Newton:
- Python 3.11+ (lo gestisce `uv`);
- driver NVIDIA **R580+** per il wheel standard di Warp 1.18 (macchina di lavoro: RTX A1000, driver 580.173);
  su R545-R579 serve `warp-lang==1.17.0` (vedi installation guide);
- non serve il CUDA Toolkit locale.

Struttura sull'host (macchina di lavoro, `/home/piero/Scrivania/Newton/`):
```
docker_ws/Dockerfile.newton_examples   # definizione dell'immagine
docker_ws/build.sh                     # docker build -f Dockerfile.newton_examples -t newton-examples .
run.sh                                 # avvia il container
exec.sh                                # docker exec -it newton-lab-Mutasci bash (seconda shell)
newton_ws/                             # QUESTO progetto, montato nel container
```

**Dockerfile** (clone di Newton + `uv sync`, come da development guide, metodo 1):
```dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive NVIDIA_DRIVER_CAPABILITIES=all UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential \
    libgl1 libglu1-mesa libegl1 libglx-mesa0 \
    libx11-6 libxrandr2 libxinerama1 libxcursor1 libxi6 \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
WORKDIR /workspace
RUN git clone https://github.com/newton-physics/newton.git
WORKDIR /workspace/newton
RUN uv sync --extra examples
CMD ["bash"]
```

**run.sh**: monta `newton_ws` (host) su `/workspace/project_workspace` (container). Il resto,
`/workspace/newton` (repo + `.venv`), resta quello dell'immagine e non è mappato. Usa GPU (`--gpus all`) e X11
(`xhost +local:docker`, `DISPLAY`) per le finestre. Nome container fisso: `newton-lab-Mutasci`, container `--rm`.

Uso:
```bash
./docker_ws/build.sh                                   # una volta
./run.sh                                               # shell nel container (cwd /workspace/newton)
./run.sh uv run -m newton.examples diffsim_drone       # lancia un esempio
# dentro il container, per codice del progetto:
cd /workspace/newton && uv run python /workspace/project_workspace/src/<script>.py
```

> **STATO: il build dell'immagine non è mai stato completato né verificato** (interrotto la prima volta),
> e nessun codice del progetto è stato eseguito. Tutto va provato al primo avvio.

Alternativa documentata per chi *usa* Newton come libreria (non lo sviluppa): `pip install "newton[examples]"`
in un venv (installation guide). Il flusso clone + `uv sync` è quello della development guide.
Considerata ma non adottata: Dockerfile con `uv venv` + `uv pip install "newton[examples]"`, senza clone.

Problemi noti/rischi dell'impostazione attuale:
- il `git clone` nell'immagine è in cache: un nuovo build non aggiorna Newton senza `--no-cache`;
- container `--rm` e girano come root: cache dei kernel Warp persa a ogni avvio (prima esecuzione lenta;
  soluzione: volume su `~/.cache/warp`) e file creati in `newton_ws` di proprietà root (`--user $(id -u):$(id -g)`);
- `uv run` può provare a risincronizzare il venv (`--frozen` lo evita), oppure usare `/workspace/newton/.venv/bin/python`
  o mettere `/workspace/newton/.venv/bin` nel `PATH` dell'immagine.

## 3. Concetti chiave (cosa si è chiarito)

- **uv**: un solo tool al posto di pip + venv + pyenv; non è un acronimo. `uv sync` crea il venv da `pyproject.toml`/`uv.lock`;
  `uv run` esegue nel venv; `uv pip install` ≈ `pip install`.
- **Kernel Warp**: funzione `@wp.kernel` scritta *per un solo elemento*, lanciata con `wp.launch(kernel, dim=...)` in parallelo
  sulla GPU; `wp.tid()` dà l'indice del thread. Dentro: solo tipi Warp (`wp.array`, `wp.vec3`, `wp.transform`...), niente liste/dict/numpy/torch.
  I kernel li scrive l'utente; Warp fornisce decoratore, tipi, funzioni matematiche e `wp.Tape` (gradienti).
- **Warp vs Torch**: Warp per fisica, costi, sensori; Torch per rete, ottimizzatore, training.
  Interscambio senza copie: `wp.to_torch(arr)` / `wp.from_torch(t)`. Per i gradienti attraverso la simulazione nel training
  con Torch serve avvolgere il rollout Warp in una `torch.autograd.Function` che chiama `tape.backward()`
  (da verificare nella documentazione quando si arriva al training). Mai `.numpy()` nel ciclo di training (copia su CPU, rompe i gradienti).
- **ModelBuilder**: classe di Newton; il "cantiere" a cui si aggiungono corpi/forme (`add_body`, `add_shape_box`, ...);
  `finalize()` produce il `Model` sulla GPU (massa e inerzia calcolate da forme e densità). La costruzione è Python normale su CPU, una volta sola.
- **Stato**: `State.body_q` (transform: posizione + quaternione) e `State.body_qd` (spatial vector: velocità lineare+angolare;
  **ordine delle componenti da verificare**). Indice del corpo del mondo `w`, drone `d`: `w * corpi_per_mondo + d`.
- **Decoratori**: `@dataclass`, `@property`, `@wp.kernel`, `@wp.struct`...; `@dataclass` genera `__init__`/`__repr__`.

## 4. Come funziona l'esempio `diffsim_drone` (riferimento)

File: `newton/examples/diffsim/example_diffsim_drone.py` (nel clone dentro l'immagine). **Non è training di una rete**:
è MPC con ottimizzazione di traiettoria tramite gradiente attraverso la fisica.
- **Azioni**: 4 spinte dei propeller in [0.1, 1.0]. Variabile ottimizzata: 4 punti di controllo × 4 motori (distanza 10 step, orizzonte 30 step = 0.5 s a 60 fps),
  interpolati linearmente (`interpolate_control_linear`), moltiplicati per il gain 0.8, poi `compute_prop_wrenches`:
  forza = dir·max_thrust·comando; coppia di reazione = dir·max_torque·comando·verso + braccio×forza; sul corpo come `body_f`.
- **Costo** (`drone_cost`, per step): distanza² dal target (peso 1000), quota fuori da [0.25, 0.75] (100), inclinazione `1 - dot(su, z)` (10),
  velocità² (0.1), comando² (0.05); normalizzato per la somma dei pesi e per `0.8^(H-step-1)/H²` (conta di più la fine dell'orizzonte).
  Più `collision_cost` con SDF dal palo (margine = raggio, peso 1e4).
- **Ciclo per frame**: 16 rollout campionati attorno alla traiettoria migliore (rumore σ=0.15) → 20 iterazioni di forward (30 step, 16 mondi in parallelo)
  + backward (`wp.Tape`) + SGD (lr 1e-2) + clamp → si sceglie il rollout a costo minimo → si simula **solo il primo step** sul drone vero (receding horizon).
  Forward/backward in un CUDA graph (`wp.ScopedCapture`). Campionamento + gradiente: ibrido perché il gradiente attraverso i contatti è inaffidabile.
- Note: il rollout salva tutti gli stati intermedi (memoria); l'esempio usa `SolverSemiImplicit`.
  Stranezza: `rps = max_rpm / fps` (dimensionalmente dovrebbe essere `/60`; funziona perché fps=60).
- Massa dell'esempio: non esplicita, deriva da densità 1750 kg/m³ e due scatole (semi-estensioni 0.01×0.2×0.01, `drone_size=0.2`):
  stimata ≈ **0.56 kg** (mio calcolo, da verificare con `model.body_mass`). Spinta max ≈ 4.18 N/rotore, spinta/peso ≈ 3, hover a comando ≈ 0.33.

## 5. Architettura del progetto (`src/`)

Regola: file per responsabilità; `@dataclass` per i parametri; kernel e funzioni per la fisica; **classi solo dove c'è stato** (World, policy, trainer).
Tutto lo stato in `wp.array` con indici di mondo/drone (niente oggetti per drone); nessun `.numpy()` nel rollout differenziabile; forward/backward in CUDA graph.

```
src/
  drone_model/            # com'è fatto il drone, indipendente dal compito
    params.py             # DroneParams (SI)                                  [FATTO]
    actuators.py          # struct Propeller [FATTO]; kernel forze/coppie [DA FARE]
    builder.py            # add_drone(builder, params, label, collision_group) [FATTO]
    sensors.py            # osservazioni (inizialmente stato esatto)          [DA FARE]
  scene/
    world.py              # classe World: ModelBuilder/finalize, N mondi x M droni, Model, State, Control, solver, step/reset  [DA FARE]
    task.py               # costo/reward come kernel (target, quota, collisioni, separazione tra droni) [DA FARE]
    reset.py              # reset per mondo con condizioni iniziali randomizzate [DA FARE]
  train/
    policy.py             # rete PyTorch                                      [DA FARE]
    rollout.py            # rollout differenziabile con wp.Tape               [DA FARE]
    trainer.py            # ciclo di training                                 [DA FARE]
  eval/
    evaluate.py           # valutazione/visualizzazione                       [DA FARE]
  configs/                # parametri in un file                              [DA FARE]
```

### Stato del codice scritto (non eseguito)
- `params.py`: `DroneParams` con geometria (semi-estensioni), densità, coefficienti propeller dell'esempio, limiti del comando;
  proprietà derivate `max_rps` (= max_rpm/60), `max_thrust` (Ct·ρ·n²·D⁴), `max_torque` (Cp·ρ·n²·D⁵/2π).
- `actuators.py`: `@wp.struct Propeller` (body, pos, dir, max_thrust, max_torque, turning_direction).
- `builder.py`: `add_drone` aggiunge il corpo, due barre a croce (box, densità da `params`), 4 propeller a ±x/±y con versi alternati
  (-1, +1, +1, -1); ritorna `(DroneHandle(body), lista_propeller)`. `World` concatenerà i propeller di tutti i droni in un `wp.array`.
  `ModelBuilder` viene creato e chiuso (`finalize`) da `World`, non dal builder.

### Test da fare per primo (nel container)
```bash
cd /workspace/newton
uv run python - <<'E'
import sys; sys.path.insert(0, "/workspace/project_workspace/src")
import newton
from drone_model.params import DroneParams
from drone_model.builder import add_drone
p = DroneParams(); b = newton.ModelBuilder()
h, props = add_drone(b, p, "drone_0"); m = b.finalize()
print("massa", m.body_mass.numpy(), "kg")
print("spinta/peso", 4*p.max_thrust / (m.body_mass.numpy()[0]*9.81))
E
```
Atteso: massa ≈ 0.56 kg, spinta/peso ≈ 3 (la massa potrebbe differire un po' se il builder non sottrae la sovrapposizione delle barre).

## 6. Prossimi passi

1. Far partire il build dell'immagine e verificare il test qui sopra; lanciare `diffsim_drone` per vedere l'esempio.
2. `actuators.py`: kernel `compute_prop_wrenches` (comando → forze/coppie), poi prova di caduta libera a motori spenti e hover a ≈0.33.
3. `scene/world.py`: N mondi in parallelo × M droni, con `collision_group` per mondo; reset per mondo.
4. `scene/task.py`: costo/reward come kernel, con termini multi-agent (separazione tra droni, formazione).
5. `train/`: policy in PyTorch, rollout brevi differenziabili, training con backward attraverso la simulazione
   (famiglia di metodi tipo APG/SHAC, citati dalla memoria: da cercare in letteratura).
6. Rischi da provare presto: quanti mondi/droni entrano in memoria GPU (il backward salva tutti gli stati); stabilità dei gradienti su orizzonti lunghi e nelle collisioni; **quali solver di Newton sono differenziabili** (l'esempio usa `SolverSemiImplicit`, il backend principale è MuJoCo Warp: da controllare).

## 7. Decisioni ancora aperte

- **Interfaccia di controllo**: la policy produce la spinta dei 4 motori direttamente (come l'esempio), oppure spinta collettiva + velocità angolari
  (controllore di rate sotto), oppure setpoint di velocità/posizione? Cambia `actuators.py` e l'eventuale `controller.py`.
- Realismo: dinamica dei motori (filtro del primo ordine), drag aerodinamico, vento/disturbi, effetti tra droni vicini: da aggiungere uno alla volta.
- Sensori: stato esatto all'inizio; IMU rumorosa e osservazioni relative dei vicini solo se serve il trasferimento su hardware reale.
- Randomizzazione dei parametri fisici tra i mondi paralleli.
- Massa/inerzia: calcolate dalla geometria (attuale) oppure date direttamente da valori misurati?

## 8. Riferimenti

- Repo: https://github.com/newton-physics/newton (esempio in `newton/examples/diffsim/example_diffsim_drone.py`)
- Installazione utente: `pip install "newton[examples]"` in un venv, poi `python -m newton.examples --list` (guide/installation)
- Sviluppo da sorgente: `git clone` + `uv sync --extra dev` (guide/development)
- Documentazione: https://newton-physics.github.io/newton/latest/
- Pacchetto PyPI: `newton` (vecchio nome `newton-physics`, deprecato)
