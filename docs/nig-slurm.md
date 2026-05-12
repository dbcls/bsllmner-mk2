# NIG Slurm

Running bsllmner-mk2 as a Slurm job on the NIG GPU compute environment.

## Prerequisites

### SSH

Add to `~/.ssh/config`:

```
Host nig-gw-3
    HostName gwa3.ddbj.nig.ac.jp
    Port 22
    User <your-username>
    IdentityFile ~/.ssh/<your-key>.pem

Host nig-gpu
    HostName h200-01
    Port 22
    User <your-username>
    ProxyJump nig-gw-3
```

### Software on the Compute Node

- Docker with NVIDIA Container Toolkit
- Slurm

## Setup

### 1. Clone Repository

```bash
cd /home/<your-username>/git/github.com/dbcls
git clone https://github.com/dbcls/bsllmner-mk2.git
cd bsllmner-mk2
```

### 2. Generate slurm.sh

```bash
./init-slurm.sh                     # defaults: 1 GPU, h200, 168 h
./init-slurm.sh -g 4 -t 72:00:00    # 4 GPUs, 72 h
./init-slurm.sh -g 8 -m 256G        # 8 GPUs, 256 GB memory
./init-slurm.sh -b                  # also build the Docker image on job start
```

| Flag | Default | Description |
|---|---|---|
| `-g, --gpus` | `1` | GPU count. |
| `-p, --partition` | `h200` | Slurm partition. |
| `-c, --cpus` | `32` | CPUs per task. |
| `-m, --mem` | `128G` | Memory allocation. |
| `-t, --time` | `168:00:00` | Time limit. |
| `-b, --build` | off | Pass `--build` to `docker compose` when the job starts. |
| `-f, --force` | off | Overwrite existing `slurm.sh` without prompting. |

`slurm.sh` is gitignored.

### 3. Create the Docker Network

```bash
docker network create bsllmner-mk2-network
```

### 4. Create the Ollama Data Directory

```bash
mkdir -p ollama-data
```

### 5. Run Preparation Scripts

Prepare ChIP-Atlas BioSample entries (see [ChIP-Atlas](chip-atlas.md)) before submitting the Slurm job:

```bash
docker compose -f compose.yml up app -d --build
docker compose -f compose.yml exec app python3 scripts/prepare_bs_entries.py --genome-assembly hg38
docker compose -f compose.yml down
```

Ontology setup (download + subset build + NCBI Gene OWL) is documented in [Ontology Preparation](ontology.md).

## Running Slurm Jobs

```bash
sbatch slurm.sh             # submit
squeue -u $USER             # status
scancel <job-id>            # cancel
tail -f slurm-logs/bsllmner2-ollama-<job-id>.out     # stdout
tail -f slurm-logs/bsllmner2-ollama-<job-id>.err     # stderr
```

`slurm.sh` reads `SLURM_JOB_GPUS`, rewrites the `__DEVICE_IDS__` placeholder in `compose.slurm.yml.template` into a JSON array, and brings up `bsllmner-mk2-app` + `bsllmner-mk2-ollama` via `docker compose -f compose.slurm.yml up -d --force-recreate`. The job stays alive (`tail -f /dev/null`) so subsequent `docker exec` calls can run `bsllmner2_extract` / `bsllmner2_select` inside the container. On exit the trap runs `docker compose -f compose.slurm.yml down`.

## Running the Application

```bash
docker exec -it bsllmner-mk2-app bash

# Inside the app container
bsllmner2_select \
  --bs-entries tests/data/example_biosample.json \
  --model llama3.1:70b \
  --select-config ./scripts/select-config-hg38.json \
  --run-name small-test \
  --debug
```

GPU sanity check inside the ollama container:

```bash
docker exec -t bsllmner-mk2-ollama nvidia-smi
```

## Troubleshooting

**GPUs not visible.** Verify Slurm GPU allocation and `--gres=gpu:N`:

```bash
env | grep -E 'SLURM_.*GPU|CUDA_VISIBLE_DEVICES'
```

`SLURM_JOB_GPUS` or `SLURM_STEP_GPUS` must be set; otherwise `slurm.sh` aborts.

**Container won't start.** Confirm the network exists, remove stale containers, and check that `compose.slurm.yml` was rewritten:

```bash
docker network ls | grep bsllmner-mk2-network
docker rm -f bsllmner-mk2-app bsllmner-mk2-ollama 2>/dev/null
grep device_ids compose.slurm.yml
```

**Ollama not responding.** Inspect the container logs and GPU memory:

```bash
docker logs bsllmner-mk2-ollama
docker exec -t bsllmner-mk2-ollama nvidia-smi
```

## File Reference

| File | Description |
|---|---|
| `init-slurm.sh` | Generates `slurm.sh` from the template. |
| `slurm.sh.template` | Slurm job script template (placeholders `__NUM_GPUS__`, `__PARTITION__`, ...). |
| `slurm.sh` | Generated job script (gitignored). |
| `compose.slurm.yml.template` | Docker Compose template with a `__DEVICE_IDS__` placeholder. |
| `compose.slurm.yml` | Generated at job start (gitignored). |
| `slurm-logs/` | Slurm stdout / stderr (gitignored). |
