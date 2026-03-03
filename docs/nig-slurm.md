# Running on NIG Slurm Environment

Instructions for running bsllmner-mk2 as a Slurm job on NIG computing environment.

## Prerequisites

### SSH Configuration

Add the following to `~/.ssh/config`:

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

### Required Software

The compute node must have:

- Docker (with NVIDIA Container Toolkit)
- Slurm

## Setup

### 1. Clone Repository

```bash
cd /home/<your-username>/git/github.com/dbcls
git clone https://github.com/dbcls/bsllmner-mk2.git
cd bsllmner-mk2
```

### 2. Generate slurm.sh

Run `init-slurm.sh` to generate `slurm.sh` with your desired configuration:

```bash
# Default: 2 GPUs, h200 partition, 168 hours
./init-slurm.sh

# Custom: 4 GPUs, 72 hour limit
./init-slurm.sh -g 4 -t 72:00:00

# Custom: 8 GPUs, 256GB memory
./init-slurm.sh -g 8 -m 256G

# First run (with docker build)
./init-slurm.sh -b
```

Available options:

| Option | Description | Default |
|--------|-------------|---------|
| `-g, --gpus` | Number of GPUs | 2 |
| `-p, --partition` | Slurm partition | h200 |
| `-c, --cpus` | CPUs per task | 32 |
| `-m, --mem` | Memory allocation | 128G |
| `-t, --time` | Time limit | 168:00:00 |
| `-b, --build` | Enable docker image build | false |
| `-f, --force` | Overwrite without prompt | false |

The generated `slurm.sh` is gitignored, so local changes won't affect the repository.

### 3. Create Docker Network

```bash
docker network create bsllmner-mk2-network
```

### 4. Create Ollama Data Directory

```bash
mkdir -p ollama-data
```

### 5. Run Preparation Scripts

```bash
docker compose -f compose.yml up app -d --build
docker compose -f compose.yml exec app bash

# Inside container
cd scripts
python3 prepare_bs_entries.py --genome-assembly mm10  # or hg38
exit

docker compose -f compose.yml down
```

Ontology files setup (download + OBO→OWL conversion) is described in [Quick Start §2](getting-started.md#2-download-ontology-files).

## Running Slurm Jobs

### Submit Job

```bash
sbatch slurm.sh
```

### Check Job Status

```bash
squeue -u $USER
```

### Cancel Job

```bash
scancel <job-id>
```

### View Logs

```bash
# stdout
tail -f slurm-logs/bsllmner2-ollama-<job-id>.out

# stderr
tail -f slurm-logs/bsllmner2-ollama-<job-id>.err
```

## Running Applications

### Access Container

```bash
docker exec -it bsllmner-mk2-app bash
```

### Example Commands

```bash
# Select mode
bsllmner2_select \
  --debug \
  --bs-entries tests/data/example_biosample.json \
  --model llama3.1:70b \
  --select-config ./scripts/select-config.json \
  --run-name small-test

# Extract mode
bsllmner2_extract \
  --debug \
  --bs-entries tests/data/example_biosample.json \
  --model llama3.1:70b \
  --run-name small-test
```

### Verify GPU Allocation

```bash
docker exec -t bsllmner-mk2-ollama nvidia-smi
```

## Troubleshooting

### GPUs Not Visible

1. Check Slurm GPU allocation:

   ```bash
   env | grep -E 'SLURM_.*GPU|CUDA_VISIBLE_DEVICES'
   ```

2. Verify `SLURM_JOB_GPUS` or `SLURM_STEP_GPUS` is set

3. Confirm `--gres=gpu:N` option is correctly specified

### Container Won't Start

1. Check Docker network exists:

   ```bash
   docker network ls | grep bsllmner-mk2-network
   ```

2. Remove existing containers:

   ```bash
   docker ps -a | grep bsllmner-mk2
   docker rm -f bsllmner-mk2-app bsllmner-mk2-ollama
   ```

3. Verify `compose.slurm.yml` was generated correctly:

   ```bash
   cat compose.slurm.yml | grep device_ids
   # Expected: device_ids: [ "2", "3" ] (or similar)
   ```

### Ollama Not Responding

1. Check Ollama container logs:

   ```bash
   docker logs bsllmner-mk2-ollama
   ```

2. Check GPU memory usage:

   ```bash
   docker exec -t bsllmner-mk2-ollama nvidia-smi
   ```

## File Reference

| File | Description |
|------|-------------|
| `init-slurm.sh` | Setup script (generates slurm.sh with options) |
| `slurm.sh.template` | Slurm job script template |
| `slurm.sh` | Generated Slurm job script (gitignored) |
| `compose.slurm.yml.template` | Docker Compose template |
| `compose.slurm.yml` | Generated Compose file at runtime (gitignored) |
| `slurm-logs/` | Job log output directory (gitignored) |
