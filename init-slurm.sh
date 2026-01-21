#!/usr/bin/env bash
# Generate slurm.sh from slurm.sh.template with configurable parameters

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
NUM_GPUS=2
PARTITION="h200"
CPUS=32
MEM="128G"
TIME="168:00:00"
BUILD=false
FORCE=false

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Generate slurm.sh from template with specified parameters.

Options:
  -g, --gpus NUM       Number of GPUs (default: $NUM_GPUS)
  -p, --partition NAME Slurm partition (default: $PARTITION)
  -c, --cpus NUM       CPUs per task (default: $CPUS)
  -m, --mem SIZE       Memory allocation (default: $MEM)
  -t, --time TIME      Time limit (default: $TIME)
  -b, --build          Enable docker image build on job start
  -f, --force          Overwrite existing slurm.sh without prompt
  -h, --help           Show this help message

Examples:
  $(basename "$0")                    # Use defaults (2 GPUs, h200 partition)
  $(basename "$0") -g 4 -t 72:00:00   # 4 GPUs, 72 hour limit
  $(basename "$0") -g 8 -m 256G       # 8 GPUs, 256GB memory
  $(basename "$0") -b                 # Enable docker build
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -g|--gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    -p|--partition)
      PARTITION="$2"
      shift 2
      ;;
    -c|--cpus)
      CPUS="$2"
      shift 2
      ;;
    -m|--mem)
      MEM="$2"
      shift 2
      ;;
    -t|--time)
      TIME="$2"
      shift 2
      ;;
    -b|--build)
      BUILD=true
      shift
      ;;
    -f|--force)
      FORCE=true
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

if [[ -f "$SCRIPT_DIR/slurm.sh" && "$FORCE" != true ]]; then
  echo "slurm.sh already exists. Overwrite? (y/N)"
  read -r answer
  if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
    echo "Aborted."
    exit 0
  fi
fi

# Set build flag
BUILD_FLAG=""
if [[ "$BUILD" == true ]]; then
  BUILD_FLAG="--build"
fi

# Replace placeholders
sed \
  -e "s|__ROOT__|${SCRIPT_DIR}|g" \
  -e "s|__NUM_GPUS__|${NUM_GPUS}|g" \
  -e "s|__PARTITION__|${PARTITION}|g" \
  -e "s|__CPUS__|${CPUS}|g" \
  -e "s|__MEM__|${MEM}|g" \
  -e "s|__TIME__|${TIME}|g" \
  -e "s|__BUILD_FLAG__|${BUILD_FLAG}|g" \
  "$SCRIPT_DIR/slurm.sh.template" > "$SCRIPT_DIR/slurm.sh"

chmod +x "$SCRIPT_DIR/slurm.sh"

echo "Generated: $SCRIPT_DIR/slurm.sh"
echo ""
echo "Configuration:"
echo "  Partition: $PARTITION"
echo "  GPUs:      $NUM_GPUS"
echo "  CPUs:      $CPUS"
echo "  Memory:    $MEM"
echo "  Time:      $TIME"
echo "  Build:     $BUILD"
echo ""
echo "Next steps: see ./docs/nig-slurm.md"
