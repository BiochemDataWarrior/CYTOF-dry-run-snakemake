#!/bin/bash -l
#SBATCH --job-name=cytof-dry
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=cytof-dry-%j.out
#SBATCH --error=cytof-dry-%j.err
# usage : sbatch run-pipeline.sh [-n | --forceall | ...]
set -eo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
ENV="${CYTOF_ENV:-cytof}"
conda activate "$ENV" || { echo "ERROR: env '$ENV' not found"; conda env list; exit 1; }
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg
echo "dir: $PWD | env: $CONDA_PREFIX | snakemake: $(command -v snakemake || echo MISSING)"

snakemake --snakefile snakefile --configfile config.yaml \
    --cores "${SLURM_CPUS_PER_TASK:-16}" --rerun-incomplete --printshellcmds --latency-wait 30 "$@"

grep -h "gating:" results/logs/01_preprocess_*.log || true
grep -h "ARI\|DA:\|DS:" results/logs/02_*.log results/logs/03_*.log || true
echo "done $(date)"
