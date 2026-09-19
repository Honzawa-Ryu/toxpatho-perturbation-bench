#!/bin/bash
#SBATCH --job-name=centering_ablation
#SBATCH --partition=large-andre01
#SBATCH --output=/workspace/andre01/honzawa/02-playground/toxpatho-perturbation-bench/logs/notebooks/%j_centering_ablation.out
#SBATCH --error=/workspace/andre01/honzawa/02-playground/toxpatho-perturbation-bench/logs/notebooks/%j_centering_ablation.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --time=04:00:00
# CPU-only, same shape and model count as Exp 0006 (one variant's 500k x D
# embeddings held at a time, plus D x D CKA buffers), so mem=128g is carried
# over from there -- a 5-model run of this script peaked at 68GB. --time is
# pinned at the 240min floor the cluster's job_submit plugin enforces
# regardless of partition (see experiments/0006_.../run_slurm.sh); the
# 5-model run took 20min, so 23 models should land near 90min.
#
# Not routed through scripts/slurm_entry.sh: that helper requires an
# experiments/${EXP_NAME}/run_slurm.sh to read its time limit from, and this
# is a notebooks/ analysis rather than a pipeline experiment.

set -euo pipefail

PROJECT_ROOT="/workspace/andre01/honzawa/02-playground/toxpatho-perturbation-bench"

apptainer exec \
    --bind "/scratch/${USER}" \
    "${PROJECT_ROOT}/env.sif" \
    bash -c "
        set -euo pipefail
        source ${PROJECT_ROOT}/.venv/bin/activate
        cd ${PROJECT_ROOT}
        python notebooks/centering_ablation.py
    "
