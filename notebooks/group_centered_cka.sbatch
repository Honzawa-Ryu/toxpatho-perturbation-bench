#!/bin/bash
#SBATCH --job-name=group_centered_cka
#SBATCH --partition=large-andre01
#SBATCH --output=/workspace/andre01/honzawa/02-playground/toxpatho-perturbation-bench/logs/notebooks/%j_group_centered_cka.out
#SBATCH --error=/workspace/andre01/honzawa/02-playground/toxpatho-perturbation-bench/logs/notebooks/%j_group_centered_cka.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --time=04:00:00
# Needs the perturbed rows, so this reads the full 154GB like
# centering_ablation.py did (47min there, and this computes a comparable
# number of CKAs) -- mem=128g carried over, where that run peaked at 63GB.
# --time is both the plugin's 240min floor and large-andre01's ceiling.

set -euo pipefail

PROJECT_ROOT="/workspace/andre01/honzawa/02-playground/toxpatho-perturbation-bench"

apptainer exec \
    --bind "/scratch/${USER}" \
    "${PROJECT_ROOT}/env.sif" \
    bash -c "
        set -euo pipefail
        source ${PROJECT_ROOT}/.venv/bin/activate
        cd ${PROJECT_ROOT}
        python notebooks/group_centered_cka.py
    "
