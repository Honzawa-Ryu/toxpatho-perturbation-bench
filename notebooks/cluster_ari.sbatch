#!/bin/bash
#SBATCH --job-name=cluster_ari
#SBATCH --partition=large-andre01
#SBATCH --output=/workspace/andre01/honzawa/02-playground/toxpatho-perturbation-bench/logs/notebooks/%j_cluster_ari.out
#SBATCH --error=/workspace/andre01/honzawa/02-playground/toxpatho-perturbation-bench/logs/notebooks/%j_cluster_ari.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --time=04:00:00
# CPU-only. Same embedding-loading shape as notebooks/centering_ablation.py
# (one variant's 500k x D array at a time, which peaked at 63GB there), plus
# PCA and k-means on 20k x 50 which are small next to that. --time is both
# the plugin's 240min floor and large-andre01's ceiling, which is why the
# script clusters level 3 only and uses MiniBatchKMeans -- see its docstring.

set -euo pipefail

PROJECT_ROOT="/workspace/andre01/honzawa/02-playground/toxpatho-perturbation-bench"

apptainer exec \
    --bind "/scratch/${USER}" \
    "${PROJECT_ROOT}/env.sif" \
    bash -c "
        set -euo pipefail
        source ${PROJECT_ROOT}/.venv/bin/activate
        cd ${PROJECT_ROOT}
        python notebooks/cluster_ari.py
    "
