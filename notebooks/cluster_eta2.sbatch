#!/bin/bash
#SBATCH --job-name=cluster_eta2
#SBATCH --partition=medium-andre01
#SBATCH --output=/workspace/andre01/honzawa/02-playground/toxpatho-perturbation-bench/logs/notebooks/%j_cluster_eta2.out
#SBATCH --error=/workspace/andre01/honzawa/02-playground/toxpatho-perturbation-bench/logs/notebooks/%j_cluster_eta2.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=02:00:00
# Reads only the unperturbed rows (20k per file), like study_knn_purity.py,
# and works on 20k x 50 after PCA. In a one-model test k-means took 3.4s at
# K=1000 and the 100-shuffle baselines were negligible, so 48 model/variant
# passes fit well inside 2h; mem=32g covers the largest model's 20k x 4608.

set -euo pipefail

PROJECT_ROOT="/workspace/andre01/honzawa/02-playground/toxpatho-perturbation-bench"

apptainer exec \
    --bind "/scratch/${USER}" \
    "${PROJECT_ROOT}/env.sif" \
    bash -c "
        set -euo pipefail
        source ${PROJECT_ROOT}/.venv/bin/activate
        cd ${PROJECT_ROOT}
        python notebooks/cluster_eta2.py
    "
