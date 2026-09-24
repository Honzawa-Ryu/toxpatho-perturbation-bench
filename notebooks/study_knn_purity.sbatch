#!/bin/bash
#SBATCH --job-name=study_knn_purity
#SBATCH --partition=large-andre01
#SBATCH --output=/workspace/andre01/honzawa/02-playground/toxpatho-perturbation-bench/logs/notebooks/%j_study_knn_purity.out
#SBATCH --error=/workspace/andre01/honzawa/02-playground/toxpatho-perturbation-bench/logs/notebooks/%j_study_knn_purity.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=04:00:00
# Far lighter than the other notebooks jobs: only the unperturbed rows are
# read (20k of each file's 500k), so both variants of one model fit in well
# under a GB and the run is dominated by the 20k x 20k cosine matmuls rather
# than I/O -- mem=32g instead of 128g. --time is the plugin's 240min floor,
# not an estimate; the run is expected to take well under an hour.

set -euo pipefail

PROJECT_ROOT="/workspace/andre01/honzawa/02-playground/toxpatho-perturbation-bench"

apptainer exec \
    --bind "/scratch/${USER}" \
    "${PROJECT_ROOT}/env.sif" \
    bash -c "
        set -euo pipefail
        source ${PROJECT_ROOT}/.venv/bin/activate
        cd ${PROJECT_ROOT}
        python notebooks/study_knn_purity.py
    "
