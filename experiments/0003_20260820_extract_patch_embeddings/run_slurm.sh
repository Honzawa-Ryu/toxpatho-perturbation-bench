#!/bin/bash
#SBATCH --job-name=0003_20260820_extract_patch_embeddings
#SBATCH --partition=x-large-andre01
#SBATCH --output=/workspace/andre01/honzawa/02-playground/toxpatho-perturbation-bench/logs/0003_20260820_extract_patch_embeddings/%A_%a_0003_20260820_extract_patch_embeddings.out
#SBATCH --error=/workspace/andre01/honzawa/02-playground/toxpatho-perturbation-bench/logs/0003_20260820_extract_patch_embeddings/%A_%a_0003_20260820_extract_patch_embeddings.out
#SBATCH --array=0-29
#SBATCH --signal=B:USR1@144
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=04:00:00

# 他の実験のジョブに依存させたい場合、有効化してjob_idを埋める
# （job_idは outputs/{依存先exp}/latest_job_id.txt を参照。投入のたびに
#  変わりうる値なので、都度手動で書き換えること）:
# #SBATCH --dependency=afterok:<job_id>

# ⚠️ 注意: リソース(--gres/--cpus-per-task/--mem/--time)を変更したら、
#          --partition と --signal のマージンも合わせて手動で見直すこと
#          （make create_exp 実行時に一度だけ計算されたもので、自動追従しない）。

export PROJECT_ROOT="/workspace/andre01/honzawa/02-playground/toxpatho-perturbation-bench"
export EXP_NAME="0003_20260820_extract_patch_embeddings"
export SIF_PATH="${PROJECT_ROOT}/env.sif"

# =====================================================
# Storage
# This experiment reads from outputs/0001_.../ and outputs/0002_.../
# (not data/), so staging data/ (644GB) to scratch would be pure waste.
# =====================================================

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=1

# =====================================================
# python path
# =====================================================

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"

# =====================================================
# Array run: 1 job per TRIDENT patch encoder (30 models -- the full
# encoder_factory roster minus the two Gemma-4 vision-tower models,
# which are a different weight class and not the kind of patch/SSL
# encoder this benchmark targets). Non-gated and gated (HF_TOKEN-backed)
# models are mixed together deliberately: each array task is independent,
# so a gated model we don't actually have access to just fails its own
# task without affecting the rest.
# =====================================================

RUN_MODE="array"
BASE_COMMAND="python ${PYTHON_PATH}"
GRID_ARGS=(
    "--model"
)
GRID_VALUES=(
    "conch_v1 conch_v15 uni_v1 uni_v2 ctranspath phikon phikon_v2 resnet50 keep gigapath gigapath-flash virchow virchow2 virchow2-cls hoptimus0 hoptimus1 h0-mini phaet mascaret musk openmidnight gpfm hibou_l kaiko-vitb8 kaiko-vitb16 kaiko-vits8 kaiko-vits16 kaiko-vitl14 lunit-vits8 genbio-pathfm"
)

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
