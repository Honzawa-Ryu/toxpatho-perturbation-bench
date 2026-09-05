#!/bin/bash
#SBATCH --job-name=0006_20260904_eval_representation_shift
#SBATCH --partition=large-andre01
#SBATCH --output=/workspace/andre01/honzawa/02-playground/toxpatho-perturbation-bench/logs/0006_20260904_eval_representation_shift/%j_0006_20260904_eval_representation_shift.out
#SBATCH --error=/workspace/andre01/honzawa/02-playground/toxpatho-perturbation-bench/logs/0006_20260904_eval_representation_shift/%j_0006_20260904_eval_representation_shift.out
#SBATCH --signal=B:USR1@144
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --time=04:00:00
# CPU-only (reads pre-computed embeddings from Exp 0003, no encoder forward
# pass) -- no --gres=gpu. mem=128g: per model we hold one full variant's
# (N=500k, D<=4608) embeddings.parquet in memory at a time (same shape Exp
# 0004 budgeted 96g for), plus the D x D covariance/CKA computation's
# transient buffers on top. time=04:00:00 (large-andre01's own ceiling):
# 23 models x 2 variants x ~25 (kind,level) groups, each a D x D matmul
# (dominated by D<=4608) -- comfortable margin over Exp 0004's proven
# 02:00:00 for a single-variant pass. (A 03:00:00 request was rejected by
# the cluster's job_submit plugin with "x-large-andre01 requires >= 240
# minutes" even when --partition=large-andre01 was set explicitly --
# empirically only time_limit >= 240min got past it; the deployed plugin
# apparently differs from /etc/slurm/job_submit.lua as read from this node.)

# 他の実験のジョブに依存させたい場合、有効化してjob_idを埋める
# （job_idは outputs/{依存先exp}/latest_job_id.txt を参照。投入のたびに
#  変わりうる値なので、都度手動で書き換えること）:
# #SBATCH --dependency=afterok:<job_id>

# Array run にする場合、上の3行の --output/--error/この直後の --array を
# 以下の2行に置き換える（%j→%A_%a、--array=0-N を追加。Nの決め方は下記参照）:
# #SBATCH --output=/workspace/andre01/honzawa/02-playground/toxpatho-perturbation-bench/logs/0006_20260904_eval_representation_shift/%A_%a_0006_20260904_eval_representation_shift.out
# #SBATCH --error=/workspace/andre01/honzawa/02-playground/toxpatho-perturbation-bench/logs/0006_20260904_eval_representation_shift/%A_%a_0006_20260904_eval_representation_shift.out
# #SBATCH --array=0-N
#
# ⚠️ 注意: リソース(--gres/--cpus-per-task/--mem/--time)を変更したら、
#          --partition と --signal のマージンも合わせて手動で見直すこと
#          （make create_exp 実行時に一度だけ計算されたもので、自動追従しない）。
# ⚠️ 注意: シェル上での for/while ループによる複数組み合わせ実行は推奨しない。
#          下記の Array run / Seq run の使用を推奨。

export PROJECT_ROOT="/workspace/andre01/honzawa/02-playground/toxpatho-perturbation-bench"
export EXP_NAME="0006_20260904_eval_representation_shift"

# Apptainer image used by scripts/slurm_entry.sh (--nv + .venv activate inside
# the container). Hardcoded here (rather than relying on the submitting
# shell's env) so a job never silently falls back to running on the bare host
# just because SIF_PATH wasn't exported at submission time.
export SIF_PATH="${PROJECT_ROOT}/env.sif"

# =====================================================
# Storage
# /workspace はNFS（遅い）、/scratch はノード付属のm.2 SSD（速い・ジョブ終了時に
# 自動削除）。デフォルトで有効。NFS越しに直接読み書きしたい場合のみ0にする
# （例: 出力を実行中にリアルタイムで/workspace側から監視したい等）。
# =====================================================

# This experiment reads from outputs/0003_.../ (not data/), so staging
# data/ (644GB) to scratch would be pure waste.
USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=1

# =====================================================
# python path
# =====================================================

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"

# =====================================================
# Single run（デフォルト）
# =====================================================

RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

# =====================================================
# Array run にしたい場合
#
# 1. 上の RUN_MODE="single" と RUN_COMMAND=... をコメントアウトする
# 2. 下のブロックを有効化する
# 3. ファイル先頭の --output/--error/--array の3行を%A_%a版に切り替える
#    （Nは GRID_VALUES の組み合わせ数-1。make preflight が一致を検証する）
#
# GRID_ARGS[i] と GRID_VALUES[i] が対応し、直積が CONFIGS として展開される。
# 例:
#   GRID_ARGS=("--model" "--dataset")
#   GRID_VALUES=("bert roberta" "pubmed pmc")
#   → --model bert --dataset pubmed / --model bert --dataset pmc / ...
# =====================================================

# RUN_MODE="array"
# BASE_COMMAND="python ${PYTHON_PATH}"
# GRID_ARGS=(
#     "--model"
#     "--dataset"
# )
# GRID_VALUES=(
#     "google/gemma-4-31b-it meta-llama/Llama-3-8b-it"
#     "BC5CDR BIORED"
# )

# =====================================================
# Seq run にしたい場合（1ジョブ内でGRIDを順次実行）
#
# 上と同様に RUN_MODE="seq" にし、BASE_COMMAND/GRID_ARGS/GRID_VALUES を設定する。
# こちらは #SBATCH --array は不要（1ジョブでループするため）。
# =====================================================

# RUN_MODE="seq"
# BASE_COMMAND="python ${PYTHON_PATH}"
# GRID_ARGS=(
#     "--model"
# )
# GRID_VALUES=(
#     "bert roberta"
# )

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
