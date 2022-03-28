#!/usr/bin/env bash
# -*- coding: utf-8 -*-

set -e 

##### example call:
#####   bash scripts/run_conversion.sh /net/cephfs/scratch/tkew/muss/experiments/fairseq/slurmjob_6443932 facebook/bart-large-cn

model_dir=$1 # "/net/cephfs/scratch/tkew/muss/experiments/fairseq/slurmjob_6443932"
config=$2 # /net/cephfs/scratch/tkew/fudge/generators/bart_large_paraNMT_filt_fr/

scripts=$(dirname "$(readlink -f "$0")")

echo "converting model..."

python $scripts/convert_bart_original_pytorch_checkpoint_to_pytorch.py \
    $model_dir/checkpoints/checkpoint_best.pt \
    $model_dir/hf_models \
    --hf_config $config

echo "testing model..."

python $scripts/test_hf_model.py \
    $model_dir/hf_models

echo "done!"