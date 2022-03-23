#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# Wrapper for applying a trained MUSS model on Newsela English corpus
# This scripts performs the following operations:
# 1. organises newsela tsv data for muss
# 2. runs param optimisation on first 50 lines of dev sets
# 3. runs inference with muss simplifier on dev and test sets with best params

# Author: Tannon Kew // kew@cl.uzh.ch


muss="/srv/scratch6/kew/ats/muss"
aligned_data="/srv/scratch6/kew/ats/data/en/aligned"

results=$muss/outputs

budget=50 # parameterization budget, i.e. number of lines in validation file

export CUDA_VISIBLE_DEVICES=$1

# Step 1: organise newsela tsv data for muss processing
for level in 1 2 3 4; do
    dataset=$muss/resources/datasets/newsela_v0_v$level
    for split in train test dev; do
        mkdir -p $dataset
        cut -f 1 $aligned_data/newsela_manual_v0_v${level}_${split}.tsv >| $dataset/$split.complex
        cut -f 2 $aligned_data/newsela_manual_v0_v${level}_${split}.tsv >| $dataset/$split.simple
    done
    head -n $budget $dataset/dev.complex >| $dataset/valid.complex
    head -n $budget $dataset/dev.simple >| $dataset/valid.simple
    echo "$dataset"
    ls $dataset
done
echo "finished data prep"

# Step 2: run param optimisation on validation set (number of lines for processing should corresponde to $budget)
for level in 1 2 3 4; do
    python param_search.py $muss/resources/models/muss_en_mined $results newsela_v0_v$level $budget
done
echo "finished param search..."

# Step 3: run inference with params learnt in step 2
for level in 1 2 3 4; do
    for split in dev test; do
        python simplify_file.py \
            $aligned_data/newsela_manual_v0_v${level}_${split}.tsv \
            --out_path $results \
            --model-name muss_en_mined \
            --param_file $results/finetune_newsela_v0_v${level}_${budget}_params.json
    done
done
echo "finished simplification..."

echo "done!"