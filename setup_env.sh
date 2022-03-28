#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# conda create -n muss python=3.8
# pip install -e .  # Install package
# python -m spacy download en_core_web_md fr_core_news_md es_core_news_md  # Install required spacy models

module load volta
module load anaconda3
module load cuda/10.2
module load gcc/7.4.0

conda activate muss

# srun -p generic --pty -n 1 -c 32 --time=08:00:00 --mem=32G bash -l
