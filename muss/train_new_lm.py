#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Before running, set the environment tmp dir to avoid running out of tmp memory

mkdir /srv/scratch6/kew/tmp
export TMPDIR=/srv/scratch6/kew/tmp
python train_new_lm.py
"""

from kenlm import train_kenlm_language_model

input_data_paths = ['/srv/scratch6/kew/ats/data/de/wiki_dumps/dewiki/dewiki_sents.txt']
output_model_dir = '/srv/scratch6/kew/ats/muss/resources/models/language_models/kenlm_dewiki'

train_kenlm_language_model(input_data_paths, output_model_dir)
print("done!")