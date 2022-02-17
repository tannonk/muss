#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from kenlm import train_kenlm_language_model

input_data_paths = ['/srv/scratch6/kew/ats/data/de/wiki_dumps/dewiki/dewiki_sents.txt']
output_model_dir = '/srv/scratch6/kew/ats/muss/resources/tools/kenlm/models'

train_kenlm_language_model(input_data_paths, output_model_dir)
print("done!")