#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Isolates the final step of the mine_sequences.py pipeline. 

Example Call (in interactive session):

    python wrap_up_paraphrases.py /scratch/tkew/muss/resources/datasets/uts/de de

"""

import sys
from pathlib import Path
from muss.mining.preprocessing import get_sentences_paths
from muss.mining.nn_search import get_cache_dir, wrap_up_paraphrases
    
dataset_dir = Path(sys.argv[1])
language = sys.argv[2]
import pdb; pdb.set_trace()
db_sentences_paths = get_sentences_paths(dataset_dir)
query_sentences_paths = db_sentences_paths
cache_dir = get_cache_dir(dataset_dir)
pairs_dir = cache_dir / 'laser_de' / 'pairs'

# hardcoded variables - copied from `mine_sequences.py`
topk = 8
nprobe = 16
filter_kwargs = {'density': 0.6,'distance': 0.05,'levenshtein': 0.2,'simplicity': 0.0,'filter_ne': False}

wrap_up_paraphrases(
    query_sentences_paths,
    db_sentences_paths,
    topk, 
    nprobe, 
    filter_kwargs,
    pairs_dir, 
    language
    )