# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Adapted from original `scripts/train_model.py` to search for optimal params
on a Newsela Corpus dev split.

Author: Tannon Kew // kew@cl.uzh.ch
"""

import sys
from pathlib import Path
import json
from muss.fairseq.main import finetune_on_dataset
from muss.mining.training import get_bart_kwargs

exp_dir = Path(sys.argv[1]) #'/srv/scratch6/kew/ats/muss/resources/models/muss_en_mined')
out_dir = Path(sys.argv[2]) #'/srv/scratch6/kew/ats/muss/outputs')
# NOTE: dataset should exist in resources/datasets/ and contain the following files:
# train.complex, train.simple, valid.complex, valid.simple, test.complex, test.simple
dataset = sys.argv[3] # newsela_v0_v1, newsela_v0_v2, newsela_v0_v3, newsela_v0_v14
budget = int(sys.argv[4]) # 50

def save_kwargs(kwargs, kwargs_file):
    with open(kwargs_file, 'w', encoding='utf8') as outf:
        json.dump(kwargs, outf, ensure_ascii=False, indent=4) # doesn't handle PosixPaths in values
    print(f'wrote optimized kwargs to {kwargs_file}')

kwargs = get_bart_kwargs(dataset=dataset, language='en', use_access=True)

kwargs['train_kwargs']['ngpus'] = 1  # Set this to 1 for local training
kwargs['train_kwargs']['max_tokens'] = 128  # Lower this number to prevent OOM
kwargs['parametrization_budget'] = budget # must be set to the number of valid lines we have!
kwargs['out_dir'] = out_dir

params = finetune_on_dataset(dataset, exp_dir, **kwargs)

save_kwargs(params, out_dir / f"finetune_{dataset}_{kwargs['parametrization_budget']}_params.json")


