# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from muss.fairseq.main import finetune_and_predict_on_dataset
from muss.mining.training import get_bart_kwargs, get_score_rows
from muss.resources.prepare import prepare_wikilarge_detokenized, prepare_asset
from muss.resources.datasets import create_smaller_dataset



# This dataset should exist in resources/datasets/ and contain the following files:
# train.complex, train.simple, valid.complex, valid.simple, test.complex, test.simple
# prepare_wikilarge_detokenized()
# prepare_asset()
datasets = [
    'newsela_v0_v1', 
    'newsela_v0_v2', 
    'newsela_v0_v3', 
    'newsela_v0_v4'
    ]

exp_dir = Path('/srv/scratch6/kew/ats/muss/resources/models/muss_en_mined')
out_dir = Path('/srv/scratch6/kew/ats/muss/outputs')
for dataset in datasets:
    kwargs = get_bart_kwargs(dataset=dataset, language='en', use_access=True)

    # kwargs['train_kwargs']['ngpus'] = 1  # Set this from 8 to 1 for local training
    kwargs['train_kwargs']['max_tokens'] = 128  # Lower this number to prevent OOM
    kwargs['parametrization_budget'] = 50 # must be set to the number of valid lines we have!
    # kwargs['fast_parametrization_search'] = True
    kwargs['out_dir'] = out_dir

    finetune_and_predict_on_dataset(dataset, exp_dir, **kwargs)
