# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""

Creating the initial faiss index takes quite some time. 

For 10**7 sentences:
    Computing embeddings completed after 7882.18s. (~2.5 hours)
    Training index completed after 39061.53s. (~ 11 hours)

"""

import sys
import faiss

from muss.mining.preprocessing import create_base_index, get_index_name, get_sentences_paths, load_from_pickle, pickle_train_sentences
from muss.utils.helpers import yield_lines
from muss.laser import get_laser_embeddings
from muss.resources.paths import get_dataset_dir

# import pdb;pdb.set_trace()

# Create index
language = sys.argv[1]

print('Language:', language)

dataset_dir = get_dataset_dir('uts') / language
train_sentences = dataset_dir / 'train_sentences.pkl'
base_index_dir = dataset_dir / f'base_indexes/'
base_index_dir.mkdir(exist_ok=True, parents=True)

print('Base index dir:', base_index_dir)
    
get_embeddings = lambda sentences: get_laser_embeddings(
    sentences, max_tokens=3000, language=language, n_encoding_jobs=10
)  # noqa: E731

create_base_index(train_sentences, get_index_name(), get_embeddings, faiss.METRIC_L2, base_index_dir)
