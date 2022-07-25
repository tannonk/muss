
"""
Annotate a corpus with ACCESS features as done in MUSS model training.

Creates new versions of *.complex *.simple files with ACCESS features.

"""

from muss.fairseq.main import fairseq_train_and_evaluate_with_parametrization
from muss.mining.training import get_bart_kwargs, get_score_rows
from muss.resources.datasets import create_smaller_dataset, create_preprocessed_dataset
from muss.preprocessors import get_preprocessors

# # This dataset should exist in resources/datasets/ and contain the following files:
# # train.complex, train.simple, valid.complex, valid.simple, test.complex, test.simple
dataset = '/scratch/tkew/muss/resources/datasets/muss_mined_paraphrases/en_mined_paraphrases/'

kwargs = get_bart_kwargs(
        dataset=dataset, language='en', use_access=True
    )

preprocessors_kwargs = kwargs.get('preprocessors_kwargs', {})
preprocessors = get_preprocessors(preprocessors_kwargs)

# NOTE features are added sequentially and a new version of the dataset is created for each added feature.
dataset = create_preprocessed_dataset(dataset, preprocessors, n_jobs=8)

print('done.')