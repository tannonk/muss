
from pathlib import Path
from tqdm import tqdm

from muss.mining.training import get_bart_kwargs
from muss.preprocessors import get_preprocessors

# # This dataset should exist in resources/datasets/ and contain the following files:
# # train.complex, train.simple, valid.complex, valid.simple, test.complex, test.simple
def compute_features(src, tgt, preprocessors):

    data = {'src': src, 'tgt': tgt}

    for preprocessor in preprocessors:
        try:
            # equivalent to preprocessor.encode_sentence_pair(c, s), but get intermediate results
            feat_val = preprocessor.get_feature_value(src, tgt)
            bucketed_feat_val = preprocessor.bucketize(feat_val)
            special_token = preprocessor.get_feature_token(bucketed_feat_val)
            # preprocessor.get_feature_token(preprocessor.bucketize())
            # print(preprocessor.prefix, preprocessor.bucketize(preprocessor.get_feature_value(c, s)))
            # print(preprocessor.prefix, preprocessor.get_feature_value(c, s))
            data[f'{preprocessor.prefix}_score'] = feat_val
            data[f'{preprocessor.prefix}_bucket'] = bucketed_feat_val
            data[f'{preprocessor.prefix}_token'] = special_token
    
        except AttributeError:
            pass
    
    return data

if __name__ == '__main__':

    # import pdb;pdb.set_trace()
    dataset = '/scratch/tkew/muss/resources/datasets/muss_mined_paraphrases/en_mined_paraphrases/'

    kwargs = get_bart_kwargs(
            dataset=None, language='en', use_access=True
        )

    preprocessors_kwargs = kwargs.get('preprocessors_kwargs', {})
    if 'GPT2BPEPreprocessor' in preprocessors_kwargs:
        preprocessors_kwargs.pop('GPT2BPEPreprocessor')
    preprocessors = get_preprocessors(preprocessors_kwargs)

    with open(Path(dataset) / 'train_head200.csv', 'r', encoding='utf8') as f:
        for line in tqdm(f):
            src, tgt = line.strip().split('\t')
            data = compute_features(src, tgt, preprocessors)
            print(data)
    
    print('done')