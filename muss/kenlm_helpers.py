# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
from pathlib import Path

import kenlm
# from transformers import PreTrainedTokenizerFast #, SentencePieceBPETokenizerFast
from tokenizers import SentencePieceBPETokenizer, Tokenizer

from muss.utils.helpers import get_temp_filepaths, read_lines, write_lines, log_action, run_command

BUF_SIZE=1024*8

def rm_existing(filepath):
    filepath = Path(filepath)
    if filepath.exists():
        print(f'[!] overwriting existing file {filepath}')
        filepath.unlink()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.touch()
    return filepath

def add_lines(lines, filepath=None):
    with filepath.open('a+', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
    return filepath


def train_kenlm_language_model(input_data_paths, output_model_dir):
    print(f'Processing data from {", ".join(input_data_paths)} ...')
    output_model_dir = Path(output_model_dir)
    output_model_dir.mkdir(exist_ok=True, parents=True)
    output_model_path = output_model_dir / 'kenlm_model.arpa'
    tokenizer_path = output_model_dir / 'spm_tokenizer.json'
    merges_file = output_model_dir / 'spm_tokenizer-merges.txt'
    vocab_file = output_model_dir / 'spm_tokenizer-vocab.json'
    if merges_file.exists() and vocab_file.exists():
        print(f'Loading existing tokenizer from {merges_file} and {vocab_file} ...')
        # tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer = SentencePieceBPETokenizer.from_file(str(vocab_file), str(merges_file))
    elif tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        with log_action('Training tokenizer'):
            tokenizer = SentencePieceBPETokenizer()
            tokenizer.train([str(path) for path in input_data_paths], vocab_size=20000)
            # NOTE: this is unstable and not recommended as can lead to issues when reloading model files. see https://github.com/huggingface/tokenizers/issues/588
            # tokenizer.save_model(str(output_model_dir), 'spm_tokenizer') # saves separate vocab + mergs files
            tokenizer.save(str(output_model_dir / 'spm_tokenizer.json')) # saves single json for loading with PreTrainedTokenizerFast
    
    with log_action('Tokenizing'):
        # tokenized_data_paths = get_temp_filepaths(len(input_data_paths))
        # for tokenized_data_path, input_data_path in zip(tokenized_data_paths, input_data_paths):
        #     encodings = tokenizer.encode_batch(read_lines(input_data_path))
        #     write_lines([' '.join(encoding.tokens) for encoding in encodings], tokenized_data_path)
        
        # process in chunks for memory-friendly compute... 
        tokenized_data_paths = []
        for filepath in input_data_paths:
            filepath = Path(filepath)
            tok_file_path = filepath.parent / f'{filepath.stem}.tok'
            tokenized_data_paths.append(tok_file_path)
        for tokenized_data_path, input_data_path in zip(tokenized_data_paths, input_data_paths):
            
            if tokenized_data_path.exists():
                rm_existing(tokenized_data_path)
                
            # process chunks of lines at a time...
            print(f'Writing tokenized lines to tmp file at {tokenized_data_path} ...')
            with open(input_data_path, 'r', encoding='utf-8') as inf:
                tmp_lines = inf.readlines(BUF_SIZE)
                while tmp_lines:
                    encodings = tokenizer.encode_batch([line.strip() for line in tmp_lines])
                    add_lines([' '.join(encoding.tokens) for encoding in encodings], tokenized_data_path)
                    tmp_lines = inf.readlines(BUF_SIZE)
    
    
    with log_action('Training language model'):
        # kenlm_path = input('Please provide the path to the lmplz script (install at https://github.com/kpu/kenlm): ')
        # kenlm_path = '/home/user/kew/INSTALLS/kenlm/build/bin/lmplz' # rattle
        kenlm_path = '/data/tkew/INSTALLS/kenlm/build/bin/lmplz' # s3it
        command = (
            f'cat {" ".join([str(path) for path in tokenized_data_paths])} | {kenlm_path} -o 3 -T /scratch/tkew/tmp -S 80% > {output_model_path}'
        )
        run_command(command, mute=False)
    [path.unlink() for path in tokenized_data_paths]
    return output_model_dir


@lru_cache(maxsize=10)
def get_spm_tokenizer(model_dir):
    assert model_dir.exists(), 'You can download models at https://dl.fbaipublicfiles.com/muss/muss_mining_filtering_kenlm_language_models.tar.gz'
    merges_file = model_dir / 'spm_tokenizer-merges.txt'
    vocab_file = model_dir / 'spm_tokenizer-vocab.json'
    tokenizer_json = model_dir / 'spm_tokenizer.json'
    if merges_file.exists() and vocab_file.exists():
        tokenizer = SentencePieceBPETokenizer.from_file(str(vocab_file), str(merges_file))
    elif tokenizer_json.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_json))
        # tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json))
        # tokenizer = SentencePieceBPETokenizerFast(tokenizer_file=str(tokenizer_json))
    else:
        raise RuntimeError(f'Could not find tokenizer files in directory {model_dir}')
    return tokenizer

@lru_cache(maxsize=10)
def get_kenlm_model(model_dir):
    assert model_dir.exists(), 'You can download models at https://dl.fbaipublicfiles.com/muss/muss_mining_filtering_kenlm_language_models.tar.gz'
    model_file = model_dir / 'kenlm_model.arpa'
    return kenlm.Model(str(model_file))


def get_kenlm_log_prob(text, model_dir, *args, **kwargs):
    tokenizer = get_spm_tokenizer(model_dir)
    kenlm_model = get_kenlm_model(model_dir)
    encoded_text = ' '.join(tokenizer.encode(text).tokens)
    return kenlm_model.score(encoded_text, *args, **kwargs)
