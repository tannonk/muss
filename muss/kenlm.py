# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
from pathlib import Path

import kenlm
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
    if tokenizer_path.exists():
        print(f'[!] Loading existing tokenizer from {tokenizer_path} ...')
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        with log_action('Training tokenizer'):
            tokenizer = SentencePieceBPETokenizer()
            tokenizer.train([str(path) for path in input_data_paths], vocab_size=20000)
            #    # tokenizer.save(str(output_model_dir), 'spm_tokenizer')
            tokenizer.save(str(tokenizer_path))
    with log_action('Tokenizing'):
        # tokenized_data_paths = get_temp_filepaths(len(input_data_paths))
        tokenized_data_paths = []
        for filepath in input_data_paths:
            filepath = Path(filepath)
            tok_file_path = filepath.parent / f'{filepath.stem}.tok'
            tokenized_data_paths.append(tok_file_path)
        for tokenized_data_path, input_data_path in zip(tokenized_data_paths, input_data_paths):
            # process chunks of lines at a time...
            # rm_existing(tokenized_data_path)
            if tokenized_data_path.exists():
                print(f'{tokenized_data_path} already exists! Assuming data has already been tokenized and skipping...') 
            else:
                print(f'Writing tokenized lines to tmp file at {tokenized_data_path} ...')
                with open(input_data_path, 'r', encoding='utf-8') as inf:
                    tmp_lines = inf.readlines(BUF_SIZE)
                    while tmp_lines:
                        # process([line for line in tmp_lines])
                        encodings = tokenizer.encode_batch([line.strip() for line in tmp_lines])
                        add_lines([' '.join(encoding.tokens) for encoding in encodings], tokenized_data_path)
                        tmp_lines = inf.readlines(BUF_SIZE)

    with log_action('Training language model'):
        # kenlm_path = input('Please provide the path to the lmplz script (install at https://github.com/kpu/kenlm): ')
        kenlm_path = '/home/user/kew/INSTALLS/kenlm/build/bin/lmplz'
        command = (
            f'cat {" ".join([str(path) for path in tokenized_data_paths])} | {kenlm_path} -o 3 -S 80% > {output_model_path}'
        )
        run_command(command, mute=False)
    [path.unlink() for path in tokenized_data_paths] # remove tokenized files once lm is trained
    return output_model_dir


@lru_cache(maxsize=10)
def get_spm_tokenizer(model_dir):
    assert model_dir.exists(), 'You can download models at https://dl.fbaipublicfiles.com/muss/muss_mining_filtering_kenlm_language_models.tar.gz'
    merges_file = model_dir / 'spm_tokenizer-merges.txt'
    vocab_file = model_dir / 'spm_tokenizer-vocab.json'
    return SentencePieceBPETokenizer(vocab_file=str(vocab_file), merges_file=str(merges_file))


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
