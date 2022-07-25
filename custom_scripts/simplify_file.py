# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Adapted from original `scripts/simplify.py` to facilitate file processing

Author: Tannon Kew // kew@cl.uzh.ch

New features:

- 22/07/23: accepts a file with hardcoded control tokens (preprocessors)
    - this is modfied in muss/preprocessors.py, if the line contains a preprocessor's prefix, we don't encode it with the preprocessor.

"""

import json
import argparse
from pathlib import Path

from muss.simplify import simplify_sentences

def read_split_lines(infile):
    lines = []
    with open(infile, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split('\t')
            lines.append(line[0]) # src text
    return lines

def parse_json_args(json_file):
    with open(json_file, 'r', encoding='utf8') as f:
        d = json.load(f)
    
    processor_args = {
        'len_ratio': round(d['LengthRatioPreprocessor']['target_ratio'], 2),
        'lev_sim': round(d['ReplaceOnlyLevenshteinPreprocessor']['target_ratio'], 2), 
        'word_rank': round(d['WordRankRatioPreprocessor']['target_ratio'], 2), 
        'tree_depth': round(d['DependencyTreeDepthRatioPreprocessor']['target_ratio'], 2),
        }

    return processor_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simplify a file line by line.')
    parser.add_argument('filepath', type=Path, help='File for simplification. Can be one sentence per line or tsv with src\tref...')
    parser.add_argument('--model_name', type=str, default='muss_en_mined', help='Muss compatible model trained with fairseq')
    parser.add_argument('--out_file', type=Path, default=None, help='if provided, outputs are written one-per-line to provided file path')
    parser.add_argument('--out_path', type=Path, default=None, help='if provided, output file name is inferred and translations are written one-per-line to file')
    
    # setting preprocessors / ctrl tokens for inference
    parser.add_argument('--len_ratio', type=float, default=0.75, help='default value according to original paper')
    parser.add_argument('--lev_sim', type=float, default=0.65, help='default value according to original paper')
    parser.add_argument('--word_rank', type=float, default=0.75, help='default value according to original paper')
    parser.add_argument('--tree_depth', type=float, default=0.4, help='default value according to original paper')
    parser.add_argument('--param_file', type=str, required=False, default=None, help='path to json file containing optimum param settings')
    args = parser.parse_args()
    
    # added processor arguments to commandline for experimentation
    if args.param_file is not None:
        processor_args = argparse.Namespace(**parse_json_args(args.param_file))
    else:
        processor_args = argparse.Namespace(**{k: v for k, v in args._get_kwargs()
                                if k in ['len_ratio', 'lev_sim', 'word_rank', 'tree_depth']})
    
    # print('Processor args:', processor_args)
    source_sentences = read_split_lines(args.filepath)
    print(f'Loaded {len(source_sentences)} sentences from {args.filepath}')
    
    # run translation
    pred_sentences = simplify_sentences(source_sentences, processor_args, model_name=args.model_name)
    print(f'Simplified {len(pred_sentences)} sentences')

    if args.out_file is not None:
        out_file = Path(args.out_file)
    elif args.out_path is not None:
        out_file = Path(args.out_path) / f"{args.filepath.stem}_lr{processor_args.len_ratio}_ls{processor_args.lev_sim}_wr{processor_args.word_rank}_td{processor_args.tree_depth}.pred"

    if out_file:
        out_file.parent.mkdir(exist_ok=True, parents=True)
        with open(out_file, 'w', encoding='utf8') as outf:
            for s in pred_sentences:
                outf.write(f'{s}\n')
        print(f'Outputs written to {out_file}')

    else: # original behaviour
        for c, s in zip(source_sentences, pred_sentences):
            print('-' * 80)
            print(f'Original:   {c}')
            print(f'Simplified: {s}')
