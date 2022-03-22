# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path

from muss.simplify import ALLOWED_MODEL_NAMES, simplify_sentences
from muss.utils.helpers import read_lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simplify a file line by line.')
    parser.add_argument('filepath', type=str, help='File containing the source sentences, one sentence per line.')
    parser.add_argument(
        '--model-name',
        type=str,
        default=ALLOWED_MODEL_NAMES[0],
        choices=ALLOWED_MODEL_NAMES,
        help=f'Model name to generate from. Models selected with the highest validation SARI score.',
    )
    parser.add_argument('--outfile', type=str, default=None, help='if provided, outputs are written one-per-line to provided file path')
    parser.add_argument('--len_ratio', type=float, default=0.75)
    parser.add_argument('--lev_sim', type=float, default=0.65)
    parser.add_argument('--word_rank', type=float, default=0.75)
    parser.add_argument('--tree_depth', type=float, default=0.4)
    args = parser.parse_args()
    # added processor arguments to commandline for experimentation
    processor_args = argparse.Namespace(**{k: v for k, v in args._get_kwargs()
                              if k in ['len_ratio', 'lev_sim', 'word_rank', 'tree_depth']})
    source_sentences = read_lines(args.filepath)
    pred_sentences = simplify_sentences(source_sentences, processor_args, model_name=args.model_name)
    if args.outfile is not None:
        Path(args.outfile).parent.mkdir(exist_ok=True, parents=True)
        with open(args.outfile, 'w', encoding='utf8') as outf:
            for s in pred_sentences:
                outf.write(f'{s}\n')
    else: # original behaviour
        for c, s in zip(source_sentences, pred_sentences):
            print('-' * 80)
            print(f'Original:   {c}')
            print(f'Simplified: {s}')
