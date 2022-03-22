#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
import argparse
from genericpath import exists
import pandas as pd
import numpy as np
from tqdm import tqdm
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from easse import quality_estimation

from muss.feature_extraction import (get_lexical_complexity_score, get_levenshtein_similarity,
                                       get_dependency_tree_depth)

"""
Example call:
    python scripts/compute_label_scores_on_corpus.py \
        --infile /srv/scratch6/kew/ats/data/en/aligned/newsela_manual_v0_v4_dev.tsv /srv/scratch6/kew/ats/data/en/aligned/newsela_manual_v0_v4_train.tsv \
        --df_path /srv/scratch6/kew/ats/data/en/data_stats

    python scripts/compute_label_scores_on_corpus.py \
        --infile /srv/scratch6/kew/ats/data/en/aligned/newsela_manual_v0_v3_dev.tsv /srv/scratch6/kew/ats/data/en/aligned/newsela_manual_v0_v3_train.tsv \
        --df_path /srv/scratch6/kew/ats/data/en/data_stats

    python scripts/compute_label_scores_on_corpus.py \
        --infile /srv/scratch6/kew/ats/data/en/aligned/newsela_manual_v0_v2_dev.tsv /srv/scratch6/kew/ats/data/en/aligned/newsela_manual_v0_v2_train.tsv \
        --df_path /srv/scratch6/kew/ats/data/en/data_stats

    python scripts/compute_label_scores_on_corpus.py \
        --infile /srv/scratch6/kew/ats/data/en/aligned/newsela_manual_v0_v1_dev.tsv /srv/scratch6/kew/ats/data/en/aligned/newsela_manual_v0_v1_train.tsv \
        --df_path /srv/scratch6/kew/ats/data/en/data_stats

"""

def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--infiles', type=str, nargs='*', default=None, required=False)
    # ap.add_argument('--plot_path', type=Path, default=None, required=False)
    ap.add_argument('--df_path', type=Path, default=None, required=False)
    return ap.parse_args()

def read_lines(files, sep='\t'):
    src_lines, tgt_lines = [], []
    for file in files:
        print(f'reading lines from {file} ...')
        with open(file, 'r', encoding='utf8') as inf:
            for line in inf:
                line = line.strip().split(sep)
                src_lines.append(line[0])
                tgt_lines.append(line[1]) # TODO check for multi-references
    return src_lines, tgt_lines

def score_sentences(src_sents, tgt_sents):

    scores = {
        'complex': [],
        'simple': [],
        'length_ratio': [],
        'lex_complexity': [],
        'levenshtein': [],
        'dep_tree_depth': []

    }

    for src, tgt in tqdm(zip(src_sents, tgt_sents), total=len(src_sents)):
        scores['complex'].append(src)
        scores['simple'].append(tgt)
        scores['length_ratio'].append(len(tgt) / len(src))
        scores['lex_complexity'].append(get_lexical_complexity_score(tgt) / get_lexical_complexity_score(src))
        scores['levenshtein'].append(get_levenshtein_similarity(src, tgt))
        scores['dep_tree_depth'].append(get_dependency_tree_depth(tgt) / get_dependency_tree_depth(src))

    qe = quality_estimation.sentence_quality_estimation(src_sents, tgt_sents)
    # import pdb;pdb.set_trace()
    scores.update(qe)

    return scores

def to_df(scores, outfile=None):
    df = pd.DataFrame.from_dict(scores)
    if outfile:
        df.to_csv(outfile, sep='\t', header=True, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f'saved dataframe to {outfile}')
    return df

def make_plots(df, title='', outfile=None):

    sns.set_context('talk')

    fig, axes = plt.subplots(1, 4, figsize=(16, 8), sharey=True)
    ax0 = sns.histplot(df, x='length_ratio', ax=axes[0], discrete=False, stat='density')
    ax1 = sns.histplot(df, x='lex_complexity', ax=axes[1], discrete=False, stat='density')
    ax2 = sns.histplot(df, x='levenshtein', ax=axes[2], discrete=False, stat='density')
    ax3 = sns.histplot(df, x='dep_tree_depth', ax=axes[3], discrete=False, stat='density')

    # import pdb;pdb.set_trace()
    for ax in axes:
        ax.set_xlim(left=0.0, right=2.0)

    # plt.legend(bbox_to_anchor=(1.05, 1.5), loc=2, borderaxespad=0.)
    fig.suptitle(title)
    fig.set_tight_layout(True)

    plt.savefig(outfile, dpi=300)
    print(f'saved plot to {outfile}')
    return

def test():
    src_sents = [
        'VIRGINIA CITY, Nev. — One wonders what Mark Twain himself would make of the news: The Gold Rush-era newspaper for which he once penned stories and witticisms on frontier life as a fledgling journalist is once again in print after a decadeslong hiatus.',
        'Would Twain use Twitter to bemoan the deplorable state of the press, as he once did by pen?',
        ]

    tgt_sents = [
        'VIRGINIA CITY, Nev. — A 19th-century newspaper that Mark Twain once wrote for is back in print, after a break of many years. What would the famous writer and humorist think?',
        'Would Twain use Twitter to bemoan the sad state of the press, as he once did with pen and ink?',
    ]

    scores = score_sentences(src_sents, tgt_sents)
    for k, v in scores.items():
        print(k, v)

def parse_title(filenames):
    stems = [Path(filename).stem for filename in filenames]
    stems = [re.sub('(_dev|_train)', '', stem) for stem in stems]
    if len(set(stems)) > 1:
        raise RuntimeError(f'File stems don\'t match. {stems}')
    return stems[0]

def main():

    args = set_args()
    
    if not args.infiles:
        test()
        sys.exit()
    
    src_sents, tgt_sents = read_lines(args.infiles)

    title = parse_title(args.infiles)

    args.df_path.mkdir(parents=True, exist_ok=True)
    df_path = args.df_path / f'{title}.tsv'
    df = to_df(score_sentences(src_sents, tgt_sents), df_path)
    
    # NOTE: see notebook script `analyse_target_levels.ipynb` for plotting
    # if args.plot_path:
    #     outfile = args.plot_path / f'{title}.png'
    #     make_plots(df, title, outfile)

if __name__ == "__main__":
    main()