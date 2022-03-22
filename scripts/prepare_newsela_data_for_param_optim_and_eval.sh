#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# organises newsela tsv data for muss param optimisation

muss_data="/srv/scratch6/kew/ats/muss/resources/datasets"
aligned_data="/srv/scratch6/kew/ats/data/en/aligned"

for level in 1 2 3 4; do
    outdir=$muss_data/newsela_v0_v$level
    for split in train test dev; do
        mkdir -p $outdir
        cut -f 1 $aligned_data/newsela_manual_v0_v${level}_${split}.tsv >| $outdir/$split.complex
        cut -f 2 $aligned_data/newsela_manual_v0_v${level}_${split}.tsv >| $outdir/$split.simple
    done
    head -n 50 $outdir/dev.complex >| $outdir/valid.complex
    head -n 50 $outdir/dev.simple >| $outdir/valid.simple

    echo "$outdir"
    ls $outdir

done

echo "done!"