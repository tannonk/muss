#!/usr/bin/env bash
# -*- coding: utf-8 -*-

lang=$1
use_resource_monitor=$2
module load generic anaconda3
module list
echo "activating conda env"
source activate muss
module purge
echo "purging modules:"
module list

py_ex=$(which python)

if [[ $py_ex != '/home/cluster/tkew/.conda/envs/muss/bin/python' ]]; then
    echo "python called from $py_ex"
    echo "check that 'muss' conda env is loaded..." && exit 1
fi

if [[ ! -z $use_resource_monitor ]]; then
    echo "running with resource monitor"
    resource_monitor -O  mining_job -i 600 -- python scripts/mine_sequences.py $lang
else
    python scripts/mine_sequences.py $lang
fi