#!/bin/bash
: '
Evaluate a series of fine-tuned models on the `original` dataset
based on their `run_id` from Weights & Biases specified via
command-line arguments.
'

run_ids=( "$@" )

for run_id in "${run_ids[@]}"
do
    poetry run eval run_id=$run_id
done
