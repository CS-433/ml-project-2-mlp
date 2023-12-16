#!/bin/bash
: '
Delete all processed data and labels of all labelers
for a specified dataset (e.g. "original", "ours")
'

dataset=$1

if [ -z "$dataset" ]; then
    echo "Usage: ./clean.sh <dataset>"
    exit 1
fi

# Remove all processed data
rm -rf data/features/$dataset

# Remove labels from all labelers
labelers=$(ls data/labels)

for labeler in $labelers; do
    rm data/labels/$labeler/$dataset.json
done