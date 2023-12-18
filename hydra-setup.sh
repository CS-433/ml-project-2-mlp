#!/bin/bash
: '
Duplicates the configurations for `labeler` and `data` in folders
`trainlabeler`, `testlabeler` and `traindata` and `testdata` respectively.
'

# Labeler
path="conf/labeler"
tr_path="conf/train_labeler"
te_path="conf/test_labeler"
mkdir -p $tr_path
mkdir -p $te_path
ln $path/*.yaml $tr_path
ln $path/*.yaml $te_path

dirs=( "context" "model" "shot" )
for dir in "${dirs[@]}"
do
    mkdir -p $tr_path/$dir
    mkdir -p $te_path/$dir
    ln $path/$dir/*.yaml $tr_path/$dir
    ln $path/$dir/*.yaml $te_path/$dir
done

# Data
path="conf/data"
tr_path="conf/train_data"
te_path="conf/test_data"
mkdir -p $tr_path
mkdir -p $te_path
ln $path/*.yaml $tr_path
ln $path/*.yaml $te_path

# Dataodules
path="conf/datamodule"
tr_path="conf/train_datamodule"
te_path="conf/test_datamodule"
mkdir -p $tr_path
mkdir -p $te_path
ln $path/*.yaml $tr_path
ln $path/*.yaml $te_path
