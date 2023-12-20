#!/bin/bash
: '
Computes the baseline performance of the pre-trained 
Homepage2Vec model on the `original` (crowdsourced) dataset (80%) and fine-tunes Homepage2Vec given annotations from the GPT annotators.
'

# Pretrained Performance
poetry run train experiment=pretrained

# Finetune
labelers=( "gpt3.5-oneshot-context2" "gpt4-zeroshot-context2" )

for labeler in "${labelers[@]}"
do
    poetry run train \
        experiment=finetuned \
        train_labeler=$labeler
done