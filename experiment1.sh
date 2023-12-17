#!/bin/bash
: '
Fine-tune and evaluate Homepage2Vec model on the `original` dataset
with annotations from all GPT labelers and the crowdsourced human
labels.
'

tuned=false
group="exp1"

# Pre-trained Homepage2Vec model
poetry run train \
    data=original \
    labeler=human \
    logger=wandb \
    logger.wandb.name=$group-pretrained \
    finetune=false \
    eval=true \
    group=$group \

# Fine-tune on human labels
poetry run train \
    data=original \
    labeler=human \
    logger=wandb \
    logger.wandb.name=$group-human \
    eval=true \
    group=$group

# Fine-tune on GPT labels
# labelers=$(ls data/labels)
labelers=( 
    "gpt3.5-zeroshot-context1" 
    "gpt3.5-oneshot-context1" 
    "gpt3.5-zeroshot-context2" 
    "gpt3.5-oneshot-context2" 
    "gpt3.5-zeroshot-context3" 
    "gpt3.5-oneshot-context3" 
    )

for labeler in "${labelers[@]}"
do
    poetry run train \
        data=original \
        labeler=$labeler \
        logger=wandb \
        logger.wandb.name=$group-$labeler \
        group=$group
done
