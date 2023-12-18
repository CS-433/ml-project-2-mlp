#!/bin/bash
: '
Fine-tune and evaluate Homepage2Vec model on the `original` dataset
with annotations from all GPT labelers and the crowdsourced human
labels.
'

labelers=( 
    "human"
    "gpt3.5-zeroshot-context1" 
    "gpt3.5-oneshot-context1" 
    "gpt3.5-zeroshot-context2" 
    "gpt3.5-oneshot-context2" 
    "gpt3.5-zeroshot-context3" 
    "gpt3.5-oneshot-context3" 
    "gpt4-zeroshot-context2" 
    "gpt4-oneshot-context2"
    )

poetry run train experiment=exp1-pretrained

for labeler in "${labelers[@]}"
do
    poetry run train experiment=exp1-finetuned train_labeler=$labeler
done