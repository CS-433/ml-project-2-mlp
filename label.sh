#!/bin/bash
: '
Label the entire `original` (crowdsourced) dataset with all
GPT annotators.
'

labelers=( 
    "gpt3.5-zeroshot-context1" 
    "gpt3.5-oneshot-context1" 
    "gpt3.5-zeroshot-context2" 
    "gpt3.5-oneshot-context2" 
    "gpt3.5-zeroshot-context3" 
    "gpt3.5-oneshot-context3" 
    "gpt4-zeroshot-context1" 
    "gpt4-oneshot-context1" 
    "gpt4-zeroshot-context2" 
    "gpt4-oneshot-context2" 
    "gpt4-zeroshot-context3" 
    "gpt4-oneshot-context3" 
    )

for labeler in "${labelers[@]}"
do
    poetry run label data=original labeler=$labeler
done