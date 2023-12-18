#!/bin/bash
: '
'

labelers=( "gpt3.5-oneshot-context2" )

poetry run train experiment=exp2-pretrained

for labeler in "${labelers[@]}"
do
    poetry run train experiment=exp2-finetuned train_labeler=$labeler
done