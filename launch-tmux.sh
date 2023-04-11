#!/bin/bash
INPUT=$1
NAME=$2
COUNT=1
while IFS= read -r line
do
    SESH="$NAME-$COUNT"
    echo "starting $SESH"
    echo $line
    tmux new-session -d -s $SESH "conda activate master && $CMD"
    COUNT=$((COUNT + 1))
done < "$INPUT"