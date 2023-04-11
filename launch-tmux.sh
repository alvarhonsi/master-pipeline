#!/bin/bash
INPUT=$1
NAME=$2
COUNT=1
while IFS= read -r line
do
    SESH="$NAME-$COUNT"
    echo "starting $SESH"
    echo $line
    tmux new-session -d -s $SESH "conda activate master"
    sleep 1
    tmux send-keys -t $SESH $CMD Enter
    COUNT=$((COUNT + 1))
done < "$INPUT"