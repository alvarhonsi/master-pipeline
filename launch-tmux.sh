#!/bin/bash
INPUT=$1
NAME=$2
COUNT=1

IFS=''
while read -r line
do
    SESH="$NAME-$COUNT"
    echo "starting $SESH"
    echo $line
    tmux new-session -d -s $SESH
    tmux send-keys -t $SESH "conda activate master" Enter
    sleep 0.5
    tmux send-keys -t $SESH $line Enter
    COUNT=$((COUNT + 1))
done < "$INPUT"