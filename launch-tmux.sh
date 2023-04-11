#!/bin/bash
INPUT=$1
NAME=$2
COUNT=1
while IFS= read -r line
do
    SESH="$NAME-$COUNT"
    echo "starting $SESH"
    echo $line
    tmux new-session -d -s $SESH
    tmux send-keys -t $SESH "conda run -n master $line" Enter
    COUNT=$((COUNT + 1))
done < "$INPUT"