#!/bin/bash
INPUT=$1
NAME=$2
COUNT=1
while IFS= read -r line
do
    SESH="$NAME-$COUNT"
    echo "starting $SESH"
    echo $line
    tmux new-session -d -s $SESH -n myWindow
    tmux send-keys -t $SESH:myWindow "conda activate master" Enter
    tmux send-keys -t $SESH:myWindow $CMD Enter
    COUNT=$((COUNT + 1))
done < "$INPUT"