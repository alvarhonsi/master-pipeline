#!/bin/bash
NAME=$2
COUNT=1
for CMD in `cat $1`
do
    SESH=$NAME$COUNT
    echo "starting $SESH"
    echo $CMD
    tmux new-session -d -s $SESH -n myWindow
    tmux send-keys -t $SESH:myWindow "conda activate master" Enter
    tmux send-keys -t $SESH:myWindow $CMD Enter
    COUNT=$((COUNT + 1))
done