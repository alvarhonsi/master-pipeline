#!/bin/bash
PROPSFILE=$1
NAME=$2
COUNT=1

# array declaration
declare -A props

while IFS='= ' read -r key value; do
    props["$key"]="$value"
done < "$PROPSFILE"

DIR=${props["dir"]}
IFS=', ' read -r -a DEVICES <<< "${props["devices"]}"
IFS=',' read -r -a PROFILES <<< "${props["profiles"]}"

for i in "${!PROFILES[@]}"
do
    DEV=${DEVICES[$i]}
    PROF=${PROFILES[$i]}
    RUNCMD="python run-pipeline.py -dir $DIR --train --eval -p $PROF --device $DEV"

    conda activate master
    python run-pipeline.py -dir $DIR --generate

    SESH="$NAME-$COUNT"
    echo "starting $SESH"
    echo $CMD
    tmux new-session -d -s $SESH
    tmux send-keys -t $SESH "conda activate master" Enter
    sleep 2
    tmux send-keys -t $SESH $RUNCMD Enter
    COUNT=$((COUNT + 1))
done
