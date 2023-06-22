#!/bin/bash
PROPSFILE=$1
START_COUNT=$2

if [ -z "$START_COUNT" ]
then
    COUNT=1
else
    COUNT=$START_COUNT
fi

# array declaration
declare -A props

while IFS='= ' read -r key value; do
    props["$key"]="$value"
done < "$PROPSFILE"

DIR=${props["dir"]}
NAME=${props["name"]}
IFS=', ' read -r -a DEVICES <<< "${props["devices"]}"
IFS=',' read -r -a PROFILES <<< "${props["profiles"]}"

mkdir -p $DIR/logs

for i in "${!PROFILES[@]}"
do
    DEV=${DEVICES[$i]}
    PROF=${PROFILES[$i]}
    RUNCMD="python src/run-pipeline.py -dir $DIR --train --eval -p $PROF --device $DEV"


    SESH="$NAME-$COUNT"
    echo "starting $SESH"
    echo $RUNCMD
    tmux new-session -d -s $SESH
    tmux send-keys -t $SESH "conda activate master" Enter
    sleep 2
    tmux send-keys -t $SESH "python" Space "src/run-pipeline.py" Space "-dir" Space "$DIR" Space "--train" Space "--eval" Space "-p" Space "$PROF" Space "--device" Space "$DEV" Space "2>&1" Space "|" Space "tee" Space "$DIR/logs/$SESH.log" Enter
    COUNT=$((COUNT + 1))
done
