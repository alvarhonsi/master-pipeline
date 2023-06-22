#!/bin/sh
PROPSFILE=$1

# array declaration
declare -A props

while IFS='= ' read -r key value; do
    props["$key"]="$value"
done < "$PROPSFILE"

DIR=${props["dir"]}
mkdir -p $DIR/logs
conda run -n master python src/run-pipeline.py -dir $DIR --generate > $DIR/logs/generate.log 2>&1
