#!/bin/sh
PROPSFILE=$1

# array declaration
declare -A props

while IFS='= ' read -r key value; do
    props["$key"]="$value"
done < "$PROPSFILE"

DIR=${props["dir"]}
conda run -n master python tyxe_runfiles/run-pipeline.py -dir $DIR --generate
