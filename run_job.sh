#!/bin/bash
#
# This script runs the complete learning job
#
# names of files in s3 bucket:
export WEIGHTS_IN=M16-hiragana_weights.h5
export WEIGHTS_OUT=M16-hiragana_weights_out.h5

# on aws set environment and get data from s3
if [[ "$HOSTNAME" == "ip-"* ]]; then
    source activate tensorflow_p36

    # get ETLC data
    aws s3 cp "s3://clausdata/ETLC.zip" ETLC.zip
    retVal=$?
    if [ $retVal -ne 0 ]; then
        echo "Error getting ETLC.zip from s3"
        exit 1
    fi
    unzip ETLC.zip

    # make weights dir
    mkdir -p weights
    # copy weights from s3 if they exist
    aws s3 ls "s3://clausdata/${WEIGHTS_IN}"
    retVal=$?
    if [ $retVal -eq 0 ]; then
        echo "Using existing weights ${WEIGHTS_IN} from s3"
        aws s3 cp "s3://clausdata/${WEIGHTS_IN}" weights/weights_in.h5
    else
        echo "Weights ${WEIGHTS_IN} not found on s3"
    fi
fi
# run job
RES=$(python example_job.py &> output.txt)
# on aws ..
if [[ "$HOSTNAME" == "ip-"* ]]; then
    aws s3 cp output.txt s3://clausdata/output.txt 
    # save to s3 if terminated normally
    if [ $RES -eq 0 ]; then
        echo "saving weights ${WEIGHTS_OUT} to s3..."
        aws s3 cp weights/weights_out.h5 "s3://clausdata/${WEIGHTS_OUT}" 
    else
        echo "Error running job, check output.txt on s3"
    fi
fi