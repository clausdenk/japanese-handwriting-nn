#!/bin/bash
#
# This script runs the complete learning job
#
# CONFIGURE YOUR INPUT AND OUTPUT FILES ON S3 HERE:
export WEIGHTS_IN=M71-kanji_weights.h5
export WEIGHTS_OUT=M71-kanji_weights.h5
export OUTPUT_TXT=output.txt
export S3_BUCKET=s3://clausdata   # include dir, no trailing /

# on aws, set environment and get data from s3
if [[ "$HOSTNAME" == "ip-"* ]]; then
    source activate tensorflow_p36

    # get ETLC data
    aws s3 cp "${S3_BUCKET}/ETLC.zip" ETLC.zip --quiet
    retVal=$?
    if [ $retVal -ne 0 ]; then
        echo "Error getting ETLC.zip from s3"
        exit 1
    fi
    unzip ETLC.zip

    # make weights dir
    mkdir -p weights
    # copy weights from s3 if they exist
    aws s3 ls "${S3_BUCKET}/${WEIGHTS_IN}"
    retVal=$?
    if [ $retVal -eq 0 ]; then
        echo "Using existing weights ${S3_BUCKET}/${WEIGHTS_IN} from s3"
        aws s3 cp "${S3_BUCKET}/${WEIGHTS_IN}" weights/weights_in.h5 --quiet
    else
        echo "Weights ${S3_BUCKET}/${WEIGHTS_IN} not found on s3"
    fi
fi
# run job
time python example_job.py &> output.txt
retVal=$?
# on aws ..
if [[ "$HOSTNAME" == "ip-"* ]]; then
    aws s3 cp output.txt ${S3_BUCKET}/${OUTPUT_TXT} --quiet
    # save to s3 if terminated normally
    if [ $retVal -eq 0 ]; then
        echo "saving weights to ${S3_BUCKET}/${WEIGHTS_OUT}"
        aws s3 cp weights/weights_out.h5 "${S3_BUCKET}/${WEIGHTS_OUT}" 
    else
        echo "Error running job, check ${S3_BUCKET}/${OUTPUT_TXT} on s3"
    fi
fi