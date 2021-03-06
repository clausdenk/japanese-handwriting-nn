#!/bin/bash
#
# This script runs the complete learning job, getting data and weights from s3 and saving after rnu
#
# CONFIGURE YOUR INPUT AND OUTPUT FILES ON S3 HERE:
export WEIGHTS_IN=mobile-kanji_weights_in.h5
export WEIGHTS_OUT=mobile-kanji_weights_out.h5
export OUTPUT_TXT=output.txt
export S3_BUCKET=s3://clausdata   # include dir, no trailing /

# on aws, set environment and get data from s3
if [[ $(hostname) == "ip-"* ]]; then

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
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate tensorflow_p36
time python example_mobile.py &> output.txt
retVal=$?
# on aws ..
if [[ $(hostname) == "ip-"* ]]; then
    aws s3 cp output.txt ${S3_BUCKET}/${OUTPUT_TXT} --quiet
    # save to s3 if terminated normally
    if [ $retVal -eq 0 ]; then
        echo "saving weights to ${S3_BUCKET}/${WEIGHTS_OUT}"
        aws s3 cp weights/weights_out.h5 "${S3_BUCKET}/${WEIGHTS_OUT}" 
    else
        echo "Error running job, check ${S3_BUCKET}/${OUTPUT_TXT} on s3"
    fi
fi