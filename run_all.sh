#!/bin/bash
# This is just a wrapper to execute run_job.sh so that we can save output to s3
#
export S3_BUCKET=s3://clausdata  
./run_job.sh &> run_job.out.txt
if [[ $(hostname) == "ip-"* ]]; then
    aws s3 cp run_job.out.txt ${S3_BUCKET}/run_job.out.txt --quiet
    # stop (and possibly terminate) instance
    sudo halt
fi