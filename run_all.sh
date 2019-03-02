#!/bin/bash
# This is just a wrapper to execute run_job.sh so that we can save output to s3
./run_job.sh &> run_job.out.txt
if [[ "$HOSTNAME" == "ip-"* ]]; then
    aws s3 cp run_job.out.txt s3://clausdata/run_job.out.txt
    # stop (and possibly terminate) instance
    sudo halt
fi