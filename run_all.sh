#!/bin/bash
#
./run_job.sh &> run_job.out.txt
if [[ "$HOSTNAME" == "ip-"* ]]; then
    aws s3 cp run_job.out.txt s3://clausdata/run_job.out 
    # stop (and possibly terminate) instance
    sudo halt
fi