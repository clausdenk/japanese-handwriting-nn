#!/bin/bash
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate tensorflow_p36
python3 -c 'import tensorflow as tf; print(tf.__version__)'