# japanese-handwriting-nn

This is a fork of [https://github.com/charlietsai/japanese-handwriting-nn]. I have updated the model definitions
to keras2, also some models were inconsistent with respect to zero padding. Tsai's paper states that "Following [15], a small receptive field of 3 x 3 with a stride of 1 was used. This preserves the image size throughout the neural network", which (preservation of image size) would require zero padding for each convolutional layer. Also, model "M7_2" was named "M7_1" in the original repo ("M7_1" shows up twice there). This model also had 
2 more 256 layers as described in table 1 of the paper.

I have also addded explicit initializations to be of type "he_normal" as this seems to be what Tsai has been 
using in his paper. A LearningRateScheduler has been added to reduce the learning rate every 20 epochs by a factor
of 0.1 as described in the paper. Just a few models use batch normalization in the original repo, the paper seems
to suggest that batch normlization has been used in every model ("with batch normalization [8] after each weight
layer and before each activation layer").

There is an [ansible playbook](aws_start_spot_job.yml) to run a job on a spot instance on AWS. 
Input/output files are stored on a bucket in s3, adapt files [run_all.sh] and [run_job.sh]. 

The input images come from the ETL Character database [http://etlcdb.db.aist.go.jp/?page_id=56], I have
used the following directory structure for the input data:

    ETLC
    ├── ETL8B
    │   ├── ETL8B2C1
    │   ├── ETL8B2C2
    │   ├── ETL8B2C3
    │   └── ETL8INFO
    └── ETL9B
        ├── ETL9B_1
        ├── ETL9B_2
        ├── ETL9B_3
        ├── ETL9B_4
        ├── ETL9B_5
        └── ETL9INFO

---------
Original README.md:

Handwritten japanese character recognition using neural networks.

Read the corresponding paper [here](writeup.pdf).

An example job running the M16 model on the hiragana dataset is included [here](example_job.py). 

You will need to obtain the ETL Character Database [here](http://etlcdb.db.aist.go.jp/) and make sure the `ETL_path` in [`/preprocessing/data_utils.py`](/preprocessing/data_utils.py) is correct.
