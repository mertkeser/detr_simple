#!/bin/bash

echo "activate environment"
source activate pytorch_p38

# required to avoid a library export error (if occurs)
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ec2-user/anaconda3/lib

echo "run main"
python main.py