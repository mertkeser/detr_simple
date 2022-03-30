#!/bin/bash

echo "activate environment"
source activate pytorch_p38


echo "run profiling"
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true -o my_profile python model_profiling.py
