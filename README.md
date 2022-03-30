# Helpful links

* Paper [End-to-End Object Detection with Transformers](https://arxiv.org/pdf/2005.12872.pdf)

* Official [Code](https://github.com/facebookresearch/detr)

* Python version should be 3.8

# Setup for nuimages dataset
- "sudo ./setup.sh" for downloading the nuimages dataset
- "pip install -r requirements.txt" for downloading the required python packages

# code setup

## on JUWELS Booster

(tested for Python 3.9)

- what also works (but requires a `requirements.txt` without matplotlib): 
    ```
    $ ml Stages/2022 GCC/11.2.0 CUDA/11.5
    $ ml Python/3.9.6
    ```

- go to your repo 
```
$ cd /path/to/repo
$ git clone http://psteinb@github.com/mertkeser/detr_simple.git
$ cd detr_simple
```

- set up `venv` folder:
```
$ python -m venv local-py38
$ source local-py38/bin/activate
(local-py38) [steinbach1@jwlogin24 detr_simple]$
```

- download requirements:
```
$ python -m pip install -U pip #important: update pip first, proved essential!
$ python -m pip install --extra-index-url https://download.pytorch.org/whl/cu113 -r dev-requirements.txt
```

- create folders for data:
```
$ mkdir -p data/sets/nuimage
$ cd data/sets/
$ wget -N https://www.nuscenes.org/data/nuimages-v1.0-mini.tgz
$ tar xf nuimages-v1.0-mini.tgz -C nuimage
```

# Training

When you are logged into a single node with 4 GPUs, the training can be performed with:

```
$ python ./main.py --epochs 5 --batch_size 64 --gpu 0 1 2 3
#...
distributed_backend=nccl
All distributed processes registered. Starting with 4 processes
----------------------------------------------------------------------------------------------------

LOCAL_RANK: 1 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 3 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 2 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3]

  | Name         | Type         | Params
----------------------------------------------
0 | backbone     | ResNet       | 11.2 M
1 | conv         | Conv2d       | 65.7 K
2 | transformer  | Transformer  | 3.8 M
3 | linear_class | Linear       | 1.4 K
4 | linear_bbox  | Linear       | 516
5 | loss_func    | SetCriterion | 0
----------------------------------------------
15.0 M    Trainable params
0         Non-trainable params
15.0 M    Total params
60.084    Total estimated model params size (MB)
Epoch 0:   0%|                                                                                                                                       | 0/328 [00:00<?, ?it/s]Epoch 0:  19%|████████████████████▎                                                                                     | 63/328 [02:01<08:31,  1.93s/it, loss=2.76, v_num=2]

```

# License

We are not using this software for any commercial purpose

Used software:

* PyTorch - BSD
* NuImages - Apache 2.0
* DETR - Apache 2.0
