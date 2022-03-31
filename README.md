# Helpful links

* Paper [End-to-End Object Detection with Transformers](https://arxiv.org/pdf/2005.12872.pdf)

* Official [Code](https://github.com/facebookresearch/detr)

* Python version should be 3.8

# Setup for nuimages dataset
- "sudo ./setup.sh" for downloading the nuimages dataset
- "pip install -r requirements.txt" for downloading the required python packages

# License

We are not using this software for any commercial purpose

Used software:

* PyTorch - BSD
* NuImages - Apache 2.0
* DETR - Apache 2.0

# Running the IO sandbox

Either submit the following:

```
$ sbatch -A training2203 --gres=gpu:1 --time=0-00:30 -o sum-image-val.out -p develbooster --wrap="python ./sum-image.py --ds_version v1.0-val --epoch 1"
```

or run locally:

```
$ python ./sum-image.py --ds_version v1.0-val --epoch 1
```

You should see some output like:

```
Namespace(lr=0.0001, wd=0.0001, epochs=1, num_classes=10, ds_length=100, batch_size=32, gpu=[0, 1, 2, 3], cp='', ds_version='v1.0-val', ds_path='/p/scratch/training2203/heat
ai/data/sets/nuimage/', print_trace=False)
0 :: 13346871296.0
Training ended in 211.628s
```
