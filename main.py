from operator import ge
from losses.object_detection_losses import SetCriterion
from models.simple_detr import SimpleDETR
from datasets import DummyDataset, build_CocoDetection
from torch.optim import AdamW
from torch.utils.data import DataLoader
from nuim_dataloader.nuim_dataloader import (
    NuimDataset,
    Rescale,
    transforms,
    collate_fn_nuim,
)
from utils.funcs import format_nuim_targets, generate_trace_report
import argparse
import sys
import torch
import os
import time
from torch import nn
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy as ddp

# from pytorch_lightning.strategies import DDPShardedStrategy as dshard
# from pytorch_lightning.strategies import DeepSpeedStrategy as dspeed

# lr = 1e-4
# wd = 1e-4
# epochs = 10
# num_classes = 15

# batch_size = 10


def get_cuda_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, folder, name="recent.pth"):
    if len(folder) == 0:
        return

    if not os.path.exists(folder):
        os.makedirs(folder)

    print("Saving model to: {}".format(os.path.join(folder, name)))
    dst = os.path.join(folder, name)
    torch.save(model.state_dict(), dst)


def load_model(num_classes, device, folder, name="recent.pth"):
    model = SimpleDETR(num_classes=num_classes)
    croot = Path(folder)
    if croot.exists() and (croot / name).exists():
        path = croot / name
        print("Loading model from: {}".format(path))
        model.load_state_dict(
            torch.load(str(path), map_location=lambda storage, loc: storage)
        )
        return model

    return model


def main(args):

    pl.seed_everything(args.seed)

    start_time = time.time()

    # Introducing the arguments
    lr = args.lr
    wd = args.wd
    epochs = args.epochs
    # num_classes contains exact number of classes in dataset
    # DETR requires no object (∅) class, it is added in SimpleDETR and SetCriterion classes
    num_classes = args.num_classes
    batch_size = args.batch_size
    gpus = [int(i) for i in (args.gpu)]
    device = get_cuda_device()
    run_on_gpu = "gpu" in str(device) or len(gpus) > 0
    folder = args.cp
    dsversion = args.ds_version
    dspath = args.ds_path

    # dataset = build_CocoDetection('val', 'C:\Code\_datasets\coco', True)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # path_to_ds = r'C:\Code\_datasets\nuimages'
    # path_to_ds = r"/p/scratch/training2203/heatai/data/sets/nuimage"

    # consider porting this to a Lightning Data Module
    # Test training dataset
    train_nuim_dataset = NuimDataset(
        dspath, version=dsversion, transform=transforms.Compose([Rescale((800, 800))])
    )

    train_dataloader = DataLoader(
        train_nuim_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
        * len(gpus),  # 4 produces 4 threads per GPU to shuffle the data to device
        collate_fn=collate_fn_nuim,
        pin_memory=True if run_on_gpu else False, persistent_workers=True
    )

    # Test training dataset
    test_nuim_dataset = NuimDataset(
        dspath, version="v1.0-val", transform=transforms.Compose([Rescale((800, 800))])
    )

    test_dataloader = DataLoader(
        test_nuim_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
        * len(gpus),  # 4 produces 4 threads per GPU to shuffle the data to device
        collate_fn=collate_fn_nuim,
        pin_memory=True if run_on_gpu else False, persistent_workers=True
    )

    criterion = SetCriterion(num_classes=num_classes)

    # default is to train from scratch
    # we might want to consider loading a checkpoint

    model = None
    if args.reference:
        refpath = Path(args.reference)
        if refpath.exists() and refpath.stat().st_size > 0:
            model = SimpleDETR.load_from_checkpoint(checkpoint_path=str(refpath))
    else:
        model = SimpleDETR(num_classes=num_classes, lr=lr, wd=wd, loss_func=criterion)

    # model = load_model(num_classes, device, folder)
    strategy = ddp(find_unused_parameters=False)
    # the following is a placeholder, use these lines of code if you like to integrate
    #   deepspeed (https://pypi.org/project/deepspeed/) or
    #   fairscale (https://pypi.org/project/fairscale/)
    # if "speed" in args.strategy.lower():
    #     print("using deepspeed strategy")
    #     strategy = dspeed(pin_memory=True)
    #     details: https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.strategies.DeepSpeedStrategy.html#pytorch_lightning.strategies.DeepSpeedStrategy
    # elif "shard" in args.strategy.lower():
    #     print("using DDP sharded strategy")
    #     strategy = dshard(pin_memory=True)
    #     details: https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.strategies.DDPShardedStrategy.html#pytorch_lightning.strategies.DDPShardedStrategy
    # else:
    #     print("using ddp strategy")
    nconfig = [int(item) for item in args.nodeconfig.split("x")]
    config = {}
    if nconfig[0] < 2:
        config = dict(gpus=gpus[: nconfig[-1]])
    else:
        config = dict(num_nodes=nconfig[0], devices=nconfig[-1])
    print(f"deduced DDP configuration {config} from {nconfig}")

    trainer = pl.Trainer(
        max_epochs=epochs,
        strategy=strategy,
        **config
        ##logger=TensorBoardLogger("./logs", name="best_model_ever"),
        # log every nth batch, default 50
        ##log_every_n_steps=10,
        # logger=None,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_dataloader,
    )

    if len(folder) > 1:
        dst = Path(folder) / "last.pth"
        model.save_checkpoint(str(dst))

    print("Training ended in {:.3f}s".format(time.time() - start_time))


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=1e-4, type=float, help="Initial learning rate")
    parser.add_argument("--wd", default=1e-4, type=float, help="wd")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs")
    parser.add_argument("--num_classes", default=10, type=int, help="Number of Classes")

    parser.add_argument("--batch_size", default=32, type=int, help="Number of Classes")
    parser.add_argument(
        "--seed",
        default=3,
        type=int,
        help="random seed to init the model and shuffle the data",
    )
    parser.add_argument(
        "--reference",
        default=None,
        type=str,
        help="path of a model checkpoint to start from (if empty or None, train from scratch)",
    )
    parser.add_argument(
        "--gpu",
        default=[0, 1, 2, 3],
        help="GPU device number, ignored if absent",
        nargs="+",
    )
    parser.add_argument(
        "--nodeconfig",
        default="1x4",
        help="node configuration for multi_node training (4 nodes with 2 gpus each: 4x2, 2 nodes with 3 gpus each: 2x3)",
    )

    parser.add_argument(
        "--cp",
        default="",
        type=str,
        help="Checkpoints folder (note: if the folder does not exist, checkpointing is skipped)",
    )
    # parser.add_argument(
    #     "--strategy",
    #     default="ddp",
    #     choices="ddp,dshard,dspeed".split(","),
    #     type=str,
    #     help="strategy for parallelization, for details see https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html#training-on-accelerators",
    # )
    parser.add_argument(
        "--ds_version", default="v1.0-train", type=str, help="dataset version"
    )
    parser.add_argument(
        "--ds_path",
        default="/p/scratch/training2203/heatai/data/sets/nuimage/",
        type=str,
        help="dataset path",
    )

    try:
        parsed = parser.parse_args(argv)
        print(parsed)
    except argparse.ArgumentError:
        parser.print_help()
        sys.exit(2)
    else:
        return parsed


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
