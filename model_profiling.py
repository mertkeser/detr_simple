import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import AdamW
from losses.object_detection_losses import SetCriterion

from models.simple_detr import SimpleDETR
import argparse
import os
import sys
import numpy as np

def get_cuda_device():
    return torch.device('cuda' if torch.cuda.is_available() else "cpu")

def load_model(num_classes, device, folder, name='recent.pth'):
    model = SimpleDETR(num_classes=num_classes)
    path = os.path.join(folder, name)
    if os.path.exists(path):
        print('Loading model from: {}'.format(path))
        model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        return model
    print('Initializing model: {}'.format(path))
    return model

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate')
    parser.add_argument('--wd', default=1e-4, type=float, help='wd')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('--num_classes', default=10, type=int, help='Number of Classes')
    parser.add_argument('--ds_length', default=100, type=int, help='Length of the ds')
    parser.add_argument('--batch_size', default=32, type=int, help='Number of Classes')
    parser.add_argument('--gpu', default=[0,1,2,3], help='GPU device number, ignored if absent', nargs='+')
    parser.add_argument('--cp', default='checkpoints', type=str, help='Checkpoints folder, ignored if it doesn\'t exist')
    parser.add_argument('--ds_version', default='v1.0-train', type=str, help='dataset version')
    parser.add_argument('--ds_path', default='/p/scratch/training2203/heatai/data/sets/nuimage', type=str, help='dataset path')

    try:
        parsed = parser.parse_args(argv)
        print(parsed)
    except argparse.ArgumentError:
        parser.print_help()
        sys.exit(2)
    else:
        return parsed

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])

    #Introducing the arguments
    lr = args.lr
    wd = args.wd
    epochs = args.epochs
    # num_classes contains exact number of classes in dataset
    # DETR requires no object (∅) class, it is added in SimpleDETR and SetCriterion classes
    num_classes = args.num_classes
    ds_length = args.ds_length
    batch_size = args.batch_size
    gpus = [int(i) for i in (args.gpu)]
    device = get_cuda_device()
    run_on_gpu = "gpu" in str(device) or len(gpus) > 0
    folder = args.cp

    # setup
    device = get_cuda_device()
    model = load_model(num_classes, device, folder)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = SetCriterion(num_classes=num_classes)

    data = torch.randn(batch_size, 3, 800, 800, device=device)

    # Generate the target
    targets = []
    for i_batch in range(batch_size):
        number_of_objs = np.random.randint(low=1, high=10)
        tgt_dict = {'labels': [], 'boxes': []}

        labels = np.random.randint(low=1, high=9, size=number_of_objs)
        boxes = np.random.uniform(low=0, high=1, size=(number_of_objs, 4))

        labels = torch.from_numpy(labels).to(device)
        boxes = torch.from_numpy(boxes).to(device)

        tgt_dict['labels'] = labels
        tgt_dict['boxes'] = boxes

        targets.append(tgt_dict)

    model = model.to(device)
    model = nn.DataParallel(model, device_ids=gpus)

    nb_iters = 11
    warmup_iters = 10
    for i in range(nb_iters):
        optimizer.zero_grad()

        # start profiling after 10 warmup iterations
        if i == warmup_iters: torch.cuda.cudart().cudaProfilerStart()

        # push range for current iteration
        if i >= warmup_iters: torch.cuda.nvtx.range_push("iteration{}".format(i))

        # push range for forward
        if i >= warmup_iters: torch.cuda.nvtx.range_push("forward")
        output = model(data)
        if i >= warmup_iters: torch.cuda.nvtx.range_pop()

        loss = criterion(output, targets)

        if i >= warmup_iters: torch.cuda.nvtx.range_push("backward")
        loss.backward()
        if i >= warmup_iters: torch.cuda.nvtx.range_pop()

        if i >= warmup_iters: torch.cuda.nvtx.range_push("opt.step()")
        optimizer.step()
        if i >= warmup_iters: torch.cuda.nvtx.range_pop()

        # pop iteration range
        if i >= warmup_iters: torch.cuda.nvtx.range_pop()

    torch.cuda.cudart().cudaProfilerStop()