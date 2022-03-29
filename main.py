from operator import ge
from losses.object_detection_losses import SetCriterion
from models.simple_detr import SimpleDETR
from datasets import DummyDataset, build_CocoDetection
from torch.optim import AdamW
from torch.utils.data import DataLoader
from nuim_dataloader.nuim_dataloader import NuimDataset, Rescale, transforms, collate_fn_nuim
from utils.funcs import format_nuim_targets, generate_trace_report
import argparse
import sys
import torch
import os
import time
from torch import nn
from pathlib import Path

#lr = 1e-4
#wd = 1e-4
#epochs = 10
#num_classes = 15

#ds_length = 100
#batch_size = 10

def get_cuda_device():
    return torch.device('cuda' if torch.cuda.is_available() else "cpu")

def save_model(model, folder, name='recent.pth'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    print('Saving model to: {}'.format(os.path.join(folder, name)))
    torch.save(model.state_dict(), os.path.join(folder, name))


def load_model(num_classes, device, folder, name='recent.pth'):
    model = SimpleDETR(num_classes=num_classes)
    croot = Path(folder)
    if croot.exists() and (croot / name).exists():
        path = croot / name
        print('Loading model from: {}'.format(path))
        model.load_state_dict(torch.load(str(path), map_location=lambda storage, loc: storage))
        return model

    return model


def main(args):
    start_time = time.time()

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
    dsversion = args.ds_version
    dspath = args.ds_path

    #dataset = DummyDataset(ds_length, num_classes=num_classes)
    #dataset = build_CocoDetection('val', 'C:\Code\_datasets\coco', True)
    #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #path_to_ds = r'C:\Code\_datasets\nuimages'
    #path_to_ds = r"/p/scratch/training2203/heatai/data/sets/nuimage"
    
    # Test training dataset
    nuim_dataset = NuimDataset(dspath, version=dsversion, transform=transforms.Compose([Rescale((800, 800))]))
    
    dataloader = DataLoader(nuim_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4*len(gpus), #4 produces 4 threads per GPU to shuffle the data to device
                            collate_fn=collate_fn_nuim,
                            pin_memory = True if run_on_gpu else False)

    #model = SimpleDETR(num_classes=num_classes)

    model = load_model(num_classes, device, folder)

    if args.print_trace:
        generate_trace_report(model, torch.device('cpu'), batch_size=20, input_size=(800, 800), filename="trace_cpu.json")
        generate_trace_report(model, device, batch_size=20, input_size=(800, 800), filename="trace.json")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)    
    
    criterion = SetCriterion(num_classes=num_classes)

    model = model.to(device)
    
    model = nn.DataParallel(model, device_ids = gpus)

    for epoch in range(epochs):

        epoch_start_time = time.time()

        epoch_loss = 0.

        batches = len(dataloader.dataset) // dataloader.batch_size

        for batch_num, (img, tgts) in enumerate(dataloader):

            batch_start_time = time.time()

            img = img.to(device)
            tgts_formatted_to_device = format_nuim_targets(tgts, device)

            pred = model(img)

            batch_losses_dict, total_batch_loss = criterion(pred, tgts_formatted_to_device)

            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()

            batch_loss_np = total_batch_loss.detach().cpu().numpy()

            epoch_loss += total_batch_loss.detach().cpu().numpy()

            #save_model(model, folder, name='e{}b{}.pth'.format(epoch, batch_num))
            batch_delta_time = time.time() - batch_start_time
            print('({:.3f}s) Batch [{}/{}] loss:'.format(batch_delta_time, batch_num + 1, batches), batch_loss_np)

        #save_model(model, folder, name='e{}.pth'.format(epoch))
        if Path(folder).exists():
            print(f'checkpoint written to {folder}')
            save_model(model, folder)
        else:
            print('skipping writing checkpoints')

        epoch_delta_time = time.time() - epoch_start_time
        print('({:.3f}s) Epoch [{}/{}] loss:'.format(epoch_delta_time, epoch + 1, epochs), epoch_loss)

    print('Training ended in {:.3f}s'.format(time.time() - start_time))


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate')
    parser.add_argument('--wd', default=1e-4, type=float, help='wd')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('--num_classes', default=10, type=int, help='Number of Classes')
    parser.add_argument('--ds_length', default=100, type=int, help='Length of the ds')
    parser.add_argument('--batch_size', default=32, type=int, help='Number of Classes')
    parser.add_argument('--gpu', default=[0,1,2,3], help='GPU device number, ignored if absent', nargs='+')
    parser.add_argument('--cp', default="", type=str, help='Checkpoints folder (note: if the folder does not exist, checkpointing is skipped)')
    parser.add_argument('--ds_version', default='v1.0-train', type=str, help='dataset version')
    parser.add_argument('--ds_path', default='/p/scratch/training2203/heatai/data/sets/nuimage/', type=str, help='dataset path')
    parser.add_argument('--print_trace', default=False, action='store_true', help='print a trace report')

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
    main(args)
