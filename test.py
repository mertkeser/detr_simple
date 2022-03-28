from operator import ge
from losses.object_detection_losses import SetCriterion
from models.simple_detr import SimpleDETR
from datasets import DummyDataset, build_CocoDetection
from torch.optim import AdamW
from torch.utils.data import DataLoader
from nuim_dataloader.nuim_dataloader import NuimDataset, Rescale, transforms, collate_fn_nuim
from utils.funcs import format_nuim_targets
import argparse
import sys
import torch
import os
import time


parser = argparse.ArgumentParser()
parser.add_argument("--save_data_dir", type=str, default="/output/results/", help="directory for saving images")
parser.add_argument("--checkpoint_dir", type=str, default="/output/results/checkpoints/", help="directory of checkpoints")
parser.add_argument("--input_data_dir", type=str, default="/output/data/", help="directory of input images")
parser.add_argument("--dataset_name", type=str, default="nuscenes", help="Name of the Dataset")
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
opt = parser.parse_args()
print(opt)

# set gpu ids
str_ids = opt.gpu_ids.split(',')
opt.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        opt.gpu_ids.append(id)
if len(opt.gpu_ids) > 0:
    torch.cuda.set_device(opt.gpu_ids[0])

device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
print(device)

# Create sample and checkpoint directories
if not os.path.exists("%s/%s" % (opt.save_data_dir, opt.dataset_name)):
    os.makedirs("%s/%s" % (opt.save_data_dir, opt.dataset_name))

def load_model(num_classes, device, folder, name='recent.pth'):
    model = SimpleDETR(num_classes=num_classes)
    path = os.path.join(folder, name)
    if os.path.exists(path):
        print('Loading model from: {}'.format(path))
        model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        return model
    print('Initializing model: {}'.format(path))
    return model

def main(args):

    #path_to_ds = r'C:\Code\_datasets\nuimages'
    path_to_ds = r'./data/sets/nuimage'
    
    # Test training dataset
    nuim_dataset = NuimDataset(path_to_ds, version='v1.0-mini', transform=transforms.Compose([Rescale((800, 800))]))
    dataloader = DataLoader(nuim_dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn_nuim)    

    #model = SimpleDETR(num_classes=num_classes)

    model = load_model(num_classes, device, folder)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)    
    
    criterion = SetCriterion(num_classes=num_classes)

    model = model.to(device)

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
        save_model(model, folder)

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
    parser.add_argument('--batch_size', default=15, type=int, help='Number of Classes')
    parser.add_argument('--gpu', default=0, type=int, help='GPU device number, ignored if absent')
    parser.add_argument('--cp', default='checkpoints', type=str, help='Checkpoints folder')

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
