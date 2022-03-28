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

def main():

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
    
    nuim_dataset = NuimDataset(path_to_ds, version='v1.0-mini', transform=transforms.Compose([Rescale((800, 800))]))
    dataloader = DataLoader(nuim_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn_nuim)    

    for batch_num, (img, tgts) in enumerate(dataloader):

        img = img.to(device)
        tgts_formatted_to_device = format_nuim_targets(tgts, device)

        pred = model(img)
        
        new_path = os.path.join(opt.save_data_dir, opt.dataset_name, batch_num)
        save_image(pred, "%s" % (new_path), normalize=True)

if __name__ == '__main__':    
    main()
