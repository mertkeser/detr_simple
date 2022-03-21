from losses.object_detection_losses import SetCriterion
from models.simple_detr import SimpleDETR
from datasets import DummyDataset, build_CocoDetection
from torch.optim import AdamW
from torch.utils.data import DataLoader
from nuim_dataloader.nuim_dataloader import NuimDataset, Rescale, transforms, collate_fn_nuim
from utils.funcs import format_nuim_targets
import argparse
import sys

#lr = 1e-4
#wd = 1e-4
#epochs = 10
#num_classes = 15

#ds_length = 100
#batch_size = 10

def main(args):

    #Introducing the arguments
    lr = args.lr
    wd = args.wd
    epochs = args.epochs
    # num_classes contains exact number of classes in dataset
    # DETR requires no object (∅) class, it is added in SimpleDETR and SetCriterion classes
    num_classes = args.num_classes
    ds_length = args.ds_length
    batch_size = args.batch_size

    #dataset = DummyDataset(ds_length, num_classes=num_classes)
    #dataset = build_CocoDetection('val', 'C:\Code\_datasets\coco', True)
    #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #path_to_ds = r'C:\Code\_datasets\nuimages'
    path_to_ds = r'/p/home/jusers/keser1/juwels/detr_simple/data/sets/nuimage'
    
    # Test training dataset
    nuim_dataset = NuimDataset(path_to_ds, version='v1.0-mini', transform=transforms.Compose([Rescale((800, 800))]))
    dataloader = DataLoader(nuim_dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn_nuim)    

    model = SimpleDETR(num_classes=num_classes)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)    
    
    criterion = SetCriterion(num_classes=num_classes)

    for epoch in range(epochs):

        epoch_loss = 0.

        for img, tgts in dataloader:

            pred = model(img)

            targets_new = format_nuim_targets(tgts)

            batch_losses_dict, total_batch_loss = criterion(pred, targets_new)

            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()

            print('Batch loss:', total_batch_loss.detach().cpu().numpy())

            epoch_loss += total_batch_loss.detach().cpu().numpy()

        print('{} epoch loss:'.format(epoch), epoch_loss)

    pass


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate')
    parser.add_argument('--wd', default=1e-4, type=float, help='wd')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('--num_classes', default=10, type=int, help='Number of Classes')
    parser.add_argument('--ds_length', default=100, type=int, help='Length of the ds')
    parser.add_argument('--batch_size', default=15, type=int, help='Number of Classes')

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
