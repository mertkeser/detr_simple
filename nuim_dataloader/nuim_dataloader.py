from __future__ import print_function, division
import os
import torch
import numpy as np
from skimage import transform
from torch.utils.data import Dataset
from torchvision import transforms
from nuimages import NuImages
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, target = sample

        h, w = image.size()[-2:]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = torch.as_tensor(transform.resize(image, (image.size(0), new_h, new_w)))

        # Update the bounding box points
        # target[:, 1:] = torch.mm(target[:, 1:].to(torch.float32), torch.diag(torch.tensor([new_w / w, new_h / h, new_w / w, new_h / h])))

        return img, target


class NuimDataset(Dataset):
    """NuImage dataset."""

    def __init__(self, root_dir = "", version = 'v1.0-mini', set='train', train_ratio=1.0, transform=None):
        """
        Args:
            root_dir (string): Directory with NuImage Dataset.
            version (string): Version of NuImage Dataset.
            set (string): 'train' or 'test'
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.set = set
        self.nuim = NuImages(dataroot=root_dir, version=version, verbose=False, lazy=False)
        self.split_index = int(len(self.nuim.sample) * train_ratio)
        self.convert_tensor = transforms.ToTensor()
        self.class_to_number_table = {
            'movable_object.barrier':               0,
            'vehicle.bicycle':                      1,
            'vehicle.bus.bendy':                    2,
            'vehicle.bus.rigid':                    2,
            'vehicle.car':                          3,
            'vehicle.construction':                 4,
            'vehicle.motorcycle':                   5,
            'human.pedestrian.adult':               6,
            'human.pedestrian.child':               6,
            'human.pedestrian.construction_worker': 6,
            'human.pedestrian.police_officer':      6,
            'movable_object.trafficcone':           7,
            'vehicle.trailer':                      8,
            'vehicle.truck':                        9
        }
        """self.class_to_number_table = {
            'movable_object.barrier':               10,
            'vehicle.bicycle':                      20,
            'vehicle.bus.bendy':                    30,
            'vehicle.bus.rigid':                    30,
            'vehicle.car':                          40,
            'vehicle.construction':                 50,
            'vehicle.motorcycle':                   60,
            'human.pedestrian.adult':               70,
            'human.pedestrian.child':               70,
            'human.pedestrian.construction_worker': 70,
            'human.pedestrian.police_officer':      70,
            'movable_object.trafficcone':           80,
            'vehicle.trailer':                      90,
            'vehicle.truck':                        100
        }"""

    def __len__(self):
        if self.set == 'train':
            return self.split_index
        elif self.set == 'test':
            return self.nuim.sample.__len__() - self.split_index

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.set == 'test':
            idx = idx + self.split_index

        # Get image
        sample = self.nuim.sample[idx]
        key_camera_token = sample['key_camera_token']
        sample_data = self.nuim.get('sample_data', key_camera_token)
        im_path = os.path.join(self.nuim.dataroot, sample_data['filename'])
        image = self.convert_tensor(Image.open(im_path))

        # Get target
        interested_ann = [o for o in self.nuim.object_ann if o['sample_data_token'] == key_camera_token]
        num_of_obj = len(interested_ann)

        target = np.zeros((num_of_obj, 5))

        for idx, ann in enumerate(interested_ann):
            target[idx, 0] = self.class_to_number(self.nuim.get('category', ann['category_token'])['name'])
            target[idx, 1:] = ann['bbox']
        target[:, [1, 3]] = target[:, [1, 3]] / image.size(2)
        target[:, [2, 4]] = target[:, [2, 4]] / image.size(1)

        target = torch.as_tensor(target)

        sample = image, target

        if self.transform:
            sample = self.transform(sample)

        return sample

    def class_to_number(self, obj_class):
        if obj_class in self.class_to_number_table.keys():
            return self.class_to_number_table[obj_class]

        return 0


def collate_fn_nuim(batch):
    """
    Args:
        batch: Batch data

    Output:
        batch: batch[0] is images (4D tensor) with dimension [B, C, H, W]
        batch[1] is LIST of targets with [num_of_obj, parameters], parameters[0] = Classification, parameters[1:] = boundbox_points
    """
    # Note that batch is a list
    batch = list(map(list, zip(*batch)))

    numel = sum([x.numel() for x in batch[0]])
    storage = batch[0][0].storage()._new_shared(numel)
    out = batch[0][0].new(storage)

    batch[0] = torch.stack(batch[0], 0, out=out)

    return batch
