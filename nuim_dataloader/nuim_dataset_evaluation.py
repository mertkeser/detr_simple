import numpy as np

from nuim_dataloader import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Test training dataset
    versions = ['v1.0-train', 'v1.0-val', 'v1.0-test']
    classes = ['barrier',
               'bicycle',
               'bus',
               'car',
               'construction',
               'motorcycle',
               'pedestrian',
               'traffic_cone',
               'trailer',
               'truck']
    num_of_class = len(classes)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    for i, version in enumerate(versions):
        nuim_dataset = NuimDataset('/p/scratch/training2203/heatai/data/sets/nuimage', version=version, transform=transforms.Compose([
                                                   Rescale((900, 800))]))

        dataloader = DataLoader(nuim_dataset, batch_size=4,
                                shuffle=False, num_workers=5, collate_fn=collate_fn_nuim)

        box_counter = np.zeros(len(nuim_dataset))
        box_size_counter = dict((el, []) for el in range(num_of_class))
        class_counter = np.zeros(list(nuim_dataset.class_to_number_table.values())[-1] + 1)


        image_counter = 0

        for i_batch, batch in enumerate(dataloader):
            image_batched = batch[0]
            target_batched = batch[1]

            # Loop through all images
            for idx, (_, target) in enumerate(zip(image_batched, target_batched)):
                box_counter[image_counter] = target.shape[0]
                image_counter += 1
                class_nums = target[:, 0]
                for i_c, class_num in enumerate(class_nums):
                    class_counter[int(class_num)] += 1
                    box_diag_length = np.linalg.norm((target[i_c, 1] - target[i_c, 2],  target[i_c, 3] - target[i_c, 4]))
                    box_size_counter[int(class_num)].append(box_diag_length)

        ax1.bar(i, np.mean(box_counter), 0.5, yerr=np.std(box_counter), edgecolor='white', label=version, alpha=0.5, ecolor='black', capsize=10)

        ax2.bar(np.arange(num_of_class) + (i-1)*0.2, class_counter / len(nuim_dataset), 0.2, edgecolor='white', label = version)

        box_size_summary = np.array([(np.mean(v), np.std(v)) for k, v in box_size_counter.items()])

        ax3.bar(np.arange(num_of_class) + (i-1)*0.2, box_size_summary[:, 0], 0.2, yerr= box_size_summary[:, 1],  alpha=0.5, ecolor='black', capsize=10, label = version)

    ax1.set_title('Number Of Boxes In One Image')
    ax1.set_ylabel('Number of Boxes')
    ax1.set_xticks(np.arange(len(versions)))
    ax1.set_xticklabels(versions)
    fig1.set_size_inches((12, 9), forward=False)
    fig1.savefig('Number Of Boxes In One Image.png', dpi=500)

    ax2.set_title('Avg Number Of Objects In One Image')
    ax2.set_ylabel('Avg Number')
    ax2.set_xlabel('Class')
    ax2.set_xticks(np.arange(num_of_class))
    ax2.set_xticklabels(classes)
    ax2.legend()
    fig2.set_size_inches((12, 9), forward=False)
    fig2.savefig('Avg Number Of Objects In One Image.png', dpi=500)

    ax3.set_title('Avg Box Size Of Class')
    ax3.set_ylabel('Normalised Avg Box Size')
    ax3.set_xlabel('Class')
    ax3.set_xticks(np.arange(num_of_class))
    ax3.set_xticklabels(classes)
    ax3.legend()
    fig3.set_size_inches((12, 9), forward=False)
    fig3.savefig('Avg Box Size Of Class', dpi=500)