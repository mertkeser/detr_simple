import numpy as np

from nuim_dataloader import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
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
    fig4, ax4 = plt.subplots()
    fig5, ax5 = plt.subplots()
    fig6, ax6 = plt.subplots()

    for i, version in enumerate(versions):
        print(version)
        nuim_dataset = NuimDataset('/p/scratch/training2203/heatai/data/sets/nuimage', version=version)

        dataloader = DataLoader(nuim_dataset, batch_size=20,
                                shuffle=False, num_workers=10, collate_fn=collate_fn_nuim)

        box_counter = np.zeros(len(nuim_dataset))
        box_size_norm_counter = dict((el, []) for el in range(num_of_class))
        box_area_counter = dict((el, []) for el in range(num_of_class))
        box_width_counter = dict((el, []) for el in range(num_of_class))
        box_height_counter = dict((el, []) for el in range(num_of_class))
        class_counter = np.zeros(list(nuim_dataset.class_to_number_table.values())[-1] + 1)

        image_counter = 0

        # Loop throught all batchs
        for i_batch, batch in enumerate(tqdm(dataloader)):
            image_batched = batch[0]
            target_batched = batch[1]

            image_width, image_height = image_batched.size(-1), image_batched.size(-2)

            # Loop through all images in one batch
            for idx, (_, target) in enumerate(zip(image_batched, target_batched)):
                box_counter[image_counter] = target.shape[0]
                image_counter += 1
                class_nums = target[:, 0]

                # count all results in one batch
                for i_c, class_num in enumerate(class_nums):
                    class_counter[int(class_num)] += 1

                    # Calculate results related to bounding box
                    box_width = (target[i_c, 3] - target[i_c, 1]) * image_width
                    box_height = (target[i_c, 4] - target[i_c, 2]) * image_height
                    box_area = box_width*box_height
                    box_diag_length = np.linalg.norm((target[i_c, 3] - target[i_c, 1],  target[i_c, 4] - target[i_c, 2]))

                    # Add result to counters
                    box_width_counter[int(class_num)].append(box_width)
                    box_height_counter[int(class_num)].append(box_height)
                    box_area_counter[int(class_num)].append(box_area)
                    box_size_norm_counter[int(class_num)].append(box_diag_length)

        ax1.bar(i, np.mean(box_counter), 0.5, yerr=np.std(box_counter), edgecolor='white', label=version, alpha=0.5, ecolor='black', capsize=10)

        ax2.bar(np.arange(num_of_class) + (i-1)*0.2, class_counter / len(nuim_dataset), 0.2, edgecolor='white', label = version)

        box_width_summary = np.array([(np.mean(v), np.std(v)) for k, v in box_width_counter.items()])
        box_height_summary = np.array([(np.mean(v), np.std(v)) for k, v in box_height_counter.items()])
        box_area_summary = np.array([(np.mean(v), np.std(v)) for k, v in box_area_counter.items()])
        box_size_summary = np.array([(np.mean(v), np.std(v)) for k, v in box_size_norm_counter.items()])

        ax3.bar(np.arange(num_of_class) + (i-1)*0.2, box_size_summary[:, 0], 0.2, alpha=0.5, ecolor='black', capsize=10, label = version)
        ax4.bar(np.arange(num_of_class) + (i - 1) * 0.2, box_width_summary[:, 0], 0.2, alpha=0.5, ecolor='black', capsize=10, label=version)
        ax5.bar(np.arange(num_of_class) + (i - 1) * 0.2, box_height_summary[:, 0], 0.2, alpha=0.5, ecolor='black', capsize=10, label=version)
        ax6.bar(np.arange(num_of_class) + (i - 1) * 0.2, box_area_summary[:, 0], 0.2, alpha=0.5, ecolor='black', capsize=10, label=version)

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
    fig3.savefig('Avg Box Size Of Class without std', dpi=500)

    ax4.set_title('Avg Box Width Of Class (Pixel)')
    ax4.set_ylabel('Avg Box Width (Pixel)')
    ax4.set_xlabel('Class')
    ax4.set_xticks(np.arange(num_of_class))
    ax4.set_xticklabels(classes)
    ax4.legend()
    fig4.set_size_inches((12, 9), forward=False)
    fig4.savefig('Avg Box Width Of Class without std', dpi=500)

    ax5.set_title('Avg Box Height Of Class (Pixel)')
    ax5.set_ylabel('Avg Box Height (Pixel)')
    ax5.set_xlabel('Class')
    ax5.set_xticks(np.arange(num_of_class))
    ax5.set_xticklabels(classes)
    ax5.legend()
    fig5.set_size_inches((12, 9), forward=False)
    fig5.savefig('Avg Box Height Of Class without std', dpi=500)

    ax6.set_title('Avg Box Area Of Class (Pixel)')
    ax6.set_ylabel('Avg Box Area (Pixel)')
    ax6.set_xlabel('Class')
    ax6.set_xticks(np.arange(num_of_class))
    ax6.set_xticklabels(classes)
    ax6.legend()
    fig6.set_size_inches((12, 9), forward=False)
    fig6.savefig('Avg Box Area Of Class without std', dpi=500)