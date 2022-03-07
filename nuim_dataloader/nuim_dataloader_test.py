from nuim_dataloader import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import patches


if __name__ == "__main__":
    # Test training dataset
    nuim_dataset = NuimDataset('data/sets/nuimages', version='v1.0-mini', transform=transforms.Compose([
                                               Rescale((900, 800))]))

    dataloader = DataLoader(nuim_dataset, batch_size=4,
                            shuffle=True, num_workers=5, collate_fn=collate_fn_nuim)

    print('train')
    for i_batch, batch in enumerate(dataloader):
        image_batched = batch[0]
        target_batched = batch[1]
        print(image_batched.size(), len(target_batched))

        # observe 4th batch and stop.
        if i_batch == 3:

            w1, h1, w2, h2 = target_batched[0][0][1] * image_batched.size(-1), \
                             target_batched[0][0][2] * image_batched.size(-2), \
                             target_batched[0][0][3] * image_batched.size(-1),\
                             target_batched[0][0][4] * image_batched.size(-2)

            fig, ax = plt.subplots()

            # Plot image
            ax.imshow(image_batched[0].permute(1, 2, 0))

            # Plot one bounding box
            rect = patches.Rectangle((w1, h1), w2-w1, h2-h1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.show()
            break

    # Test test dataset
    nuim_dataset_test = NuimDataset('data/sets/nuimages', version='v1.0-mini', transform=transforms.Compose([
        Rescale((900, 800))]), set='test')

    dataloader_test = DataLoader(nuim_dataset_test, batch_size=1,
                            shuffle=True, num_workers=1, collate_fn=collate_fn_nuim)
    print('test')
    for i_batch, batch in enumerate(dataloader_test):
        image_batched = batch[0]
        target_batched = batch[1]
        print(image_batched.size(), len(target_batched))
