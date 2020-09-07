
import matplotlib.pyplot as plt
import os
from os.path import join
import cv2

if __name__ == "__main__":

    dirs = [
        # '.\\output\\101_augm0_0030',
        # '.\\output\\101_augm0_0060',
        # '.\\output\\101_augm0_0090',
        # '.\\output\\101_augm1_0030',
        # '.\\output\\101_augm1_0060',
        # '.\\output\\101_augm1_0090',
        # '.\\output\\50_augm0_0030',
        # '.\\output\\50_augm0_0060',
        # '.\\output\\50_augm0_0090',
        # '.\\output\\50_augm1_0030',
        # '.\\output\\50_augm1_0060',
        # '.\\output\\50_augm1_0090'
        '.\\output\\50_augm1_07_0030',
        '.\\output\\50_augm1_07_0060',
        '.\\output\\50_augm1_07_0090'
    ]
    dirs_lst = [dirs]#[dirs[:3], dirs[3:6], dirs[6:9], dirs[9:12]]

    for dirs_paths in dirs_lst:
        fig, axis = plt.subplots(3, 2*4, figsize=(48, 18))
        rows = 3
        columns = 2
        for epoch in range(3):
            root = dirs_paths[epoch]
            files = os.listdir(root)
            for row in range(rows):
                for col in range(columns):
                    img_index = col + row * columns
                    mask_name = "mask_{}.png".format(img_index)
                    if mask_name in files:
                        img = cv2.imread(join(root, mask_name))
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    else:
                        img = cv2.imread(join(root, "org_{}.png".format(img_index)))
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    axis[row, epoch * columns + col].imshow(img)
                    axis[row, epoch * columns + col].tick_params(
                        axis='both',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        bottom=False,  # ticks along the bottom edge are off
                        top=False,  # ticks along the top edge are off
                        left=False,
                        right=False,
                        labelleft=False,
                        labelbottom=False)  # labels along the bottom edge are off
        for row in range(rows):
            for col in range(columns):
                epoch = 3
                img_index = col + row * columns
                mask_name = "mask_{}_gt.png".format(img_index)
                img = cv2.imread(join(root, mask_name))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                axis[row, epoch * columns + col].imshow(img)
                axis[row, epoch * columns + col].tick_params(
                    axis='both',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    left=False,
                    right=False,
                    labelleft=False,
                    labelbottom=False)  # labels along the bottom edge are off

        plt.subplots_adjust(hspace=0.1, wspace=0.1)
        name = root.split("_")[0] +"_" + root.split("_")[1]+"_seq.png"
        plt.savefig(name)

    a=0
