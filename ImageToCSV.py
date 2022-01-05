import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import argparse
import os
from scipy import ndimage as ndi
from Constants import SIZE, RADIUS
import CellFromIllustrator
import scipy.ndimage
import matplotlib.patches as patches
"""
This program takes as input an image, with red dots on a white background.
It creates a csv file, with the location of each of the dots
If you want to manually count cells, boutons, or other structures in histological tissue, open the file in
Photoshop and create a new "counting" layer. Place a red dot with the brush tool above every structure you want to count.
After you are finished counting, export this counting layer to a separate image. An example of one of these images
can be found "R:\Fillan\BoutonCount\05\Fillan.jpg"
The output csv file will look like:
IMAGE_NAME
IMAGE_NAME,IMAGE_SHAPE_X,IMAGE_SHAPE_Y,"Cell",BOUTON_BOX_LEFT_X,BOUTON_BOX_TOP_Y,BOUTON_BOX_RIGHT_X,BOUTON_BOX_BOTTOM_Y
IMAGE_NAME,IMAGE_SHAPE_X,IMAGE_SHAPE_Y,"Cell",BOUTON_BOX_LEFT_X,BOUTON_BOX_TOP_Y,BOUTON_BOX_RIGHT_X,BOUTON_BOX_BOTTOM_Y
IMAGE_NAME,IMAGE_SHAPE_X,IMAGE_SHAPE_Y,"Cell",BOUTON_BOX_LEFT_X,BOUTON_BOX_TOP_Y,BOUTON_BOX_RIGHT_X,BOUTON_BOX_BOTTOM_Y
IMAGE_NAME,IMAGE_SHAPE_X,IMAGE_SHAPE_Y,"Cell",BOUTON_BOX_LEFT_X,BOUTON_BOX_TOP_Y,BOUTON_BOX_RIGHT_X,BOUTON_BOX_BOTTOM_Y
...
Where the first number is the X-coordinate, and the second number is the Y-coordinate
Although most of the code of this program is the same as in BoutonCount/ImageToCSV, this version outputs in a slightly
different format intended to be used in GenerateTrainData.py
"""


class Search:
    """
    Humans are inaccurate at placing boutons at the exact local minima
    This class moves any given point to the nearby local minima
    """
    def __init__(self, image_path, radius=3):
        self.radius = radius
        self.mask = np.zeros((radius * 2 + 1, radius * 2 + 1))
        for i in range(self.mask.shape[0]):
            for j in range(self.mask.shape[1]):
                distance = (i - radius) ** 2 + (j - radius) ** 2
                self.mask[i, j] = distance * 3
        self.unblurred_image = CellFromIllustrator.file_to_array(image_path)
        self.image = scipy.ndimage.gaussian_filter(self.unblurred_image, 0.5)

    def find_local_minima(self, x, y, show=False):
        subset = self.image[x-self.radius:x+self.radius+1, y-self.radius:y+self.radius+1]
        if not subset.size == 4 * (self.radius * self.radius) + 4 * self.radius + 1:
            return x, y
        d_x, d_y = np.unravel_index(np.argmin(subset + self.mask), subset.shape)
        if show:
            big_radius = self.radius * 2
            big_subset = self.unblurred_image[x-big_radius:x+big_radius+1, y-big_radius:y+big_radius+1]
            if big_subset.size == 4 * (big_radius * big_radius) + 4 * big_radius + 1:
                plt.imshow(big_subset, cmap="Greys_r", vmin=0, vmax=255)
                ax = plt.gca()
                rect = patches.Rectangle((big_radius - .5, big_radius - .5), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                rect = patches.Rectangle((big_radius + d_y - .5 - self.radius, big_radius + d_x - .5 - self.radius), 1, 1, linewidth=1, edgecolor='g', facecolor='none')
                ax.add_patch(rect)
                plt.show()
        x += d_x - self.radius
        y += d_y - self.radius
        return x, y


def get_adjacent_indices(i, j, shape):
    adjacent_indices = []
    for m in range(i - 1, i + 2):
        for n in range(j - 1, j + 2):
            adjacent_indices.append((m, n))
    return adjacent_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manually identify boutons")
    parser.add_argument("-f", "--folder", type=str, help="Folder for three following arguments")
    parser.add_argument("-i", "--image", type=str, help="Input image file path")
    parser.add_argument("-l", "--labels", type=str, help="Input image file path")
    parser.add_argument("-c", "--csv", type=str, help="Output csv file path")
    args = parser.parse_args()
    assert args.image and args.csv and args.folder and args.labels, "Command line arguments are None"
    assert os.path.exists(os.path.join(args.folder, args.labels)), "Image does not exist: %s" % args.image
    image = matplotlib.image.imread(os.path.join(args.folder, args.labels))
    image = image[:, :, -3:]
    if np.max(image) <= 1:
        image = (image * 255).astype(np.int16)
    arr = 255 - np.min(image, axis=2)
    arr_binary = arr > 100
    center_points = np.zeros(shape=arr_binary.shape, dtype=bool)
    labels, num_objects = ndi.label(arr_binary)
    for i in range(1, num_objects + 1):
        if np.sum(labels == i) < 9:
            continue
        while np.sum(labels == i) > 0:
            # Find the top left most point that is that label
            item_index = np.where(labels == i)
            x, y = item_index[0][0], item_index[1][0]

            # Find a list of possible "center points"
            adjacent_indices = get_adjacent_indices(x, y, labels.shape)

            # Greedy algorithm to find the best "center point"
            best_point = (None, None)
            best_point_score = 0
            for adjacent_index in adjacent_indices:
                if adjacent_index[0] <= 0 or adjacent_index[0] >= arr.shape[0] or adjacent_index[1] <= 0 or adjacent_index[1] >= arr.shape[1]:
                    continue
                box_around = np.zeros(shape=labels.shape, dtype=bool)
                box_around[adjacent_index[0] - 1: adjacent_index[0] + 2, adjacent_index[1] - 1: adjacent_index[1] + 2] = np.True_
                if np.any(np.multiply(box_around, labels == 0)):
                    continue
                score = np.sum(np.multiply(box_around, labels == i))
                if score > best_point_score:
                    best_point_score = score
                    best_point = adjacent_index
            # Tombstone (by marking as -1) all points around the center point
            if best_point_score == 0:
                print(x, y)
                print("On edge")
                break
            labels[best_point[0] - 1: best_point[0] + 2, best_point[1] - 1: best_point[1] + 2] = -1
            center_points[best_point[0], best_point[1]] = True
            progress_bar_length = int(80 * np.sum(labels == -1) / np.sum(arr_binary))
            if i % 100 == 0:
                print("[" + "-" * progress_bar_length + " " * (80 - progress_bar_length) + "]")
    arr = arr_binary * 255 - 120 * center_points
    plt.imshow(arr, cmap="gray")
    plt.show()
    search = Search(os.path.join(args.folder, args.image))
    with open(os.path.join(args.folder, args.csv), "w+") as f:
        for point in np.argwhere(center_points):
            x, y = point
            x, y = search.find_local_minima(x, y, show=False)
            if x - RADIUS > 0 and y - RADIUS > 0 and x + RADIUS < arr.shape[0] and y + RADIUS < arr.shape[1]:
                f.write("%s,%i,%i,Cell,%i,%i,%i,%i\n" % (os.path.join(args.folder, args.image), arr.shape[0], arr.shape[1], x - RADIUS, y - RADIUS, RADIUS * 2, RADIUS * 2))
