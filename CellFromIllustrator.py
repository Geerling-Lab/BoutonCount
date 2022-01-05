import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PIL.Image
import argparse
import scipy.ndimage
import os
from Constants import SIZE, RADIUS


def read_metadata_file(filepath):
    target = None
    Images = {}
    with open(filepath) as f:
        for line in f:
            words = line.lower().split(":")
            if words[0] == "target":
                target = float(words[1])
            else:
                Images[os.path.join(os.path.dirname(filepath), words[0])] = float(words[1]) / target
    return Images


def find_image_files(image_name):
    labeled = None
    unlabeled = None
    folder = os.path.dirname(image_name)
    for file_name in os.listdir(folder):
        file_name_lower = file_name.lower()
        if os.path.basename(image_name) in file_name_lower and "labeled" in file_name_lower and not "unlabeled" in file_name_lower:
            labeled = os.path.join(folder, file_name)
        if os.path.basename(image_name) in file_name_lower and "unlabeled" in file_name_lower:
            unlabeled = os.path.join(folder, file_name)
    assert labeled is not None
    assert unlabeled is not None
    return labeled, unlabeled


def create_distance_array(arr):
    """
    :param arr: binary array
    :return: array of same size with each element the distance to the closest 0 element
    """
    distance_array = arr.astype(np.int8)
    distance = 1
    while arr.sum() > 0:
        distance += 1
        arr = scipy.ndimage.morphology.binary_erosion(arr)
        distance_array[arr == 1] = distance
    return distance_array


def file_to_array(filepath, average=True):
    im = PIL.Image.open(filepath)
    arr = np.array(im)
    if average:
        if len(arr.shape) > 2:
            arr = np.sum(arr, axis=2)
            arr = arr * (255 / arr.max())
    return arr.astype(np.int32)


def calculate_mean_values(arr):
    """
    Calculates the mean center value of array
    """
    x = int(np.sum(arr.sum(axis=1) * np.arange(arr.shape[0])) / arr.sum() + 0.5)
    y = int(np.sum(arr.sum(axis=0) * np.arange(arr.shape[1])) / arr.sum() + 0.5)
    return x, y


def wobble_point(fuzzy, point_x, point_y, wobble_distance=20):
    try:
        fuzzy_surround = fuzzy[point_x - wobble_distance: point_x + wobble_distance,
                         point_y - wobble_distance: point_y + wobble_distance]
        # maximum_point = np.unravel_index(fuzzy_surround.argmax(), fuzzy_surround.shape)
        maximum_point = calculate_mean_values(fuzzy_surround)
        point_x += maximum_point[0] - wobble_distance
        point_y += maximum_point[1] - wobble_distance
        return point_x, point_y
    except ValueError as v:
        return point_x, point_y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bounding boxes around cells")
    parser.add_argument("-m", "--metadata", type=str, help="Metadata filepath")
    parser.add_argument("-c", "--csv", type=str, help="Filepath of output csv file")
    args = parser.parse_args()
    if os.path.exists(args.csv):
        os.remove(args.csv)
    for image_name, scaling in read_metadata_file(args.metadata).items():
        labeled_name, unlabeled_name = find_image_files(image_name)
        labeled = file_to_array(labeled_name, average=False)
        unlabeled = file_to_array(unlabeled_name, average=False)

        difference = np.sum(np.abs(labeled - unlabeled), axis=2) > 5
        difference = scipy.ndimage.binary_fill_holes(difference)
        difference = scipy.ndimage.morphology.binary_erosion(difference, iterations=1)
        lbl, num_islands = scipy.ndimage.label(difference)
        points = scipy.ndimage.measurements.center_of_mass(difference, lbl, range(1, num_islands))
        unlabeled = np.average(unlabeled, axis=2).astype(np.int16)
        unlabeled = ((unlabeled - unlabeled.min()) * (255 / (unlabeled.max() - unlabeled.min()))).astype(np.uint8)

        # Normalize to between 0 and 255
        im = PIL.Image.fromarray(unlabeled)
        im = im.resize(size=(int(unlabeled.shape[1] * scaling), int(unlabeled.shape[0] * scaling)), resample=PIL.Image.BICUBIC)
        rescaled_name = unlabeled_name.replace("Unlabeled", "Rescaled").replace(".png", ".tif")
        im.save(rescaled_name)
        unlabeled = np.array(im)

        output = unlabeled
        fuzzy = scipy.ndimage.filters.gaussian_filter(unlabeled, sigma=3)

        fig, ax = plt.subplots(1)
        plt.imshow(output, cmap="gray")

        print("%s had %s cells" % (image_name, len(points)))
        with open(args.csv, "a+") as f:
            if len(points) == 0:
                f.write(rescaled_name)
                f.write(os.linesep)
            else:
                for point_x, point_y in points:
                    point_x = int(point_x * scaling + 0.5)
                    point_y = int(point_y * scaling + 0.5)

                    """
                    The point in illustrator was probably not placed exactly on top of the cell
                    This allows the center point to move slightly to be better placed
                    """

                    point_x, point_y = wobble_point(fuzzy, point_x, point_y)
                    """
                    show_radius = 70
                    try:
                        show = unlabeled[point_x - show_radius: point_x + show_radius, point_y - show_radius: point_y + show_radius]
                        show[show_radius + RADIUS, show_radius - RADIUS: show_radius + RADIUS] = 40
                        show[show_radius - RADIUS, show_radius - RADIUS: show_radius + RADIUS] = 40
                        show[show_radius - RADIUS: show_radius + RADIUS, show_radius - RADIUS] = 40
                        show[show_radius - RADIUS: show_radius + RADIUS, show_radius + RADIUS] = 40
                    except IndexError as e:
                        pass
                    """
                    circle = patches.Circle(xy=(point_y, point_x), radius=3, facecolor='r')
                    ax.add_patch(circle)
                    if point_x < RADIUS or point_x > unlabeled.shape[0] - RADIUS or point_y < RADIUS or point_y > unlabeled.shape[1]:
                        continue
                    f.write(",".join(map(str, [rescaled_name, unlabeled.shape[0], unlabeled.shape[1], "Cell", point_x - RADIUS, point_y - RADIUS, RADIUS * 2, RADIUS * 2])))
                    f.write(os.linesep)

        plt.title(image_name)
        plt.show()