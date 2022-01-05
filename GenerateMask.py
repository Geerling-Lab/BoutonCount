import PIL.Image
import numpy as np
import argparse
import os
import scipy.spatial
import CellFromIllustrator
import GenerateTrainData
import matplotlib.pyplot as plt
from Constants import SIZE, RADIUS
import sklearn.metrics
import itertools
import NetworkEvaluate
from NetworkTrain import load_json_model


class ManageKeyPress():
    def __init__(self):
        self.X = []
        self.Y = []
        self.arr = None
        self.last = False

    def proposed_arrays(self, arr1, arr2):
        if arr1 is None or arr2 is None:
            self.arr = None
        else:
            self.arr = np.stack([arr1, arr2], axis=2)

    def on_key_press(self, event):
        if event.key == "1":
            self.last = True
            if self.arr is not None:
                y = [1, 0]
                xs, ys = GenerateTrainData.data_augmentation(self.arr, y, rotate=True)
                self.X.extend(xs)
                self.Y.extend(ys)
        elif event.key == "2":
            self.last = False
            if self.arr is not None:
                y = [0, 1]
                xs, ys = GenerateTrainData.data_augmentation(self.arr, y, rotate=True)
                self.X.extend(xs)
                self.Y.extend(ys)

    def save(self):
        folder = "Overlap"
        filepath = NetworkEvaluate.get_next_number(folder)
        print("Saving to %s" % filepath)
        if len(self.X) > 0:
            np.savez_compressed(filepath, X=self.X, Y=self.Y)


def dice(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make output image")
    parser.add_argument("-s", "--symbol", type=str, help="Symbol folder name")
    parser.add_argument("-i", "--input", type=str, help="Folder with csv files")
    args = parser.parse_args()
    csv_files = []
    for file in os.listdir(args.input):
        if file.endswith("csv"):
            csv_files.append(os.path.splitext(file)[0])
    csv_files.sort()
    shape = PIL.Image.open(os.path.join(args.input, "%s.tif" % csv_files[0])).size

    """
    Generate names of all intersections
    FoxP2, Lmx1b -> FoxP2 + Lmx1b, FoxP2, Lmx1b
    """

    csv_combinations = []
    for i in range(2 ** len(csv_files) - 1, 0, -1):
        s = format(i, "0%sb" % len(csv_files))
        c = []
        for j in range(len(csv_files)):
            if s[j] == "1":
                c.append(csv_files[j])
        csv_combinations.append(c)

    already_counted = np.zeros(shape=shape, dtype=np.bool)  # e.g. so a FoxP2+Lmx1b cell isn't counted 3 times
    counts = {}
    model = load_json_model(os.path.join("Overlap", "Model"))
    mkp = ManageKeyPress()
    for combination in csv_combinations:
        str_combination = "+".join(combination)
        if "Fluorogold" not in str_combination:
            continue
        print(str_combination)
        counts[str_combination] = 0
        output_img = PIL.Image.new('RGBA', shape, (0, 0, 0, 0))
        symbol_file = os.path.join(args.symbol, str_combination + ".png")
        symbol = PIL.Image.open(symbol_file)
        width, height = symbol.size
        if len(combination) == 1:
            file_name = combination[0]
            with open(os.path.join(args.input, "%s.csv" % file_name)) as f:
                for line in f:
                    x, y = map(int, line.split(","))
                    if not already_counted[x, y]:
                        counts[str_combination] += 1
                        output_img.paste(symbol, (x - width // 2, y - height // 2))
        else:
            image_arrays = []
            cells = []
            for file_name in combination:
                cells.append(np.genfromtxt(os.path.join(args.input, "%s.csv" % file_name), delimiter=","))
                image_arrays.append(CellFromIllustrator.file_to_array(os.path.join(args.input, "%s.tif" % file_name)))
            distance_arrays = np.zeros(shape=(cells[0].shape[0], len(cells) - 1))
            for j in range(1, len(cells)):
                distance = scipy.spatial.distance.cdist(cells[0], cells[j])
                distance_arrays[:, j - 1] = np.min(distance, axis=1)
            min_distances = np.max(distance_arrays, axis=1)
            proposed_overlapping_cells = np.where(min_distances < 20)[0].tolist()
            for index in proposed_overlapping_cells:
                x, y = map(int, cells[0][index, :].tolist())
                if not already_counted[x, y]:
                    MASK_LEVEL = 0.5
                    sub_image_arrays = [image_array[y-RADIUS:y+RADIUS, x-RADIUS:x+RADIUS] for image_array in image_arrays]

                    fig, ax = plt.subplots(1, len(image_arrays) + 1)
                    fig.set_size_inches(12 * (len(image_arrays) + 1), 10)
                    for i in range(len(image_arrays) + 1):
                        ax[i].set_xticks([])
                        ax[i].set_yticks([])
                    fig.tight_layout()
                    fig.canvas.set_window_title("1=Overlap, 2=No overlap")
                    a = np.zeros(shape=(SIZE, SIZE, 3))
                    for i, arr in enumerate(sub_image_arrays):
                        ax[i].imshow(arr, cmap='gray', vmin=0, vmax=255)
                        a[:, :, i] = sub_image_arrays[i] / 255
                    ax[len(image_arrays)].imshow(a)
                    if len(sub_image_arrays) == 2:
                        mkp.proposed_arrays(sub_image_arrays[0], sub_image_arrays[1])
                    else:
                        mkp.proposed_arrays(None, None)
                    cid = fig.canvas.mpl_connect("key_press_event", mkp.on_key_press)
                    plt.draw()
                    plt.waitforbuttonpress(0)
                    plt.close(fig)
                    if mkp.last:
                        output_img.paste(symbol, (x - width // 2, y - height // 2))
                        already_counted[x-RADIUS:x+RADIUS, y-RADIUS:y+RADIUS] = True
                        counts[str_combination] += 1
                    """
                    input = np.stack([arr1.reshape((1, SIZE, SIZE)), arr2.reshape((1, SIZE, SIZE))], axis=1)
                    prediction = model.predict(input)[0]
                    if prediction[0] < prediction[1]:
                        break
                    """
        output_img.save(os.path.join(args.input, "Output_" + str_combination + ".tif"))
        print("%s has %d cells" % (str_combination, counts[str_combination]))
    mkp.save()