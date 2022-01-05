import NetworkTrain
import CellFromIllustrator
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.segmentation
import scipy.ndimage
import numpy as np
import math
from GenerateTrainData import data_augmentation
from Constants import SIZE, RADIUS
import PIL.Image, PIL.ImageDraw


"""
This program is intended to find cells or boutons in the tissue, evaluating the NeuralNetwork
However, it has been almost entirely superseded by BoutonCount.py, and is obsolete"""

def get_next_number(folder):
    i = 0
    while True:
        i += 1
        if not os.path.exists(os.path.join(folder, "%02d.npz" % i)):
            break
    return os.path.join(folder, "%02d" % i)


def detect_local_maxima(arr, mask=None):
    scale = 20
    dilated = scipy.ndimage.morphology.grey_dilation(arr, size=(scale, scale))
    eroded = scipy.ndimage.morphology.grey_erosion(arr, size=(scale, scale))
    x1 = arr == dilated
    x2 = arr > eroded + 0.1
    x = np.logical_and(x1, x2)
    if mask is not None:
        x = x * mask
    x[:RADIUS, :] = False
    x[-RADIUS:, :] = False
    x[:, :RADIUS] = False
    x[:, -RADIUS:] = False
    return np.where(x)


def detect_local_minima(arr, mask=None):
    scale = 4
    dilated = scipy.ndimage.morphology.grey_dilation(arr, size=(scale, scale))
    eroded = scipy.ndimage.morphology.grey_erosion(arr, size=(scale, scale))
    x1 = arr == eroded
    x2 = arr < dilated - .01
    x = np.logical_and(x1, x2)
    if mask is not None:
        x = x * mask
    x[:RADIUS, :] = False
    x[-RADIUS:, :] = False
    x[:, :RADIUS] = False
    x[:, -RADIUS:] = False
    return np.where(x)


def local_maxima_generate_points(arr, mask=None, find_maxima=True):
    float_arr = arr.astype(np.float32)
    scale = 3
    fuzzy = scipy.ndimage.gaussian_filter(float_arr, sigma=scale)
    if find_maxima:
        maxima = np.array(detect_local_maxima(fuzzy, mask)).T
        print("%s putative cells detected" % np.shape(maxima)[0])
    else:
        maxima = np.array(detect_local_minima(fuzzy, mask)).T
    assert maxima.shape[0] > 0
    X = np.zeros(shape=(maxima.shape[0], 1, SIZE, SIZE), dtype=np.float32)
    for i in range(maxima.shape[0]):
        x, y = maxima[i, :]
        X[i, 0, :, :] = arr[x - RADIUS:x + RADIUS, y - RADIUS:y + RADIUS]
    return X, maxima


def felzenszwalb_generate_points(arr):
    fuzzy = scipy.ndimage.filters.gaussian_filter(arr, sigma=10)
    felzenszwalb = skimage.segmentation.felzenszwalb(arr, scale=.1, sigma=0.8, min_size=100)

    coms = scipy.ndimage.measurements.center_of_mass(arr, felzenszwalb, range(1, np.max(felzenszwalb)))
    X = []
    COMs = []
    for com in coms:
        if math.isnan(com[0]) or math.isnan(com[1]):
            continue
        x = int(com[0])
        y = int(com[1])
        if arr[x, y] > fuzzy[x, y]:
            x, y = CellFromIllustrator.wobble_point(fuzzy, x, y)
            sub_arr = arr[x-RADIUS:x+RADIUS, y-RADIUS:y+RADIUS]
            if sub_arr.shape == (SIZE, SIZE):
                X.append(sub_arr)
                COMs.append((x, y))
    X = np.array(X).reshape(-1, 1, 80, 80)
    COMs = np.array(COMs)
    return X, COMs


def take_sub_array(arr, x1, x2, y1, y2):
    place_x1 = 0
    place_x2 = x2 - x1
    place_y1 = 0
    place_y2 = y2 - y1
    sub_array = np.zeros(shape=(x2 - x1, y2 - y1))
    if x1 < 0:
        place_x1 = -x1
        x1 = 0
    if y1 < 0:
        place_y1 = -y1
        y1 = 0
    if x2 > arr.shape[0]:
        place_x2 = arr.shape[0] - x1
        x2 = arr.shape[0]
    if y2 > arr.shape[1]:
        place_y2 = arr.shape[1] - y1
        y2 = arr.shape[1]
    sub_array[place_x1: place_x2, place_y1: place_y2] = arr[x1: x2, y1: y2]
    return sub_array


class CreateSVG():
    def __init__(self, output_filename, artboard_size_xy, input_image):
        self.output_filename = output_filename
        x = artboard_size_xy[1]
        y = artboard_size_xy[0]
        self.string = \
"""<?xml version="1.0" encoding="utf-8"?>
<!-- Generator: Adobe Illustrator 23.0.3, SVG Export Plug-In . SVG Version: 6.00 Build 0)  -->
<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"
	 viewBox="0 0 %s %s" style="enable-background:new 0 0 %s %s;" xml:space="preserve">
<style type="text/css">
	.st0{fill:#FF0000;}
</style>
<symbol  id="Bouton" viewBox="-1.9 -1.9 3.8 3.8">
	<circle class="st0" cx="0" cy="0" r="1.9"/>
</symbol>
<g id="Image">
	<g id="_x30_2.tif_1_">
		<image style="overflow:visible;" width="%s" height="%s" id="_x30_2" xlink:href="%s" >
		</image>
	</g>
</g>
<g id="Boutons">""" % (x, y, x, y, x, y, input_image)

    def add_symbol(self, location_xy):
        self.string += \
"""		<use xlink:href="#Bouton"  width="3.8" height="3.8" x="-1.9" y="-1.9" transform="matrix(1.0026 0 0 -1.0026 %s %s)" style="overflow:visible;enable-background:new    ;"/>
""" % (location_xy[0] + 5, location_xy[1] + 5)

    def output(self):
        self.string += \
"""</g>
</svg>"""
        with open(self.output_filename, 'w+') as f:
            f.write(self.string)


class ManageMaskClick():
    def __init__(self, fig, ax, h, w):
        self.fig = fig
        self.ax = ax
        self.h = h
        self.w = w
        self.lines = []
        self.points = []

    def on_release(self, event):
        if event.button == 3:  # Right click
            if len(self.lines) > 0:
                self.lines[-1].remove()
                del self.lines[-1]
            if len(self.points) > 0:
                del self.points[-1]
            plt.draw()
        else:  # Left click
            if len(self.points) != 0:
                self.lines.append(ax.annotate("", xy=(self.points[-1]), xycoords="data", xytext=(event.xdata, event.ydata),
                                          textcoords="data",
                                          arrowprops=dict(arrowstyle="-", edgecolor="blue", linewidth=5, alpha=0.65,
                                                          connectionstyle="arc3,rad=0.")))
            self.points.append((event.xdata, event.ydata))
            plt.draw()

    def get_border_and_mask(self):
        img = PIL.Image.new("L", (self.h, self.w), 0)
        PIL.ImageDraw.Draw(img).polygon(self.points, outline=1, fill=1)
        mask = np.array(img)
        border = np.bitwise_and(mask, np.invert(scipy.ndimage.morphology.binary_erosion(mask)))
        return border, mask


class ManageClick():
    def __init__(self, rectangles, fig, ax, arr):
        self.rectangles = rectangles
        self.fig = fig
        self.ax = ax
        self.arr = arr
        self.fuzzy = scipy.ndimage.filters.gaussian_filter(arr, sigma=3)
        self.X = []
        self.Y = []
        self.click_x = None
        self.click_y = None

    def on_release(self, event):
        if event.xdata is None or event.ydata is None or event.xdata != self.click_x or event.ydata != self.click_y:
            return
        if event.button == 3:
            self.remove_rectangle(event)
        else:
            self.add_rectangle(event)

    def on_click(self, event):
        self.click_x = event.xdata
        self.click_y = event.ydata

    def remove_rectangle(self, event):
        remove = []
        for i, rectangle in enumerate(self.rectangles):
            x, y = rectangle.xy
            if x < event.xdata < x + SIZE and y < event.ydata < y + SIZE:
                rectangle.remove()
                remove.append(i)
                plt.draw()
                subset = take_sub_array(arr, y, y + SIZE, x, x + SIZE)
                subset = (subset * 255).astype(np.uint8)
                Xs, Ys = data_augmentation(subset, [0, 1], rotate=True, repeat=4)
                self.X.extend(Xs)
                self.Y.extend(Ys)
        for i, r in enumerate(remove):
            del self.rectangles[r - i]

    def add_rectangle(self, event):
        x = int(event.xdata)
        y = int(event.ydata)
        y, x = CellFromIllustrator.wobble_point(self.fuzzy, y, x)
        rect = patches.Rectangle(xy=(x - RADIUS, y - RADIUS), width=SIZE, height=SIZE,
                                 edgecolor=edgecolor, facecolor='none')
        subset = self.arr[x - RADIUS: x + RADIUS, y - RADIUS: y + RADIUS]
        subset = (subset * 255).astype(np.uint8)
        Xs, Ys = data_augmentation(subset, [1, 0], rotate=True, repeat=4)
        self.X.extend(Xs)
        self.Y.extend(Ys)
        self.ax.add_patch(rect)
        self.rectangles.append(rect)
        plt.draw()


def generate_border_and_mask(mask_file_path):
    if mask_file_path is None:
        return None, None
    else:
        mask = CellFromIllustrator.file_to_array(mask_file_path)
        mask = mask > 128
        border = np.bitwise_and(mask, np.invert(scipy.ndimage.morphology.binary_erosion(mask)))
        return mask, border


if __name__ == "__main__":
    mng = plt.get_current_fig_manager()

    parser = argparse.ArgumentParser(description="This program uses the pretrained NN to search for cells")
    parser.add_argument("-m", "--metadata", type=str, help="Metadata files with images to be analyzed")
    parser.add_argument("-a", "--mask", type=str, help="Mask file for images")
    args = parser.parse_args()

    dir_path = os.path.dirname(args.metadata)
    training_data_X = []
    training_data_Y = []
    mask, border = generate_border_and_mask(args.mask)
    with open(args.metadata) as fi:
        for line in fi:
            file, output_name, model_name = line.split(",")
            file = file.strip()
            output_name = output_name.strip()
            model_name = model_name.strip()
            model = NetworkTrain.load_json_model(model_name)
            if file.endswith(".tif") or file.endswith(".png"):
                arr = CellFromIllustrator.file_to_array(os.path.join(dir_path, file)).astype(np.float32)
                arr = (arr - arr.min()) / (arr.max() - arr.min())

                output_arr = np.stack([arr, arr, arr], axis=2)
                if mask is None:
                    fig, ax = plt.subplots(1)
                    mmc = ManageMaskClick(fig, ax, output_arr.shape[1], output_arr.shape[0])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    fig.tight_layout()
                    fig.canvas.set_window_title("Draw Mask")
                    ax.imshow(output_arr, cmap="gray")
                    cid = fig.canvas.mpl_connect('button_release_event', mmc.on_release)
                    try:
                        mng.frame.Maximize(True)
                    except AttributeError as e:
                        manager = plt.get_current_fig_manager()
                        manager.window.showMaximized()
                    plt.show()
                    border, mask = mmc.get_border_and_mask()

                output_arr[border == 1, 0] = 0
                output_arr[border == 1, 1] = 0
                output_arr[border == 1, 2] = 1  # Make border blue
                fig, ax = plt.subplots(1)
                ax.imshow(output_arr)
                X, COMs = local_maxima_generate_points(arr, mask, find_maxima=True)
                output = model.predict(X)
                prediction = output[:, 0] > 0.5
                rectangles = []
                for i in range(prediction.size):
                    if prediction[i]:
                        edgecolor = (output[i, 0] / 2, 0, 0)
                        x, y = COMs[i]
                        rect = patches.Rectangle(xy=(y - RADIUS, x - RADIUS), width=SIZE, height=SIZE,
                                                 edgecolor=edgecolor, facecolor='none')
                        rectangles.append(rect)
                        ax.add_patch(rect)
                    else:
                        pass
                        """
                        x, y = COMs[i]
                        circ = patches.Circle(xy=(y, x), radius=1, facecolor='b')
                        ax.add_patch(circ)
                        """
                mc = ManageClick(rectangles, fig, ax, arr)
                cid = fig.canvas.mpl_connect('button_press_event', mc.on_click)
                cid2 = fig.canvas.mpl_connect('button_release_event', mc.on_release)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.tight_layout()
                fig.canvas.set_window_title(model_name)
                try:
                    mng.frame.Maximize(True)
                except AttributeError as e:
                    manager = plt.get_current_fig_manager()
                    manager.window.showMaximized()
                plt.show()
                if len(mc.X) > 0:
                    if np.array(mc.X).ndim < 3:
                        print(len(mc.X))
                        print(mc.X)
                    np.savez_compressed(get_next_number(model_name), X=np.array(mc.X), Y=np.array(mc.Y))
                fuzzy = scipy.ndimage.filters.gaussian_filter(arr, sigma=3)
                print("%s had %s cells" % (file, len(rectangles)))
                if output_name.endswith("csv"):
                    with open(os.path.join(dir_path, output_name), 'w') as fo:
                        for rect in rectangles:
                            x, y = rect.xy
                            x, y = CellFromIllustrator.wobble_point(fuzzy=fuzzy, point_x=x + RADIUS, point_y=y + RADIUS, wobble_distance=20)
                            fo.write("%02d,%02d" % (x, y))
                            fo.write(os.linesep)
                elif output_name.endswith("svg"):
                    svg = CreateSVG(os.path.join(dir_path, output_name), arr.shape, file)
                    for rect in rectangles:
                        x, y = rect.xy
                        svg.add_symbol(location_xy=(x + RADIUS, y + RADIUS))
                    svg.output()
