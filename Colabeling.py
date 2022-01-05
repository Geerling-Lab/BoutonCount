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
from NetworkEvaluate import generate_border_and_mask, local_maxima_generate_points, ManageClick, get_next_number

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
            if file.endswith(".tif"):
                arr = CellFromIllustrator.file_to_array(os.path.join(dir_path, file)).astype(np.float32)
                arr = (arr - arr.min()) / (arr.max() - arr.min())

                output_arr = np.stack([arr, arr, arr], axis=2)
                if mask is not None:
                    output_arr[border, 0] = 0
                    output_arr[border, 1] = 0
                    output_arr[border, 2] = 1  # Make border blue
                fig, ax = plt.subplots(1)
                ax.imshow(output_arr, cmap="gray")
                X, COMs = local_maxima_generate_points(arr, mask)
                output = model.predict(X)
                prediction = output[:, 0] > output[:, 1]
                rectangles = []
                for i in range(prediction.size):
                    if prediction[i]:
                        edgecolor = (output[i, 0] / 2, 0, 0)
                        x, y = COMs[i]
                        rect = patches.Rectangle(xy=(y - RADIUS, x - RADIUS), width=SIZE, height=SIZE,
                                                 edgecolor=edgecolor, facecolor='none')
                        rectangles.append(rect)
                        ax.add_patch(rect)
                        #ax.text(y - RADIUS, x - RADIUS, "%.2f" % output[i, 0], color='w')
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
                np.savez_compressed(get_next_number(model_name), X=mc.X, Y=mc.Y)
                fuzzy = scipy.ndimage.filters.gaussian_filter(arr, sigma=3)
                with open(os.path.join(dir_path, output_name), 'w') as fo:
                    for rect in rectangles:
                        x, y = rect.xy
                        x, y = CellFromIllustrator.wobble_point(fuzzy=fuzzy, point_x=x, point_y=y, wobble_distance=20)
                        fo.write("%02d,%02d" % (x + RADIUS, y + RADIUS))
                        fo.write(os.linesep)