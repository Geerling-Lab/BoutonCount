import numpy as np
import scipy.ndimage
import os
import matplotlib.pyplot as plt
import sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import model_from_json
from keras import backend as K
import NetworkTrain
from Constants import SIZE, RADIUS
from GenerateMask import dice


def baseline_model():
    num_classes = 2
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(2, SIZE, SIZE), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


if __name__ == "__main__":
    X = []
    Y = []
    for file in os.listdir("Overlap"):
        if file.endswith("npz"):
            loaded = np.load(os.path.join("Overlap", file))
            X.extend(loaded['X'])
            Y.extend(loaded['Y'])
            os.remove(os.path.join("Overlap", file))
    X = np.stack(X, axis=0)
    Y = np.stack(Y, axis=0)
    np.savez_compressed(os.path.join("Overlap", "00.npz"), X=X, Y=Y)
    X, Y = NetworkTrain.shuffle_in_unison_scary(X, Y)
    MASK_LEVEL = 0.5
    xs = []
    ys = []
    colors = []
    for i in range(X.shape[0]):
        arr1 = X[i, 0, :, :]
        arr2 = X[i, 1, :, :]
        mask1 = arr1 > MASK_LEVEL * arr1.max() + (1 - MASK_LEVEL) * arr1.min()
        mask2 = arr2 > MASK_LEVEL * arr2.max() + (1 - MASK_LEVEL) * arr2.min()
        dice_coefficient = dice(mask1.flatten(), mask2.flatten())
        xs.append(dice_coefficient)
        mutual_information = sklearn.metrics.mutual_info_score(arr1.flatten(), arr2.flatten())
        ys.append(mutual_information)
        if Y[i, 0] > Y[i, 1]:
            colors.append('g')
        else:
            colors.append('r')
    plt.scatter(xs, ys, c=colors)
    plt.show()
    X = (X / 255).astype(np.float32)

    print("%s data samples" % Y.shape[0])
    X = np.moveaxis(X, [1, 2, 3], [-2, -1, -3])
    test_cohort_size = X.shape[0] // 20
    test_X = np.array(X[:test_cohort_size, :, :, :])
    test_Y = np.array(Y[:test_cohort_size, :])
    train_X = np.array(X[test_cohort_size:, :, :, :])
    train_Y = np.array(Y[test_cohort_size:, :])
    model = baseline_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_X, train_Y, validation_data=(test_X, test_Y), epochs=30, batch_size=40)
    scores = model.evaluate(test_X, test_Y, verbose=0)
    print("CNN Error: %.2f%%" % (100-scores[1]*100))
    NetworkTrain.model_to_json(model, os.path.join("Overlap", "Model"))
    for i in range(test_X.shape[0]):
        im1 = test_X[i, 0, :, :]
        im2 = test_X[i, 1, :, :]
        plt.imshow(np.stack([im1, im2, np.zeros(im1.shape)], axis=2))
        predict = model.predict(test_X[i:i+1, :, :, :])[0]
        if predict[0] > predict[1]:
            plt.title("Overlap")
        else:
            plt.title("No")
        plt.show()