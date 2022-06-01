from multiprocessing import Process, Manager, cpu_count
import numpy as np
from imutils import paths
import math
import cv2
from omegaconf import DictConfig


def make_batches(config: DictConfig):
    all_images = list(paths.list_images(config.data_dir))
    batch_size = int(np.ceil(len(all_images) / float(cpu_count())))

    m = len(all_images)
    mini_batches = []

    # perm = list(np.random.permutation(m))
    # shuffled_data = all_images[perm]

    complete_mini_batches = math.floor(m / batch_size)
    for k in range(0, complete_mini_batches):
        mini_b = all_images[k * batch_size:(k + 1) * batch_size]
        mini_batches.append(mini_b)

    if m % batch_size != 0:
        mini_b = all_images[int(m / batch_size) * batch_size:]
        mini_batches.append(mini_b)

    return mini_batches

def loader(batches):
    data = []
    labels = []
    for image_path in batches:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        label = image_path.split('\\')[-2]
        data.append(image)
        labels.append(label)
    return data, labels

