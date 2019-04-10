import torch
import cv2
import numpy as np
from waiguan.utils.show import *

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def base_transform(image, size, mean):
    # showCVImage2(image,"origin",0)
    x = cv2.resize(image, (size, size)).astype(np.float32)
    #showCVImage2(x,0,"test2")
    x -= mean
    x = x.astype(np.float32)
    #showCVImage2(x,0,"test3")
    # x1 = x + mean
    # x1 = x1.astype(np.uint8)
    # showCVImage2(x1, "x1", 0)
    return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels