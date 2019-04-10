import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms

def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)  # 255也可以改为256

def showImage(pil_image):
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    cv2.imshow("test",cv_image)
    cv2.waitKey(1000)

def showCVImage(img, time=1000):
    cv2.imshow("test", img)
    cv2.waitKey(time)

def showCVImage2(img, name, time=1000):
    cv2.imshow(name, img)
    cv2.waitKey(time)

def showCVImage_WithBox(img, bboxes, time=1000):
    for box in bboxes:
        cv2.rectangle(img, (int(box[0]),int(box[1])), (int(box[2]), int(box[3])),(0,255,0),2)
    showCVImage(img,time)

def showTensorImage(tensor_image):
    pil_image = transforms.ToPILImage()(tensor_image).convert('RGB')
    showImage(pil_image)

def showTensorImage2(tensor_image):
    pil_image = transforms.ToPILImage()(tensor_image).convert('RGB')
    showImage(pil_image)

def showTensorImage_WithCV(tensor_image, time=1000):
    tensor_image = tensor_image.numpy().transpose(1, 2, 0)
    img = tensor_image.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("test", img)
    cv2.waitKey(time)

def showTensorImage_WithCV_Means(tensor_image, time=1000, means = (104, 117, 123)):
    tensor_image = tensor_image.numpy().transpose(1, 2, 0)
    img = cv2.cvtColor(tensor_image, cv2.COLOR_RGB2BGR)
    mean = np.array(means, dtype=np.float32)
    img += mean
    img = img.astype(np.uint8)
    cv2.imshow("test", img)
    cv2.waitKey(time)

def showTensorImage_WithCV_Means_Labels(tensor_image, labels,time=1000, means = (104, 117, 123)):
    tensor_image = tensor_image.numpy().transpose(1, 2, 0)
    img = cv2.cvtColor(tensor_image, cv2.COLOR_RGB2BGR)
    mean = np.array(means, dtype=np.float32)
    img += mean
    img = img.astype(np.uint8)
    for item in labels:
        cv2.rectangle(img, (int(item[0]), int(item[1])), (int(item[2]), int(item[3])), (0, 255, 0), 2)
    cv2.imshow("test", img)
    cv2.waitKey(time)

def showImageWithLabels(pil_image, labels):
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    for i in np.arange(labels.shape[0]):
        target = labels[i]
        height = cv_image.shape[0]
        width = cv_image.shape[1]
        target[0] = target[0] * width
        target[1] = target[1] * height
        target[2] = target[2] * width
        target[3] = target[3] * height
        cv2.rectangle(cv_image, (int(target[0]),int(target[1])),(int(target[2]),int(target[3])),(0,255,0),2)
    cv2.imshow("test", cv_image)
    cv2.waitKey(1000)

def showTensorImageWithLabels(tensor_image, labels):
    pil_image = transforms.ToPILImage()(tensor_image).convert('RGB')
    showImageWithLabels(pil_image, labels)