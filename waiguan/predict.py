from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
#from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
#from data import VOC_CLASSES as labelmap
from waiguan.data.dataset_waiguan import WAIGUAN_ROOT, VOCAnnotationTransform, VOCDetection_Waiguan, BaseTransform
from waiguan.data.dataset_waiguan import WAIGUAN_CLASSES as labelmap
import torch.utils.data as data
from waiguan.utils.show import *

from waiguan.ssd import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default='weights/step2.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=WAIGUAN_ROOT,
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')

args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

dataset_mean = (104, 117, 123)
set_type = 'test'

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05):
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    chebiao_num =0
    for i in range(num_images):
        print("index: {}/{}".format(i,num_images))
        im, gt, h, w = dataset.pull_item(i)

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        scale = torch.Tensor([im.shape[2], im.shape[1],im.shape[2],im.shape[1]])
        show_box = list()
        # skip j = 0, because it's the background class
        for i in range(detections.size(1)):
            j = 0
            while detections[0,i,j,0] >= 0.3:
                score = detections[0,i,j,0]
                label_name = labelmap[i-1]
                pt = (detections[0,i,j,1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                show_box.append(coords)
                j += 1
                chebiao_num += 1

        # for j in range(1, detections.size(1)):
        #     dets = detections[0, j, :]
        #     mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
        #     dets = torch.masked_select(dets, mask).view(-1, 5)
        #     if dets.size(0) == 0:
        #         continue
        #     boxes = dets[:, 1:]
        #     boxes[:, 0] *= w
        #     boxes[:, 2] *= w
        #     boxes[:, 1] *= h
        #     boxes[:, 3] *= h
        #     scores = dets[:, 0].cpu().numpy()
        #     cls_dets = np.hstack((boxes.cpu().numpy(),
        #                           scores[:, np.newaxis])).astype(np.float32,
        #                                                          copy=False)
        #     all_boxes[j][i] = cls_dets

        # print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,num_images, detect_time))
        showTensorImage_WithCV_Means_Labels(im, show_box, time=1000)
        # showTensorImage_WithCV_Means(im)
    print("chebiao： {}/{}".format(chebiao_num,num_images))


if __name__ == '__main__':
    # load net
    num_classes = len(labelmap) + 1                      # +1 for background
    net = build_ssd('test', 300, num_classes)            # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    dataset = VOCDetection_Waiguan(args.voc_root, image_sets='test',transform =BaseTransform(300, dataset_mean),target_transform=VOCAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, dataset,
             BaseTransform(net.size, dataset_mean), args.top_k, 300,
             thresh=args.confidence_threshold)
