from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
# from data import VOC_ROOT, VOC_CLASSES as labelmap
from PIL import Image
# from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from waiguan.data.dataset_waiguan import WAIGUAN_ROOT, VOCAnnotationTransform, VOCDetection_Waiguan, BaseTransform
from waiguan.data.dataset_waiguan import WAIGUAN_CLASSES as labelmap
import torch.utils.data as data
from waiguan.utils.show import *
from waiguan.ssd import build_ssd
import shutil

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='../weights/step2.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='/data_1/data/temp/waiguan/pytorch_results', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=WAIGUAN_ROOT, help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def clean_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    else:
        os.mkdir(folder)

def test_net_lines(save_folder, net, cuda, testset, transform, thresh):
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i + 1, num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        if cuda:
            x = x.cuda()
        y = net(x)  # forward pass
        detections = y.data
        target_file = "{}/{}.txt".format(save_folder, img_id)
        target_lines = list()
        pred_num = 0
        show_box = list()
        scale = torch.Tensor([300, 300, 300,300])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.1:
                score = detections[0, i, j, 0]
                label_name = labelmap[i - 1]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                show_box.append(coords)
                pred_num += 1
                target_line = "{} {} {} {} {} {}".format(label_name,score, int(pt[0]), int(pt[1]), int(pt[2]), int(pt[3]))
                #print(target_line)
                j += 1
                target_lines.append(target_line)

        with open(target_file,'w') as f:
           for item in target_lines:
               f.write("%s\n" % item)

def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'test1.txt'
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        with open(filename, mode='a') as f:
            f.write('\nGROUND TRUTH FOR: '+img_id+'\n')
            for box in annotation:
                f.write('label: '+' || '.join(str(b) for b in box)+'\n')
        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        show_box = list()
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.3:
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('PREDICTIONS: '+'\n')
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                show_box.append(coords)
                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(str(pred_num)+' label: '+label_name+' score: ' +
                            str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                j += 1
        showCVImage_WithBox(img,show_box, time=3000)

def test_voc():
    # load net
    num_classes = len(labelmap) + 1 # +1 background
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = VOCDetection_Waiguan(args.voc_root, 'test', None, VOCAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    #clean_folder(args.save_folder)
    test_net_lines(args.save_folder, net, args.cuda, testset,
        BaseTransform(net.size, (104, 117, 123)),
        thresh=args.visual_threshold)

if __name__ == '__main__':
    test_voc()
