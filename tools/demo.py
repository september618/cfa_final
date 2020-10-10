#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import itertools

#CLASSES = ('__background__',
#           'aeroplane', 'bicycle', 'bird', 'boat',
#           'bottle', 'bus', 'car', 'cat', 'chair',
#           'cow', 'diningtable', 'dog', 'horse',
#           'motorbike', 'person', 'pottedplant',
#           'sheep', 'sofa', 'train', 'tvmonitor')
""" 
CLASSES = ('__background__', # always index 0
           'day_2_yes', 'day_2_no', 'day_3_yes', 'day_3_no',
           'night_2_yes', 'night_2_no', 'night_3_yes', 'night_3_no')
""" 

CLASSES = ('__background__', 
           'day_no', 'day_break', 'day_left', 'day_right',
           'night_no', 'night_break', 'night_left', 'night_right')
  
    
NETS = {'vgg_vnn_m_1024': ('VGG_CNN_M_1024',
                  'vgg_cnn_m_1024_faster_rcnn_iter_5000.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
        'vgg16':('VGG16',
                 'ZSA3.caffemodel')}



def vis_detections(im, class_name, dets, attention, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]


    im = im[:, :, (2, 1, 0)]


    fig, ax = plt.subplots(figsize=(12, 12))
    for i in inds:
        im2 = np.zeros_like(im)
        bbox = dets[i, :4]
        #feature1 = feature[i]
        score = dets[i, -1]
        # activation map
        am = attention[i]

        am =am.reshape(14,14)
        am = cv2.resize(am, (bbox[2] - bbox[0], bbox[3] - bbox[1]))
        am = 255 * (am - np.max(am)) / (np.max(am) - np.min(am) + 1e-12)
        am = np.uint8(np.floor(am))
        am = cv2.applyColorMap(am, cv2.COLORMAP_HOT)
        
        im2[int(bbox[1]):int(bbox[1] + am.shape[0]),int(bbox[0]):int(bbox[0]+am.shape[1]) , :] = am

        im = cv2.addWeighted(im, 0.5, im2, 0.5, 0)

        # overlapped
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
    ax.imshow(im, aspect='equal')

    ax.set_title("Visualization of our attention mechanism")
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo_zsa', image_name)
    im = cv2.imread(im_file)


    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes,attention = im_detect(net, im)
    timer.toc()
  

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)

        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #print(cls_scores.shape,feature.shape,keep)
        #feature1=feature[keep,:,:,:]
        vis_detections(im, cls, dets, attention, thresh=CONF_THRESH)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_end2end', 'test.prototxt')
    caffemodel = os.path.join('output', 'faster_rcnn_end2end','voc_2007_trainval',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print('\n\nLoaded network {:s}'.format(caffemodel))

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in range(2):
        _, _, _= im_detect(net, im)

    im_names = os.listdir("/home/ubuntu/user_space/maga_faster/our_method/data/demo_zsa/")
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', im_name)
        demo(net, im_name)
       # plt.savefig('./vis/'+im_name)
        plt.show()


