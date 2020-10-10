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
                 'vgg16_faster_rcnn_iter_7000.caffemodel')}

"""our vis_detections """
"""
def vis_detections(im, class_name, dets,feature, thresh=0.5):
    """"""Draw detected bounding boxes.""""""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    im2 = np.zeros_like(im)

    fig, ax = plt.subplots(figsize=(12, 12))
    for i in inds:
        bbox = dets[i, :4]
        attention = dets[i, 4:-1]
        feature1 = feature[i]
        score = dets[i, -1]
        #att = np.clip(attention, 0, 1)
        interval_y = (bbox[2] - bbox[0]) / 7.0
        interval_x = (bbox[3] - bbox[1]) / 7.0
        attention=np.array(attention)
        attention=np.argsort(attention,axis=0)

        att=feature1[attention[-10:]]
        att=att.reshape((-1,49))
        peak=np.argmax(att,axis=1)
        x_peak=np.array(peak/7)
        y_peak=np.array(peak%7)
        x=bbox[1]+interval_x*x_peak
        y=bbox[0]+interval_y*y_peak
        x+=interval_x/2
        y+=interval_y/2
        x=np.trunc(x)
        y=np.trunc(y)

        points=np.array([[x[i],y[i]] for i in range(len(x))])
        #print(points)
        k = 4
        centroids = np.zeros((k, 2))
        for i in range(k):
            index = int(np.random.uniform(0, att.shape[0]))
            centroids[i, :] = points[index]
        clusterAssment = np.mat(np.zeros((att.shape[0], 2)))
        clusterChange = True

        while clusterChange:
            clusterChange = False

            for i in range(att.shape[0]):
                minDist = 100000.0
                minIndex = -1

                for j in range(k):
                    distance =np.sqrt(np.sum((centroids[j, :]- points[i])**2))
                    if distance < minDist:
                        minDist = distance
                        minIndex = j
                if clusterAssment[i, 0] != minIndex:
                    clusterChange = True
                    clusterAssment[i, :] = minIndex, minDist ** 2
            for j in range(k):
                pointsInCluster = points[np.nonzero(clusterAssment[:, 0].A == j)[0]]
                #print(pointsInCluster)
                print(centroids)
                centroids[j, :] = np.mean(pointsInCluster, axis=0)

        #print(centroids)
        for i in range(centroids.shape[0]):
            if not bbox[1]<centroids[i,0]<bbox[3]:
                continue
            if not bbox[0]<centroids[i,1]<bbox[2]:
                continue
            im2[centroids[i,0],centroids[i,1],:]=np.array([255,255,255],dtype=np.uint8)
        im = cv2.addWeighted(im, 0.5, im2, 0.5, 0)
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

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

"""

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

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

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    #scores, boxes,attention, feature = im_detect(net, im)
    scores, boxes= im_detect(net, im)
    timer.toc()
    print (('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.75
    NMS_THRESH = 0.15
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        #dets = np.hstack((cls_boxes,attention,
        #                  cls_scores[:, np.newaxis])).astype(np.float32)

        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)

        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #print(cls_scores.shape,feature.shape,keep)

        #feature1=feature[keep,:,:,:]
        #vis_detections(im, cls, dets,feature1, thresh=CONF_THRESH)
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

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

    print ('\n\nLoaded network {:s}'.format(caffemodel))

    #Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in range(2):
        _, _= im_detect(net, im)

    filepath1='/home/ubuntu/user_space/maga_faster/our_method/data/demo/'
    #im_names = ['0035.jpg','000456.jpg', '000542.jpg', '001150.jpg', '001763.jpg', '004545.jpg']
    im_names = os.listdir(filepath1)
    for im_name in im_names:
        print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print ('Demo for data/demo/{}'.format(im_name))
        demo(net, im_name)

        plt.savefig(im_name[:-4] + '_test.jpg')


    #plt.show()


