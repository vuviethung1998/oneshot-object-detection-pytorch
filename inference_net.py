from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
from PIL import Image

import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.utils.blob import im_list_to_blob

import pdb
import gc

def load_image( infilename ):
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='coco', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res50', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="models",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        default=True)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        default=True)
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=128, type=int)
    parser.add_argument('--s', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=10, type=int)
    parser.add_argument('--p', dest='checkpoint',
                        help='checkpoint to load network',
                        default=1663, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--seen', dest='seen',
                        help='Reserved: 1 training, 2 testing, 3 both', default=2, type=int)
    parser.add_argument('--a', dest='average', help='average the top_k candidate samples', default=1, type=int)
    parser.add_argument('--g', dest='group',
                        help='which group want to training/testing',
                        default=0, type=int) 
    args = parser.parse_args()
    return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def _get_image_blob(im, mode):
    """
        Converts an image into a network input.
        Arguments:
            im (ndarray): a color image in BGR order
        Returns:
            blob (ndarray): a data blob holding an image pyramid
            im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """

    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        if mode =='base':
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
                im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)
        elif mode=="query":
            im_scale_factors.append(1)
            processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


if __name__=="__main__":
    args = parse_args()

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    np.random.seed(cfg.RNG_SEED)
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2017_train"
        args.imdbval_name = "coco_2017_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "vg":
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

    args.cfg_file = "cfgs/{}_{}.yml".format(args.net, args.group) if args.group != 0 else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
        'faster_rcnn_{}_{}_{}_{}.pth'.format(args.batch_size, args.checksession, args.checkepoch, args.checkpoint))

    drug_classes = np.asarray([i for  i in range(10)])

      # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(drug_classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(drug_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(drug_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(drug_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    if args.cuda > 0:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    # pdb.set_trace()

    print("load checkpoint %s" % (load_name))

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    query =  torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    catgory = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        cfg.CUDA = True
        fasterRCNN.cuda()
        im_data = im_data.cuda()
        query = query.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    query = Variable(query)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    catgory = Variable(catgory)
    gt_boxes = Variable(gt_boxes)

    start = time.time()
    # visualization
    vis = args.vis
    if vis:
        thresh = 0.05
    else:
        thresh = 0.0
    max_per_image = 100

    fasterRCNN.eval()

    base_img_file = './test1.png'
    query_img_file = './test2.png' # sau nay chuyen query_img sang dang feature vector 

    base_img = load_image(base_img_file)
    query_img = load_image(query_img_file)
    
    with torch.no_grad():
        im_data, im_scale = _get_image_blob(base_img,mode="base")
        # im_data = im_data.transpose(0, 3, 2, 1)
        im_data = torch.from_numpy(im_data)
        im_data = im_data.permute(0, 3, 2, 1).contiguous().cpu().numpy()
        im_data = torch.from_numpy(im_data)
        print(type(im_data))
        # print('im_data: {}'.format(im_data))
        print('im_data shape: {}'.format(im_data.shape))

        query, query_scale = _get_image_blob(query_img,mode="query")
        query = torch.from_numpy(query)
        query = query.permute(0, 3, 2, 1).contiguous().cpu().numpy()
        print('query shape: {}'.format(query.shape))
        
        im_info =  np.array([[im_data.shape[1], im_data.shape[2], im_scale[0]]],dtype=np.float32)
        catgory.resize_(1)
        gt_boxes.resize_((1,3,5)) 

    # Run testing 
    det_tic = time.time()
    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, _, RCNN_loss_bbox, \
    rois_label, weight = fasterRCNN(im_data, query, im_info, gt_boxes, catgory)

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]

    # Apply bounding-box regression
    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4)
            else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))


    # Resize to original ratio
    pred_boxes /= data[2][0][2].item()

    # Remove batch_size dimension
    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()

    # Record time
    det_toc = time.time()
    detect_time = det_toc - det_tic
    misc_tic = time.time()

    # Post processing
    inds = torch.nonzero(scores>thresh).view(-1)
    if inds.numel() > 0:
        # remove useless indices
        cls_scores = scores[inds]
        cls_boxes = pred_boxes[inds, :]
        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)

        # rearrange order
        _, order = torch.sort(cls_scores, 0, True)
        cls_dets = cls_dets[order]

        # NMS
        keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
        cls_dets = cls_dets[keep.view(-1).long()]
    misc_toc = time.time()
    nms_time = misc_toc - misc_tic
    sys.stdout.write('im_detect: {:.3f}s {:.3f}s   \r' \
        .format(detect_time, nms_time))
    sys.stdout.flush()

    print('----------------------------------')
    print('Visualize image')
    im2show = cv2.imread(base_img)
    im2show = vis_detections(im2show, 'shot', cls_dets.cpu().numpy(), 0.7)

    o_query = cv2.imread(query_img)

    (h,w,c) = im2show.shape
    o_query = cv2.resize(o_query, (h, h),interpolation=cv2.INTER_LINEAR)
    im2show = np.concatenate((im2show, o_query), axis=1)

    output_path = './'
    cv2.imwrite(output_path + 'predict.png', im2show)

