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
# from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.utils.blob import im_list_to_blob, prep_im_for_blob

# from scipy.misc import imread
from imageio import imread

import pdb
import gc

def load_image( infilename ):
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def _get_image_blob(im_np, mode):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    processed_ims = []
    im_scales = []
    # im = np.array(imread(im))

    if len(im_np.shape) == 2:
        im_np = im_np[:,:,np.newaxis]
        im_np = np.concatenate((im_np,im_np,im_np), axis=2)
    # flip the channel, since the original one using cv2
    # rgb -> bgr
    im_np = im_np[:,:,::-1]

    for target_size in cfg.TEST.SCALES:
        if mode =='base':
            im, im_scale = prep_im_for_blob(im_np, cfg.PIXEL_MEANS, target_size,
                            cfg.TRAIN.MAX_SIZE)
            im_scales.append(im_scale)
            processed_ims.append(im)
        elif mode=="query":
            im, im_scale = prep_im_for_blob(im_np, cfg.PIXEL_MEANS, cfg.TRAIN.query_size,
                            cfg.TRAIN.MAX_SIZE)
            im_scales.append(im_scale)
            processed_ims.append(im)
    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    return blob, im_scales

def vis_detections(im, im_scale, class_name, dets, thresh=0.5):
    '''
        Rescale detected images then return the dectections bounding box after rescaling
        im: image after performing detection 
        original_im: original image 
        scale_factor: scale factor after scaling to 800-600
        class_name: name of class
        dets: results after performing detection
        thresh: confidence score 
    '''
    ''' CV2 rectangle 
        cv2.rectangle(img, (x1, y1), (x2, y2), color in BGR (255,0,0), thickness: 2)
        x1,y1 ------
        |          |
        --------x2,y2
    '''
    '''
        Step 1: rescale to original image size 
        Step 2: recalcualte the bbox coordination 
        Step 3: return original image and recalculated bbox 
    '''

    bboxs = []
    # rescale image to the original size 
    # print(im_scale)
    im_base = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    # print('{}, {}'.format(im_base.shape[0], im_base.shape[1]))
    
    # get rescaled image shape 
    if len(im.shape) == 3:
        im_height, im_width ,_ = im.shape
        # print(im_width)
        # print(im_height)
    else:
        im = im.squeeze()        

    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        bbox = np.asarray(bbox).reshape(1,-1)

        score = dets[i, -1]
        if score > thresh: # confidence threshold
            output_size = (im.shape[0], im.shape[1])
            print('height: {}'.format(im.shape[0]))
            print('width: {}'.format(im.shape[1]))
            # bbox = scale_coords(im_base, bbox, output_size)
            bbox = bbox.astype('int32').squeeze()
            print('bbox: {}'.format(bbox))
            bboxs.append(tuple(bbox))
            
            # draw bbox 
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 110, 255), 1)

            # text = '%.3f' % (score)
            # (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.2, thickness=2)[0]

            # cv2.rectangle(im, (bbox[0], bbox[1] ), (bbox[0]+text_width, bbox[1] + text_height), (0, 255, 251), -1)

            # cv2.putText(im,text , (bbox[0], bbox[1]+text_height), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 0, 0), thickness=2)
    # return im, bboxs
    return im 

def scale_coords(im, bbox, output_size, thresh=0.5):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        im: output image 
        bbox: bbox location in 2d np array form of output 
        output_size: the desired (height, width) resolution in tuple.
    Returns:
        Instances: the resized output from the model, based on the output resolution
    """

    # Converts integer tensors to float temporaries
    #   to ensure true division is performed when
    #   computing scale_x and scale_y.
    def scale(bbox, scale_x: float, scale_y: float) -> None:
        """
        Scale the box with horizontal and vertical scaling factors
        """
        bbox = bbox.astype('float32')
        bbox[:, 0::2] *= scale_x
        bbox[:, 1::2] *= scale_y
        return bbox 
    
    def clip(bbox, height_bbox, width_bbox) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        Args:
            box_size (height, width): The clipping box's size.
        """
        h, w = height_bbox, width_bbox
        bbox[:, [0, 2]] = bbox[:, [0, 2]].clip(0, w)  # x1, x2
        bbox[:, [1, 3]] = bbox[:, [1, 3]].clip(0, h)
        return bbox 
    

    scale_x, scale_y = (
        output_size[1] / im.shape[1],
        output_size[0] / im.shape[0],
    )
    print('scale_x: {}'.format(scale_x))
    print('scale_y: {}'.format(scale_y))

    height_bbox, width_bbox =  max(0, bbox[:,3] - bbox[:,1]), max(0, bbox[:,2] - bbox[:,0])
    print('height_bbox: {}'.format(height_bbox))
    print('width_bbox: {}'.format(width_bbox))

    print('bbox 1: {}'.format(bbox.squeeze()))
    bbox = scale(bbox, scale_x, scale_y)
    print('bbox 2: {}'.format(bbox.squeeze()))
    bbox = clip(bbox,height_bbox, width_bbox)
    print('bbox 3: {}'.format(bbox))
    return bbox

def init_env(model_path='faster_rcnn_16_1_29_798.pth'):
    model_path = model_path
    np.random.seed(cfg.RNG_SEED)

    cfg_file = "cfgs/res50.yml"
    set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

    cfg_from_file(cfg_file)
    cfg_from_list(set_cfgs)
    cfg.USE_GPU_NMS = True
    np.random.seed(cfg.RNG_SEED)

    input_dir = 'models/res50/coco'
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir, model_path)

    drug_classes = np.zeros([1], dtype = int)

    # initilize the network here.
    fasterRCNN = resnet(drug_classes, 50, pretrained=False, class_agnostic=True)

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')
    print("load checkpoint %s" % (load_name)) 
    fasterRCNN.eval()
    return fasterRCNN

def get_single_oneshot_predict(base_np, im_np, label):
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    query =  torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    catgory = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

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
    vis = True
    if vis:
        thresh = 0.05
    else:
        thresh = 0.0
    max_per_image = 100

    with torch.no_grad():
        im_blob, im_scales = _get_image_blob(base_np, mode="base")
        query_blob, _ = _get_image_blob(im_np, mode="query")

        im_info_np =  np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],dtype=np.float32)
        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        query_pt = torch.from_numpy(query_blob)
        query_pt = query_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
        query.resize_(query_pt.size()).copy_(query_pt)
        im_info.resize_(im_info_pt.size()).copy_(im_info_pt)


        catgory.resize_(1).zero_()
        gt_boxes.resize_((1,3,5)).zero_() 

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
            # if args.class_agnostic:
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            box_deltas = box_deltas.view(1, -1, 4)
            # else:
            #     box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
            #             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            #     box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    # Resize to original ratio
    pred_boxes /= im_info[0][2].item()

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

    print('time_detect: {}'.format(detect_time))

    if inds.numel() > 0:
        print('----------------------------------')
        print('Visualize image')
        im2show = cv2.imread(base_img_file) # original image 
        im2show = vis_detections( im2show, im_scales[0], 'shot', cls_dets.cpu().numpy(), 0.7)

        o_query = cv2.imread(query_img_file)

        (h,w,c) = im2show.shape
        print(h)
        o_query = cv2.resize(o_query, (h, h),interpolation=cv2.INTER_LINEAR)
        im2show = np.concatenate((im2show, o_query), axis=1)

        output_path = './'
        cv2.imwrite(output_path + 'predict_{}.png'.format(label), im2show)
    else:
        print("Khong tra ve detection")

def get_multiple_oneshot_predicts(im, lst_queries):
    pass 

if __name__=="__main__":
    base_img_file = './test1.png'
    query_img_file = './test2.png' # sau nay chuyen query_img sang dang feature vector 

    base_np = np.array(imread(base_img_file))
    query_np = np.array(imread(query_img_file))
    label = '' 
    fasterRCNN = init_env()
    get_single_oneshot_predict(base_np, query_np, '1')    
    get_single_oneshot_predict(base_np, query_np, '2')    
    get_single_oneshot_predict(base_np, query_np, '3')    
