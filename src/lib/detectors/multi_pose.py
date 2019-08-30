from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

from external.nms import soft_nms_39
from models.decode import multi_pose_decode, multi_pose_decode_c, multi_pose_decode_c1
from models.utils import flip_tensor, flip_lr_off, flip_lr
from utils.image import get_affine_transform
from utils.post_process import multi_pose_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector

import copy

class MultiPoseDetector(BaseDetector):
  def __init__(self, opt):
    super(MultiPoseDetector, self).__init__(opt)
    self.flip_idx = opt.flip_idx
    if self.opt.zjb:
        self.last_dets = [0]
        self.non_person = 0

  def process(self, images, return_time=False):
    with torch.no_grad():
      torch.cuda.synchronize()
      if self.opt.zjb:
        all_out = self.model(images)
        output = all_out[-1]
        #gt
        output['hm_hp'] = self.gt[0]['hm_hp'].cuda()
        if self.opt.hm_hp and not self.opt.mse_loss:
          output['hm_hp'] = output['hm_hp'].sigmoid_()

        hm_hp = output['hm_hp'] if self.opt.hm_hp else None
        hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
        torch.cuda.synchronize()
        forward_time = time.time()

        #T-param

        
        # if self.opt.flip_test:
        #   output['hm'] = (output['hm'][0:1] + flip_tensor(output['hm'][1:2])) / 2
        #   output['wh'] = (output['wh'][0:1] + flip_tensor(output['wh'][1:2])) / 2
        #   output['hps'] = (output['hps'][0:1] +
        #     flip_lr_off(output['hps'][1:2], self.flip_idx)) / 2
        #   hm_hp = (hm_hp[0:1] + flip_lr(hm_hp[1:2], self.flip_idx)) / 2 \
        #           if hm_hp is not None else None
        #   reg = reg[0:1] if reg is not None else None
        #   hp_offset = hp_offset[0:1] if hp_offset is not None else None
        
        dets, center = multi_pose_decode_c(output, output['hps'], hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)
        dets   = dets.data.cpu().numpy().reshape(2, -1, 8)
        center = center.data.cpu().numpy().reshape(2, -1, 4)
        dets   = dets.reshape(1, -1, 8)
        center   = center.reshape(1, -1, 4)
        detections = dets
        center_points = center
        classes    = detections[..., -1]
        classes    = classes[0]
        detections = detections[0]
        center_points = center_points[0]
        
        valid_ind = detections[:,4]> -1
        valid_detections = detections[valid_ind]
        
        box_width = valid_detections[:,2] - valid_detections[:,0]
        box_height = valid_detections[:,3] - valid_detections[:,1]
        
        s_ind = (box_width*box_height <= 22500)
        l_ind = (box_width*box_height > 22500)
        
        s_detections = valid_detections[s_ind]
        l_detections = valid_detections[l_ind]
        
        s_left_x = (2*s_detections[:,0] + s_detections[:,2])/3
        s_right_x = (s_detections[:,0] + 2*s_detections[:,2])/3
        s_top_y = (2*s_detections[:,1] + s_detections[:,3])/3
        s_bottom_y = (s_detections[:,1]+2*s_detections[:,3])/3
        
        s_temp_score = copy.copy(s_detections[:,4])
        s_detections[:,4] = -1
        
        center_x = center_points[:,0][:, np.newaxis]
        center_y = center_points[:,1][:, np.newaxis]
        s_left_x = s_left_x[np.newaxis, :]
        s_right_x = s_right_x[np.newaxis, :]
        s_top_y = s_top_y[np.newaxis, :]
        s_bottom_y = s_bottom_y[np.newaxis, :]
        
        ind_lx = (center_x - s_left_x) > 0
        ind_rx = (center_x - s_right_x) < 0
        ind_ty = (center_y - s_top_y) > 0
        ind_by = (center_y - s_bottom_y) < 0
        ind_cls = (center_points[:,2][:, np.newaxis] - s_detections[:,-1][np.newaxis, :]) == 0
        ind_s_new_score = np.max(((ind_lx+0) & (ind_rx+0) & (ind_ty+0) & (ind_by+0) & (ind_cls+0)), axis = 0) == 1
        index_s_new_score = np.argmax(((ind_lx+0) & (ind_rx+0) & (ind_ty+0) & (ind_by+0) & (ind_cls+0))[:,ind_s_new_score], axis = 0)
        s_detections[:,4][ind_s_new_score] = (s_temp_score[ind_s_new_score]*2 + center_points[index_s_new_score,3])/3
       
        l_left_x = (3*l_detections[:,0] + 2*l_detections[:,2])/5
        l_right_x = (2*l_detections[:,0] + 3*l_detections[:,2])/5
        l_top_y = (3*l_detections[:,1] + 2*l_detections[:,3])/5
        l_bottom_y = (2*l_detections[:,1]+3*l_detections[:,3])/5
        
        l_temp_score = copy.copy(l_detections[:,4])
        l_detections[:,4] = -1
        
        center_x = center_points[:,0][:, np.newaxis]
        center_y = center_points[:,1][:, np.newaxis]
        l_left_x = l_left_x[np.newaxis, :]
        l_right_x = l_right_x[np.newaxis, :]
        l_top_y = l_top_y[np.newaxis, :]
        l_bottom_y = l_bottom_y[np.newaxis, :]
        
        ind_lx = (center_x - l_left_x) > 0
        ind_rx = (center_x - l_right_x) < 0
        ind_ty = (center_y - l_top_y) > 0
        ind_by = (center_y - l_bottom_y) < 0
        ind_cls = (center_points[:,2][:, np.newaxis] - l_detections[:,-1][np.newaxis, :]) == 0
        ind_l_new_score = np.max(((ind_lx+0) & (ind_rx+0) & (ind_ty+0) & (ind_by+0) & (ind_cls+0)), axis = 0) == 1
        index_l_new_score = np.argmax(((ind_lx+0) & (ind_rx+0) & (ind_ty+0) & (ind_by+0) & (ind_cls+0))[:,ind_l_new_score], axis = 0)
        l_detections[:,4][ind_l_new_score] = (l_temp_score[ind_l_new_score]*2 + center_points[index_l_new_score,3])/3
        
        detections = np.concatenate([l_detections,s_detections],axis = 0)
        detections = detections[np.argsort(-detections[:,4])] 
        classes   = detections[..., -1]

        #nms
        keep_inds  = (detections[:, 4] > 0)
        #keep_inds  = (detections[:, 4] > )
        detections = detections[keep_inds]
        classes = classes[keep_inds]
        nms_inds = py_nms(detections[:,0:5], 0.5)
        detections = detections[nms_inds]
        classes = classes[nms_inds]

        #gt
        # det_gt = self.gt[0]['gt_det']
        # detections = np.ones((len(det_gt), 8))
        # for i in range(len(det_gt)):
        #   detections[i][:4] = det_gt[i].numpy()
        #   detections[i][7] = 0
        #
        # if detections.shape[0] == 0:
        #   self.non_person += 1
        #   print("nonperson"+str(self.non_person))
        #   self.last_dets[0][0, :, 4] = 0
        #   self.last_dets[0] = np.zeros(shape=(1, 1, 6+34))
        #   if return_time:
        #     return output, self.last_dets[0], forward_time
        #   else:
        #     return output, self.last_dets[0]


        kps = multi_pose_decode_c1(output, detections, output['hps'], hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K).data.cpu().numpy()[0]


        num_j = kps.shape[1]
        dets = np.zeros(shape=(1, detections.shape[0], 6+num_j))
        dets[0, :, 0:5]=detections[:,0:5]
        dets[0, :, 5:5+num_j] = kps
        dets[0, :, -1] = detections[:, -1]

        # top_bboxes[image_id] = {}
        # for j in range(categories):
        #     keep_inds = (classes == j)
        #     top_bboxes[image_id][j + 1] = detections[keep_inds][:, 0:7].astype(np.float32)
        #     if merge_bbox:
        #         soft_nms_merge(top_bboxes[image_id][j + 1], Nt=nms_threshold, method=nms_algorithm, weight_exp=weight_exp)
        #     else:
        #         soft_nms(top_bboxes[image_id][j + 1], Nt=nms_threshold, method=nms_algorithm)
        #     top_bboxes[image_id][j + 1] = top_bboxes[image_id][j + 1][:, 0:5]
        #
        # scores = np.hstack([
        #     top_bboxes[image_id][j][:, -1]
        #     for j in range(1, categories + 1)
        # ])
        # if len(scores) > max_per_image:
        #     kth    = len(scores) - max_per_image
        #     thresh = np.partition(scores, kth)[kth]
        #     for j in range(1, categories + 1):
        #         keep_inds = (top_bboxes[image_id][j][:, -1] >= thresh)
        #         top_bboxes[image_id][j] = top_bboxes[image_id][j][keep_inds]



        self.last_dets[0]=dets
        if return_time:
          return output, dets, forward_time
        else:
          return output, dets

          
      else:
        output = self.model(images)[-1]
        output['hm'] = output['hm'].sigmoid_()
        if self.opt.hm_hp and not self.opt.mse_loss:
          output['hm_hp'] = output['hm_hp'].sigmoid_()

        reg = output['reg'] if self.opt.reg_offset else None
        hm_hp = output['hm_hp'] if self.opt.hm_hp else None
        hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
        torch.cuda.synchronize()
        forward_time = time.time()
        
        if self.opt.flip_test:
          output['hm'] = (output['hm'][0:1] + flip_tensor(output['hm'][1:2])) / 2
          output['wh'] = (output['wh'][0:1] + flip_tensor(output['wh'][1:2])) / 2
          output['hps'] = (output['hps'][0:1] + 
            flip_lr_off(output['hps'][1:2], self.flip_idx)) / 2
          hm_hp = (hm_hp[0:1] + flip_lr(hm_hp[1:2], self.flip_idx)) / 2 \
                  if hm_hp is not None else None
          reg = reg[0:1] if reg is not None else None
          hp_offset = hp_offset[0:1] if hp_offset is not None else None
        
        dets = multi_pose_decode(
          output['hm'], output['wh'], output['hps'],
          reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)

        if return_time:
          return output, dets, forward_time
        else:
          return output, dets

  def post_process(self, dets, meta, scale=1):
    if self.opt.zjb:
      dets=dets.reshape(1, -1, dets.shape[2])
    else:
      dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets = multi_pose_post_process(
      dets.copy(), [meta['c']], [meta['s']],
      meta['out_height'], meta['out_width'])
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 39)
      # import pdb; pdb.set_trace()
      dets[0][j][:, :4] /= scale
      dets[0][j][:, 5:] /= scale
    return dets[0]

  def merge_outputs(self, detections):
    results = {}
    results[1] = np.concatenate(
        [detection[1] for detection in detections], axis=0).astype(np.float32)
    if self.opt.nms or len(self.opt.test_scales) > 1:
      soft_nms_39(results[1], Nt=0.5, method=2)
    results[1] = results[1].tolist()
    return results

  def debug(self, debugger, images, dets, output, scale=1):
    dets = dets.detach().cpu().numpy().copy()
    dets[:, :, :4] *= self.opt.down_ratio
    dets[:, :, 5:39] *= self.opt.down_ratio
    img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(((
      img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
    pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
    debugger.add_blend_img(img, pred, 'pred_hm')
    if self.opt.hm_hp:
      pred = debugger.gen_colormap_hp(
        output['hm_hp'][0].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hmhp')
  
  def show_results(self, debugger, image, results):
    debugger.add_img(image, img_id='multi_pose')
    for bbox in results[1]:
      if bbox[4] > self.opt.vis_thresh:
        debugger.add_coco_bbox(bbox[:4], 0, bbox[4], img_id='multi_pose')
        debugger.add_coco_hp(bbox[5:39], img_id='multi_pose')
    debugger.show_all_imgs(pause=self.pause)



def py_nms(dets, thresh):
    """Pure Python NMS baseline."""
    #x1、y1、x2、y2、以及score赋值
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    #每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #order是按照score降序排序的
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        #计算当前概率最大矩形框与其他矩形框的相交框的坐标，会用到numpy的broadcast机制，得到的是向量
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        #计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        #找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= thresh)[0]
        #将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    return keep