from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

from models.model import create_model, load_model
from utils.image import get_affine_transform
from utils.debugger import Debugger

#gt
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math

class BaseDetector(object):
  def __init__(self, opt):
    if opt.gpus[0] >= 0:
      opt.device = torch.device('cuda')
    else:
      opt.device = torch.device('cpu')
    
    print('Creating model...')
    self.model = create_model(opt.arch, opt.heads, opt.head_conv)
    self.model = load_model(self.model, opt.load_model)
    self.model = self.model.to(opt.device)
    self.model.eval()

    self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
    self.max_per_image = 100
    self.num_classes = opt.num_classes
    self.scales = opt.test_scales
    self.opt = opt
    self.pause = True
    # self.max_objs = 100
    # self.gt = [0]

  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def pre_process(self, image, scale, meta=None):
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width  = int(width * scale)
    if self.opt.fix_res:
      inp_height, inp_width = self.opt.input_h, self.opt.input_w
      c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
      s = max(height, width) * 1.0
    else:
      inp_height = (new_height | self.opt.pad) + 1
      inp_width = (new_width | self.opt.pad) + 1
      c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
      s = np.array([inp_width, inp_height], dtype=np.float32)

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    if self.opt.flip_test:
      images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
    images = torch.from_numpy(images)
    meta = {'c': c, 's': s, 
            'out_height': inp_height // self.opt.down_ratio, 
            'out_width': inp_width // self.opt.down_ratio}
    return images, meta
    #gt
    # inp_height_h = int(inp_width/4)
    # inp_width_h = int(inp_height / 4)
    # output_res = self.opt.output_res
    # num_joints = 17
    # num_objs = min(len(anns), self.max_objs)
    # rot = 0
    # trans_output_rot = get_affine_transform(c, s, rot, [inp_height_h, inp_width_h])
    # trans_output = get_affine_transform(c, s, 0, [inp_height_h, inp_width_h])
    #
    # hm = np.zeros((self.num_classes, inp_height_h, inp_width_h), dtype=np.float32)
    # hm_hp = np.zeros((num_joints, inp_height_h, inp_width_h), dtype=np.float32)
    # dense_kps = np.zeros((num_joints, 2, inp_height_h, inp_width_h),
    #                      dtype=np.float32)
    # dense_kps_mask = np.zeros((num_joints, inp_height_h, inp_width_h),
    #                           dtype=np.float32)
    # wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    # kps = np.zeros((self.max_objs, num_joints * 2), dtype=np.float32)
    # reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    # ind = np.zeros((self.max_objs), dtype=np.int64)
    # reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    # kps_mask = np.zeros((self.max_objs, num_joints * 2), dtype=np.uint8)
    # hp_offset = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32)
    # hp_ind = np.zeros((self.max_objs * num_joints), dtype=np.int64)
    # hp_mask = np.zeros((self.max_objs * num_joints), dtype=np.int64)
    #
    # tl_heatmaps = np.zeros((self.num_classes, inp_height_h, inp_width_h), dtype=np.float32)
    # br_heatmaps = np.zeros((self.num_classes, inp_height_h, inp_width_h), dtype=np.float32)
    # ct_heatmaps = np.zeros((self.num_classes, inp_height_h, inp_width_h), dtype=np.float32)
    # tl_regrs = np.zeros((self.max_objs, 2), dtype=np.float32)
    # br_regrs = np.zeros((self.max_objs, 2), dtype=np.float32)
    # ct_regrs = np.zeros((self.max_objs, 2), dtype=np.float32)
    # tl_tags = np.zeros((self.max_objs), dtype=np.int64)
    # br_tags = np.zeros((self.max_objs), dtype=np.int64)
    # ct_tags = np.zeros((self.max_objs), dtype=np.int64)
    #
    # draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
    #   draw_umich_gaussian
    #
    # gt_det = []
    # for k in range(num_objs):
    #   ann = anns[k]
    #   bbox = self._coco_box_to_bbox(ann['bbox'])
    #   cls_id = int(ann['category_id']) - 1
    #   pts = np.array(ann['keypoints'], np.float32).reshape(num_joints, 3)
    #   bbox[:2] = affine_transform(bbox[:2], trans_output)
    #   bbox[2:] = affine_transform(bbox[2:], trans_output)
    #   bbox = np.clip(bbox, 0, output_res - 1)
    #   h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
    #
    #   xtl, ytl = bbox[0], bbox[1]
    #   xbr, ybr = bbox[2], bbox[3]
    #   xct, yct = (bbox[2] + bbox[0]) / 2., (bbox[3] + bbox[1]) / 2.
    #
    #   fxtl = (xtl)
    #   fytl = (ytl)
    #   fxbr = (xbr)
    #   fybr = (ybr)
    #   fxct = (xct)
    #   fyct = (yct)
    #
    #   xtl = int(fxtl)
    #   ytl = int(fytl)
    #   xbr = int(fxbr)
    #   ybr = int(fybr)
    #   xct = int(fxct)
    #   yct = int(fyct)
    #
    #   width = bbox[2] - bbox[0]
    #   height = bbox[3] - bbox[1]
    #
    #   width = math.ceil(width)
    #   height = math.ceil(height)
    #
    #   radius = gaussian_radius((height, width), 0.7)
    #   radius = max(0, int(radius))
    #
    #   draw_gaussian(tl_heatmaps[cls_id], [xtl, ytl], radius)
    #   draw_gaussian(br_heatmaps[cls_id], [xbr, ybr], radius)
    #   draw_gaussian(ct_heatmaps[cls_id], [xct, yct], radius)
    #
    #   tl_regrs[k, :] = [fxtl - xtl, fytl - ytl]
    #   br_regrs[k, :] = [fxbr - xbr, fybr - ybr]
    #   ct_regrs[k, :] = [fxct - xct, fyct - yct]
    #   tl_tags[k] = ytl * output_res + xtl
    #   br_tags[k] = ybr * output_res + xbr
    #   ct_tags[k] = yct * output_res + xct
    #
    #   if (h > 0 and w > 0) or (rot != 0):
    #     radius = gaussian_radius((math.ceil(h), math.ceil(w)))
    #     radius = self.opt.hm_gauss if self.opt.mse_loss else max(0, int(radius))
    #     ct = np.array(
    #       [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
    #     ct_int = ct.astype(np.int32)
    #     wh[k] = 1. * w, 1. * h
    #     ind[k] = ct_int[1] * output_res + ct_int[0]
    #     reg[k] = ct - ct_int
    #     reg_mask[k] = 1
    #     num_kpts = pts[:, 2].sum()
    #     if num_kpts == 0:
    #       hm[cls_id, ct_int[1], ct_int[0]] = 0.9999
    #       reg_mask[k] = 0
    #
    #     hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
    #     hp_radius = self.opt.hm_gauss \
    #       if self.opt.mse_loss else max(0, int(hp_radius))
    #     for j in range(num_joints):
    #       if pts[j, 2] > 0:
    #         pts[j, :2] = affine_transform(pts[j, :2], trans_output_rot)
    #         if pts[j, 0] >= 0 and pts[j, 0] < output_res and \
    #                 pts[j, 1] >= 0 and pts[j, 1] < output_res:
    #           # offset with ct
    #           kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
    #           kps_mask[k, j * 2: j * 2 + 2] = 1
    #           pt_int = pts[j, :2].astype(np.int32)
    #           hp_offset[k * num_joints + j] = pts[j, :2] - pt_int
    #           hp_ind[k * num_joints + j] = pt_int[1] * output_res + pt_int[0]
    #           hp_mask[k * num_joints + j] = 1
    #           if self.opt.dense_hp:
    #             # must be before draw center hm gaussian
    #             draw_dense_reg(dense_kps[j], hm[cls_id], ct_int,
    #                            pts[j, :2] - ct_int, radius, is_offset=True)
    #             draw_gaussian(dense_kps_mask[j], ct_int, radius)
    #           draw_gaussian(hm_hp[j], pt_int, hp_radius)
    #     draw_gaussian(hm[cls_id], ct_int, radius)
    #     gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
    #                    ct[0] + w / 2, ct[1] + h / 2, 1] +
    #                   pts[:, :2].reshape(num_joints * 2).tolist() + [cls_id])
    # gt = {'hps': kps, 'hps_mask': kps_mask, 'tl_heatmaps': tl_heatmaps, 'br_heatmaps': br_heatmaps,
    #        'ct_heatmaps': ct_heatmaps,
    #        'tl_regrs': tl_regrs, 'br_regrs': br_regrs, 'ct_regrs': ct_regrs, 'tl_tags': tl_tags, 'br_tags': br_tags,
    #        'ct_tags': ct_tags, 'tag_masks': reg_mask}
    # if self.opt.hm_hp:
    #   gt.update({'hm_hp': hm_hp})
    # if self.opt.reg_hp_offset:
    #   gt.update({'hp_offset': hp_offset, 'hp_ind': hp_ind, 'hp_mask': hp_mask})
    # meta = {'c': c, 's': s,
    #         'out_height': inp_height // self.opt.down_ratio,
    #         'out_width': inp_width // self.opt.down_ratio,
    #         'gt':gt}
    # return images, meta

  def process(self, images, return_time=False):
    raise NotImplementedError

  def post_process(self, dets, meta, scale=1):
    raise NotImplementedError

  def merge_outputs(self, detections):
    raise NotImplementedError

  def debug(self, debugger, images, dets, output, scale=1):
    raise NotImplementedError

  def show_results(self, debugger, image, results):
   raise NotImplementedError

  def run(self, image_or_path_or_tensor, meta=None):
    load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
    merge_time, tot_time = 0, 0
    debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug==3),
                        theme=self.opt.debugger_theme)
    start_time = time.time()
    pre_processed = False
    if isinstance(image_or_path_or_tensor, np.ndarray):
      image = image_or_path_or_tensor
    elif type(image_or_path_or_tensor) == type (''): 
      image = cv2.imread(image_or_path_or_tensor)
    else:
      image = image_or_path_or_tensor['image'][0].numpy()
      pre_processed_images = image_or_path_or_tensor
      pre_processed = True
    
    loaded_time = time.time()
    load_time += (loaded_time - start_time)
    
    detections = []
    for scale in self.scales:
      scale_start_time = time.time()
      if not pre_processed:
        images, meta = self.pre_process(image, scale, meta)
      else:
        # import pdb; pdb.set_trace()
        images = pre_processed_images['images'][scale][0]
        meta = pre_processed_images['meta'][scale]
        # if len(meta) == 5:
        #   self.gt[0] = meta['gt']
        #   meta = {'c': meta['c'], 's': meta['s'], 'out_height': meta['out_height'], 'out_width': meta['out_width']}
        meta = {k: v.numpy()[0] for k, v in meta.items()}
      images = images.to(self.opt.device)
      torch.cuda.synchronize()
      pre_process_time = time.time()
      pre_time += pre_process_time - scale_start_time

      if self.opt.zjb:
        images_dict = {'input': images}
        output, dets, forward_time = self.process(images_dict, return_time=True)
      else:
        output, dets, forward_time = self.process(images, return_time=True)

      torch.cuda.synchronize()
      net_time += forward_time - pre_process_time
      decode_time = time.time()
      dec_time += decode_time - forward_time
      
      if self.opt.debug >= 2:
        self.debug(debugger, images, dets, output, scale)
      
      dets = self.post_process(dets, meta, scale)
      torch.cuda.synchronize()
      post_process_time = time.time()
      post_time += post_process_time - decode_time

      detections.append(dets)
    
    results = self.merge_outputs(detections)
    torch.cuda.synchronize()
    end_time = time.time()
    merge_time += end_time - post_process_time
    tot_time += end_time - start_time

    if self.opt.debug >= 1:
      self.show_results(debugger, image, results)
    #self.show_results(debugger, image, results)
    
    return {'results': results, 'tot': tot_time, 'load': load_time,
            'pre': pre_time, 'net': net_time, 'dec': dec_time,
            'post': post_time, 'merge': merge_time}