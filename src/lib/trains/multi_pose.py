from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss, RegL1Loss, RegLoss, RegWeightedL1Loss
from models.decode import multi_pose_decode
from models.utils import _sigmoid, flip_tensor, flip_lr_off, flip_lr
from utils.debugger import Debugger
from utils.post_process import multi_pose_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer

class MultiPoseLoss(torch.nn.Module):
  def __init__(self, opt):
    super(MultiPoseLoss, self).__init__()
    self.crit = FocalLoss()
    self.crit_hm_hp = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_kp = RegWeightedL1Loss() if not opt.dense_hp else \
                   torch.nn.L1Loss(reduction='sum')
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
                    RegLoss() if opt.reg_loss == 'sl1' else None
    self.opt = opt

  def forward(self, outputs, batch):
    opt = self.opt
    if opt.zjb:
      aeloss = AELoss()
      hm_loss, pull_loss, push_loss, regr_loss = 0, 0, 0, 0
      hp_loss, hm_hp_loss, hp_offset_loss = 0, 0, 0
      for s in range(opt.num_stacks):
        output = outputs[s]
        if opt.hm_hp and not opt.mse_loss:
          output['hm_hp'] = _sigmoid(output['hm_hp'])

        a,b,c,d = aeloss(output, batch)
        hm_loss += a
        pull_loss += b
        push_loss += c
        regr_loss += d
        
        if opt.dense_hp:
          mask_weight = batch['dense_hps_mask'].sum() + 1e-4
          hp_loss += (self.crit_kp(output['hps'] * batch['dense_hps_mask'], 
                                   batch['dense_hps'] * batch['dense_hps_mask']) / 
                                   mask_weight) / opt.num_stacks
        else:
          hp_loss += self.crit_kp(output['hps'], batch['hps_mask'], 
                                  batch['ind'], batch['hps']) / opt.num_stacks
       
        if opt.reg_hp_offset and opt.off_weight > 0:
          hp_offset_loss += self.crit_reg(
            output['hp_offset'], batch['hp_mask'],
            batch['hp_ind'], batch['hp_offset']) / opt.num_stacks
        if opt.hm_hp and opt.hm_hp_weight > 0:
          hm_hp_loss += self.crit_hm_hp(
            output['hm_hp'], batch['hm_hp']) / opt.num_stacks
      loss = hm_loss + pull_loss + push_loss + regr_loss + opt.hp_weight * hp_loss + \
             opt.hm_hp_weight * hm_hp_loss + opt.off_weight * hp_offset_loss
      
      loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'pull_loss': pull_loss, 
                    'push_loss': push_loss, 'regr_loss': regr_loss,
                    'hm_hp_loss': hm_hp_loss, 'hp_offset_loss': hp_offset_loss, 'hp_loss': hp_loss}
      return loss, loss_stats

    else:
      hm_loss, wh_loss, off_loss = 0, 0, 0
      hp_loss, off_loss, hm_hp_loss, hp_offset_loss = 0, 0, 0, 0
      for s in range(opt.num_stacks):
        output = outputs[s]
        output['hm'] = _sigmoid(output['hm'])
        if opt.hm_hp and not opt.mse_loss:
          output['hm_hp'] = _sigmoid(output['hm_hp'])
        
        if opt.eval_oracle_hmhp:
          output['hm_hp'] = batch['hm_hp']
        if opt.eval_oracle_hm:
          output['hm'] = batch['hm']
        if opt.eval_oracle_kps:
          if opt.dense_hp:
            output['hps'] = batch['dense_hps']
          else:
            output['hps'] = torch.from_numpy(gen_oracle_map(
              batch['hps'].detach().cpu().numpy(), 
              batch['ind'].detach().cpu().numpy(), 
              opt.output_res, opt.output_res)).to(opt.device)
        if opt.eval_oracle_hp_offset:
          output['hp_offset'] = torch.from_numpy(gen_oracle_map(
            batch['hp_offset'].detach().cpu().numpy(), 
            batch['hp_ind'].detach().cpu().numpy(), 
            opt.output_res, opt.output_res)).to(opt.device)


        hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
        if opt.dense_hp:
          mask_weight = batch['dense_hps_mask'].sum() + 1e-4
          hp_loss += (self.crit_kp(output['hps'] * batch['dense_hps_mask'], 
                                   batch['dense_hps'] * batch['dense_hps_mask']) / 
                                   mask_weight) / opt.num_stacks
        else:
          hp_loss += self.crit_kp(output['hps'], batch['hps_mask'], 
                                  batch['ind'], batch['hps']) / opt.num_stacks
        if opt.wh_weight > 0:
          wh_loss += self.crit_reg(output['wh'], batch['reg_mask'],
                                   batch['ind'], batch['wh']) / opt.num_stacks
        if opt.reg_offset and opt.off_weight > 0:
          off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                    batch['ind'], batch['reg']) / opt.num_stacks
        if opt.reg_hp_offset and opt.off_weight > 0:
          hp_offset_loss += self.crit_reg(
            output['hp_offset'], batch['hp_mask'],
            batch['hp_ind'], batch['hp_offset']) / opt.num_stacks
        if opt.hm_hp and opt.hm_hp_weight > 0:
          hm_hp_loss += self.crit_hm_hp(
            output['hm_hp'], batch['hm_hp']) / opt.num_stacks
      loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
             opt.off_weight * off_loss + opt.hp_weight * hp_loss + \
             opt.hm_hp_weight * hm_hp_loss + opt.off_weight * hp_offset_loss
      
      loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'hp_loss': hp_loss, 
                    'hm_hp_loss': hm_hp_loss, 'hp_offset_loss': hp_offset_loss,
                    'wh_loss': wh_loss, 'off_loss': off_loss}
      return loss, loss_stats

class MultiPoseTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(MultiPoseTrainer, self).__init__(opt, model, optimizer=optimizer)
  
  def _get_losses(self, opt):
    if opt.zjb:
      loss_states = ['loss', 'hm_loss', 'pull_loss', 'push_loss', 'regr_loss', 
                     'hm_hp_loss', 'hp_offset_loss', 'hp_loss']
    else:
      loss_states = ['loss', 'hm_loss', 'hp_loss', 'hm_hp_loss', 
                     'hp_offset_loss', 'wh_loss', 'off_loss']
    loss = MultiPoseLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt
    reg = output['reg'] if opt.reg_offset else None
    hm_hp = output['hm_hp'] if opt.hm_hp else None
    hp_offset = output['hp_offset'] if opt.reg_hp_offset else None
    dets = multi_pose_decode(
      output['hm'], output['wh'], output['hps'], 
      reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])

    dets[:, :, :4] *= opt.input_res / opt.output_res
    dets[:, :, 5:39] *= opt.input_res / opt.output_res
    dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
    dets_gt[:, :, :4] *= opt.input_res / opt.output_res
    dets_gt[:, :, 5:39] *= opt.input_res / opt.output_res
    for i in range(1):
      debugger = Debugger(
        dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')

      debugger.add_img(img, img_id='out_pred')
      for k in range(len(dets[i])):
        if dets[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                 dets[i, k, 4], img_id='out_pred')
          debugger.add_coco_hp(dets[i, k, 5:39], img_id='out_pred')

      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt[i])):
        if dets_gt[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                 dets_gt[i, k, 4], img_id='out_gt')
          debugger.add_coco_hp(dets_gt[i, k, 5:39], img_id='out_gt')

      if opt.hm_hp:
        pred = debugger.gen_colormap_hp(output['hm_hp'][i].detach().cpu().numpy())
        gt = debugger.gen_colormap_hp(batch['hm_hp'][i].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hmhp')
        debugger.add_blend_img(img, gt, 'gt_hmhp')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)

  def save_result(self, output, batch, results):
    reg = output['reg'] if self.opt.reg_offset else None
    hm_hp = output['hm_hp'] if self.opt.hm_hp else None
    hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
    dets = multi_pose_decode(
      output['hm'], output['wh'], output['hps'], 
      reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    
    dets_out = multi_pose_post_process(
      dets.copy(), batch['meta']['c'].cpu().numpy(),
      batch['meta']['s'].cpu().numpy(),
      output['hm'].shape[2], output['hm'].shape[3])
    results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]


def _ae_loss(tag0, tag1, mask):
    num  = mask.sum(dim=1, keepdim=True).float()
    tag0 = tag0.squeeze(2)
    tag1 = tag1.squeeze(2)

    tag_mean = (tag0 + tag1) / 2

    tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
    tag0 = tag0[mask].sum()
    tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
    tag1 = tag1[mask].sum()
    pull = tag0 + tag1

    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)
    num  = num.unsqueeze(2)
    num2 = (num - 1) * num
    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = torch.nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()
    return pull, push

def _regr_loss(regr, gt_regr, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr    = regr[mask]
    gt_regr = gt_regr[mask]
    
    regr_loss = torch.nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

def _neg_loss(preds, gt):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

class AELoss(torch.nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, regr_weight=1, focal_loss=_neg_loss):
        super(AELoss, self).__init__()

        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.regr_weight = regr_weight
        self.focal_loss  = focal_loss
        self.ae_loss     = _ae_loss
        self.regr_loss   = _regr_loss

    def forward(self, outs, targets):
        tl_heats = []
        br_heats = []
        ct_heats = []
        tl_tags  = []
        br_tags  = []
        tl_regrs = []
        br_regrs = []
        ct_regrs = []        
        tl_heats.append(outs['tl_heat'])
        br_heats.append(outs['br_heat'])
        ct_heats.append(outs['ct_heat'])
        tl_tags.append(outs['tl_tag'])
        br_tags.append(outs['br_tag'])
        tl_regrs.append(outs['tl_regr'])
        br_regrs.append(outs['br_regr'])
        ct_regrs.append(outs['ct_regr'])

        gt_tl_heat = targets['tl_heatmaps']
        gt_br_heat = targets['br_heatmaps']
        gt_ct_heat = targets['ct_heatmaps']
        gt_mask    = targets['tag_masks']
        gt_tl_regr = targets['tl_regrs']
        gt_br_regr = targets['br_regrs']
        gt_ct_regr = targets['ct_regrs']
        
        # focal loss
        focal_loss = 0

        tl_heats = [_sigmoid(t) for t in tl_heats]
        br_heats = [_sigmoid(b) for b in br_heats]
        ct_heats = [_sigmoid(c) for c in ct_heats]

        focal_loss += self.focal_loss(tl_heats, gt_tl_heat)
        focal_loss += self.focal_loss(br_heats, gt_br_heat)
        focal_loss += self.focal_loss(ct_heats, gt_ct_heat)

        # tag loss
        pull_loss = 0
        push_loss = 0

        for tl_tag, br_tag in zip(tl_tags, br_tags):
            pull, push = self.ae_loss(tl_tag, br_tag, gt_mask)
            pull_loss += pull
            push_loss += push
        pull_loss = self.pull_weight * pull_loss
        push_loss = self.push_weight * push_loss

        regr_loss = 0
        for tl_regr, br_regr, ct_regr in zip(tl_regrs, br_regrs, ct_regrs):
            regr_loss += self.regr_loss(tl_regr, gt_tl_regr, gt_mask)
            regr_loss += self.regr_loss(br_regr, gt_br_regr, gt_mask)
            regr_loss += self.regr_loss(ct_regr, gt_ct_regr, gt_mask)
        regr_loss = self.regr_weight * regr_loss

        return (focal_loss / len(tl_heats)).unsqueeze(0), (pull_loss / len(tl_heats)).unsqueeze(0), (push_loss / len(tl_heats)).unsqueeze(0), (regr_loss / len(tl_heats)).unsqueeze(0)