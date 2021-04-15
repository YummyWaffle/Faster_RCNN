import torch
import torch.nn as nn
import configs as cfg
import sys
sys.path.append('..')
from utils.box_trans import apply_reg
from utils.nms import nms

class Proposal_Layer(nn.Module):
    def __init__(self):
        super(Proposal_Layer,self).__init__()
        #print('Enter Proposal Layer')
    def forward(self,score,reg_param,anchors,im_info):
        # Apply Regression
        rois = apply_reg(anchors,reg_param)
        rois[0::2].clamp_(0,im_info[0]-1)
        rois[1::2].clamp_(0,im_info[1]-1)
        # Pre-NMS Top-K Selection
        score_foreground = score[:,:,0].squeeze(0)
        _,order = torch.sort(score_foreground,descending=True)
        if(cfg.pre_nms_topk > 0 and cfg.pre_nms_topk < score_foreground.size(0)):
            order = order[:cfg.pre_nms_topk]
        rois = rois[order,:]
        score_foreground = score_foreground[order]
        # NMS
        nms_keep_index = nms(rois,score_foreground,cfg.rpn_nms_thr)
        rois = rois[nms_keep_index,:]
        # Aft-NMS Top-K Selection
        if(cfg.aft_nms_topk > 0 and cfg.aft_nms_topk < nms_keep_index.size(0)):
            rois = rois[:cfg.aft_nms_topk,:]
            score_foreground = score_foreground[:cfg.aft_nms_topk] 
        return rois
    """def backward(self):
        pass"""