import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision.ops.roi_pool import RoIPool
from .anchor_generation_layer import AnchorGenerator
from .proposal_layer import Proposal_Layer
from .anchor_target_layer import Anchor_Target_Layer
from .proposal_target_layer import Proposal_Target_Layer
import sys
sys.path.append('..')
import configs as cfg 

class rpn(nn.Module):
    def __init__(self):
        super(rpn,self).__init__()
        self.base_size = cfg.base_size
        self.scales = cfg.scales
        self.ratios = cfg.ratios
        self.anchor_num = len(self.ratios) * len(self.scales)
        self.inplanes = cfg.rpn_inplane
        
        # Network Part
        self.rpn_net = nn.Conv2d(self.inplanes,512,kernel_size=3,stride=1,padding=1)
        self.rpn_bn = nn.BatchNorm2d(512)
        self.rpn_relu = nn.ReLU(inplace=True)
        self.rpn_cls_score_net = nn.Conv2d(512,self.anchor_num*2,kernel_size=1,stride=1,padding=0)
        self.rpn_bbx_pred_net = nn.Conv2d(512,self.anchor_num*4,kernel_size=1,stride=1,padding=0)
        
        # Pooler
        self.pooler = RoIPool(cfg.roi_pool_size,cfg.spatial_scale)
        # RPN Layers
        self.generator = AnchorGenerator(self.base_size,self.scales,self.ratios)
        self.proposal_layer = Proposal_Layer()
        self.Anchor_Target_Layer = Anchor_Target_Layer()
        self.Proposal_Target_Layer = Proposal_Target_Layer()
        
        
    def forward(self,feat_map,im_info,gt_boxes_with_cls):
        #print(list(self.rpn_bbx_pred_net.parameters()))
        gt_boxes = gt_boxes_with_cls[:,:4]
        #print(gt_boxes)
        #print(gt_boxes_with_cls)
        # Network Predicting
        feat_info = feat_map.size()[2:]
        x = self.rpn_net(feat_map)
        x = self.rpn_bn(x)
        x = self.rpn_relu(x)
        rpn_score = self.rpn_cls_score_net(x)
        rpn_score_reshape = rpn_score.permute(0,2,3,1).contiguous().view(rpn_score.size(0),-1,2)
        rpn_bbox = self.rpn_bbx_pred_net(x)
        rpn_bbox_reshape = rpn_bbox.permute(0,2,3,1).contiguous().view(rpn_bbox.size(0),-1,4)
        # Anchor Generation - anchor (W*H*Anchor_Num,4)
        anchors = self.generator((im_info,feat_info,cfg.feat_stride))
        # Proposal Layer - 1) Apply Reg. 2) Pre-NMS Top-K Selection 3) NMS 4) Aft-NMS Top-K Selection 
        rois = self.proposal_layer(rpn_score_reshape,rpn_bbox_reshape,anchors,im_info)
        batch_index = torch.zeros(size=(rois.size(0),1)).type_as(rois)
        rois_pool = torch.cat((batch_index,rois),1).cuda()
        # roi-pooling - this module is directly loaded from torchvision.ops
        feat_pool = self.pooler(feat_map,rois_pool) 
        # This Part Is About Generating Targets & Calculate Anchor Loss
        rpn_loss = self.Anchor_Target_Layer(rpn_score_reshape.squeeze(0),rpn_bbox_reshape.squeeze(0),anchors,gt_boxes)
        # Give Cls-Reg Targets Here.
        positive_keep,negative_keep,cls_target,reg_target = self.Proposal_Target_Layer(rois,gt_boxes_with_cls,cfg.class_num)
        return feat_pool,rpn_loss,positive_keep,negative_keep,cls_target,reg_target
        
if __name__ == '__main__':
    print('---Testing RPN Module---')
    im_info = [1600,1600,3]
    gt_boxes = [[10.,20.,40.,60.]]
    model = rpn().cuda()
    tsr = torch.randn(size=(1,512,100,100)).cuda()
    out = model(tsr,im_info,gt_boxes)