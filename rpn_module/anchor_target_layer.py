import torch
import torch.nn as nn
import numpy as np
import math
import cv2
import sys
sys.path.append('..')
from utils.iou_compute import iou_compute
def single_encoder(anchor,gt):
    anchor_ctr_x = anchor[0] + (anchor[2]-anchor[0])/2.
    anchor_ctr_y = anchor[1] + (anchor[3]-anchor[1])/2.
    anchor_w = (anchor[2]-anchor[0])
    anchor_h = (anchor[3]-anchor[1])
    
    gt_ctr_x = gt[0] + (gt[2]-gt[0])/2.
    gt_ctr_y = gt[1] + (gt[3]-gt[1])/2.
    gt_w = (gt[2]-gt[0])
    gt_h = (gt[3]-gt[1])
    
    dx = (gt_ctr_x - anchor_ctr_x)/anchor_w
    dy = (gt_ctr_y - anchor_ctr_y)/anchor_h
    dw = math.log(gt_w/anchor_w)
    dh = math.log(gt_h/anchor_h)
    targets = torch.tensor([dx,dy,dw,dh]).cuda()
    return targets

class Anchor_Target_Layer(nn.Module):
    # We Want Calculate RPN-Loss Here
    # - Generate Targets Input - (anchors,gt_boxes) Output - (score_target,reg_target)
    # - Compute Loss  Input - (score_pred,reg_pred,score target,reg target) Output - (cls_loss,reg_loss)
    
    def __init__(self):
        super(Anchor_Target_Layer,self).__init__()
    def forward(self,score_pred,reg_pred,anchors,gts,sample_max_num=256,neg_thr=0.3,pos_thr=0.5,np_rate=0.5):
        # Step-1 IoU Calculate
        iou_matrix = iou_compute(anchors,gts)
        #print(iou_matrix)
        #print(iou_matrix)
        # Step-2 Calculate Positive Index & Negative Index
        # 2.1 - the max of row - max iou of each anchor
        anchor_max,anchor_max_indices=torch.max(iou_matrix,1)
        print('Anchor Target Layer')
        print(torch.sort(anchor_max,descending=True)[0][:5])
        # 2.2 - the max of col - max iou of each gts
        gt_max,gt_max_indices=torch.max(iou_matrix,0)
        # 2.3 - Positive Anchor Selection
        positive_mask = anchor_max > pos_thr
        positive_mask[gt_max_indices] = True
        positive_anchors = anchors[positive_mask]
        positive_pred_score = score_pred[positive_mask]
        positive_pred_reg = reg_pred[positive_mask]
        positive_mapper = anchor_max_indices[positive_mask]
        runtime_positive_num = positive_anchors.size(0)
        #print(anchor_max)
        #print(runtime_positive_num)
        positive_upper_bound = int(sample_max_num * np_rate)
        negative_sample_bound = int(sample_max_num * (1.-np_rate))
        if(runtime_positive_num<positive_upper_bound):
            negative_sample_bound = int((float(runtime_positive_num) / np_rate) * (1.-np_rate))
        else:
            positive_anchors = positive_anchors[:positive_upper_bound,:]
            positive_pred_score = positive_anchors[:positive_upper_bound,:]
            positive_pred_reg = positive_pred_reg[:positive_upper_bound,:]
            runtime_positive_num = positive_upper_bound
        # 2.5 - Negative Anchor Selection
        negative_mask = anchor_max < neg_thr
        negative_mask[gt_max_indices] = False
        negative_anchors = anchors[negative_mask]
        negative_pred_score = score_pred[negative_mask]
        negative_pred_reg = reg_pred[negative_mask]
        if(negative_sample_bound < negative_anchors.size(0)):
            negative_anchors = negative_anchors[:negative_sample_bound,:]
            negative_pred_score = negative_pred_score[:negative_sample_bound,:]
            negative_pred_reg = negative_pred_reg[:negative_sample_bound,:]        
        runtime_negative_num = negative_anchors.size(0)
        # Step-3 Target Generatings - Now We Got Postive Negative Samples & GTs
        # 3.1 Positive_Loss - Calculate Cls.[Cross-Entropy] & Reg. Loss[Smooth-L1-Loss]
        cls_target = torch.zeros(size=(positive_anchors.size(0),),dtype=torch.int64).cuda()
        pos_cls_loss = nn.functional.cross_entropy(positive_pred_score,cls_target,reduction='mean')
        pos_reg_loss = 0.
        for i,pos_anchor in enumerate(positive_anchors):
            mapper_gt_index = positive_mapper[i]
            pred = positive_pred_reg[i].unsqueeze(0)
            target = single_encoder(pos_anchor,gts[mapper_gt_index]).unsqueeze(0)
            pos_reg_loss += nn.functional.smooth_l1_loss(pred,target,reduction='sum')
        pos_reg_loss =  pos_reg_loss / float(positive_anchors.size(0))
        # 3.2 Negative_Loss - Only Calculate Cls. Loss
        neg_cls_target = torch.ones(size=(negative_anchors.size(0),),dtype=torch.int64).cuda()
        neg_cls_loss = nn.functional.cross_entropy(negative_pred_score,neg_cls_target,reduction='mean')
        print('rpn cls loss - ' + str(pos_cls_loss.item()+neg_cls_loss.item()))
        print('rpn reg loss - ' + str(pos_reg_loss.item()))
        return pos_cls_loss+pos_reg_loss+neg_cls_loss
    """def backward(self):
        pass"""

if __name__ == '__main__':
    print('---testing anchor target layer---')       
    anchors = torch.tensor([[10,10,50,50],[70,70,150,150],[0,0,25,30],[52,52,79,40],[130,128,150,162],[24,32,123,130],[46,23,65,77],[18,32,89,46]]).float()
    gt_boxes = torch.tensor([[13,12,53,56],[25,29,120,127],[69,72,152,146],[42,42,61,51]]).float()
    score_pred = torch.tensor([[1,0],[1,0],[0,1],[0,1],[0,1],[1,0],[0,1],[0,1]]).float()
    reg_pred = torch.tensor([[0.0732,0.0976,0,0.0931],[0.0062,-0.0123,0.0364,-0.0770],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[-0.0100,-0.0303,-0.0408,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]]).float()
    
    target_layer_instance = Anchor_Target_Layer()
    target_layer_instance(score_pred,reg_pred,anchors,gt_boxes)
    """anchors *= 5.
    gt_boxes *= 5.
    img = np.zeros((1000,1000,3),np.uint8)
    img.fill(255)
    for anchor in anchors:
        x1=int(anchor[0])
        y1=int(anchor[1])
        x2=int(anchor[2])
        y2=int(anchor[3])
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
    for gt_box in gt_boxes:
        x1=int(gt_box[0])
        y1=int(gt_box[1])
        x2=int(gt_box[2])
        y2=int(gt_box[3])
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),3)
    cv2.imshow('test',img)
    cv2.waitKey()"""
    