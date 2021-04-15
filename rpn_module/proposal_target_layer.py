import torch
import torch.nn as nn
import math

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


class Proposal_Target_Layer(nn.Module):
    def __init__(self):
        super(Proposal_Target_Layer,self).__init__()
        
    def forward(self,proposals,gts_with_cls,cls_num=21,max_sample_num=256,neg_thr=0.3,pos_thr=0.7,np_rate=1./4.):
        """
        cls_num contains background inside
        
        What We Want In Return ?
        1) Keep Mask - In Order To Make Sure OHEM(Online Hard Negative Mining) Are Executed. - (Proposal_Num,) [bool]
        2) Cls-Target - (Proposal_Num,1) [Long/Int]
        3) Regression-Target - (Proposal_Num,4) [Float]
        """
        iou_matrix = iou_compute(proposals.cuda(),gts_with_cls[:,:4].cuda())
        anchor_max,anchor_max_index = torch.max(iou_matrix,1)
        print('Proposal Target Layer')
        print(torch.sort(anchor_max,descending=True)[0][:5])
        _,gt_max_index = torch.max(iou_matrix,0)
        # Take Positive Sample Out
        Positive_Mask = anchor_max > pos_thr
        Positive_Mask[gt_max_index] = True
        #Positive_Mapper = anchor_max_index[Positive_Mask]
        Positive_Keep = (Positive_Mask==True).nonzero().squeeze(1)
        runtime_positive_num = Positive_Keep.size(0)
        #print(runtime_positive_num)
        if(runtime_positive_num > int(max_sample_num * np_rate)):
            runtime_positive_num = int(max_sample_num * np_rate)
            Positive_Keep = Positive_Keep[:runtime_positive_num]
        negative_upper_bound = int((float(runtime_positive_num) / np_rate)*(1-np_rate))
        # Take Negative Sample Out
        Negative_Mask = anchor_max < pos_thr
        Negative_Mask[gt_max_index] = False
        Negative_Keep = (Negative_Mask==True).nonzero().squeeze(1)
        runtime_negative_num = Negative_Keep.size(0)
        if(runtime_negative_num > negative_upper_bound):
            runtime_negative_num = negative_upper_bound
            Negative_Keep = Negative_Keep[:runtime_negative_num]
        #Keep_Index = torch.cat((Positive_Keep,Negative_Keep),0)
        #print(Keep_Index)
        # Generate Targets Here
        cls_target = torch.zeros(size=(proposals.size(0),),dtype=torch.int64)
        #cls_target *= cls_num
        reg_target = torch.zeros(size=(proposals.size(0),4),dtype=torch.float32)
        for i,proposal in enumerate(proposals):
            # if not in positive keep just classify it as negative_upper_bound
            if(i in Positive_Keep):
                cls_target[i] = gts_with_cls[anchor_max_index[i],4]
                single_reg_target = single_encoder(proposal,gts_with_cls[anchor_max_index[i],:4])
                reg_target[i] = single_reg_target
            else:
                cls_target[i] = cls_num - 1
        #print(cls_target)
        #print(Positive_Keep)
        #print(proposals[Positive_Keep,:])
        return Positive_Keep,Negative_Keep,cls_target,reg_target


if __name__ == '__main__':
    print('---testing proposal_target_layer---')
    proposals = torch.tensor([[10,10,50,50],[70,70,150,150],[0,0,25,30],[52,52,79,90],[130,128,150,162],[24,32,123,130],[46,23,65,77],[18,32,89,46]]).float()
    gts = torch.tensor([[13,12,53,56,0],[25,29,120,127,2],[69,72,152,146,3],[42,42,61,51,6]]).float()
    
    proposal_target_layer = Proposal_Target_Layer()
    positive_keep,negative_keep,cls_target,reg_target=proposal_target_layer(proposals,gts)