import torch
import torch.nn as nn
import numpy as np
import cv2
import sys
sys.path.append('..')
import configs as cfg

# From Center XY & WH Form
def to_corner_form(anchors):
    return torch.cat((anchors[:,:2]-0.5*(anchors[:,2:]),anchors[:,:2]+0.5*(anchors[:,2:])),1)
    
# From Conrner Form   
def to_xywh_form(anchors):
    return torch.cat((anchors[:,:2]+0.5*(anchors[:,2:]-anchors[:,:2]),anchors[:,2:]-anchors[:,:2]),1)

# Ratio Transform Equation
# w' = sqrt(w*h/ratio) = sqrt(area/ratio)
# h' = sqrt(w*h*ratio) = sqrt(area*ratio)
def ratio_trans(anchors,ratios):
    anchor_xywh=to_xywh_form(anchors)
    area = anchor_xywh[:,2] * anchor_xywh[:,3]
    ratio_anchors = []
    for ratio in ratios:
        w_ratio = torch.sqrt(area/ratio)
        h_ratio = torch.sqrt(area*ratio)
        ratio_anchor_temp = anchor_xywh.clone()
        ratio_anchor_temp[:,2] = w_ratio
        ratio_anchor_temp[:,3] = h_ratio
        ratio_anchor_temp=ratio_anchor_temp.numpy()
        ratio_anchors.extend(ratio_anchor_temp)
        #torch.cat((ratio_anchors,ratio_anchor_temp),0)
    ratio_anchors = torch.tensor(ratio_anchors)
    return to_corner_form(ratio_anchors)

# Scale Transform Equation
# w' = w * scale
# h' = h * scale
def scale_trans(anchors,scales):
    anchor_xywh=to_xywh_form(anchors)
    scale_anchors = []
    for scale in scales:
        w_scale = anchor_xywh[:,2] * scale
        h_scale = anchor_xywh[:,3] * scale
        scale_anchor_temp = anchor_xywh.clone()
        scale_anchor_temp[:,2] = w_scale
        scale_anchor_temp[:,3] = h_scale
        scale_anchor_temp = scale_anchor_temp.numpy()
        scale_anchors.extend(scale_anchor_temp)
    scale_anchors = torch.tensor(scale_anchors)
    return to_corner_form(scale_anchors)

class AnchorGenerator(nn.Module):
    def __init__(self,base_size=16,scales=[8.,16.,32.],ratios=[0.5,1.,2.]):
        super(AnchorGenerator,self).__init__()
        self.base_size = base_size
        self.scales = scales
        self.ratios = ratios
    def forward(self,input):
        # input[0] = img_size - []
        # input[1] = featmap_size - []
        # input[2] = feat_stride
        im_info = input[0]
        featmap_info = input[1]
        feat_stride = input[2]
        # Step 1: Generate Base Anchor
        # center point in (0,0) -> corner form
        base_anchor = to_corner_form(torch.tensor([0,0,self.base_size,self.base_size]).unsqueeze(0).float())
        #base_anchor = torch.tensor([0,0,self.base_size,self.base_size]).unsqueeze(0).float()
        # Step 2: ratios transformation
        ratio_anchors = ratio_trans(base_anchor,self.ratios)
        # Step 3: scales transformation
        scale_anchors = scale_trans(ratio_anchors,self.scales)
        # Step 4: Lay Out On Origin Images
        shift_x = np.arange(featmap_info[0]) * feat_stride
        shift_y = np.arange(featmap_info[1]) * feat_stride
        shift_x,shift_y = np.meshgrid(shift_x,shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(),shift_y.ravel(),shift_x.ravel(),shift_y.ravel())).transpose())
        A = len(self.ratios) * len(self.scales)
        K = shifts.size(0)
        # Broadcasting Machanism scale_anchors - (Anchor_Num,4)  shifts - (Featmap_Size,4)
        # Expected Size - (Featmap_Size*A,4) 
        # Process - [Tips：Because The Parameters Are Permuted As (BatchSize,H,W,Params.) So The Temp Result Should Be (K,A,4)] 
        #(1,Anchor_Num,4) + (Featmap_Size,1,4) - (Featmap_Size,Anchor_Num,4) - (Anchor_Num,Featmap_Size,4)
        anchors = scale_anchors.view(1,A,4) + shifts.view(K,1,4)
        anchors = anchors.view(K*A,4)
        anchors[0::2].clamp_(0,im_info[0]-1)
        anchors[1::2].clamp_(0,im_info[1]-1)
        return anchors
    """def backward(self):
        pass"""

if __name__ == '__main__':
    generator = AnchorGenerator(base_size=16,scales=[8.,16.,32.],ratios=[.5,1.,2.])
    anchors = generator(([700,700,3],[70,70,512],10))
    img = np.zeros((700,700,3),np.uint8)
    img.fill(255)
    for i,anchor in enumerate(anchors):
        x1=int(anchor[0])
        y1=int(anchor[1])
        x2=int(anchor[2])
        y2= int(anchor[3])
        #if(i>=24750 and i<24759):
        #    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),3)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),1)
    cv2.imshow('test',img)
    cv2.waitKey()
    #print(anchors)