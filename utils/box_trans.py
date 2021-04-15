import torch
import sys
import cv2
import numpy as np
sys.path.append('..')
from rpn_module.anchor_generation_layer import to_corner_form,to_xywh_form
def apply_reg(boxes,regression_params):
    # Formula - x' = x_box + w_box * param_x
    #         - y' = y_box + h_box * param_y
    #         - w' = exp{param_w} * w_box
    #         - h' = exp{param_h} * h_box
    xywh_boxes = to_xywh_form(boxes).unsqueeze(0).cuda()
    new_boxes = torch.zeros(size=xywh_boxes.size(),dtype=torch.float32)
    param_xy = regression_params[:,:,:2]
    param_wh = regression_params[:,:,2:]
    new_boxes[:,:,:2] = xywh_boxes[:,:,:2] + param_xy * xywh_boxes[:,:,2:]
    new_boxes[:,:,2:] = xywh_boxes[:,:,2:] * torch.exp(param_wh)
    #print(xywh_boxes)
    trans_boxes = to_corner_form(new_boxes.squeeze(0))
    #print(trans_boxes)
    return trans_boxes

if __name__=='__main__':
    color = [(0,0,255),(0,255,0)]
    boxes = torch.tensor([[10,10,50,50],[70,89,128,97]]).float().cuda()
    reg_para = torch.tensor([[[0.1,0.1,0.2,0.3],[0.1,0.3,0.04,0.6]]]).float().cuda()
    trans_boxes = apply_reg(boxes,reg_para)
    img = np.zeros((150,150,3),np.uint8)
    img.fill(255)
    for i,box in enumerate(boxes):
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        cv2.rectangle(img,(x1,y1),(x2,y2),color[i],1)
        
    for i,box in enumerate(trans_boxes):
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        #print(x1,y1,x2,y2)
        cv2.rectangle(img,(x1,y1),(x2,y2),color[i],2)
    #cv2.imshow('test',img)
    #cv2.waitKey()