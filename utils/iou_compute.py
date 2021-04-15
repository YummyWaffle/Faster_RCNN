import torch
def iou_compute(anchors,gts):
    A = anchors.size(0)
    B = gts.size(0)
    max_xy = torch.min(anchors[:,2:].unsqueeze(1).expand(A,B,2),
                       gts[:,2:].unsqueeze(0).expand(A,B,2))
    min_xy = torch.max(anchors[:,:2].unsqueeze(1).expand(A,B,2),
                       gts[:,:2].unsqueeze(0).expand(A,B,2))
    inter = torch.clamp((max_xy-min_xy),min=0)
    inter = inter[:,:,0] * inter[:,:,1]
    area_anchors = ((anchors[:,2]-anchors[:,0])*(anchors[:,3]-anchors[:,1])).unsqueeze(1).expand_as(inter)
    area_gts = ((gts[:,2]-gts[:,0])*(gts[:,3]-gts[:,1])).unsqueeze(0).expand_as(inter)
    return inter / (area_anchors + area_gts - inter)