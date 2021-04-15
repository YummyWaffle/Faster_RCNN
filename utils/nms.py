import torch

def nms(boxes,score,thr=0.7):
    _,order = torch.sort(score,descending=True)
    area = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    keep_indexes = []
    while order.size(0)!= 0:
        keep_index = order[0].item()
        keep_indexes.append(keep_index)
        order = order[1:]
        xx1 = torch.max(boxes[keep_index,0],boxes[order[:],0])
        yy1 = torch.max(boxes[keep_index,1],boxes[order[:],1])
        xx2 = torch.min(boxes[keep_index,2],boxes[order[:],2])
        yy2 = torch.min(boxes[keep_index,3],boxes[order[:],3])
        #print(xx1.size())
        w = (xx2-xx1+1).clamp_(min=0.)
        h = (yy2-yy1+1).clamp_(min=0.)
        inter = w * h
        iou = inter / (area[keep_index] + area[order[:]] - inter)
        not_same_indxes = iou[:] <= thr
        order = order[not_same_indxes] 
    return torch.LongTensor(keep_indexes)

if __name__ == '__main__':
    boxes = torch.tensor([[0,0,10,10],[1,1,11,11],[9,9,20,20],[2,2,5,5]]).float()
    scores = torch.tensor([0.6,0.9,0.5,0.7])
    keep = nms(boxes,scores,0.7)
    print('---testing demo---')
    print(keep)