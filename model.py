import torch
import torch.nn as nn
from base_net.vgg import vgg
from rpn_module.rpn import rpn
import configs as cfg

class faster_rcnn(nn.Module):
    def __init__(self,backbone=None,rpn=None,class_num=None):
        super(faster_rcnn,self).__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.fully_connected = nn.Sequential(nn.Linear(cfg.in_feats,2048,bias=True),
                                             nn.ReLU(inplace=True),
                                             nn.Dropout(0.5))
        self.classifier = nn.Sequential(nn.Linear(2048,class_num,bias=True),
                                        nn.ReLU(inplace=True))
        #nn.softmax = nn.Softmax
        self.regressor = nn.Sequential(nn.Linear(2048,4,bias=True),
                                       nn.ReLU(inplace=True))
        
    def forward(self,img,im_info,gts_with_cls):
        #gts_with_cls = gts_with_cls
        #gts = gts_with_cls[:,:4]
        feat_map = self.backbone(img)
        feat_size = feat_map.size()[2:]
        #print(feat_size)
        feat_pool,rpn_loss,positive_keep,negative_keep,cls_target,reg_target = self.rpn(feat_map,im_info,gts_with_cls)
        #print(rpn_loss)
        feat_pool = feat_pool.view(feat_pool.size(0),-1)
        deeper_feat = self.fully_connected(feat_pool)
        #deeper_feat = self.Dropout_layer(deeper_feat)
        #print(deeper_feat.size())
        cls_pred = self.classifier(deeper_feat)
        reg_pred = self.regressor(deeper_feat)
        #print(cls_pred.size(),reg_pred.size())
        # Calculate Loss Here
        # 1. Calculate Positive Loss
        positive_mask = torch.zeros(size=(cls_pred.size(0),)).bool()
        positive_mask[positive_keep] = True
        postive_rcnn_cls_pred = cls_pred[positive_mask,:]
        postive_rcnn_cls_target = cls_target[positive_mask].cuda()
        rcnn_postive_cls_loss = nn.functional.cross_entropy(postive_rcnn_cls_pred,postive_rcnn_cls_target,reduction='mean')
        #print(rcnn_postive_cls_loss)
        postive_rcnn_reg_pred = reg_pred[positive_mask,:]
        postive_rcnn_reg_target = reg_target[positive_mask,:].cuda()
        rcnn_postive_reg_loss = nn.functional.smooth_l1_loss(postive_rcnn_reg_pred,postive_rcnn_reg_target,reduction='sum')
        
        # 2. Calculate Negative Loss
        negative_mask = torch.zeros(size=(cls_pred.size(0),)).bool()
        negative_mask[negative_keep] = True
        negative_rcnn_cls_pred = cls_pred[negative_mask]
        negative_rcnn_cls_target = cls_target[negative_mask].cuda()
        rcnn_negative_cls_loss = nn.functional.cross_entropy(negative_rcnn_cls_pred,negative_rcnn_cls_target,reduction='mean')
        print('rcnn cls loss - ' + str(rcnn_negative_cls_loss.item() + rcnn_postive_cls_loss.item()))
        print('rcnn reg loss - ' + str(rcnn_postive_reg_loss.item()))
        return rpn_loss + rcnn_postive_cls_loss + rcnn_postive_reg_loss + rcnn_negative_cls_loss

if __name__ == '__main__':
    backbone = vgg().cuda()
    rpn_net = rpn().cuda()
    model = faster_rcnn(backbone,rpn_net,cfg.class_num).cuda()
    img = torch.randn(size=(1,3,800,800)).cuda()
    im_info = (800,800,3)
    gts = torch.tensor([[13,12,53,56,0],[25,29,120,127,2],[69,72,152,146,3],[42,42,61,51,6]]).float()
    output = model(img,im_info,gts)