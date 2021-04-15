import torch
import torch.nn as nn
import torch.optim as optim
import configs as cfg
from datasets import *
#from base_net import *
from rpn_module import *
from model import faster_rcnn
import random
from torchvision import models

# Data Inputs Here
dataloader = faster_rcnn_loader('D:\SYSU\ObjDet\TowerSet_VOC\VOCdevkit')
#dataloader = faster_rcnn_loader('D:/0426DIOR/DIOR/VOCdevkit')

# Constructe Model Here
base_net = models.vgg16(pretrained=True).features
#base_net = vgg().cuda()
rpn_net = rpn().cuda()
detector = faster_rcnn(base_net,rpn_net,cfg.class_num).cuda()
print(list(detector.named_parameters()))
#output = detector(img,im_info,gts_with_cls)

# Traning Settings
Epochs = 10


index = np.arange(len(dataloader))

optimizer = optim.SGD(detector.parameters(),lr=0.00001)
# Run Training Here
for Epoch in range(Epochs):
    random.shuffle(index)
    print('-----------------------EPOCH DIVISION----------------------')
    for batch_num,i in enumerate(index):
        img,gt_with_cls,im_info = dataloader[i]
        loss = detector(img,im_info,gt_with_cls)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(' ')
        print('Epoch: '+ str(Epoch) + ' Batch: '+ str(batch_num) + ' ' + ' Loss: '+str(loss.item()))
        print('----------------------------------------------')