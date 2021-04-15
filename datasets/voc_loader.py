import torch.utils.data as data
import cv2
import os
import torch
import numpy as np
import xml.etree.ElementTree as ET

class faster_rcnn_loader(data.Dataset):
    """CLASSES = ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney', 'dam',
               'Expressway-Service-area', 'Expressway-toll-station', 'golffield', 'groundtrackfield', 
               'harbor', 'overpass','ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 
               'vehicle','windmill','background']"""
    CLASSES = ['tower','background']
    def __init__(self,dataset_folder):
        self.img_folder = dataset_folder + '/VOC2007/JPEGImages'
        self.anno_folder = dataset_folder + '/VOC2007/Annotations'
        self.txt_folder = dataset_folder + '/VOC2007/ImageSets/Main/train.txt'
        # Read TXT Files
        self.files_name = []
        f = open(self.txt_folder,'r')
        lines = f.readlines()
        for line in lines:
            filename = line.strip().split()
            self.files_name.extend(filename)
            
    def __len__(self):
        return len(self.files_name)
        
    def __getitem__(self,index):
        # Analyze Annotation Files(XML Format)
        anno_file_name = self.anno_folder + '/' + self.files_name[index] + '.xml'
        root = ET.parse(anno_file_name).getroot()
        objs = root.findall('object')
        ground_truths = []
        # CV2 HWC Format
        img_height = int(root.find('size').find('height').text)
        img_width = int(root.find('size').find('width').text)
        img_channel = int(root.find('size').find('depth').text)
        im_info = (img_height,img_width,img_channel)
        for obj in objs:
            labels = self.CLASSES.index(obj.find('name').text)
            xmin = float(obj.find('bndbox').find('xmin').text)
            xmax = float(obj.find('bndbox').find('xmax').text)
            ymin = float(obj.find('bndbox').find('ymin').text)
            ymax = float(obj.find('bndbox').find('ymax').text)
            temps = [xmin,ymin,xmax,ymax,labels]
            #print(temps)
            ground_truths.append(temps)
        ground_truths = torch.tensor(np.array(ground_truths)).float()
        # Analyze Image Files(JPEG Foramt)
        img_file_name = self.img_folder + '/' + self.files_name[index] + '.jpg'
        img = cv2.imread(img_file_name)
        b,g,r = cv2.split(img)
        img = np.array(cv2.merge([r,g,b]))
        img_tsr = torch.tensor(img).permute(2,0,1).float().cuda()
        
        """for gt in ground_truths:
            xmin = int(gt[0])
            xmax = int(gt[2])
            ymin = int(gt[1]) 
            ymax = int(gt[3])
            label = str(self.CLASSES[gt[4]])
            cv2.putText(img,label,(xmin,ymin),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),1)
        cv2.imshow('test',img)
        cv2.waitKey()"""
        
        return img_tsr.unsqueeze(0),ground_truths,im_info


if __name__ == '__main__':
    dataloader = faster_rcnn_loader('D:/0426DIOR/DIOR/VOCdevkit')
    print(len(dataloader))
    img,gts_with_cls = dataloader[6]