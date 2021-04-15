import torch
import torch.nn as nn

class conv_bn_relu(nn.Module):
    def __init__(self,inplane,plane,kernel_size=3,stride=1,padding=1,dilation=1):
        super(conv_bn_relu,self).__init__()
        self.conv = nn.Conv2d(inplane,plane,kernel_size,stride,padding,dilation=dilation)
        self.bn = nn.BatchNorm2d(plane)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class vgg(nn.Module):
    arch_settings = {
        11:(1,1,2,2,2),
        13:(2,2,2,2,2),
        16:(2,2,3,3,3),
        19:(2,2,4,4,4)
    }
    def __init__(self,depth=16):
        super(vgg,self).__init__()
        self.inplane = 3
        self.depth = depth
        self.block_setting = self.arch_settings[depth]
        self.block1 = self._make_block(self.block_setting[0],64)
        self.block2 = self._make_block(self.block_setting[1],128)
        self.block3 = self._make_block(self.block_setting[2],256)
        self.block4 = self._make_block(self.block_setting[3],512)
        self.block5 = self._make_block(self.block_setting[4],512)
    def _make_block(self,block_num,plane):
        layers = []
        layers.append(conv_bn_relu(self.inplane,plane))
        self.inplane = plane
        for i in range(1,block_num):
            layers.append(conv_bn_relu(self.inplane,plane))
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True))
        return nn.Sequential(*layers)
        
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x
        
if __name__ == '__main__':
    model = vgg().cuda()
    tsr = torch.randn(size=(1,3,512,512)).cuda()
    out_tsr = model(tsr)
    print(out_tsr.size())