import torch
from torchvision.ops import RoIPool

img = torch.randn(size=(1,3,100,100))
box = torch.tensor([10.,10.,20.,20.]).unsqueeze(0)
#box2 = torch.tensor([30.,30.,70.,70.]).unsqueeze(0)
boxes = [box]
#print(boxes)
pooler = RoIPool(output_size=3,spatial_scale=1.)
#print(pooler)
out_tsr = pooler(img,boxes)
print(out_tsr.size())