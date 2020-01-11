import pdb
import torch
from pytorch_pyramids import MSDPyramidNet

gPyramid = MSDPyramidNet()#.cuda()

img = torch.randn(1, 1, 256, 256)#.cuda()

img_lp, img_hp = gPyramid.transform(img)
img2 = gPyramid.itransform(img_lp, img_hp)

err = ((img-img2)**20).sum()**.05

print('img size:', img.size())
print('img2 size:', img2.size())
print('img_lp size:', img_lp.size())
print('img_hp size:', img_hp.size())
print('Error:', err)

