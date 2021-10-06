import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
from torchvision import utils,transforms
from PIL import Image
class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        x = F.conv2d(x.unsqueeze(0), self.weight, padding=2, groups=self.channels)
        return x

input_img = Image.open('.jpg') #加入图片

trans = transforms.ToTensor()
input_img = trans(input_img)

gaussian_conv = GaussianBlurConv()
out_x = gaussian_conv(input_img)

input_img=input_img.unsqueeze(0)
out2 = input_img-out_x
out2 = torch.squeeze(out2)
out_x = torch.squeeze(out_x)
#out2=cv2.medianBlur(out2,5)
#cv2.imshow("out_x", out2)
out2 = transforms.ToPILImage()(out2)
img1=transforms.ToPILImage()(out_x)
out2.save("gaosi_out.jpg")
img1.save("gaosi.jpg")
