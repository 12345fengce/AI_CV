# -*- coding:utf-8 -*-
import os
import net
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as tf


PATH = "F:/MTCNN/train/48/positive/"
DEVICE = torch.device("cuda")
validation = net.HighLight().to(DEVICE)
validation.load_state_dict(torch.load("G:/Project/Code/RNN/Attention/params.pkl"))
validation.eval()
for i in range(24):
    img_name = PATH+"{}.jpg".format(i)
    img = Image.open(img_name)
    x = tf.ToTensor()(img).unsqueeze(0)
    mask, confi = validation(x.cuda())
    print("confi:", confi.item())
    _img = (mask.cpu()*x*255).data.numpy()
    _img = _img.squeeze().transpose(1, 2, 0)
    new = Image.fromarray(np.uint8(_img))
    new.show()

