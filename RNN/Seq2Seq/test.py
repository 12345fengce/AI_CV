# -*- coding:utf-8 -*-
import os
import net
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as tf


PATH = "F:/MTCNN/test/"
DEVICE = torch.device("cuda")
validation = net.MainNet().to(DEVICE)
validation.extractor.load_state_dict(torch.load("G:/Project/Code/RNN/Seq2Seq/params/extractor.pkl"))
validation.generator.load_state_dict(torch.load("G:/Project/Code/RNN/Seq2Seq/params/generator.pkl"))
validation.eval()
for i in range(24):
    img_name = PATH+"{}.jpg".format(i)
    img = Image.open(img_name)
    x = tf.ToTensor()(img).unsqueeze(0)
    y = validation(x.cuda()).squeeze()
    _img = y.cpu().detach().numpy()*255
    _img = _img.transpose(1, 2, 0)
    new = Image.fromarray(np.uint8(_img))
    new.show()

