# -*- coding:utf-8 -*-
import os
import cfg
import net
import torch
import utils
import numpy as np
import torch.nn as nn
import PIL.Image as Image


class Test:
    def __init__(self, path, params, cls_num):
        self.cls_num = cls_num
        self.device = torch.device("cuda")
        self.net = net.UNetPlusPlus(cls_num).to(self.device)
        if os.path.exists(params):
            self.net.load_state_dict(torch.load(params))
        else:
            raise FileNotFoundError("Please offer the params of Network!!!")
        self.image_origin = Image.open(path)
        self.data = utils.transform(self.image_origin).unsqueeze(dim=0)
        h, w = self.data.size(-2)//16*16, self.data.size(-1)//16*16
        self.data = nn.AdaptiveAvgPool2d((h, w))(self.data).to(self.device)

    def main(self):
        self.net.eval()
        with torch.no_grad():
            output = self.net(self.data)
            image = utils.fill(output, self.cls_num, cfg.COLOR).cpu().numpy()
            image_array = image.squeeze().transpose(1, 2, 0)
            image_separate = Image.fromarray(np.uint8(image_array))
            figure = Image.new("RGB", (640, 240), (255, 255, 255))
            figure.paste(self.image_origin, (0, 0))
            figure.paste(image_separate, (self.image_origin.size[0], 0))

        return figure.show()


if __name__ == '__main__':
    path = "./test_imgs/6.jpg"
    params = "./params/unet++.pkl"
    cls_num = 8
    Test(path, params, cls_num).main()

