# -*-coding:utf-8-*-
import sys
import torch 
import utils
import numpy as np
import PIL.Image as Image
import torchvision.transforms as tf 

class Test:
    def __init__(self, net_path, img_path):
        self.net = torch.load(net_path)
        self.net.eval()
        self.img = Image.open(img_path)
    def transform(self, out):
        out = out.transpose(1, -1)
        out = out.reshape(shape=(out.size(0), out.size(1), out.size(2), 3, -1))
        return out
    def predict(self):
        img = tf.ToTensor()(self.img).unsqueeze(dim=0)
        out_13, out_26, out_52 = self.net(img.cuda())
        out_13, out_26, out_52 = self.transform(out_13), self.transform(out_26), self.transform(out_52)
        coordinates_13 =  utils.parse(out_13, 13)
        coordinates_26 =  utils.parse(out_26, 26)
        coordinates_52 =  utils.parse(out_52, 52)
        coordinates_13.extend(coordinates_26)
        coordinates_13.extend(coordinates_52)
        return np.stack(coordinates_13)

# if __name__ == "__main__":
#     net_path = "F:/Project/Code/YoLoV3/yolo_net.pth"
#     image_path = "G:/Yolo_train/img/train0.jpg"
#     Test(net_path, image_path).predict()

        








