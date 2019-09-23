# -*- coding:utf-8 -*-
import os
import net
import time
import torch
import utils
import numpy as np 
import PIL.Image as Image
import torchvision.transforms as tf


class Test:
    """pyramid
        p: p_net
        r: r_net
        o: o_net"""
    def __init__(self, para_p, para_r, para_o, test_img):
        self.device = torch.device("cuda")
        self.test_img = test_img
        self.image = Image.open(test_img)  # 用于抠图输入下一层
        self.img = Image.open(test_img)  # 复制图片用于图像金字塔

        self.pnet = net.PNet().to(self.device)
        self.pnet.load_state_dict(torch.load(para_p))
        self.pnet.eval()

        self.rnet = net.RNet().to(self.device)
        self.rnet.load_state_dict(torch.load(para_r))
        self.rnet.eval()

        self.onet = net.ONet().to(self.device)
        self.onet.load_state_dict(torch.load(para_o))
        self.onet.eval()

    def pyramid(self, scal=0.707):
        "resize the image to smaller size"
        w, h = self.img.size
        self.img = self.img.resize((int(scal*w), int(scal*h)))
        return self.img
    
    def p(self):
        """transform out of tensor to numpy
            filter with confidence
            calculate coordinates
            filter with NMS
            crop image from original image for RNet's input
            draw"""
        r_prior, r_data = [], []  # collect RNet's prior, RNet's input
        coordinates = []  # collect coordinates for draw
        count = 0
        start_time = time.time()
        while min(self.img.size) > 12:
            scal = 0.707**count  # 缩放比例，可以还原到原图  0.707为面积的一半
            input = tf.ToTensor()(self.img).unsqueeze(dim=0)-0.5
            with torch.no_grad():
                confi, offset = self.pnet(input.cuda())
            W = offset.size(3)  # 取出图片的w值
            confi = confi.permute(0, 2, 3, 1)
            confi = confi.reshape(-1).cpu().numpy()
            offset = offset.permute(0, 2, 3, 1)  # 换轴，将四个通道数据组合到一起
            offset = offset.reshape((-1, 14)).cpu().numpy()

            o_index = np.arange(len(offset)).reshape(-1, 1)  # 特征图W_out*H_out
            offset, o_index, confi = offset[confi >= 0.9], o_index[confi >= 0.9], confi[confi >= 0.9]
           
            y_index, x_index = divmod(o_index, W)  # 索引/w  在特征图中对应索引为（x，y）=（余数， 商）
            x1, y1, x2, y2 = x_index*2/scal, y_index*2/scal, (x_index*2+12)/scal, (y_index*2+12)/scal  # 左上角=索引*步长  右上角=左上角+边长
            p_prior = np.hstack((x1, y1, x2, y2))  # 将原图坐标组合为一个二维数组
            offset, landmarks = offset[:, :4], offset[:, 4:]
            offset, landmarks = utils.transform(offset, landmarks, p_prior)
            
            boxes = np.hstack((offset, np.expand_dims(confi, axis=1), landmarks))  # 将偏移量与置信度结合，进行NMS
            boxes = utils.NMS(boxes, threshold=0.7, ismin=False)
            coordinates.extend(boxes.tolist())
            if boxes.shape[0] == 0:
                break

            data, prior = utils.crop_to_square(boxes[:, :5], 24, self.image)
            r_prior.extend(prior)
            r_data.extend(data)
            self.img = self.pyramid()  # 图像金字塔
            count += 1  

        r_prior = np.stack(r_prior, axis=0)  # 数据重组，重新装载为numpy和tensor
        r_data = torch.stack(r_data, dim=0)
        end_time = time.time()
        print("PNet create {} candidate items\ncost {}s!".format(r_data.size(0), end_time - start_time))
        utils.draw(np.stack(coordinates, axis=0), self.test_img, "PNet")
        return r_data,  r_prior
        
    def r(self):
        """transform out of tensor to numpy
            filter with confidence
            calculate coordinates
            filter with NMS
            crop image from original image for ONet's input
            draw"""
        start_time = time.time()
        data, prior = self.p()
        with torch.no_grad():
            confi, offset = self.rnet(data.cuda())
        confi = confi.cpu().numpy().flatten()
        offset = offset.cpu().numpy()

        offset, prior, confi = offset[confi >= 0.99], prior[confi >= 0.99], confi[confi >= 0.99]

        offset, landmarks = offset[:, :4], offset[:, 4:]
        offset, landmarks = utils.transform(offset, landmarks, prior)

        boxes = np.hstack((offset, np.expand_dims(confi, axis=1), landmarks))
        boxes = utils.NMS(boxes, threshold=0.6, ismin=False)

        o_data, o_prior = utils.crop_to_square(boxes[:, :5], 48, self.image)

        o_prior = np.stack(o_prior, axis=0)
        o_data = torch.stack(o_data, dim=0)
        end_time = time.time()
        print("RNet create {} candidate items\ncost {}s!".format(o_data.size(0), end_time - start_time))
        utils.draw(boxes, self.test_img, "RNet")
        return o_data, o_prior
    
    def o(self):
        """transform out of tensor to numpy
            filter with confidence
            calculate coordinates
            filter with NMS
            draw"""
        data, prior = self.r()
        confi, offset = self.onet(data.cuda())
        confi = confi.data.cpu().numpy().flatten()
        offset = offset.data.cpu().numpy()
        offset, prior, confi = offset[confi >= 0.999], prior[confi >= 0.999], confi[confi >= 0.999]
        offset, landmarks = offset[:, :4], offset[:, 4:]
        offset, landmarks = utils.transform(offset, landmarks, prior)

        boxes = np.hstack((offset, np.expand_dims(confi, axis=1), landmarks))  # 将偏移量与置信度以及landmarks结合，进行NMS
        boxes = utils.NMS(boxes, threshold=0.4, ismin=True)

        print("ONet create {} candidate items".format(boxes.shape[0]))
        utils.draw(boxes, self.test_img, "ONet")

    
if __name__ == "__main__":
    p_path = "./params/pnet.pkl"
    r_path = "./params/rnet.pkl"
    o_path = "./params/onet.pkl"
    i = 24
    while i < 31:
        test_img = "F:/MTCNN/test/{}.jpg".format(i)
        print("\ntest - {} :".format(i+1))
        print("**************************************************")
        try:
            test = Test(p_path, r_path, o_path, test_img)
            test.o()
            i += 1
        except:
            print("No faces found! Please check your code!!!")
            i += 1













            


    



