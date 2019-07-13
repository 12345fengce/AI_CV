# -*-coding:utf-8-*-
import os
import net
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
    def __init__(self, p_path, r_path, o_path, img_file):
        self.img_file = img_file
        self.image = Image.open(img_file)  # 用于抠图输入下一层
        self.img = self.image.crop((0, 0, self.image.size[0], self.image.size[1])).convert("RGB")  # 复制图片用于图像金字塔
        self.pnet = torch.load(p_path).cpu()
        self.rnet = torch.load(r_path).cpu()
        self.onet = torch.load(o_path).cpu()
    
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
        r_prior = []  # 收集PNet已知角坐标信息  R的先验
        r_data = []  # 收集R网络输入数据
        coordinates = []  # 作图坐标
        count = 0  # 计数
        while min(self.img.size) > 12:
            scal = 0.707**count  # 缩放比例，可以还原到原图  0.707为面积的一半
            input = tf.ToTensor()(self.img).unsqueeze(dim=0)  
            confi, offset = self.pnet(input)
            W = offset.size(3)  # 取出图片的w值
            confi = confi.permute(0, 2, 3, 1)
            confi = confi.reshape(-1).data.numpy()  
            offset = offset.permute(0, 2, 3, 1)  # 换轴，将四个通道数据组合到一起
            offset = offset.reshape((-1, 4)).data.numpy()
    
            o_index = np.arange(len(offset)).reshape(-1, 1)  # 特征图W_out*H_out
            offset, o_index, confi = offset[confi >= 0.9], o_index[confi >= 0.9], confi[confi >= 0.9]  
           
            y_index, x_index = divmod(o_index, W)  # 索引/w  在特征图中对应索引为（x，y）=（余数， 商）
            x1, y1, x2, y2 = x_index*2/scal, y_index*2/scal, (x_index*2+12)/scal, (y_index*2+12)/scal  # 左上角=索引*步长  右上角=左上角+边长
            p_prior = np.hstack((x1, y1, x2, y2))  # 将原图坐标组合为一个二维数组
            offset = utils.transform(offset, p_prior)  
            
            boxes = np.hstack((offset, np.expand_dims(confi, axis=1)))  # 将偏移量与置信度结合，进行NMS
            boxes = utils.NMS(boxes, threshold=0.3, ismin=False) 
            coordinates.extend(boxes.tolist())
            if boxes.shape[0] == 0:
                break

            data, prior = utils.crop_to_square(boxes, 24, self.image)
            r_prior.extend(prior)
            r_data.extend(data)
            self.img = self.pyramid()  # 图像金字塔
            count += 1  

        r_prior = np.stack(r_prior, axis=0)  # 数据重组，重新装载为numpy和tensor
        r_data = torch.stack(r_data, dim=0)
        print("PNet create {} candidate items".format(r_data.size(0)))
        utils.draw(np.stack(coordinates, axis=0), self.img_file, "PNet")
        return r_data,  r_prior
        
    def r(self):
        """transform out of tensor to numpy
            filter with confidence
            calculate coordinates
            filter with NMS
            crop image from original image for ONet's input
            draw"""
        data, prior = self.p()
        confi, offset = self.rnet(data)  
        confi = confi.data.numpy().flatten()
        offset = offset.data.numpy()

        offset, prior, confi = offset[confi >= 0.99], prior[confi >= 0.99], confi[confi >= 0.99]  
    
        offset = utils.transform(offset, prior)
        del prior, data

        boxes = np.hstack((offset, np.expand_dims(confi, axis=1)))  
        boxes = utils.NMS(boxes, threshold=0.3, ismin=False)
        coordinates = boxes
        
        o_data, o_prior = utils.crop_to_square(boxes, 48, self.image)

        o_prior = np.stack(o_prior, axis=0)  
        o_data = torch.stack(o_data, dim=0)
        print("RNet create {} candidate items".format(o_data.size(0)))
        utils.draw(np.stack(coordinates, axis=0), self.img_file, "RNet")
        return o_data, o_prior
    
    def o(self):
        """transform out of tensor to numpy
            filter with confidence
            calculate coordinates
            filter with NMS
            draw"""
        data, prior = self.r()
        confi, offset = self.onet(data)  
        confi = confi.data.numpy().flatten()
        offset = offset.data.numpy()

        offset, prior, confi = offset[confi >= 0.999], prior[confi >= 0.999], confi[confi >= 0.999] 

        offset = utils.transform(offset, prior) 

        boxes = np.hstack((offset, np.expand_dims(confi, axis=1)))  # 将偏移量与置信度结合，进行NMS
        boxes = utils.NMS(boxes, threshold=0.3, ismin=True)

        coordinates = boxes
        print("ONet create {} candidate items".format(boxes.shape[0]))
        utils.draw(np.stack(coordinates, axis=0), self.img_file, "ONet")

    
if __name__ == "__main__":
    p_path = "f:/project/code/MTCNN/pnet.pth"
    r_path = "f:/project/code/MTCNN/rnet.pth"
    o_path = "f:/project/code/MTCNN/onet.pth"
    i = 0
    while i < 11:
        img_file = "f:/project/code/MTCNN/{}.jpg".format(i)
        print("\ntest - {} :".format(i+1))
        print("**************************************************")
        try:
            test = Test(p_path, r_path, o_path, img_file)
            test.o()
            i += 1
        except:
            print("No faces found! Please check your code!!!")
            i += 1













            


    



