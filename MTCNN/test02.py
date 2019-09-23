# -*-coding:utf-8-*-
import os
import cv2
import net
import torch
import utils
import datetime
import numpy as np 
import PIL.Image as Image
import torchvision.transforms as tf


class Test:
    """pyramid
        p: p_net
        r: r_net
        o: o_net"""
    def __init__(self, test_img):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image = Image.fromarray(np.uint8(test_img[:, :, [2, 1, 0]]))  # for croped  transform cv2 to PIL
        self.img = self.image  # for pyramid
        
        self.pnet = net.PNet().to(self.device)
        self.pnet.load_state_dict(torch.load("./params/pnet.pkl"))
        self.pnet.eval()
        
        self.rnet = net.RNet().to(self.device)
        self.rnet.load_state_dict(torch.load("./params/rnet.pkl"))
        self.rnet.eval()

        self.onet = net.ONet().to(self.device)
        self.onet.load_state_dict(torch.load("./params/onet.pkl"))
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
        start_time = datetime.datetime.now()
        r_prior, r_data = [], []  # collect RNet's prior, RNet's input
        coordinates = []  # collect coordinates for draw
        count = 0
        while min(self.img.size) > 12:
            scal = 0.707**count  # 0.707 make the area half of origin image
            input = tf.ToTensor()(self.img).unsqueeze(dim=0)-0.5
            with torch.no_grad():
                confi, offset = self.pnet(input.cuda())
            confi, offset = confi.transpose(1, -1), offset.transpose(1, -1)

            mask = confi[..., 0] > 0.9
            confi = confi[mask].cpu().numpy()  # filter confi
            offset = offset[mask].cpu().numpy()  # filter offset

            index = mask.nonzero().cpu().numpy()  # index 
            x_index, y_index = index[:, 1:2], index[:, 2:3]
            x1, y1, x2, y2 = x_index*2/scal, y_index*2/scal, (x_index*2+12)/scal, (y_index*2+12)/scal  # top_left*scal=index*stride  bottom_right*scal=top_left+12
            p_prior = np.hstack(([x1, y1, x2, y2]))  # translate to numpy which ndim=2

            offset, landmarks = offset[:, :4], offset[:, 4:]
            offset, landmarks = utils.transform(offset, landmarks, p_prior)
            
            boxes = np.hstack((offset, confi, landmarks))  # [[offset+confi+landmarks]] for NMS
            boxes = utils.NMS(boxes, threshold=0.7, ismin=False)
            coordinates.extend(boxes.tolist())
            if boxes.shape[0] == 0:
                break

            data, prior = utils.crop_to_square(boxes[:, :5], 24, self.image)
            r_prior.extend(prior)
            r_data.extend(data)
            self.img = self.pyramid()  
            count += 1  

        r_prior = np.stack(r_prior, axis=0)  
        r_data = torch.stack(r_data, dim=0)
        end_time = datetime.datetime.now()
        print("PNet cost {}ms".format((end_time - start_time).microseconds/1000))
        return r_data,  r_prior
        
    def r(self):
        """transform out of tensor to numpy
            filter with confidence
            calculate coordinates
            filter with NMS
            crop image from original image for ONet's input
            draw"""
        start_time = datetime.datetime.now()
        data, prior = self.p()
        with torch.no_grad():
            confi, offset = self.rnet(data.cuda())
        confi = confi.cpu().numpy().flatten()
        offset = offset.cpu().numpy()

        offset, prior, confi = offset[confi > 0.99], prior[confi > 0.99], confi[confi > 0.99]

        offset, landmarks = offset[:, :4], offset[:, 4:]
        offset, landmarks = utils.transform(offset, landmarks, prior)

        boxes = np.hstack((offset, np.expand_dims(confi, axis=1), landmarks))
        boxes = utils.NMS(boxes, threshold=0.6, ismin=False)
        
        o_data, o_prior = utils.crop_to_square(boxes[:, :5], 48, self.image)

        o_prior = np.stack(o_prior, axis=0)  
        o_data = torch.stack(o_data, dim=0)
        end_time = datetime.datetime.now()
        print("RNet cost {}ms".format((end_time - start_time).microseconds/1000))
        return o_data, o_prior
    
    def o(self):
        """transform out of tensor to numpy
            filter with confidence
            calculate coordinates
            filter with NMS
            draw"""
        start_time = datetime.datetime.now()
        data, prior = self.r()
        with torch.no_grad():
            confi, offset = self.onet(data.cuda())
        confi = confi.cpu().numpy().flatten()
        offset = offset.cpu().numpy()

        offset, prior, confi = offset[confi >= 0.999], prior[confi >= 0.999], confi[confi >= 0.999]

        offset, landmarks = offset[:, :4], offset[:, 4:]
        offset, landmarks = utils.transform(offset, landmarks, prior)

        boxes = np.hstack((offset, np.expand_dims(confi, axis=1), landmarks))  # 将偏移量与置信度结合，进行NMS
        boxes = utils.NMS(boxes, threshold=0.4, ismin=True)
        end_time = datetime.datetime.now()
        print("ONet cost {}ms".format((end_time - start_time).microseconds/1000))
        return boxes

    
if __name__ == "__main__":
    FILE = "F:/MTCNN/test/video2.mp4"
    FUNC = Test
    utils.show(FILE, FUNC, 20)















            


    



