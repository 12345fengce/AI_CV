# -*-coding:utf-8-*-
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_img = test_img
        self.image = Image.open(test_img)  # for croped
        self.img = Image.open(test_img)  # for pyramid
        
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
            scal = 0.707**count  # 0.707 make the area half of origin image
            input = tf.ToTensor()(self.img).unsqueeze(dim=0)-0.5
            with torch.no_grad():
                confi, offset = self.pnet(input.cuda())
            confi = confi.transpose(1, -1)

            mask = confi[..., 0] >= 0.9
            confi = confi[mask].cpu().numpy()  # filter confi

            offset = offset.transpose(1, -1)
            offset = offset[mask].cpu().numpy()  # filter offset

            index = mask.nonzero().cpu().numpy()  # index 
            x_index, y_index = index[:, 1:2], index[:, 2:3]
            x1, y1, x2, y2 = x_index*2/scal, y_index*2/scal, (x_index*2+12)/scal, (y_index*2+12)/scal  # top_left*scal=index*stride  bottom_right*scal=top_left+12
            p_prior = np.hstack(([x1, y1, x2, y2]))  # translate to numpy which ndim=2

            offset, landmarks = offset[:, :4], offset[:, 4:]
            offset, landmarks = utils.transform(offset, landmarks, p_prior)
            
            boxes = np.hstack((offset, confi, landmarks))  # [[offset+confi+landmarks]] for NMS
            boxes = utils.NMS(boxes, threshold=0.3, ismin=False) 
            coordinates.extend(boxes.tolist())
            if boxes.shape[0] == 0:  # for the case which can not get any box of confi >= 0.9
                break

            data, prior = utils.crop_to_square(boxes[:, :5], 24, self.image)
            r_prior.extend(prior)
            r_data.extend(data)
            self.img = self.pyramid()  
            count += 1  

        r_prior = np.stack(r_prior, axis=0)  
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
        boxes = utils.NMS(boxes, threshold=0.3, ismin=False)
        
        o_data, o_prior = utils.crop_to_square(boxes[:, :5], 48, self.image)

        o_prior = np.stack(o_prior, axis=0)  
        o_data = torch.stack(o_data, dim=0)
        end_time = time.time()
        print("RNet create {} candidate items\ncost {}s!".format(o_data.size(0), end_time-start_time))
        utils.draw(boxes, self.test_img, "RNet")
        return o_data, o_prior
    
    def o(self):
        """transform out of tensor to numpy
            filter with confidence
            calculate coordinates
            filter with NMS
            draw"""
        start_time = time.time()
        data, prior = self.r()
        with torch.no_grad():
            confi, offset = self.onet(data.cuda())
        confi = confi.cpu().numpy().flatten()
        offset = offset.cpu().numpy()

        offset, prior, confi = offset[confi >= 0.999], prior[confi >= 0.999], confi[confi >= 0.999]

        offset, landmarks = offset[:, :4], offset[:, 4:]
        offset, landmarks = utils.transform(offset, landmarks, prior)

        boxes = np.hstack((offset, np.expand_dims(confi, axis=1), landmarks))  # 将偏移量与置信度结合，进行NMS
        boxes = utils.NMS(boxes, threshold=0.3, ismin=True)
        print("ONet create {} candidate items".format(boxes.shape[0]))
        utils.draw(boxes, self.test_img, "ONet")

    
if __name__ == "__main__":
    para_p = "F:/MTCNN/test/pnet.pth"
    para_r = "F:/MTCNN/test/rnet.pth"
    para_o = "F:/MTCNN/test/onet.pth"
    i = 0
    while i < 25:
        test_img = "F:/MTCNN/test/{}.jpg".format(i)
        print("\ntest - {} :".format(i+1))
        print("**************************************************")
        try:
            test = Test(para_p, para_r, para_o, test_img)
            test.o()
            i += 1
        except:
            print("No faces found! Please check your code!!!")
            i += 1













            


    



