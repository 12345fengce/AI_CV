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
        self.image = Image.open(img_file)  # for croped
        self.img = Image.open(img_file)  # for pyramid
        self.pnet = torch.load(p_path).eval()
        self.rnet = torch.load(r_path).eval()
        self.onet = torch.load(o_path).eval()
    
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
        r_prior = []  # collect RNet's prior
        r_data = []  # collect RNet's input
        coordinates = []  # collect coordinates for draw
        count = 0  
        while min(self.img.size) > 12:
            scal = 0.707**count  # 0.707 make the area half of origin image
            input = tf.ToTensor()(self.img).unsqueeze(dim=0)  
            confi, offset = self.pnet(input.cuda())
            confi = confi.transpose(1, -1)
            mask = confi[..., 0] >= 0.9
            confi = confi[mask].data.cpu().numpy()  # filter confi

            offset = offset.transpose(1, -1)
            offset = offset[mask].data.cpu().numpy()  # filter offset
            index = mask.nonzero().cpu().numpy()  # index 
            x_index, y_index = index[:, 1:2], index[:, 2:3]
            x1, y1, x2, y2 = x_index*2/scal, y_index*2/scal, (x_index*2+12)/scal, (y_index*2+12)/scal  # top_left*scal=index*stride  bottom_right*scal=top_left+12
            p_prior = np.hstack(([x1, y1, x2, y2]))  # translate to numpy which ndim=2
            offset = utils.transform(offset, p_prior)  
            
            boxes = np.hstack((offset, confi))  # [[offset+confi]] for NMS
            boxes = utils.NMS(boxes, threshold=0.3, ismin=False) 
            coordinates.extend(boxes.tolist())
            if boxes.shape[0] == 0:  # for the case which can not get any box of confi >= 0.9
                break

            data, prior = utils.crop_to_square(boxes, 24, self.image)
            r_prior.extend(prior)
            r_data.extend(data)
            self.img = self.pyramid()  
            count += 1  

        r_prior = np.stack(r_prior, axis=0)  
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
        confi, offset = self.rnet(data.cuda())  
        confi = confi.data.cpu().numpy().flatten()
        offset = offset.data.cpu().numpy()

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
        confi, offset = self.onet(data.cuda())  
        confi = confi.data.cpu().numpy().flatten()
        offset = offset.data.cpu().numpy()

        offset, prior, confi = offset[confi >= 0.999], prior[confi >= 0.999], confi[confi >= 0.999] 

        offset = utils.transform(offset, prior) 

        boxes = np.hstack((offset, np.expand_dims(confi, axis=1)))  # 将偏移量与置信度结合，进行NMS
        boxes = utils.NMS(boxes, threshold=0.3, ismin=True)
        print(boxes[..., -1])
        coordinates = boxes
        print("ONet create {} candidate items".format(boxes.shape[0]))
        utils.draw(np.stack(coordinates, axis=0), self.img_file, "ONet")

    
if __name__ == "__main__":
    p_path = "G:/for_MTCNN/test/pnet.pth"
    r_path = "G:/for_MTCNN/test/rnet.pth"
    o_path = "G:/for_MTCNN/test/onet.pth"
    i = 0
    while i < 21:
        img_file = "G:/for_MTCNN/test/{}.jpg".format(i)
        print("\ntest - {} :".format(i+1))
        print("**************************************************")
        try:
            test = Test(p_path, r_path, o_path, img_file)
            test.o()
            i += 1
        except:
            print("No faces found! Please check your code!!!")
            i += 1













            


    



