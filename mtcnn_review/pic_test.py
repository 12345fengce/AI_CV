import os
import cfg
import net
import time
import torch
import functions
import PIL.Image as Image


class Test:
    def __init__(self):
        "Device"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        "Net"
        self.pnet = net.PNet().to(self.device)
        self.rnet = net.RNet().to(self.device)
        self.onet = net.ONet().to(self.device)
        if not os.path.exists("./params"):
            raise FileNotFoundError
        self.pnet.load_state_dict(torch.load("./params/pnet.pkl"))
        self.rnet.load_state_dict(torch.load("./params/rnet.pkl"))
        # self.onet.load_state_dict(torch.load("./params/onet.pkl"))

    def detect(self, image: str):
        "PNet"
        start_time = time.time()
        pboxes = self.p_detect(image)
        if len(pboxes) == 0:
            return "No faces found! Please check your image!!!"
        end_time = time.time()
        print("PNet cost {}ms!!!".format(end_time-start_time)*1000)
        "RNet"
        start_time = time.time()
        rprior, rdata = self.__crop2square__(image, pboxes, cfg.NET["rnet"])
        rboxes = self.ro_detect(rprior, rdata, self.rnet)
        if len(rboxes) == 0:
            return "No faces found! Please check your PNet'images!!!"
        end_time = time.time()
        print("RNet cost {}ms!!!".format(end_time - start_time) * 1000)
        "ONet"
        # start_time = time.time()
        # oprior, odata = self.__crop2square__(image, rboxes, cfg.NET["onet"])
        # oboxes = self.ro_detect(oprior, odata, self.onet)
        # if len(oboxes) == 0:
        #     return "No faces found! Please check your RNet'images!!!"
        # end_time = time.time()
        # print("ONet cost {}ms!!!".format(end_time - start_time) * 1000)

    def p_detect(self, image: str):
        self.pnet.eval()
        pboxes = torch.Tensor([]).to(self.device)
        image = Image.open(image)
        while min(image.size) > 12:
            "Scale"
            scale = 1
            "Data"
            data = functions.transform(image).unsqueeze(0).to(self.device)
            "Net"
            confi, offset, _ = self.pnet(data)
            confi, offset = confi.permute(0, 2, 3, 1), offset.permute(0, 2, 3, 1)  # (N, W, H, C)
            mask = confi[..., 0] > cfg.CONFI["pnet"]
            index = torch.nonzero(mask)
            "ROI"
            side, stride = cfg.NET["pnet"]
            x1, y1 = index[:, 1]*stride, index[:, 2]*stride
            x2, y2 = x1+side, y1+stride
            roi = torch.stack([x1, y1, x2, y2], dim=-1)
            "Origin"
            confi, offset = confi[mask], offset[mask]
            origin = offset*side/scale+roi.float()
            box = torch.cat((confi, origin), dim=-1)
            pboxes = torch.cat((pboxes, box), dim=0)
            "Pyramid"
            scale *= 0.707
            image = image.resize((image.size[0], image.size[1]))
        return functions.nms(pboxes)

    def ro_detect(self, prior, data, model):
        prior, data = prior.to(self.device), data.to(self.device)
        confi, offset, landmark = model(data)
        confi, offset, landmark = confi.permute(0, 2, 3, 1), offset.permute(0, 2, 3, 1), landmark.permute(0, 2, 3, 1)
        mask = confi[..., 0] > cfg.CONFI[str(model).lower()]
        confi, offset, landmark = confi[mask], offset[mask], landmark[mask]
        side = prior[:, 2:3]-prior[:, :1]
        coordinate = offset*side+prior
        cx, cy = (prior[:, :1]+prior[:, 2:3])/2, (prior[:, 1:2]+prior[3:])/2
        landmark[:, ::2] = landmark[:, ::2]*side+cx
        landmark[:, 1::2] = landmark[:, 1::2]*side+cy
        boxes = torch.cat((confi, coordinate, landmark), dim=-1)
        return functions.nms(boxes)

    def __crop2square__(self, image, boxes, size):
        data, prior = [], []
        image = Image.open(image)
        x1, y1, x2, y2 = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
        cx, cy = (x2 - x1) / 2, (y2 - y1) / 2
        mside = max((x2 - x1), (y2 - y1))
        __x1, __y1 = (cx - m / 2).int(), (cy - m / 2).int()
        __x2, __y2 = __x1 + mside, __y1 + mside
        for i in range(boxes.size(0)):
            prior.append([__x1[i], __y1[i], __x2[i], __y2[i]])
            __data = image.crop(prior[-1])
            __data = __data.resize((size, size))
            __data = functions.transform(__data).unsqueeze(0)
            data.append(__data)
        return torch.stack(prior), torch.stack(data)


if __name__ == '__main__':
    image = "F:/mtcnn_v1/test/4.jpg"
    Test().detect(image)