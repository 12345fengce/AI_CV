# -*- coding:utf-8 -*-
import os
import cfg
import net
import torch 
import utils
import numpy as np
import PIL.Image as Image


class Test:
    def __init__(self, test_file: str):
        "Device"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        "Net"
        self.net = net.MobileNet(len(cfg.NAME)).to(self.device)
        if not os.path.exists("./params/10_20.pkl"):
            raise FileNotFoundError
        self.net.load_state_dict(torch.load("./params/10_20.pkl"))
        "Data"
        self.test_file = test_file
        image = Image.open(test_file)
        self.data = utils.transform(image).unsqueeze(dim=0).to(self.device)

    def __transform__(self, out):
        out = out.transpose(1, -1)
        out = out.reshape(shape=(out.size(0), out.size(1), out.size(2), 3, -1))  # (N, W, H, 3, 5+cls)
        return out

    def __parse__(self, out):
        mask = out[..., 0] > cfg.THRESHOLD
        index = mask.nonzero()
        if len(index) == 0:
            return torch.tensor([]).to(self.device)
        info = out[mask]  # ndim = 2
        "Confi -- MSELoss"
        confi = info[:, 0]
        "Center -- MSELoss"
        cx_offset, cy_offset = info[:, 1], info[:, 2]
        cx_int, cy_int = index[:, 1].float(), index[:, 2].float()
        scale = 2**cfg.DOWN_NUM
        cx, cy = (cx_int+cx_offset)*scale, (cy_int+cy_offset)*scale
        "W, H -- MSELoss"
        w_offset, h_offset = info[:, 3], info[:, 4]
        anchor_index = index[:, -1]  # nums of anchor_boxes
        anchor_boxes = torch.Tensor(cfg.ANCHOR_BOX).to(self.device)
        w_anchor, h_anchor = anchor_boxes[anchor_index, 0], anchor_boxes[anchor_index, 1]
        w, h = w_anchor * torch.exp(w_offset), h_anchor * torch.exp(h_offset)
        "Coordinate"
        x1, y1, x2, y2 = cx-w/2, cy-h/2, cx+w/2, cy+h/2
        "Cls -- CrossEntropyLoss"
        cls = torch.argmax(info[:, 5:], dim=-1)
        return torch.stack([x1, y1, x2, y2, cls.float(), confi], dim=-1)

    def __select__(self, boxes):
        bbox = []
        cls_num = len(cfg.NAME)
        for cls in range(cls_num):
            Cboxes = boxes[boxes[:, -2] == cls]  # 倒数第二位为cls
            if len(Cboxes) != 0:
                bbox.extend(utils.nms_filter(Cboxes, 0.3))
            else:
                continue
        return bbox

    def predict(self):
        self.net.eval()
        outputs = self.net(self.data)
        outputs = self.__transform__(outputs)
        info = self.__parse__(outputs)
        if len(info) == 0:
            raise Exception("Warning! no boxes on the current threshold!!!")
        boxes = self.__select__(info.cpu())
        return utils.draw(boxes, self.test_file)


if __name__ == '__main__':
    path = "F:/Face/img"
    for img_name in os.listdir(path):
        img_path = path+"/"+img_name
        Test(img_path).predict()
        








