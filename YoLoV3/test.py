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
        self.net = net.YoLoNet().to(self.device)
        if not os.path.exists("./params/yolo.pkl"):
            raise FileNotFoundError
        self.net.load_state_dict(torch.load("./params/yolo.pkl"))
        "Data"
        self.test_file = test_file
        image = Image.open(test_file)
        self.data = utils.transform(image).unsqueeze(dim=0).to(self.device)

    def __transform__(self, out):
        out = out.transpose(1, -1)
        out = out.reshape(shape=(out.size(0), out.size(1), out.size(2), 3, -1))  # (N, W, H, 3, 5+cls)
        return out

    def __parse__(self, out, size):
        mask = out[..., 0] > cfg.THRESHOLD
        index = mask.nonzero()
        if len(index) == 0:
            return torch.tensor([]).to(self.device)
        info = out[mask]  # ndim = 2
        "Confi -- BCEWithLogitsLoss"
        confi = torch.sigmoid(info[:, 0])
        "Center -- BCEWithLogitsLoss"
        cx_offset, cy_offset = torch.sigmoid(info[:, 1]), torch.sigmoid(info[:, 2])
        cx_int, cy_int = index[:, 1].float(), index[:, 2].float()
        scale = cfg.IMG_SIZE/size
        cx, cy = (cx_int+cx_offset)*scale, (cy_int+cy_offset)*scale
        "W, H -- MSELoss"
        w_offset, h_offset = info[:, 3], info[:, 4]
        anchor_index = index[:, -1]  # nums of anchor_boxes
        anchor_boxes = torch.Tensor(cfg.ANCHOR_BOX[size]).to(self.device)
        w_anchor, h_anchor = anchor_boxes[anchor_index, 0], anchor_boxes[anchor_index, 1]
        w, h = w_anchor * torch.exp(w_offset), h_anchor * torch.exp(h_offset)
        "Coordinate"
        x1, y1, x2, y2 = cx-w//2, cy-h//2, cx+w//2, cy+h//2
        "Cls -- CrossEntropyLoss"
        cls = torch.argmax(info[:, 5:], dim=-1)
        return torch.stack([x1, y1, x2, y2, confi, cls.float()], dim=-1)

    def __select__(self, boxes):
        bbox = []
        cls_num = len(cfg.COCO_CLASS)
        for cls in range(cls_num):
            Cboxes = boxes[boxes[:, -1] == cls]
            if len(Cboxes) != 0:
                bbox.extend(utils.nms(Cboxes).astype(int))
            else:
                continue
        return bbox

    def predict(self):
        self.net.eval()
        out_13, out_26, out_52 = self.net(self.data)
        out_13, out_26, out_52 = self.__transform__(out_13), self.__transform__(out_26), self.__transform__(out_52)
        info_13 = self.__parse__(out_13, 13)
        info_26 = self.__parse__(out_26, 26)
        info_52 = self.__parse__(out_52, 52)
        info = torch.cat((info_13, info_26, info_52), dim=0)
        if len(info) == 0:
            raise Exception("Warning! no boxes on the current threshold!!!")
        boxes = self.__select__(info.cpu().detach().numpy())
        return utils.draw(boxes, self.test_file)


if __name__ == '__main__':
    file = "F:/YOLO_V3/train/000000103723.jpg"
    Test(file).predict()
        








