# -*-coding:utf-8-*-
import os
import cfg
import json
import utils
import torch
import numpy as np
import PIL.Image as Image
import torch.utils.data as data


def image_process(path, label, img_size):
    with open(label, 'w') as f:
        for filename in os.listdir(path+"/outputs"):
            filepath = path+"/outputs/"+filename
            imgname = filename.split(".")[0]+".jpg"
            imgpath = path+"/img/"+imgname

            file = json.load(open(filepath, "r"))
            message = file["outputs"]["object"][0]["bndbox"]
            x1, y1, x2, y2 = message.values()

            image = Image.open(imgpath)
            w, h = image.size
            w_scale, h_scale = w/img_size, h/img_size
            x1, y1, x2, y2 = x1/w_scale, y1/h_scale, x2/w_scale, y2/h_scale
            cx, cy = (x1+x2)/2, (y1+y2)/2
            w, h = x2-x1, y2-y1

            f.write("{} {} {} {} {} {}\n".format(imgname, filename[0], int(cx), int(cy), int(w), int(h)))


# if __name__ == '__main__':
#     path = "F:/L"
#     label = "F:/Face/label.txt"
#     img_size = 416
#     image_process(path, label, img_size)

    
class MyData(data.Dataset):
    def __init__(self, path):
        super(MyData, self).__init__()
        self.path = path
        label_path = path+"/label.txt"
        with open(label_path) as f:
            self.label = f.readlines()

    def __len__(self):
        return len(self.label)

    def __getlabel__(self, data):
        down_scale = 2**cfg.DOWN_NUM
        feature_map = int(cfg.IMG_SIZE/down_scale)

        label = np.zeros((feature_map, feature_map, 3, 5+1))

        cls, x, y, w, h = data
        "Center offset"
        (x_offset, x_index), (y_offset, y_index) = np.modf(x/down_scale), np.modf(y/down_scale)

        for i, anchor in enumerate(cfg.ANCHOR_BOX):
            "W, H offset"
            w_offset, h_offset = np.log(w/anchor[0]), np.log(h/anchor[1])  # b/p
            "Iou"
            inter_area = min(w, anchor[0]) * min(h, anchor[1])
            union_area = w * h + cfg.ANCHOR_AREA[i] - inter_area
            iou = inter_area / union_area
            "Lable"
            label[int(x_index), int(y_index), i] = \
                np.array([iou, x_offset, y_offset, w_offset, h_offset, int(cls)])
        return label

    def __getitem__(self, index):
        img_name, *data = self.label[index].split()
        img_path = self.path+"/img/"+img_name
        img = Image.open(img_path)
        img = utils.transform(img)
        data = np.array(data, dtype=np.float)
        label = self.__getlabel__(data)
        return img, torch.Tensor(label)


TRAIN = data.DataLoader(MyData(cfg.PATH), batch_size=cfg.BATCH, shuffle=True)





