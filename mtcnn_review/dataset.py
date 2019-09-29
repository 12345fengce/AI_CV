# -*- coding:utf-8 -*-
import os
import cfg
import cv2
import torch
import random
import functions
import numpy as np
import PIL.Image as Image
import torch.utils.data as data
import torchvision.transforms as tf


def gen(data: str, target: str, landmark: str, type: str, goal: str):
    # for size in cfg.NET.values():
    for size in [48, 24, 12]:
        "创建文件夹"
        path = goal+"/%d/"%size+type
        if not os.path.exists(path):
            os.makedirs(path)
        label = open(path+".txt", "a")
        "读取标签信息"
        with open(target) as f:
            targets = f.readlines()
        with open(landmark) as f:
            landmarks = f.readlines()
        for i in range(len(targets)):
            if i < 2:
                continue
            __name, *__box = targets[i].split()

            __image = Image.open(os.path.join(data, __name))

            __box = np.array( __box, dtype=np.int)
            __min = min(__box[2:])
            if np.sum(__box < 0) > 0:
                continue
            __box[2], __box[3] = __box[0]+__box[2], __box[1]+__box[3]
            __midx, __midy = (__box[0]+__box[2])/2, (__box[1]+__box[3])/2

            __landmark = np.array(landmarks[i].split()[1:], dtype=np.float)
            j = 0
            time = 0
            while j < 1:
                if time > 30:
                    j += 1
                "标签框滑动"
                __x_offset = random.randint(int(-0.3*__min), int(0.3*__min))
                __y_offset = random.randint(int(-0.3*__min), int(0.3*__min))
                __newx, __newy = __midx + __x_offset, __midy + __y_offset
                __side = random.randint(int(0.6*__min), int(1.2*__min))
                "生成坐标"
                x1, y1, x2, y2 = __newx-__side/2,  __newy-__side/2, __newx+__side/2,  __newy+__side/2
                box = np.array([x1, y1, x2, y2], dtype=np.int)
                "偏移量（标签-图片）"
                box_offset = (__box-box)/__side
                landmark_offset = np.zeros(__landmark.shape)
                landmark_offset[0::2] = (__landmark[0::2]-__newx)/__side
                landmark_offset[1::2] = (__landmark[1::2]-__newy)/__side
                "IOU"
                __x1, __y1, __x2, __y2 = __box
                x1, y1, x2, y2 = box
                __area = (__x2-__x1)*(__y2-__y1)
                area = (x2-x1)*(y2-y1)
                inter_area = (min(x2, __x2)-max(x1, __x1))*((min(y2, __y2)-max(y1, __y1)))
                iou = inter_area/(__area+area-inter_area)
                "图片"
                image = __image.crop(box)
                image = image.resize((size, size))
                "保存"
                if type == "positive" and iou > cfg.INTERVAL[type]:
                    image.save(path+"/%d.png"%i)
                    "numpy对象会自动在中间插入换行符"
                    label.write("/positive/{0}.png,{1},{2},{3}\n".
                                format(i, 1, box_offset, str(landmark_offset).replace("\n", " ")))
                    j += 1
                elif type == "part" and iou < 0.3:
                    image.save(path + "/%d.png" % i)
                    "numpy对象会自动在中间插入换行符"
                    label.write("/part/{0}.png,{1},{2},{3}\n".
                                format(i, 2, box_offset, str(landmark_offset).replace("\n", " ")))
                    j += 1
                time += 1
            if i % 10000 == 0:
                print("i'm runing, {}w has Done! ! !".format(i))
        label.close()


def ne(data: str, target: str, goal: str):
    # for size in cfg.NET.values():
    for size in [48, 24, 12]:
        "创建文件夹"
        path = goal + "/%d/" % size + "negative"
        if not os.path.exists(path):
            os.makedirs(path)
        label = open(path + ".txt", "a")
        "读取标签信息"
        with open(target) as f:
            targets = f.readlines()
        for i in range(len(targets)):
            if i < 2:
                continue
            __name, *__box = targets[i].split()

            __image = Image.open(os.path.join(data, __name))
            __W, __H = __image.size

            __x1, __y1, __w, __h = list(map(int, __box))
            __x2, __y2 = __x1+__w, __y1+__h

            if __w <= 0 or __h <= 0:
                continue

            top = __image.crop((0, 0, __x1, __y1))
            left = __image.crop((0, __y2, __x1, __H))
            right = __image.crop((__x2, __y2, __W, __H))

            for j, pic in enumerate([top, left, right]):
                box = np.zeros(4)
                landmark = np.zeros(10)
                pic = pic.resize((size, size))
                pic.save(path + "/{}{}.png".format(i, j))
                "numpy对象会自动在中间插入换行符"
                label.write("/negative/{0}{1}.png,{2},{3},{4}\n".
                            format(i, j, 0, box, str(landmark).replace("\n", " ")))

            if i * 3 % 10000 == 0:
                print("i'm running, {}w has done!!!".format(i*3))
        label.close()


def check(label):
    with open(label) as f:
        lines = f.readlines()
    for line in lines:
        size = int(label.split("/")[-2])
        filename, _, box, landmark = line.split(",")

        file = label[:-4]+"/"+filename.split("/")[-1]
        image = cv2.imread(file)
        box = np.array(box.strip()[1:-1].split(), dtype=np.float)
        landmark = np.array(landmark.strip()[1:-1].split(), dtype=np.float)

        box[:2] = box[:2]*size
        box[2:] = box[2:]*size+size
        landmark = landmark*size+size/2
        box, landmark = box.astype(int), landmark.astype(int)

        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0))
        for i in range(len(landmark)):
            if i % 2 != 0:
                cv2.circle(image, (landmark[i-1], landmark[i]), 2, (0, 0, 255))
        cv2.imshow(filename.split("/")[-1], image)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()


class MyData(data.Dataset):
    def __init__(self, mode: str, path: str, size: int):
        super(MyData, self).__init__()
        self.path = path
        self.size = size
        self.database = []
        for type in ["positive", "part", "negative"]:
            self.target = path+"/{}/{}.txt".format(size[0], type)
            with open(self.target, "r") as f:
                if mode == "train":
                    self.database.extend(f.readlines()[:-10000])
                elif mode == "test":
                    self.database.extend(f.readlines()[-10000:])

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        transform = tf.Compose([
                                tf.ToTensor(),
                                tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
        filename, confi, offset, landmark = self.database[idx].split(",")

        filepath = self.path+"/%d"%self.size+filename
        data = transform(Image.open(filepath))

        confi = torch.Tensor(np.array([confi], dtype=np.int))
        offset = torch.Tensor(np.array(offset.strip()[1:-1].split(), dtype=np.float))
        landmark = torch.Tensor(np.array(landmark.strip()[1:-1].split(), dtype=np.float))

        return data, confi, offset, landmark


ptrain = MyData("train", cfg.PATH, cfg.NET["pnet"])
rtrain = MyData("train", cfg.PATH, cfg.NET["rnet"])
otrain = MyData("train", cfg.PATH, cfg.NET["onet"])

P_TRAIN = data.DataLoader(ptrain, batch_size=cfg.BATCH["pnet"], shuffle=True, drop_last=True, num_workers=4)
R_TRAIN = data.DataLoader(rtrain, batch_size=cfg.BATCH["rnet"], shuffle=True, drop_last=True, num_workers=4)
O_TRAIN = data.DataLoader(otrain, batch_size=cfg.BATCH["onet"], shuffle=True, drop_last=True, num_workers=4)

ptest = MyData("test", cfg.PATH, cfg.NET["pnet"])
rtest = MyData("test", cfg.PATH, cfg.NET["rnet"])
otest = MyData("test", cfg.PATH, cfg.NET["onet"])

P_TEST = data.DataLoader(ptest, batch_size=cfg.TEST_BATCH, shuffle=True, drop_last=True)
R_TEST = data.DataLoader(rtest, batch_size=cfg.TEST_BATCH, shuffle=True, drop_last=True)
O_TEST = data.DataLoader(otest, batch_size=cfg.TEST_BATCH, shuffle=True, drop_last=True)

