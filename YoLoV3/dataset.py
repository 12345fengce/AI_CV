# -*-coding:utf-8-*-
import json
import torch
import utils
import numpy as np
import PIL.Image as Image
import torch.utils.data as data
import torchvision.transforms as tf


class PreProccess:
    """resize images to size of 416*416
        create the new labels"""
    def __init__(self, label_file):
        with open(label_file, "r") as self.f:
            self.label = self.f.readlines()
    def resize(self, path):
        f = open(path+"/new_label.txt", "w")
        for i in range(len(self.label)):
            img_name, *data = self.label[i].split()
            img_path = path+"/"+img_name
            img = Image.open(img_path)
            side_len = max(img.size)
            new_img = Image.new("RGB", (side_len, side_len), (255, 255, 255))
            w, h = img.size
            coordinate =((side_len-w)//2, 0) if w < h else (0, (side_len-h)//2)
            new_img.paste(img, coordinate)
            new_img = new_img.resize((416, 416))
            new_img.save(path+"/n{}.jpg".format(i))
            data = np.array(data, dtype=np.float)
            data = np.array(np.split(data, data.size//5))
            data[:, 1], data[:, 2] = data[:, 1]+coordinate[0], data[:, 2]+coordinate[1]
            scal = side_len/416
            data[:, 1:] = data[:, 1:]/scal
            f.write("n{}.jpg".format(i))
            for j in range(data.shape[0]):
                target = data[j]
                f.write(" {0} {1} {2} {3} {4}".format(int(target[0]), target[1], target[2], target[3], target[4]))
            f.write("\n")
        f.close()

    
class MyData(data.Dataset):
    """transform image to Tensor(C, H, W)
        create labels: type=dict  labels has 3 keys: 13, 26, 52
        type[key]=numpy.ndarray
        shape=(W, H, 3, 7)
        3: every grad has 3 anchor_boxes
        7: [iou, x_offset, y_offset, w_offset, h_offset, cls:one_hot]"""
    def __init__(self, path):
        super(MyData, self).__init__()
        self.path = path
        label_path = path+"/label.txt"
        with open(label_path) as f:
            self.label = f.readlines()
    def __len__(self):
        return len(self.label)
    def __getitem__(self, index):
        img_path, *data = self.label[index].split()
        img_path = self.path+"/"+img_path
        img = Image.open(img_path)
        W, H = img.size
        img = tf.ToTensor()(img)
        data = np.array(data, dtype=np.float)
        data = np.array(np.split(data, len(data)//5))
        label_13, label_26, label_52 = utils.normalize(data, W)
        return torch.Tensor(label_13), torch.Tensor(label_26), torch.Tensor(label_52), img


if __name__ == "__main__":
    file = "G:/Coco2017/annotations/instances_train2017.json"
    with open(file, "r") as file:
        lst = json.load(file)
    i = 0
    for key, item in lst.items():
        if ".jpg" in item:
            print(key, item)
            break
        else:
            continue
        
