# -*-coding:utf-8-*-
import os
import torch
import PIL.Image as Image
import torch.utils.data as data
import torchvision.transforms as tf


transform = tf.Compose([
                        tf.ToTensor(),
                        tf.Normalize((0.5, 0.5, 0.5), (1, 1, 1))
                        ])


class MyData(data.Dataset):
    def __init__(self, path):
        super(MyData, self).__init__()
        self.path = path
        self.database = []
        for type in ["/positive", "/negative"]:
            with open(path+type+".txt", "r") as f:
                self.database.extend(f.readlines())

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        target = torch.tensor([int(self.database[idx].split()[1])], dtype=torch.float)
        img_name = self.database[idx].split()[0]
        img_path = self.path+"/"+img_name
        img = Image.open(img_path)
        img = transform(img)
        return img, target


# if __name__ == '__main__':
#     path = "F:/MTCNN/validation/48"
#     mydata = MyData(path)
#     print(mydata[10000])