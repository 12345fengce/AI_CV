import os
import net
import torch
import random
import PIL.Image as Image
import torchvision.transforms as tf


save_path = "G:/Project/Code/RNN/Verify/union.pkl"
img_path = "F:/RNN/VERIFY/test/"
detection = net.UnionNet()
detection.load_state_dict(torch.load(save_path))
detection = detection.cpu()

for i in range(10):
    label = os.listdir(img_path)[random.randint(0, 1200)]
    img = img_path+label
    img = Image.open(img)
    x = (tf.ToTensor()(img)-0.5).unsqueeze(dim=0)
    output = detection(x)
    predict = torch.argmax(output, dim=-1).squeeze()
    print("Test - {}:\nPredict: {}\nTruth:   {}".format(i, [j.item() for j in predict],
                                                      list(map(int, label.split(".")[0]))))
    print("*"*88)




