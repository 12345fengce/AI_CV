# -*-coding:utf-8-*-
import os
import net
import torch
import dataset
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import darknet53


class MyTrainer:
    """set net, set train_data, set optimizer, set loss
        translate out(N, C, H, W) to (N, W, H, 3, 5+cls)  
        [iou, x_offset, y_offset, w_offset, h_offset, cls:one_hot]
        optimize paramters of YoLoNet"""
    def __init__(self, save_path, data_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = save_path
        if os.path.exists(save_path):
            self.net = torch.load(save_path)
        else:
            self.net = darknet53.MainNet(2).to(self.device)
        self.net.train()
        self.train_data = data.DataLoader(dataset.MyData(data_path), batch_size=1, shuffle=True)
        self.opt = optim.Adam(self.net.parameters())
        # self.loss_confi = nn.BCEWithLogitsLoss(weight=None, size_average=False, reduce=True)
        # self.loss_centre = nn.BCEWithLogitsLoss(weight=None, size_average=False, reduce=True)
        self.loss_confi = nn.MSELoss()
        self.loss_centre = nn.MSELoss()
        self.loss_side = nn.MSELoss()
        self.loss_cls = nn.CrossEntropyLoss()
    def optimize(self):
        count = 1
        while True:
            for i, (label_13, label_26, label_52, img) in enumerate(self.train_data):
                out_13, out_26, out_52 = self.net(img.to(self.device))
                label_13, label_26, label_52 = label_13.to(self.device), label_26.to(self.device), label_52.to(self.device)
                loss_13 = self.translate(out_13, label_13)
                loss_26 = self.translate(out_26, label_26)
                loss_52 = self.translate(out_52, label_52)
                loss = loss_13+loss_26+loss_52
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            count += 1
            if count%10 == 0:
                print("[epoche] - {} - loss_13:{} + loss_26:{} + loss_52:{} = Loss:{}".format(count, loss_13, loss_26, loss_52, loss))
                torch.save(self.net, self.save_path)
    def translate(self, out, label, alpha=0.9):
        out = out.transpose(1, -1)
        out = out.reshape(out.size(0), out.size(1), out.size(2), 3, -1)
        mask_positive, mask_negative = label[..., 0] > 0, label[..., 0] == 0

        label_positive, label_negative = label[mask_positive], label[mask_negative]
        out_positive, out_negative = out[mask_positive], out[mask_negative]
        
        label_confi_positive, label_centre_positive, label_side_positive, label_cls_positive = label_positive[:, :1], label_positive[:, 1:3], label_positive[:, 3:5], label_positive[:, 5:]
        out_confi_positive, out_centre_positive, out_side_positive, out_cls_positive = out_positive[:, :1], out_positive[:, 1:3], out_positive[:, 3:5], out_positive[:, 5:]
        label_confi_negative, out_confi_negative = label_negative[:, :1], out_negative[:, :1]
        
        loss_confi = self.loss_confi(out_confi_positive, label_confi_positive)
        loss_centre = self.loss_centre(out_centre_positive, label_centre_positive)
        loss_side = self.loss_side(out_side_positive, label_side_positive)
        loss_cls = self.loss_cls(out_cls_positive, torch.argmax(label_cls_positive, dim=1))
        loss = alpha*(loss_confi+loss_centre+loss_side+loss_cls)+(1-alpha)*self.loss_confi(out_confi_negative, label_confi_negative)
        return loss


# if __name__ == "__main__":
#     trainer = MyTrainer("F:/Project/Code/YoLoV3/yolo3.pt", "G:/Yolo_train")
#     for x, y, z, img in trainer.train_data:
#         print(x[..., 5:][0][x[..., 0][0]>0], x[..., 0][0][x[..., 0][0]>0])
#         break
    # trainer.optimize()
    # sys.exit()
    # net = torch.load("F:/Project/Code/YoLoV3/yolo_net.pth")
    # print(list(net.parameters()))






        

        