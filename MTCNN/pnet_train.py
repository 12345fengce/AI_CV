# -*-coding:utf-8-*-
import net
import trainer 


pnet = net.PNet()
save_path = "F:/Project/Code/MTCNN/pnet.pth"
data_path = "F:/Project/DataSet/celebre"
img_size = 12
if __name__ == "__main__":
    trainer.Trainer(pnet, data_path, save_path, img_size).train()