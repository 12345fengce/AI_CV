# -*-coding:utf-8-*-
import net
import trainer 


onet = net.ONet()
save_path = "F:/Project/Code/MTCNN/onet.pth"
data_path = "F:/Project/DataSet/celebre"
img_size = 48
if __name__ == "__main__":
    trainer.Trainer(onet, data_path, save_path, img_size).train()
