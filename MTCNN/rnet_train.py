# -*-coding:utf-8-*-
import net
import trainer 


rnet = net.RNet()
save_path = "F:/Project/Code/MTCNN/rnet.pth"
data_path = "F:/Project/DataSet/celebre"
img_size = 24
if __name__ == "__main__":
    trainer.Trainer(rnet, data_path, save_path, img_size).train()