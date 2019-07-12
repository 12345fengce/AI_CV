# -*-coding:utf-8-*-
import net
import trainer 


onet = net.ONet()
save_path = "F:/Python/Project/MTCNN/onet.pth"
data_path = "F:/Python/DataSet/celebre"
img_size = 48
if __name__ == "__main__":
    trainer.Trainer(onet, data_path, save_path, img_size).train()