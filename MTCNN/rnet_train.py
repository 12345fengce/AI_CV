# -*-coding:utf-8-*-
import net
import trainer 


rnet = net.RNet()
save_path = "F:/Python/Project/MTCNN/rnet.pth"
data_path = "F:/Python/DataSet/celebre"
img_size = 24
if __name__ == "__main__":
    trainer.Trainer(rnet, data_path, save_path, img_size).train()