# -*-coding:utf-8-*-
import net
import trainer 


pnet = net.PNet()
save_path = "F:/Python/Project/MTCNN/pnet.pth"
data_path = "F:/Python/DataSet/celebre"
img_size = 12
if __name__ == "__main__":
    trainer.Trainer(pnet, data_path, save_path, img_size).train()