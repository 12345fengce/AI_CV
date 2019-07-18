# -*-coding:utf-8-*-
import net
import trainer 


pnet = net.PNet()
save_path = "F:/Project/Code/MTCNN/pnet.pth"
train_path = "F:/Project/DataSet/celebre"
validation_path = "G:/for_MTCNN/train"
img_size = 12
if __name__ == "__main__":
    trainer.Trainer(pnet, train_path, validation_path, save_path, img_size).train()