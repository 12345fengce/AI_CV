# -*-coding:utf-8-*-
import net
import trainer 


onet = net.ONet()
save_path = "F:/Project/Code/MTCNN/onet.pth"
train_path = "G:/for_Mtcnn/train"
validation_path = "G:/for_MTCNN/validation"
img_size = 48
if __name__ == "__main__":
    trainer.Trainer(onet, train_path, validation_path, save_path, img_size).train()
