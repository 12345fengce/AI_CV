# -*-coding:utf-8-*-
import net
import torch
import trainer 


pnet = net.PNet()
save_path = "F:/Project/Code/MTCNN/pnet.pth"
train_path = "G:/for_Mtcnn/train"
validation_path = "G:/for_MTCNN/validation"
img_size = 12
if __name__ == "__main__":
    trainer.Trainer(pnet, train_path, validation_path, save_path, img_size).train()
  
        
    