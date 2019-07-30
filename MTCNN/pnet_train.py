# -*-coding:utf-8-*-
import net
import torch
import trainer 


model = net.PNet()
save = "G:/for_MTCNN/test/pnet.pth"
train = "G:/for_MTCNN/train"
validation = "G:/for_MTCNN/validation"
size = 12
if __name__ == "__main__":
    trainer.Trainer(model, train, validation, save, size).main()
  
        
    