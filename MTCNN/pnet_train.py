# -*-coding:utf-8-*-
import os
import net
import torch
import trainer 


model = net.PNet()
save = "F:/MTCNN/test/pnet.pth"
train = "F:/MTCNN/train"
validation = "F:/MTCNN/validation"
size = 12
if __name__ == "__main__":
    trainer.Trainer(model, train, validation, save, size).main()
  
        
    