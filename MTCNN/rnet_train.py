# -*-coding:utf-8-*-
import net
import trainer 


model = net.RNet()
save = "G:/for_MTCNN/test/rnet.pth"
train = "G:/for_MTCNN/train"
validation = "G:/for_MTCNN/validation"
size = 24
if __name__ == "__main__":
    trainer.Trainer(model, train, validation, save, size).main()