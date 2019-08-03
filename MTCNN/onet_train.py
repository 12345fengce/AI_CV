# -*-coding:utf-8-*-
import net
import trainer 


model = net.ONet()
save = "F:/MTCNN/test/onet.pth"
train = "F:/MTCNN/train"
validation = "F:/MTCNN/validation"
size = 48
if __name__ == "__main__":
    trainer.Trainer(model, train, validation, save, size).main()
