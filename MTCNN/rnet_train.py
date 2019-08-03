# -*-coding:utf-8-*-
import net
import trainer 


model = net.RNet()
save = "F:/MTCNN/test/rnet.pth"
train = "F:/MTCNN/train"
validation = "F:/MTCNN/validation"
size = 24
if __name__ == "__main__":
    trainer.Trainer(model, train, validation, save, size).main()