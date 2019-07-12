# -*-coding:utf-8-*-
import trainer 
if __name__ == "__main__":
    save_path = "F:/Python/Project/YoLoV3/yolo_net.pth"
    data_path = "G:/Yolo_train"
    trainer.MyTrainer(save_path, data_path).optimize()
    