# -*-coding:utf-8-*-
import os
import test
import utils
import trainer 

if __name__ == "__main__":
    MODE = input("please Keyboard input 0 or 1:\n0 to train\n1 to test")
    if MODE == "0":
        save_path = "F:/Project/Code/YoLoV3/yolo_net.pth"
        data_path = "G:/Yolo_train"
        trainer.MyTrainer(save_path, data_path).optimize()
    elif MODE == "1":
        net_path = "F:/Project/Code/YoLoV3/yolo_net.pth"
        image_path = "G:/Yolo_train/img"
        for image in os.listdir(image_path):
            img_path = image_path+"/"+image
            boxes = test.Test(net_path, img_path).predict()
            try:
                boxes = utils.NMS(boxes, threshold=0.3, ismin=False)
                boxes = utils.NMS(boxes, threshold=0.3, ismin=True)
                utils.draw(boxes, img_path)
            except:
                continue
            
