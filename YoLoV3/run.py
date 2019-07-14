# -*-coding:utf-8-*-
import os, sys
import test
import utils
import trainer 

if __name__ == "__main__":
    """choose MODE:
        "0": train  model.train()
        "1": test    model.eval()"""
    MODE = input("0 to train\n1 to test\nplease Keyboard input model:")
    if MODE == "0":
        save_path = "F:/Project/Code/YoLoV3/yolo_net.pth"
        data_path = "G:/Yolo_train"
        trainer.MyTrainer(save_path, data_path).optimize()
    elif MODE == "1":
        net_path = "F:/Project/Code/YoLoV3/yolo_net.pth"
        image_path = "G:/Yolo_train/img_test"
        for image in os.listdir(image_path):
            img_path = image_path+"/"+image
            boxes = test.Test(net_path, img_path).predict()
            try:
                boxes = utils.NMS(boxes, threshold=0.3, ismin=False)
                boxes = utils.NMS(boxes, threshold=0.3, ismin=True)
                utils.draw(boxes, img_path)
            except:
                continue
             
