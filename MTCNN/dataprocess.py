# -*-coding:utf-8-*-
import cv2
import os, sys
import utils
import random
import numpy as np
import PIL.Image as Image
from PIL import ImageFile


def clean(path_save):
    path_save.replace(r"\\", r"/")
    try:
        os.remove(path_save)
    except:
        for file in os.listdir(path_save):
            flag = False
            try:
                os.remove(file)
                flag = True
            except:
                pass
            if not flag:
                Clean(path_save+r"/"+file)
    print("Cleaning ... ... Clean Done!")


def dataprocess(read, save, target, landmark):
    # 循环制作三个尺寸的图片
    for img_size in [12, 24, 48]:
        # 图片存储目录
        positive_path = save+r"/"+str(img_size)+r"/positive"
        part_path = save+r"/"+str(img_size)+r"/part"
        for path in [positive_path, part_path]:
            if not os.path.exists(path):
                os.makedirs(path)
        # 标签存储文件
        p = open(positive_path+".txt", "w")
        part = open(part_path+".txt", "w")
        # 计数，作为图片名，每个img_size都从0开始
        count = 0
        # 分行读取原标签，读取图片
        landmarks = open(landmark, "r").readlines()
        for i, line in enumerate(open(target)):
            if i < 2:
                continue
            # 取出角点与宽高
            img, *line = line.split()
            x, y, w, h = map(int, line)
            img = Image.open(read + "/" + img)
            # 取出5个标记点
            landmark_points = list(map(int, landmarks[i].replace(" ", ",").split(",")[1:]))  # 便于生成列表形式
            if w <= 0 or h <= 0:
                continue
            m = min(w, h)
            xm, ym = x+w/2, y+h/2
            box = [x, y, x+w, y+h]
            # 中心点随机移动生成5张随机边长正方形
            i = 0
            while i < 3:
                bias_x = random.randint(int(-m*0.3), int(m*0.3))
                bias_y = random.randint(int(-m*0.3), int(m*0.3))
                _xm, _ym = xm+bias_x, ym+bias_y
                side = random.randint(int(m*0.6), int(1.2*m))
                # 随机方形角坐标，计算偏移量
                _x1, _y1 = _xm-side//2, _ym-side//2
                _x2, _y2 = _xm+side//2, _ym+side//2
                boxes = list(map(int, [_x1, _y1, _x2, _y2]))
                offset = list(map(lambda x, y: (x-y)/side, boxes, box))  # 外框-标签框
                _landmark = landmark_points.copy()  # 采用列表copy  因为列表本身可变
                _landmark[::2] = list(map(lambda x: (_x2-x)/side, _landmark[::2]))
                _landmark[1::2] = list(map(lambda x: (_y2-x)/side, _landmark[1::2]))
                # 切出图片并进行缩放
                _img = img.crop((_x1, _y1, _x2, _y2))
                _img = _img.resize((img_size, img_size))
                # 根据IOU保存图片，写入标签
                iou = utils.IOU(np.array(box), np.array([boxes]))
                # 正样本  保存图片命名加路径
                if iou > 0.61:
                    _img.save(positive_path+r"/{}.jpg".format(count))
                    p.write("positive/{0}.jpg {1} {2} {3}\n".format(count, 1, offset, _landmark))
                    i += 1
                    count += 1
                # 部分样本  保存图片命名加路径
                if iou < 0.3:
                    _img.save(part_path+r"/{}.jpg".format(count))
                    part.write("part/{0}.jpg {1} {2} {3}\n".format(count, 2, offset, _landmark))
                    i += 1
                    count += 1
            if count % 10000 == 0:
                print("making ... ... {}w+".format(count/10000))
        p.close()
        part.close()
        print("size: {}*{} Saved Succeed!, total {} files".format(img_size, img_size, count))


def Negative(path_read, path_save):
    for img_size in [12, 24, 48]:
        count = 0
        n =  open(path_save+"/"+str(img_size)+"/negative.txt", "a+")
        for file in os.listdir(path_read):
            path = path_read+"/"+file
            img = Image.open(path)
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            img = img.convert("RGB")
            w, h = img.size
            # 中心点
            x, y = w/2, h/2
            w = min(w, h)
            for i in range(10):
                # 设置随机量
                bias_x = random.randint(int(-0.5*w), int(0.5*w))
                bias_y = random.randint(int(-0.5*w), int(0.5*w))
                _x, _y = x+bias_x, y+bias_y
                side = random.randint(int(w*0.3), int(w*0.5))
                _x1, _y1 = _x-side//2, _y-side//2
                _x2, _y2 = _x+side//2, _y+side//2
                # 生成目标图片进行保存  保存图片命名加路径
                _img = img.crop((_x1, _y1, _x2, _y2))
                _img = _img.resize((img_size, img_size))
                _img.save(path_save+"/"+str(img_size)+"/negative/add_{}.jpg".format(count))
                n.write("negative/add_{0}.jpg {1} {2} {3} {4} {5}\n".format(count, 0, 0, 0, 0, 0))
                count += 1
            if count % 10000 == 0:
                print("making ... ... {}w+".format(count/10000))
        n.close()
        print("size: {}*{} Saved Succeed!, total {} files".format(img_size, img_size, count))


def Sign(path_read, path_save, file):
    "make validation set with wilder face"
    with open(file, "r") as f:
        label = f.readlines()
    for size in [48, 24, 12]:
        positive_path = path_save+"/{}".format(size)+"/positive"
        part_path = path_save+"/{}".format(size)+"/part"
        for path in [positive_path, part_path]:
            if not os.path.exists(path):
                os.makedirs(path)
        positive_label = open(positive_path+".txt", "a+")
        part_label = open(part_path+".txt", "a+")
        count = 0
        for i, string in enumerate(label):
            if ".jpg" in string:
                img_path = path_read+"/"+string[:-1]
                img = Image.open(img_path)
            else:
                lst = string.split()
                if len(lst) < 2:
                    continue
                else:
                    x, y, w, h, blur = [int(ele) for ele in lst[:5]]
                    if int(blur) < 2 and w > 15 and h > 15:
                        cx, cy = x+w/2, y+h/2
                        cx, cy = random.uniform(cx-0.5*w, cx+0.5*w), random.uniform(cy-0.5*h, cy+0.5*h)
                        w, h = random.uniform(0.8*w, 1.2*w), random.uniform(0.8*h, 1.2*h)
                        m = max(w, h)
                        x1, y1, x2, y2 = int(cx-m/2), int(cy-h/2), int(cx+m/2), int(cy+h/2)
                        image = img.crop((x1, y1, x2, y2))
                        image = image.resize((size, size))
                        x1_offset, y1_offset, x2_offset, y2_offset = (x1-x)/m, (y1-y)/m, (x2-x-w)/m, (y2-y-h)/m
                        inter_area = (min(x2, x+w)-max(x1, x))*(min(y2, y+h)-max(y1, y))
                        union_area = w*h+(y2-y1)*(x2-x1)-inter_area
                        iou = inter_area/union_area
                        if iou > 0.7:
                            image.save(positive_path+"/add_8_{}.jpg".format(count))
                            positive_label.write("positive/add_8_{}.jpg {} {} {} {} {}\n".format(count, 1, x1_offset, y1_offset, x2_offset, y2_offset))
                            count += 1
                        elif iou < 0.4:
                            image.save(part_path+"/add_8_{}.jpg".format(count))
                            part_label.write("part/add_8_{}.jpg {} {} {} {} {}\n".format(count, 2, x1_offset, y1_offset, x2_offset, y2_offset))
                            count += 1
                        if count%10000 == 0:
                            print("I am running, {}w done!!!".format(count/10000))
        positive_label.close()
        part_label.close()
       
 
if __name__ == "__main__":
    read = "G:/celebre/img_celeba"
    # path_read = "F:/Python/DataSet/celebre/img_negative"
    save = "G:/for_MTCNN/train"
    target = "G:/Dataset/celebre/Anno/list_bbox_celeba.txt"
    landmarks = "G:/Dataset/celebre/Anno/list_landmarks_celeba.txt"

    # file = open(landmarks, "r")
    # file = file.readlines()
    # img, *landmark = file[2].split()
    # img = "G:/Dataset/celebre/img_celeba/"+img
    # img = cv2.imread(img)
    # for i, point in enumerate(landmark):
    #     if i%2 != 0:
    #         cv2.circle(img, (int(landmark[i-1]), int(point)), 2, (255, 0, 0))
    # cv2.imshow("landmark", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 清空存储空间
    # Clean(path_save)

    # 制作图片
    # dataprocess(read, save, target, landmark)

    # 补充负样本
    # path_read = "G:/for_Mtcnn/img"
    # path_save = "G:/for_Mtcnn/train"
    # Negative(path_read, path_save)
    
    # wilder face
    # path_read = "G:/Wider_face/WIDER_train/images"
    # path_save = "G:/for_Mtcnn/train"
    # file = "G:/Wider_face/wider_face_split/wider_face_train_bbx_gt.txt"
    # Sign(path_read, path_save, file)
