# -*-coding:utf-8-*-
import os
import cv2
import time
import numpy as np
import torchvision.transforms as tf


def IOU(box, boxes, ismin=False):
    """calculate the inter area between multi boxes
        box: numpy.ndarray ndim=1
        boxes: numpy.ndarray ndim=2"""
    # 计算各自面积
    box_area = (box[2]-box[0])*(box[3]-box[1])
    boxes_area = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
    # 计算重叠面积
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    side_1 = np.maximum(0, x2-x1)
    side_2 = np.maximum(0, y2-y1)
    inter_area = side_1*side_2
    # 计算IOU
    if ismin:
        IOU = inter_area/np.minimum(box_area, boxes_area)
    else:
        area = np.maximum(box_area+boxes_area-inter_area, 1)
        IOU = inter_area/area
    return IOU


def NMS(boxes, threshold=0.3, ismin=False):
    """Non-Maximum Suppression,NMS
        boxes: numpy.ndarray ndim=2"""
    # 依据confidence的大小对boxes进行排序
    boxes = boxes[boxes[:, 4].argsort()[::-1]]
    get_box = []
    # 循环计算IOU，根据阈值划分目标，同一目标去最大值
    while boxes.shape[0] > 1:
        box = boxes[0]
        boxes = boxes[1:]
        get_box.append(box.tolist())
        iou = IOU(box, boxes, ismin)
        iou = np.maximum(0.1, iou)
        boxes = boxes[np.where(iou<threshold)]
    if boxes.shape[0] > 0:
        get_box.append(boxes[0].tolist())
    return np.array(get_box)


def transform(offset, landmark, coordinate):
    "将偏移量转换为框坐标"
    side_len = (coordinate[:, 2:3]-coordinate[:, 0:1])
    offset = (coordinate-offset*side_len).astype(int)
    x, y = coordinate[:, 2:3], coordinate[:, 3:]
    landmark[:, ::2] = x-landmark[:, ::2]*side_len
    landmark[:, 1::2] = y-landmark[:, 1::2]*side_len
    return offset, landmark


def crop_to_square(coordinates, size, image):
    """crop images of square with coordinates
        coordinates: numpy.ndarray
        size: int
        img: Image.open(path)"""
    priors = []
    data = []
    x_centre, y_centre = (coordinates[:, 0]+coordinates[:, 2])/2, (coordinates[:, 1]+coordinates[:, 3])/2  # 计算中心点
    side_max = np.maximum(coordinates[:, 2]-coordinates[:, 0], coordinates[:, 3]-coordinates[:, 1])  # 根据最大边重新构建方形候选项
    coordinates[:, 0] = (x_centre-side_max/2).astype(int)
    coordinates[:, 1] = (y_centre-side_max/2).astype(int)
    coordinates[:, 2] = (x_centre+side_max/2).astype(int)
    coordinates[:, 3] = (y_centre+side_max/2).astype(int)
    for coordinate in coordinates:  # 只能单个处理数据（尝试cv2）
        prior = coordinate[:-1].tolist()  # 先验收集
        priors.append(prior)  # 数据制作收集
        img = image.crop(prior)  # 从原图扣取数据图片
        img = img.resize((size, size))
        img = tf.ToTensor()(img)-0.5
        data.append(img)
    return data, priors


def draw(coordinates, img_path:str, net:str):
        """draw rectangles on the image
            coordinates: numpy.ndarray"""
        img = cv2.imread(img_path)
        for tangle in coordinates:
            tangle, confi, landmark = tangle[:4].astype(int), tangle[4], tangle[5:].astype(int)
            if max(landmark[::2]) > tangle[2] or min(landmark[::2]) < tangle[0] or\
               max(landmark[1::2]) > tangle[3] or min(landmark[1::2]) < tangle[1]:
                continue
            cv2.rectangle(img,  (tangle[0], tangle[1]), (tangle[2], tangle[3]), (0, 0, 255), 1)
            cv2.putText(img, str(confi), (tangle[0], tangle[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            for i, point in enumerate(landmark):
                if i % 2 != 0:
                    cv2.circle(img, (landmark[i-1], point), 1, (0, 0, 255))
        cv2.imshow("{}".format(net), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

def show(file: str, func: str, frame: int):
    video = cv2.VideoCapture(file)
    f = 1
    while video.isOpened():
        _, image = video.read()
        start_time = time.time()
        try:
            coordinates = func(image).o()
            for tangle in coordinates:
                tangle, confi, landmark = tangle[:4].astype(int), tangle[4], tangle[5:].astype(int)
                if max(landmark[::2]) > tangle[2] or min(landmark[::2]) < tangle[0] or \
                        max(landmark[1::2]) > tangle[3] or min(landmark[1::2]) < tangle[1]:
                    continue
                cv2.rectangle(image, (tangle[0], tangle[1]), (tangle[2], tangle[3]), (0, 0, 255), 1)
                cv2.putText(image, str(confi), (tangle[0], tangle[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                for i, point in enumerate(landmark):
                    if i % 2 != 0:
                        cv2.circle(image, (landmark[i - 1], point), 1, (0, 0, 255))
            cv2.imshow("", image)
            k = cv2.waitKey(1000//frame)
        except:
            cv2.imshow("", image)
            k = cv2.waitKey(1000 // frame)
        finally:
            end_time = time.time()
            print("Frame - {}: time - {}ms".format(f, (end_time-start_time)*1000))
            f += 1
            if (k & 0xff == ord('q')):
                break
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    path = "F:/MTCNN/test/0.jpg"
    img = Image.open(path)
    _img = cv2.imread(path)
    _img = _img[:, :, [2, 1, 0]]
    print(np.array(img))
    print(_img)










    







