# -*-coding:utf-8-*-
import os, sys
import cv2
import math
import torch
import numpy as np
import PIL.Image as Image


# priors
img_size = (416, 416)
prior_boxes = {
    13: [[116, 90], [156, 198], [373, 326]],
    26: [[30, 61], [62, 45], [59, 119]],
    52: [[10, 13], [16, 30], [33, 23]]
}
prior_areas = {
    13: [x*y for x, y in prior_boxes[13]],
    26: [x*y for x, y in prior_boxes[26]],
    52: [x*y for x, y in prior_boxes[52]]
}


# functions
def IOU(box, boxes, ismin=False):
    """calculate the inter area between multi boxes
        box: numpy.ndarray ndim=1
        boxes: numpy.ndarray ndim=2"""
    box_area = (box[2]-box[0])*(box[3]-box[1])
    boxes_area = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    side_1 = np.maximum(0, x2-x1)
    side_2 = np.maximum(0, y2-y1)
    inter_area = side_1*side_2
    if ismin:
        IOU = inter_area/np.minimum(box_area, boxes_area)
    else:
        area = np.maximum(box_area+boxes_area-inter_area, 1)
        IOU = inter_area/area
    return IOU


def NMS(boxes, threshold=0.3, ismin=False):
    """Non-Maximum Suppression,NMS
        boxes: numpy.ndarray ndim=2"""
    boxes = boxes[boxes[:, 4].argsort()[::-1]]
    get_box = []
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


def one_hot(cls_max, cls_num):
    """act on cls
        translate cls to one_hot"""
    label = np.zeros((cls_max))
    label[cls_num] = 1
    return label


def normalize(data, size=416):
    """act on dataset:
        create label with 3 keys: 13, 26, 52
        every key has zeros_array shape=(feature_map_size, feature_map_size, 3, 5+cls)
        while the grid has goal's centre, then fill (w, h, i, 5+cls) with offset"""
    label = {}
    for key, item in prior_boxes.items():
        label[key] = np.zeros(shape=(key, key, 3, 7))  # 符合要求的格子（极个别）重新填入数据
        for box in data:
            cls, x, y, w, h = box
            (x_offset, x_index), (y_offset, y_index) = math.modf(x*key/size), math.modf(y*key/size)  # 余数为offset，商为index
            for i, anchor in enumerate(item):
                w_offset, h_offset = np.log(w/anchor[0]), np.log(h/anchor[1])  # log(实际框/建议框)  利用log压缩数据，使偏移量位于梯度较大处
                inter_area = min(w, anchor[0])*min(h, anchor[1])  # calc iou
                union_area = w*h+prior_areas[key][i]-inter_area
                iou = inter_area/union_area
                cls_num = one_hot(2, cls.astype(int)).astype(int)  # one_hot
                label[key][int(x_index), int(y_index), i] = np.array([iou, x_offset, y_offset, w_offset, h_offset, cls_num[0], cls_num[1]])  # fill label[key] with offset  label[key]: shape=(w, h, 3, 5+cls_num)  
    return label[13], label[26], label[52]


def parse(out, size):
    """act on out from out 
        note index which grid's iou > 0
        get original centre with index and offset
        get original side with offset and prior's side
        regroup them to a list has length (5+cls)"""
    index = torch.sigmoid(out[..., 0]) > 0.5
    index = index.nonzero()
    coordinates = []
    for idx in index:
        box = out[idx[0], idx[1], idx[-1]]  # get box of dim=1 with index of dim=3 from out: (W, H, 3, 5+cls)
        iou, x_offset, y_offset, w_offset, h_offset, *cls = box
        iou = iou.item()  # first step: iou 
        x_offset, y_offset = x_offset+idx[0], y_offset+idx[1]  # second step: centre(x, y)
        x, y = x_offset/(size/416), y_offset/(size/416)
        w_anchor, h_anchor = prior_boxes[size][idx[-1]]  # third step: w, h
        w, h = w_anchor*torch.exp(w_offset), h_anchor*torch.exp(h_offset)
        x1, y1 = int(x-w/2), int(y-h/2)  # translate centre, w, h to coordinates of top left corner and bottom right corner
        x2, y2 = x1+int(w), y1+int(h)
        cls = np.array([x.item() for x in cls])  # last step: cls
        cls = np.where(cls > 0.5, 1, 0)
        coordinates.append([x1, y1, x2, y2, iou, cls])
    return coordinates


def draw(boxes, path):
    """draw on the image with boxes
        every box has cls
        argmax(cls)=0  sign the cat on image with red line
        argmax(cls)=1  sign the dog on image with green line"""
    img = cv2.imread(path)
    for box in boxes:
        x1, y1, x2, y2, iou, cls = box
        if np.sum(cls) == 1 and np.argmax(cls) == 0:
            print(np.argmax(cls), (x1+x2)/2, (y1+y2)/2, (x2-x1), (y2-y1))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(img, "Cat", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        elif np.sum(cls) == 1 and np.argmax(cls) == 1:
            print(np.argmax(cls), (x1+x2)/2, (y1+y2)/2, (x2-x1), (y2-y1))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(img, "Dog", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow(path.split("/")[-1], img)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()


def resize(path):
    """while image's size can not exact division 32
        resize
        MODE 416: 416*416
        MODE 608: 608*608
        MODE others: others"""
    MODE = input("416: square side_len=416\n608: square side_len=608\nothers: rectangle not square\nplease Keyboard input model:")
    for image in os.listdir(path):
        img_path = path+"/"+image
        img = Image.open(img_path)
        w, h = img.size
        if w%32 == 0 and h%32 == 0:  # 跳过之前已经调整好的图片
            continue
        if MODE == "416":
            m = max(w, h)
            new_img = Image.new("RGB", (m, m), (255, 255, 255))
            coordinate = ((m-w)//2, 0) if w < h else (0, (m-h)//2)
            new_img.paste(img, coordinate)
            new_img = new_img.resize((416, 416))
            new_img.save(path+"/416_"+image)
            print("images of 416 save succeeded!!!")
        elif MODE == "608":
            m = max(w, h)
            new_img = Image.new("RGB", (m, m), (255, 255, 255))
            coordinate = ((m-w)//2, 0) if w < h else (0, (m-h)//2)
            new_img.paste(img, coordinate)
            new_img = new_img.resize((608, 608))
            new_img.save(path+"/608_"+image)
            print("images of 608 save succeeded!!!")
        elif MODE == "others":
            w_factor, h_factor = w//32, h//32
            w_max, h_max = 32*(w_factor+1), 32*(h_factor+1)
            new_img = Image.new("RGB", (w_max, h_max), (255, 255, 255))
            coordinate = ((w_max-w)//2, (h_max-h)//2)
            new_img.paste(img, coordinate)
            new_img.save(path+"/others_"+image)
            print("images of others save succeeded!!!")
        else:
            print("Input Error, Please Keyboard input word from '416', '608' and 'others'!!!")


# if __name__ == "__main__":
#     path = "g:/yolo_train/img"
#     resize(path)


        



