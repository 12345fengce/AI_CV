import torch
import numpy as np
from torchvision import transforms
import time
import mtcnn_net
from test.test_show import Img_Video_Show
import init
from cython.nms import NMS
from utils import _NMS


class Detector:
    def __init__(self, p_net_path=r"..\models\P_Net.pth", r_net_path=r"..\models\R_Net.pth",
                 o_net_path=r"..\models\O_Net.pth"):
        # 实例化设备类型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        # 实例化类型转换
        self.image_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # 实例化网络
        self.p_net = mtcnn_net.P_Net().to(self.device)
        self.r_net = mtcnn_net.R_Net().to(self.device)
        self.o_net = mtcnn_net.O_Net().to(self.device)
        # # 加载整个网络模型
        # self.p_net = torch.load(p_net_path)
        # self.r_net = torch.load(r_net_path)
        # self.o_net = torch.load(o_net_path)
        # 加载网络模型参数
        self.p_net.load_state_dict(torch.load(p_net_path, map_location="cuda"))
        self.r_net.load_state_dict(torch.load(r_net_path, map_location="cuda"))
        self.o_net.load_state_dict(torch.load(o_net_path, map_location="cuda"))
        # 网络模型设置为eval模式
        self.p_net.eval()
        self.r_net.eval()
        self.o_net.eval()

    def detect(self, img):
        # P网络输出结果
        p_net_start_time = time.time()
        p_net_boxes = self.pnet_detect(img, 0.7, 18, self.p_net, 0.7, 0.6, False)
        if p_net_boxes.shape[0] == 0:
            p_net_boxes = np.array([])
        p_net_end_time = time.time()
        print("p_net_time: {0:.3f}s------P_net_Face_Number: {1}"
              .format(p_net_end_time - p_net_start_time, len(p_net_boxes)))

        # R网络输出结果
        r_net_start_time = time.time()
        r_net_boxes = self.r_o_net_detect(img, p_net_boxes, 24, self.r_net, 0.9, 0.4, False)
        if r_net_boxes.shape[0] == 0:
            r_net_boxes = np.array([])
        r_net_end_time = time.time()
        print("r_net_time: {0:.3f}s------R_net_Face_Number: {1}"
              .format(r_net_end_time - r_net_start_time, len(r_net_boxes)))

        # O网络输出结果
        o_net_start_time = time.time()
        o_net_boxes = self.r_o_net_detect(img, r_net_boxes, 48, self.o_net, 0.9995, 0.3, True)
        if o_net_boxes.shape[0] == 0:
            o_net_boxes = np.array([])
        o_net_end_time = time.time()
        print("o_net_time: {0:.3f}s------O_net_Face_Number: {1}"
              .format(o_net_end_time - o_net_start_time, len(o_net_boxes)))
        return p_net_boxes, r_net_boxes, o_net_boxes

    def pnet_detect(self, img, ratio, size, net, conf, threshold, contain):
        p_net_box = np.array([])
        img_w, img_h = img.size[0], img.size[1]  # 获取原图片宽和高
        img_min_side = np.minimum(img_w, img_h)  # 获取原图片宽和高中的较小值
        scale = 1.0  # 图片缩放比例
        while img_min_side >= size:  # 原图片宽和高中较小值大于等于12时执行循环
            _img_data = self.image_transform(img)  # 将图片PILImage格式转换为Tensor格式,_img_data格式为CHW
            img_data = _img_data.to(self.device).unsqueeze(0)  # 选择设备 img_data升维操作，升维后格式为NCHW
            _face_class, _bounding_box, _land_mark = net(img_data)  # 将数据输入pnet，获取置信度和偏移量
            face_class, bounding_box, land_mark = _face_class[0, 0], _bounding_box[0], _land_mark[0]  # 获取图片上所有的置信度和偏移量
            face_class_index = torch.nonzero(torch.gt(face_class, conf))  # 获取置信度大于0.6的置信度索引
            _box = self.box_detect(face_class_index.cpu().detach().numpy(), face_class.cpu().data.numpy(),
                                   bounding_box.cpu().data.numpy(), land_mark.cpu().data.numpy(), scale)
            p_net_box = np.append(p_net_box, _box)
            scale *= ratio  # 缩小图片
            _img_w = int(img_w * scale)
            _img_h = int(img_h * scale)
            img = img.resize((_img_w, _img_h))
            img_min_side = np.minimum(_img_w, _img_h)
        _p_net_box = p_net_box.reshape((-1, 15))

        return NMS.nms(_p_net_box, threshold, contain, False, 5)  # Cython NMS

    def r_o_net_detect(self, img, net_box, size, net, conf, threshold, contain):
        _net_img_dataset = []
        net_input_boxes = self.convert_to_square(net_box)  # 获取输入O网络的图片坐标
        for net_box in net_input_boxes:  # 遍历坐标并剪切、压缩为size*size
            x_min, y_min, x_max, y_max = int(net_box[0]), int(net_box[1]), int(net_box[2]), int(net_box[3])
            net_img = img.crop((x_min, y_min, x_max, y_max))
            net_img = net_img.resize((size, size))
            net_img_data = self.image_transform(net_img).to(self.device)
            _net_img_dataset.append(net_img_data)
        net_input_boxes = torch.from_numpy(net_input_boxes).to(self.device)
        net_img_dataset = torch.stack(_net_img_dataset)  # 叠加为tensor类型的数组
        _face_class, _bounding_box, _landmark = net(net_img_dataset)  # 将图片批量输入网络

        face_class_index = torch.gt(_face_class, conf).cpu().detach().numpy()  # 获取置信度大于conf的索引
        net_conf = _face_class[face_class_index[:, 0]][:, 0].cpu().data.numpy()  # 获取置信度大于conf的置信度
        net_coord_offset = _bounding_box[face_class_index[:, 0]].cpu().data.numpy()  # 获取置信度大于conf的坐标偏移量
        net_landmark_offset = _landmark[face_class_index[:, 0]].cpu().data.numpy()
        net_coord = net_input_boxes[face_class_index[:, 0]].cpu().detach().numpy()

        _box = np.array([net_coord[:, 0], net_coord[:, 1], net_coord[:, 2], net_coord[:, 3]])  # 输入R网络的图片坐标
        _side = np.array([net_coord[:, 2] - net_coord[:, 0], net_coord[:, 3] - net_coord[:, 1]])
        _centre = np.array([net_coord[:, 0] + (net_coord[:, 2] - net_coord[:, 0]) / 2,
                            net_coord[:, 1] + (net_coord[:, 3] - net_coord[:, 1]) / 2])

        net_output_box = np.array([(_box[i] + _side[i % 2] * net_coord_offset[:, i]) for i in range(len(_box))])
        net_output_landmark = np.array([
            (_centre[i] + _side[i] * net_landmark_offset[:, i::2].transpose(1, 0)) for i in range(len(_centre))
        ])
        net_output_coord = np.stack(
            [net_output_box[0], net_output_box[1], net_output_box[2], net_output_box[3], net_conf,
             net_output_landmark[0, 0], net_output_landmark[1, 0], net_output_landmark[0, 1], net_output_landmark[1, 1],
             net_output_landmark[0, 2], net_output_landmark[1, 2], net_output_landmark[0, 3], net_output_landmark[1, 3],
             net_output_landmark[0, 4], net_output_landmark[1, 4]], axis=1)

        return _NMS().nms(net_output_coord, threshold, contain, False, 5)  # Python NMS

    def convert_to_square(self, _p_net_box):
        p_net_box_w = _p_net_box[:, 2] - _p_net_box[:, 0]  # 获取p net输出框图的宽
        p_net_box_h = _p_net_box[:, 3] - _p_net_box[:, 1]  # 获取p net输出框图的高
        square_side = np.maximum(p_net_box_w, p_net_box_h)  # 获取p net输出框图的宽和高中的较大值作为输入r net正方形图片的边长
        centre_x = p_net_box_w / 2 + _p_net_box[:, 0]  # 获取p net输出框图的中心点横坐标
        centre_y = p_net_box_h / 2 + _p_net_box[:, 1]  # 获取p net输出框图的中心点纵坐标
        square_x_min = centre_x - square_side / 2  # 获取输入r net正方形图片左上角横坐标
        square_y_min = centre_y - square_side / 2  # 获取输入r net正方形图片左上角纵坐标
        square_x_max = square_x_min + square_side  # 获取输入r net正方形图片右下角横坐标
        square_y_max = square_y_min + square_side  # 获取输入r net正方形图片右下角纵坐标
        square_coord = np.stack([square_x_min, square_y_min, square_x_max, square_y_max], axis=1)
        return square_coord

    def box_detect(self, index, conf, offset, land_mark, scale, stride=2, side=12):
        face_conf = conf[index[:, 0], index[:, 1]]
        # 特征图上先验框左上角,右下角横坐标，纵坐标
        _box = np.array(
            [index[:, 1] * stride, index[:, 0] * stride, index[:, 1] * stride + side, index[:, 0] * stride + side],
            dtype=float)
        # 原图框图左上角,右下角横坐标，纵坐标(5, ...)
        box = np.array([(_box[i] + offset[i][index[:, 0], index[:, 1]] * side) / scale for i in range(len(offset))])
        # 特征图上先验框中心点横坐标，纵坐标
        _landmark = np.array([index[:, 1] * stride, index[:, 0] * stride], dtype=float) + side / 2
        # 原图上5个关键点横坐标，纵坐标(2, 5, ...)
        landmark = np.array(
            [(_landmark[i] + land_mark[i::2, index[:, 0], index[:, 1]] * side) / scale for i in range(2)]
        )

        return np.stack(
            [box[0], box[1], box[2], box[3], face_conf, landmark[0, 0], landmark[1, 0], landmark[0, 1], landmark[1, 1],
             landmark[0, 2], landmark[1, 2], landmark[0, 3], landmark[1, 3], landmark[0, 4], landmark[1, 4]], axis=1)


if __name__ == '__main__':
    detector = Detector()
    face_detection_show = Img_Video_Show(detector, r"..\test\video\3.wmv", init.IMG_PATH)
    start_time = time.time()
    face_detection_show.show_test(state=1, select=0, img="5.jpg")
    """
    state: 0:视频显示   1:PIL显示图片   2: OpenCV2显示图片
    select: 0:单张显示   1：循环显示图片
    img: 单张显示图片的名称
    """
    end_time = time.time()
    print("time:{}min".format((end_time - start_time) / 60))
