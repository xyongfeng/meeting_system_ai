import os
import time

import cv2
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input

from nets.retinaface import RetinaFace
from utils.anchors import Anchors
from utils.config import cfg
from utils.utils import letterbox_image, Alignment
from utils.utils_bbox import BBoxUtility, retinaface_correct_boxes

from nets.facenet import facenet
from config import config
import matplotlib.pyplot as plt


# retinaface 与 facenet 模型组合
class ModelCombination(object):
    # 初始化
    def __init__(self, **kwargs):
        print("model start")
        # retinaface模型权重位置
        self.retinaface_model_path = config['retinface_weight']
        # 只有得分大于置信度的预测框会被保留下来
        self.confidence = 0.9
        # 非极大抑制所用到的nms_iou大小
        self.nms_iou = 0.45
        # 开启图像大小限制
        # 开启后会将输入图像大小限制到input_shape，否则将会按照原图进行预测
        # 因为keras实现的mobilnet存在小bug，所以输入图像的高宽必须为32的倍数，不然会导致偏差
        self.letterbox_image = True
        # 设置限制图像的大小
        self.input_shape = [640, 640, 3]
        # 设置配置信息
        self.cfg = cfg
        # 设置工具箱
        self.bbox_util = BBoxUtility(nms_thresh=self.nms_iou)
        # 生成先验框
        self.anchors = Anchors(self.cfg, image_size=(self.input_shape[0], self.input_shape[1])).get_anchors()

        # 载入模型和权重
        self.retinaface = RetinaFace(self.cfg)
        self.retinaface.load_weights(self.retinaface_model_path, by_name=True)
        # print(self.retinaface.summary())
        # facenet模型权重
        self.facenet_model_path = config['facenet_weight']

        # facenet所使用的输入图片大小
        self.facenet_input_shape = (160, 160, 3)

        # 载入模型和权重
        self.facenet = facenet(self.facenet_input_shape, mode='predict')
        self.facenet.load_weights(self.facenet_model_path, by_name=True)
        # print(self.facenet.summary())

    def __detect_face(self, image):
        """
        找出图像中的人脸，并返回相关坐标
        :param image:
        :return:
        """
        # 计算图片长和宽
        h, w, _ = image.shape
        # 计算scale，将预测框转换成原图的高宽
        # 将原图大小保存，最后将与预测结果相乘，得到绝对坐标
        scale = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]

        # letterbox_image可以给图像增加灰条，实现不失真的resize，将图片等比缩小或放大至input_shape，不足的部分用灰条填充
        # 增加灰条会将图片变成input_shape的大小，在初始化的时候就已经对input_shape生成了先验框了
        # 否则因为输入图片大小可能变了，避免与先验框不一致，所以按照这个图片的大小再生成先验框
        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(h, w)).get_anchors()

        # 图片进行归一化,再增加一维度
        image = np.expand_dims(preprocess_input(image), 0)

        # 将图片传入网络
        preds = self.retinaface.predict(image)

        # 解码
        results = self.bbox_util.detection_out(preds, self.anchors, confidence_threshold=self.confidence)
        # 如果没有检测到物体则返
        if len(results) == 0:
            raise Exception('没有检测到人脸')
        results = np.array(results)
        # 如果之前增加了灰条，要把灰条的部分去掉
        if self.letterbox_image:
            results = retinaface_correct_boxes(results, np.array([self.input_shape[0], self.input_shape[1]]),
                                               np.array([h, w]))
        # 将results的相对坐标变成绝对坐标
        results[:, :4] = results[:, :4] * scale
        results[:, 5:] = results[:, 5:] * scale_for_landmarks
        return results

    def __get_max_face(self, old_image, results):
        """
        只返回图片中面积最大的人脸
        :param old_image:
        :param results:
        :return:
        """
        max_aera_face = np.array([[0]])
        max_aera = 1
        for i, result in enumerate(results):
            # 先进行人脸矫正，将人的眼睛水平显示
            result = np.maximum(result, 0)
            # 将含有人脸的部分选出来
            crop_img = np.array(old_image)[int(result[1]):int(result[3]), int(result[0]):int(result[2])]
            # 人脸特征点部分
            landmark = np.reshape(result[5:], (5, 2)) - np.array([int(result[0]), int(result[1])])
            # 进行旋转
            crop_img = Alignment(crop_img, landmark)
            # print(crop_img.shape)
            h, w = crop_img.shape[:2]
            if h * w > max_aera:
                max_aera = h * w
                max_aera_face = crop_img

        # 将最大人脸添加灰条，再进行归一化
        max_aera_face = np.array(
            letterbox_image(np.uint8(max_aera_face), (self.facenet_input_shape[1], self.facenet_input_shape[0]))) / 255
        max_aera_face = np.expand_dims(max_aera_face, 0)
        return max_aera_face

    def __get_face_feature(self, max_face):

        face_encoding = self.facenet.predict(max_face)[0]
        return face_encoding

    def __draw_face(self, old_image, results):
        """
        将预测框画在原图上
        :param old_image:
        :param results:
        :return:
        """
        for i, b in enumerate(results):
            # print(i, b)
            # 读取得分
            text = "{:.4f}".format(b[4])
            # 全部转成int
            b = list(map(int, b))
            # 0 ~ 3 人脸框左上角右下角坐标
            # 画框
            cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            # 将text打印在图上
            cv2.putText(old_image, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            # 5 ~ 14 人脸特征点坐标
            cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)
            # name = face_names[i]
            # 写中文需要转成PIL来写
            # old_image = cv2ImgAddText(old_image, name, b[0] + 5, b[3] - 25)
        return old_image

    def get_face_feature(self, img):
        old_img = img.copy()
        try:
            results = self.__detect_face(img)
        except Exception:
            return None

        max_face = self.__get_max_face(old_img, results)
        max_face_feature = self.__get_face_feature(max_face)
        return max_face_feature

    def detect_draw(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        old_img = img.copy()
        try:
            results = self.__detect_face(img)
        except Exception:
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img = self.__draw_face(old_img, results)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
