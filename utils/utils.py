from functools import reduce

import cv2
import numpy as np
import math
import tensorflow as tf
from PIL import Image


# 可以查看该网址的讲解
# https://blog.csdn.net/deliberate_cha/article/details/105874309
def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


#   对输入图像进行resize
def letterbox_image(image, size):
    # 原图长宽
    ih, iw, _ = np.shape(image)
    # 目标长宽
    w, h = size
    # 取较小的目标与原图的长宽比率,要先把与目标近的边扩充，与目标远的边则用灰条填充
    scale = min(w / iw, h / ih)
    # 还能保持原图长宽比的有色图片大小
    nw = int(iw * scale)
    nh = int(ih * scale)
    # 转换
    image = cv2.resize(image, (nw, nh))
    # 生成一个目标大小的灰色图
    new_image = np.ones([size[1], size[0], 3]) * 128
    # 再将转换后的有色图覆盖中灰度图中间
    new_image[(h - nh) // 2:nh + (h - nh) // 2, (w - nw) // 2:nw + (w - nw) // 2] = image
    return new_image


# 人脸矫正
def Alignment(img, landmark):
    # 获得xy轴眼睛偏移量
    x = landmark[0, 0] - landmark[1, 0]
    y = landmark[0, 1] - landmark[1, 1]

    if x == 0:
        angle = 0
    else:
        # 计算它的角度
        angle = math.atan(y / x) * 180 / math.pi

    # 定义旋转中心
    center = (img.shape[1] // 2, img.shape[0] // 2)

    # 创建一个旋转矩阵
    RotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)
    # 仿射
    new_img = cv2.warpAffine(img, RotationMatrix, (img.shape[1], img.shape[0]))

    # 计算改变后的5个特征点
    # RotationMatrix = np.array(RotationMatrix)
    # new_landmark = []
    # for i in range(landmark.shape[0]):
    #     pts = []
    #     pts.append(RotationMatrix[0, 0] * landmark[i, 0] + RotationMatrix[0, 1] * landmark[i, 1] + RotationMatrix[0, 2])
    #     pts.append(RotationMatrix[1, 0] * landmark[i, 0] + RotationMatrix[1, 1] * landmark[i, 1] + RotationMatrix[1, 2])
    #     new_landmark.append(pts)
    #
    # new_landmark = np.array(new_landmark)

    return new_img  # ,new_landmark


# 计算人脸间的特征距离（说白了就是方差）
def face_distance(face_encodings, face_to_compare):
    # if len(face_encodings) == 0:
    #     return np.empty(0)
    # 计算所有人脸和当前人脸的特征向量的欧式距离，
    # np.linalg.norm 求范数，默认为2范数
    face_encodings, face_to_compare = np.array(face_encodings), np.array(face_to_compare)
    # 判断是多对一，还是一对一
    axis = 1 if len(face_encodings.shape) == 2 else None

    return np.linalg.norm(face_encodings - face_to_compare, axis=axis)


# 比较
def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=1):
    # (n)
    dis = face_distance(known_face_encodings, face_encoding_to_check)
    # 返回距离是否小于tolerance的list和距离

    return dis <= tolerance, dis


def get_num_classes(annotation_path):
    with open(annotation_path) as f:
        dataset_path = f.readlines()

    labels = []
    for path in dataset_path:
        path_split = path.split(";")
        labels.append(int(path_split[0]))
    num_classes = np.max(labels) + 1
    return num_classes


# ---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
# ---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

    # ---------------------------------------------------#


#   对输入图像进行resize
# ---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def preprocess_input(image):
    image /= 255.0
    return image


def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


# -------------------------------------------------------------------------------------------------------------------------------#
#   From https://github.com/ckyrkou/Keras_FLOP_Estimator
#   Fix lots of bugs
# -------------------------------------------------------------------------------------------------------------------------------#
def net_flops(model, table=False, print_result=True):
    if (table == True):
        print("\n")
        print('%25s | %16s | %16s | %16s | %16s | %6s | %6s' % (
            'Layer Name', 'Input Shape', 'Output Shape', 'Kernel Size', 'Filters', 'Strides', 'FLOPS'))
        print('=' * 120)

    # ---------------------------------------------------#
    #   总的FLOPs
    # ---------------------------------------------------#
    t_flops = 0
    factor = 1e9

    for l in model.layers:
        try:
            # --------------------------------------#
            #   所需参数的初始化定义
            # --------------------------------------#
            o_shape, i_shape, strides, ks, filters = ('', '', ''), ('', '', ''), (1, 1), (0, 0), 0
            flops = 0
            # --------------------------------------#
            #   获得层的名字
            # --------------------------------------#
            name = l.name

            if ('InputLayer' in str(l)):
                i_shape = l.get_input_shape_at(0)[1:4]
                o_shape = l.get_output_shape_at(0)[1:4]

            # --------------------------------------#
            #   Reshape层
            # --------------------------------------#
            elif ('Reshape' in str(l)):
                i_shape = l.get_input_shape_at(0)[1:4]
                o_shape = l.get_output_shape_at(0)[1:4]

            # --------------------------------------#
            #   填充层
            # --------------------------------------#
            elif ('Padding' in str(l)):
                i_shape = l.get_input_shape_at(0)[1:4]
                o_shape = l.get_output_shape_at(0)[1:4]

            # --------------------------------------#
            #   平铺层
            # --------------------------------------#
            elif ('Flatten' in str(l)):
                i_shape = l.get_input_shape_at(0)[1:4]
                o_shape = l.get_output_shape_at(0)[1:4]

            # --------------------------------------#
            #   激活函数层
            # --------------------------------------#
            elif 'Activation' in str(l):
                i_shape = l.get_input_shape_at(0)[1:4]
                o_shape = l.get_output_shape_at(0)[1:4]

            # --------------------------------------#
            #   LeakyReLU
            # --------------------------------------#
            elif 'LeakyReLU' in str(l):
                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]

                    flops += i_shape[0] * i_shape[1] * i_shape[2]

            # --------------------------------------#
            #   池化层
            # --------------------------------------#
            elif 'MaxPooling' in str(l):
                i_shape = l.get_input_shape_at(0)[1:4]
                o_shape = l.get_output_shape_at(0)[1:4]

            # --------------------------------------#
            #   池化层
            # --------------------------------------#
            elif ('AveragePooling' in str(l) and 'Global' not in str(l)):
                strides = l.strides
                ks = l.pool_size

                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]

                    flops += o_shape[0] * o_shape[1] * o_shape[2]

            # --------------------------------------#
            #   全局池化层
            # --------------------------------------#
            elif ('AveragePooling' in str(l) and 'Global' in str(l)):
                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]

                    flops += (i_shape[0] * i_shape[1] + 1) * i_shape[2]

            # --------------------------------------#
            #   标准化层
            # --------------------------------------#
            elif ('BatchNormalization' in str(l)):
                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]

                    temp_flops = 1
                    for i in range(len(i_shape)):
                        temp_flops *= i_shape[i]
                    temp_flops *= 2

                    flops += temp_flops

            # --------------------------------------#
            #   全连接层
            # --------------------------------------#
            elif ('Dense' in str(l)):
                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]

                    temp_flops = 1
                    for i in range(len(o_shape)):
                        temp_flops *= o_shape[i]

                    if (i_shape[-1] == None):
                        temp_flops = temp_flops * o_shape[-1]
                    else:
                        temp_flops = temp_flops * i_shape[-1]
                    flops += temp_flops

            # --------------------------------------#
            #   普通卷积层
            # --------------------------------------#
            elif ('Conv2D' in str(l) and 'DepthwiseConv2D' not in str(l) and 'SeparableConv2D' not in str(l)):
                strides = l.strides
                ks = l.kernel_size
                filters = l.filters
                bias = 1 if l.use_bias else 0

                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]

                    if (filters == None):
                        filters = i_shape[2]
                    flops += filters * o_shape[0] * o_shape[1] * (ks[0] * ks[1] * i_shape[2] + bias)

            # --------------------------------------#
            #   逐层卷积层
            # --------------------------------------#
            elif ('Conv2D' in str(l) and 'DepthwiseConv2D' in str(l) and 'SeparableConv2D' not in str(l)):
                strides = l.strides
                ks = l.kernel_size
                filters = l.filters
                bias = 1 if l.use_bias else 0

                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]

                    if (filters == None):
                        filters = i_shape[2]
                    flops += filters * o_shape[0] * o_shape[1] * (ks[0] * ks[1] + bias)

            # --------------------------------------#
            #   深度可分离卷积层
            # --------------------------------------#
            elif ('Conv2D' in str(l) and 'DepthwiseConv2D' not in str(l) and 'SeparableConv2D' in str(l)):
                strides = l.strides
                ks = l.kernel_size
                filters = l.filters

                for i in range(len(l._inbound_nodes)):
                    i_shape = l.get_input_shape_at(i)[1:4]
                    o_shape = l.get_output_shape_at(i)[1:4]

                    if (filters == None):
                        filters = i_shape[2]
                    flops += i_shape[2] * o_shape[0] * o_shape[1] * (ks[0] * ks[1] + bias) + \
                             filters * o_shape[0] * o_shape[1] * (1 * 1 * i_shape[2] + bias)
            # --------------------------------------#
            #   模型中有模型时
            # --------------------------------------#
            elif 'Model' in str(l):
                flops = net_flops(l, print_result=False)

            t_flops += flops

            if (table == True):
                print('%25s | %16s | %16s | %16s | %16s | %6s | %5.4f' % (
                    name[:25], str(i_shape), str(o_shape), str(ks), str(filters), str(strides), flops))

        except:
            pass

    t_flops = t_flops * 2
    if print_result:
        show_flops = t_flops / factor
        print('Total GFLOPs: %.3fG' % (show_flops))
    return t_flops
