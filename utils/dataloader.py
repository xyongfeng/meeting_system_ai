import math
import random
from random import shuffle
import os
import cv2
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from PIL import Image
from .utils import cvtColor, preprocess_input, resize_image
from tensorflow.keras.utils import to_categorical


class Generator(keras.utils.Sequence):
    def __init__(self, txt_path, img_size, batch_size, bbox_util):
        self.img_size = img_size
        self.txt_path = txt_path
        self.batch_size = batch_size
        self.bbox_util = bbox_util

        self.imgs_path, self.words = self.process_labels()

    def __len__(self):
        return math.ceil(len(self.imgs_path) / float(self.batch_size))

    def get_len(self):
        return len(self.imgs_path)

    def on_epoch_end(self):
        shuffle_index = np.arange(len(self.imgs_path))
        shuffle(shuffle_index)
        self.imgs_path = np.array(self.imgs_path)[shuffle_index]
        self.words = np.array(self.words)[shuffle_index]

    def __getitem__(self, index):
        inputs = []
        regression_targets = []
        classification_targets = []
        landms_targets = []

        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            i = i % self.get_len()
            img = Image.open(self.imgs_path[i])

            labels = self.words[i]
            annotations = np.zeros((0, 15))
            for idx, label in enumerate(labels):
                annotation = np.zeros((1, 15))
                # ------------------------------#
                #   框的坐标
                # ------------------------------#
                annotation[0, 0] = label[0]  # x1
                annotation[0, 1] = label[1]  # y1
                annotation[0, 2] = label[0] + label[2]  # x2
                annotation[0, 3] = label[1] + label[3]  # y2

                # ------------------------------#
                #   人脸关键点的坐标
                # ------------------------------#
                annotation[0, 4] = label[4]  # l0_x
                annotation[0, 5] = label[5]  # l0_y
                annotation[0, 6] = label[7]  # l1_x
                annotation[0, 7] = label[8]  # l1_y
                annotation[0, 8] = label[10]  # l2_x
                annotation[0, 9] = label[11]  # l2_y
                annotation[0, 10] = label[13]  # l3_x
                annotation[0, 11] = label[14]  # l3_y
                annotation[0, 12] = label[16]  # l4_x
                annotation[0, 13] = label[17]  # l4_y
                if annotation[0, 4] < 0:
                    annotation[0, 14] = -1
                else:
                    annotation[0, 14] = 1
                annotations = np.append(annotations, annotation, axis=0)
            target = np.array(annotations)

            img, target = self.get_random_data(img, target, [self.img_size, self.img_size])

            # ----------------------------------------------------------#
            #   计算真实框对应的先验框，与这个先验框应当有的预测结果
            # ----------------------------------------------------------#
            assignment = self.bbox_util.assign_boxes(target)

            regression = assignment[:, : 5]
            classification = assignment[:, 5: 8]
            landms = assignment[:, 8:]

            inputs.append(img)
            regression_targets.append(regression)
            classification_targets.append(classification)
            landms_targets.append(landms)
        # preprocess_input 归一化预处理，第一个为图片，第二为图片中人脸参数
        return preprocess_input(np.array(inputs, np.float32)), [np.array(regression_targets, dtype=np.float32),
                                                                np.array(classification_targets, dtype=np.float32),
                                                                np.array(landms_targets, dtype=np.float32)]

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    # 数据增强
    def get_random_data(self, image, targes, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4):
        iw, ih = image.size
        h, w = input_shape
        box = targes

        # ------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        # ------------------------------------------#
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        PRE_SCALES = [3.33, 2.22, 1.67, 1.25, 1.0]
        scale = random.choice(PRE_SCALES)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # ------------------------------------------#
        #   将图像多余的部分加上灰条
        # ------------------------------------------#
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        flip = self.rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data = np.array(image, np.uint8)
        # ---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        # ---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # ---------------------------------#
        #   将图像转到HSV上
        # ---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        # ---------------------------------#
        #   应用变换
        # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        # ---------------------------------#
        #   对真实框进行调整
        # ---------------------------------#
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2, 4, 6, 8, 10, 12]] = box[:, [0, 2, 4, 6, 8, 10, 12]] * nw / iw + dx
            box[:, [1, 3, 5, 7, 9, 11, 13]] = box[:, [1, 3, 5, 7, 9, 11, 13]] * nh / ih + dy
            if flip:
                box[:, [0, 2, 4, 6, 8, 10, 12]] = w - box[:, [2, 0, 6, 4, 8, 12, 10]]
                box[:, [5, 7, 9, 11, 13]] = box[:, [7, 5, 9, 13, 11]]

            center_x = (box[:, 0] + box[:, 2]) / 2
            center_y = (box[:, 1] + box[:, 3]) / 2

            box = box[
                np.logical_and(np.logical_and(center_x > 0, center_y > 0), np.logical_and(center_x < w, center_y < h))]

            box[:, 0:14][box[:, 0:14] < 0] = 0
            box[:, [0, 2, 4, 6, 8, 10, 12]][box[:, [0, 2, 4, 6, 8, 10, 12]] > w] = w
            box[:, [1, 3, 5, 7, 9, 11, 13]][box[:, [1, 3, 5, 7, 9, 11, 13]] > h] = h

            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

        box[:, 4:-1][box[:, -1] == -1] = 0
        box[:, [0, 2, 4, 6, 8, 10, 12]] /= w
        box[:, [1, 3, 5, 7, 9, 11, 13]] /= h
        box_data = box
        return image_data, box_data

    def process_labels(self):
        imgs_path = []
        words = []
        f = open(self.txt_path, 'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = self.txt_path.replace('label.txt', 'images/') + path
                imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)
        words.append(labels)
        return imgs_path, words


# ------------------------------------#
#   数据加载器
# ------------------------------------#
class FacenetDataset(keras.utils.Sequence):
    def __init__(self, input_shape, lines, batch_size, num_classes, random):
        self.input_shape = input_shape
        self.lines = lines
        self.length = len(lines)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.random = random

        # ------------------------------------#
        #   路径和标签
        # ------------------------------------#
        self.paths = []
        self.labels = []

        self.load_dataset()

    def __len__(self):
        return math.ceil(self.length / float(self.batch_size))

    def __getitem__(self, index):
        # ------------------------------------#
        #   创建全为零的矩阵
        # ------------------------------------#
        images = np.zeros((self.batch_size // 3, 3, self.input_shape[0], self.input_shape[1], 3))
        labels = np.zeros((self.batch_size // 3, 3))

        for i in range(self.batch_size // 3):
            # ------------------------------------#
            #   先获得两张同一个人的人脸
            #   用来作为anchor和positive
            # ------------------------------------#
            c = random.randint(0, self.num_classes - 1)
            selected_path = self.paths[self.labels[:] == c]
            while len(selected_path) < 2:
                c = random.randint(0, self.num_classes - 1)
                selected_path = self.paths[self.labels[:] == c]

            # ------------------------------------#
            #   随机选择两张
            # ------------------------------------#
            image_indexes = np.random.choice(range(0, len(selected_path)), 2)
            # ------------------------------------#
            #   打开图片并放入矩阵
            # ------------------------------------#
            image = cvtColor(Image.open(selected_path[image_indexes[0]]))
            # ------------------------------------------#
            #   翻转图像
            # ------------------------------------------#
            if self.rand() < .5 and self.random:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox_image=True)
            image = preprocess_input(np.array(image, dtype='float32'))

            images[i, 0, :, :, :] = image
            labels[i, 0] = c

            image = cvtColor(Image.open(selected_path[image_indexes[1]]))
            # ------------------------------------------#
            #   翻转图像
            # ------------------------------------------#
            if self.rand() < .5 and self.random:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox_image=True)
            image = preprocess_input(np.array(image, dtype='float32'))

            images[i, 1, :, :, :] = image
            labels[i, 1] = c

            # ------------------------------#
            #   取出另外一个人的人脸
            # ------------------------------#
            different_c = list(range(self.num_classes))
            different_c.pop(c)

            different_c_index = np.random.choice(range(0, self.num_classes - 1), 1)
            current_c = different_c[different_c_index[0]]
            selected_path = self.paths[self.labels == current_c]
            while len(selected_path) < 1:
                different_c_index = np.random.choice(range(0, self.num_classes - 1), 1)
                current_c = different_c[different_c_index[0]]
                selected_path = self.paths[self.labels == current_c]

            # ------------------------------#
            #   随机选择一张
            # ------------------------------#
            image_indexes = np.random.choice(range(0, len(selected_path)), 1)
            image = cvtColor(Image.open(selected_path[image_indexes[0]]))
            # ------------------------------------------#
            #   翻转图像
            # ------------------------------------------#
            if self.rand() < .5 and self.random:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox_image=True)
            image = preprocess_input(np.array(image, dtype='float32'))

            images[i, 2, :, :, :] = image
            labels[i, 2] = current_c

        # --------------------------------------------------------------#
        #   假设batch为32
        #   0,32,64   属于一个组合 0和32是同一个人，0和64不是同一个人
        # --------------------------------------------------------------#
        images1 = np.array(images)[:, 0, :, :, :]
        images2 = np.array(images)[:, 1, :, :, :]
        images3 = np.array(images)[:, 2, :, :, :]
        images = np.concatenate([images1, images2, images3], 0)

        labels1 = np.array(labels)[:, 0]
        labels2 = np.array(labels)[:, 1]
        labels3 = np.array(labels)[:, 2]
        labels = np.concatenate([labels1, labels2, labels3], 0)

        labels = to_categorical(np.array(labels), num_classes=self.num_classes)

        return images, {'Embedding': np.zeros_like(labels), 'Softmax': labels}

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def load_dataset(self):
        for path in self.lines:
            path_split = path.split(";")
            self.paths.append(path_split[1].split()[0])
            self.labels.append(int(path_split[0]))
        self.paths = np.array(self.paths, dtype=np.object)
        self.labels = np.array(self.labels)


class LFWDataset():
    def __init__(self, dir, pairs_path, input_shape, batch_size):
        super(LFWDataset, self).__init__()
        self.input_shape = input_shape
        self.pairs_path = pairs_path
        self.batch_size = batch_size
        self.validation_images = self.get_lfw_paths(dir)

    def generate(self):
        images1 = []
        images2 = []
        issames = []
        for annotation_line in self.validation_images:
            (path_1, path_2, issame) = annotation_line
            image1, image2 = Image.open(path_1), Image.open(path_2)
            image1 = resize_image(image1, [self.input_shape[1], self.input_shape[0]], letterbox_image=True)
            image2 = resize_image(image2, [self.input_shape[1], self.input_shape[0]], letterbox_image=True)

            image1, image2 = preprocess_input(np.array(image1, np.float32)), preprocess_input(
                np.array(image2, np.float32))

            images1.append(image1)
            images2.append(image2)
            issames.append(issame)
            if len(images1) == self.batch_size:
                yield np.array(images1), np.array(images2), np.array(issames)
                images1 = []
                images2 = []
                issames = []

        yield np.array(images1), np.array(images2), np.array(issames)

    def read_lfw_pairs(self, pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)
        return np.array(pairs)

    def get_lfw_paths(self, lfw_dir, file_ext="jpg"):
        pairs = self.read_lfw_pairs(self.pairs_path)

        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []

        for i in range(len(pairs)):
            # for pair in pairs:
            pair = pairs[i]
            if len(pair) == 3:
                path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
                path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)
                issame = True
            elif len(pair) == 4:
                path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
                path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
                issame = False
            if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                path_list.append((path0, path1, issame))
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1
        if nrof_skipped_pairs > 0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)

        return path_list
