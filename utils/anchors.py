from itertools import product as product
from math import ceil

import numpy as np


class Anchors(object):
    def __init__(self, cfg, image_size=None):
        # 先验框基础边长，注意这里并不是代表一个先验框的宽高，而是代表两个先验框的边长，应该是表示一格有两个边长不同的先验框
        self.min_sizes = cfg['min_sizes']
        # 三个特征层的压缩倍数
        self.steps = cfg['steps']
        # 结果是否为0~1之间
        self.clip = cfg['clip']
        # 图片大小
        self.image_size = image_size
        # 这里表示将原图的xy轴分别划分成了多少份，结合起来就是将原图划分成了许多个格子
        self.feature_maps = [
            [ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)]
            for step in self.steps
        ]

        # print('feature_maps', self.feature_maps)

    def get_anchors(self):
        anchors = []
        # 遍历这些格子，在格子中生成先验框
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            # 这里对三个划分格子方案中的其中一个进行操作
            # 循环遍历每个格子
            # print(list(product(range(f[0]), range(f[1]))))
            for i, j in product(range(f[0]), range(f[1])):
                # 为每个格子生成min_sizes中相应边长的先验框
                for min_size in min_sizes:
                    # 宽和高转换为与原图的比例
                    # 宽
                    s_kx = min_size / self.image_size[1]
                    # 高
                    s_ky = min_size / self.image_size[0]
                    # 计算出每一个先验框的中心(注意这里除以了原图宽高，所以得到的只是比例) 加0.5是将ij四舍五入
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    # 将这些先验框添加到anchor里面

                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        anchors = np.reshape(anchors, (-1, 4))
        output = np.zeros_like(anchors[:, :4])
        # 将先验框的形式转换为左上角右下角的形式

        # 左上角的x轴
        output[:, 0] = anchors[:, 0] - anchors[:, 2] / 2
        # 左上角的y轴
        output[:, 1] = anchors[:, 1] - anchors[:, 3] / 2
        # 右下角的x轴
        output[:, 2] = anchors[:, 0] + anchors[:, 2] / 2
        # 右下角的y轴
        output[:, 3] = anchors[:, 1] + anchors[:, 3] / 2

        # 是否将output限制到0~1，默认为false
        if self.clip:
            output = np.clip(output, 0, 1)
        return output
