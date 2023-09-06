import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf


def retinaface_correct_boxes(result, input_shape, image_shape):
    """
    将输出调整为相对于原图的大小，可以将灰条带来的影响去除
    :param result: 带灰条的坐标信息
    :param input_shape: 带灰条的图片shape
    :param image_shape: 真实图片的shape
    :return: 修改之后的坐标信息
    """
    # 将image_shape按照原来的宽长比例放大到一条边与input_shape一条边相等
    new_shape = image_shape * np.min(input_shape / image_shape)

    # 求出没有灰条之后的xy坐标偏移比例
    offset = (input_shape - new_shape) / 2. / input_shape
    # input与new的shape缩放比例
    scale = input_shape / new_shape

    # 边框的左上角右下角坐标 2 + 2
    scale_for_boxs = [scale[1], scale[0], scale[1], scale[0]]
    # 检测点的5个坐标，5 * 2
    scale_for_landmarks = [scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1],
                           scale[0]]
    # 同上
    offset_for_boxs = [offset[1], offset[0], offset[1], offset[0]]
    offset_for_landmarks = [offset[1], offset[0], offset[1], offset[0], offset[1], offset[0], offset[1], offset[0],
                            offset[1], offset[0]]
    # 减去偏移量再乘上缩放比例
    result[:, :4] = (result[:, :4] - np.array(offset_for_boxs)) * np.array(scale_for_boxs)
    result[:, 5:] = (result[:, 5:] - np.array(offset_for_landmarks)) * np.array(scale_for_landmarks)

    return result


class BBoxUtility(object):
    def __init__(self, anchors=None, overlap_threshold=0.35, top_k=300, nms_thresh=0.45):
        self.anchors = anchors
        self.num_anchors = 0 if anchors is None else len(anchors)
        self.overlap_threshold = overlap_threshold
        self._top_k = top_k
        self._nms_thresh = nms_thresh


    def iou(self, box):
        # ---------------------------------------------#
        #   计算出每个真实框与所有的先验框的iou
        #   判断真实框与先验框的重合情况
        # ---------------------------------------------#
        inter_upleft = np.maximum(self.anchors[:, :2], box[:2])
        inter_botright = np.minimum(self.anchors[:, 2:4], box[2:])

        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # ---------------------------------------------#
        #   真实框的面积
        # ---------------------------------------------#
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        # ---------------------------------------------#
        #   先验框的面积
        # ---------------------------------------------#
        area_gt = (self.anchors[:, 2] - self.anchors[:, 0]) * (self.anchors[:, 3] - self.anchors[:, 1])
        # ---------------------------------------------#
        #   计算iou
        # ---------------------------------------------#
        union = area_true + area_gt - inter

        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True):
        # ---------------------------------------------#
        #   计算当前真实框和先验框的重合情况
        #   iou [self.num_anchors]
        #   encoded_box [self.num_anchors, 5]
        # ---------------------------------------------#
        iou = self.iou(box[:4])
        encoded_box = np.zeros((self.num_anchors, 4 + return_iou + 10 + 1))

        # ---------------------------------------------#
        #   找到每一个真实框，重合程度较高的先验框
        #   真实框可以由这个先验框来负责预测
        # ---------------------------------------------#
        assign_mask = iou > self.overlap_threshold

        # ---------------------------------------------#
        #   如果没有一个先验框重合度大于self.overlap_threshold
        #   则选择重合度最大的为正样本
        # ---------------------------------------------#
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True

        # ---------------------------------------------#
        #   利用iou进行赋值
        # ---------------------------------------------#
        if return_iou:
            encoded_box[:, 4][assign_mask] = iou[assign_mask]

        # ---------------------------------------------#
        #   找到对应的先验框
        # ---------------------------------------------#
        assigned_anchors = self.anchors[assign_mask]

        # ----------------------------------------------------#
        #   逆向编码，将真实框转化为Retinaface预测结果的格式
        #   先计算真实框的中心与长宽
        # ----------------------------------------------------#
        box_center = 0.5 * (box[:2] + box[2:4])
        box_wh = box[2:4] - box[:2]
        # ---------------------------------------------#
        #   再计算重合度较高的先验框的中心与长宽
        # ---------------------------------------------#
        assigned_anchors_center = 0.5 * (assigned_anchors[:, :2] +
                                         assigned_anchors[:, 2:4])
        assigned_anchors_wh = (assigned_anchors[:, 2:4] -
                               assigned_anchors[:, :2])

        # ------------------------------------------------#
        #   逆向求取应该有的预测结果
        # ------------------------------------------------#
        encoded_box[:, :2][assign_mask] = box_center - assigned_anchors_center
        encoded_box[:, :2][assign_mask] /= assigned_anchors_wh
        encoded_box[:, :2][assign_mask] /= 0.1

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_anchors_wh)
        encoded_box[:, 2:4][assign_mask] /= 0.2

        ldm_encoded = np.zeros_like(encoded_box[:, 5: -1][assign_mask])
        ldm_encoded = np.reshape(ldm_encoded, [-1, 5, 2])

        ldm_encoded[:, :, 0] = box[[4, 6, 8, 10, 12]] - np.repeat(assigned_anchors_center[:, 0: 1], 5, axis=-1)
        ldm_encoded[:, :, 1] = box[[5, 7, 9, 11, 13]] - np.repeat(assigned_anchors_center[:, 1: 2], 5, axis=-1)

        ldm_encoded[:, :, 0] /= np.repeat(assigned_anchors_wh[:, 0:1], 5, axis=-1)
        ldm_encoded[:, :, 1] /= np.repeat(assigned_anchors_wh[:, 1:2], 5, axis=-1)

        ldm_encoded[:, :, 0] /= 0.1
        ldm_encoded[:, :, 1] /= 0.1

        encoded_box[:, 5:-1][assign_mask] = np.reshape(ldm_encoded, [-1, 10])
        encoded_box[:, -1][assign_mask] = box[-1]
        return encoded_box.ravel()

    def assign_boxes(self, boxes):
        # -----------------------------------------------------#
        #   assignment分为3个部分
        #   :4      的内容为网络应该有的回归预测结果
        #   4       的内容为是否包含物体
        #
        #   5:6     的内容为包含物体的概率
        #   7       的内容为是否包含物体
        #
        #   8:-1    的内容为特征点应该有的回归预测结果
        #   -1      的内容为是否包含物体
        # -----------------------------------------------------#
        assignment = np.zeros((self.num_anchors, 4 + 1 + 2 + 1 + 10 + 1))
        # -----------------------------------------------------#
        #   序号为5的地方是为背景的概率
        # -----------------------------------------------------#
        assignment[:, 5] = 1
        if len(boxes) == 0:
            return assignment

        # -----------------------------------------------------#
        #   每一个真实框的编码后的值，和iou
        #   encoded_boxes   n, num_anchors, 16
        # -----------------------------------------------------#
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes)
        encoded_boxes = encoded_boxes.reshape(-1, self.num_anchors, 16)

        # -----------------------------------------------------#
        #   取出每个先验框重合度最大的真实框
        # -----------------------------------------------------#
        best_iou = encoded_boxes[:, :, 4].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, 4].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]

        assign_num = len(best_iou_idx)
        # -----------------------------------------------------#
        #   将编码后的真实框取出
        # -----------------------------------------------------#
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
        # -----------------------------------------------------#
        #   4、7和-1代表为当前先验框是否包含目标
        # -----------------------------------------------------#
        assignment[:, 4][best_iou_mask] = 1

        # -----------------------------------------------------#
        #   5:6     的内容为包含物体的概率
        #   7       的内容为是否包含物体
        # -----------------------------------------------------#
        assignment[:, 5][best_iou_mask] = 0
        assignment[:, 6][best_iou_mask] = 1
        assignment[:, 7][best_iou_mask] = 1

        # -----------------------------------------------------#
        #   8:-1    的内容为特征点应该有的回归预测结果
        #   -1      的内容为是否包含物体
        # -----------------------------------------------------#
        assignment[:, 8:][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), 5:]
        return assignment

    def cal_iou(self, b1, b2):
        b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)

        inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                     np.maximum(inter_rect_y2 - inter_rect_y1, 0)

        area_b1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area_b2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        iou = inter_area / np.maximum((area_b1 + area_b2 - inter_area), 1e-6)
        return iou

    def decode_boxes(self, mbox_loc, mbox_ldm, mbox_anchorbox):
        # 获得先验框的宽高
        anchor_w = mbox_anchorbox[:, 2] - mbox_anchorbox[:, 0]
        anchor_h = mbox_anchorbox[:, 3] - mbox_anchorbox[:, 1]

        # 获得先验框的中心点
        anchor_center_x = 0.5 * (mbox_anchorbox[:, 2] + mbox_anchorbox[:, 0])
        anchor_center_y = 0.5 * (mbox_anchorbox[:, 3] + mbox_anchorbox[:, 1])
        # 真实框距离先验框中心的xy轴偏移情况
        decode_bbox_center_x = anchor_center_x + mbox_loc[:, 0] * anchor_w * 0.1
        decode_bbox_center_y = anchor_center_y + mbox_loc[:, 1] * anchor_h * 0.1

        # 真实框的宽与高的求取
        decode_bbox_width = np.exp(mbox_loc[:, 2] * 0.2) * anchor_w
        decode_bbox_height = np.exp(mbox_loc[:, 3] * 0.2) * anchor_h

        # 获取真实框的左上角与右下角
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        anchor_width = np.expand_dims(anchor_w, -1)
        anchor_height = np.expand_dims(anchor_h, -1)
        anchor_center_x = np.expand_dims(anchor_center_x, -1)
        anchor_center_y = np.expand_dims(anchor_center_y, -1)
        # 对先验框的中心进行调整获得五个人脸关键点
        mbox_ldm = mbox_ldm.reshape([-1, 5, 2])
        decode_ldm = np.zeros_like(mbox_ldm)

        # print(decode_ldm.shape,anchor_width.shape,mbox_loc.shape,anchor_center_x.shape)
        decode_ldm[:, :, 0] = np.repeat(anchor_width, 5, axis=-1) * mbox_ldm[:, :, 0] * 0.1 + np.repeat(anchor_center_x,
                                                                                                        5, axis=-1)

        decode_ldm[:, :, 1] = np.repeat(anchor_height, 5, axis=-1) * mbox_ldm[:, :, 1] * 0.1 + np.repeat(
            anchor_center_y, 5, axis=-1)

        # 将真实框左上角和右下角进行堆叠
        # None存在与哪个维度上就代表给这个维度上再扩展一维度
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None],
                                      decode_ldm.reshape((-1, 10))), axis=-1)
        # 防止超出0与1
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        return decode_bbox

    def detection_out(self, predictions, mbox_anchorbox, confidence_threshold=0.5):
        # 回归预测结果
        mbox_loc = predictions[0][0]
        # 是否为人脸种类预测概率
        mbox_conf = predictions[1][0][:, 1:2]

        # 人脸关键点预测结果
        mbox_ldm = predictions[2][0]
        # 开始解码
        decode_bbox = self.decode_boxes(mbox_loc, mbox_ldm, mbox_anchorbox)
        # print(np.shape(decode_bbox))
        # 取出人脸置信度（概率）高的一部分
        conf_mask = (mbox_conf >= confidence_threshold)[:, 0]

        # 将预测框左上角右下角,置信度，人脸预测框堆叠起来
        # shape (n,15) 4 + 1 + 10
        detection = np.concatenate((decode_bbox[conf_mask][:, :4], mbox_conf[conf_mask],
                                    decode_bbox[conf_mask][:, 4:]), axis=-1)
        # print('detection', detection.shape)
        # NMS 非极大抑制

        idx = tf.image.non_max_suppression(detection[:, :4], detection[:, 4], self._top_k,
                                           iou_threshold=self._nms_thresh)

        best_box = detection[np.array(idx)]
        return best_box
