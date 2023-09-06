import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Activation, Add, Concatenate, Conv2D, Input, Reshape
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from utils.utils import compose

from .mobilenet2 import MobileNet2_Retinaface


# keras.initializers.
# 上采样
class UpsampleLike(keras.layers.Layer):
    # 操作input的核心方法
    def call(self, inputs, **kwargs):
        # 因为输入的是两个层，要将source按照target的形状进行上采样
        source, target = inputs
        target_shape = keras.backend.shape(target)
        # 采用上采样的方法是最近的邻居插值
        # tf.image.resize()
        return tf.image.resize(source, (target_shape[1], target_shape[2]),
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # 计算图层的输出形状。
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)


# Conv2D + BatchNormalization + LeakyReLU
def Conv2D_BN_Leaky(*args, **kwargs):
    leaky = 0.1
    try:
        leaky = kwargs["leaky"]
        del kwargs["leaky"]
    except:
        pass
    return compose(
        Conv2D(*args, **kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=leaky))


# Conv2D + BatchNormalization
def Conv2D_BN(*args, **kwargs):
    return compose(
        Conv2D(*args, **kwargs),
        BatchNormalization())


# 使用了三个并行结构,多尺度加强感受野
def SSH(inputs, out_channel, leaky=0.1):
    # 假设 inputs的shape为40,40,64
    # 3x3卷积 40,40,64 -> 40,40,32
    conv3X3 = Conv2D_BN(out_channel // 2, kernel_size=3, strides=1, kernel_initializer=RandomNormal(stddev=0.02),
                        padding='same')(inputs)

    # 利用两个3x3卷积替代5x5卷积 40,40,64 -> 40,40,16
    conv5X5_1 = Conv2D_BN_Leaky(out_channel // 4, kernel_size=3, strides=1,
                                kernel_initializer=RandomNormal(stddev=0.02), padding='same', leaky=leaky)(inputs)
    conv5X5 = Conv2D_BN(out_channel // 4, kernel_size=3, strides=1, kernel_initializer=RandomNormal(stddev=0.02),
                        padding='same')(conv5X5_1)

    # 利用三个3x3卷积替代7x7卷积 40,40,64 -> 40,40,16
    conv7X7_2 = Conv2D_BN_Leaky(out_channel // 4, kernel_size=3, strides=1,
                                kernel_initializer=RandomNormal(stddev=0.02), padding='same', leaky=leaky)(conv5X5_1)
    conv7X7 = Conv2D_BN(out_channel // 4, kernel_size=3, strides=1, kernel_initializer=RandomNormal(stddev=0.02),
                        padding='same')(conv7X7_2)
    # 所有结果堆叠起来
    out = Concatenate(axis=-1)([conv3X3, conv5X5, conv7X7])
    out = Activation("relu")(out)
    return out


#   种类预测（是否包含人脸）
def ClassHead(inputs, num_anchors=2):
    outputs = Conv2D(num_anchors * 2, kernel_size=1, kernel_initializer=RandomNormal(stddev=0.02), strides=1)(inputs)
    return Activation("softmax")(Reshape([-1, 2])(outputs))


#   预测框预测
def BboxHead(inputs, num_anchors=2):
    outputs = Conv2D(num_anchors * 4, kernel_size=1, kernel_initializer=RandomNormal(stddev=0.02), strides=1)(inputs)
    outputs = Reshape([-1, 4])(outputs)
    return outputs


#   人脸关键点预测
def LandmarkHead(inputs, num_anchors=2):
    outputs = Conv2D(num_anchors * 5 * 2, kernel_size=1, kernel_initializer=RandomNormal(stddev=0.02), strides=1)(
        inputs)
    return Reshape([-1, 10])(outputs)


def RetinaFace(cfg):
    inputs = Input((None,None, 3))
    # FPN特征金字塔 从MobileNet获得3个特征层
    C3, C4, C5 = MobileNet2_Retinaface(inputs)

    leaky = 0
    if cfg['out_channel'] <= 64:
        leaky = 0.1

    #   获得三个shape的有效特征层
    #   分别是 P3 (80, 80, 64)
    #         P4 (40, 40, 64)
    #         P5 (20, 20, 64)

    P3 = Conv2D_BN_Leaky(cfg['out_channel'],
                         kernel_size=1,
                         strides=1,
                         padding='same',
                         kernel_initializer=RandomNormal(stddev=0.02),
                         name='C3_reduced',
                         leaky=leaky)(C3)
    P4 = Conv2D_BN_Leaky(cfg['out_channel'],
                         kernel_size=1,
                         strides=1,
                         padding='same',
                         kernel_initializer=RandomNormal(stddev=0.02),
                         name='C4_reduced',
                         leaky=leaky)(C4)
    P5 = Conv2D_BN_Leaky(cfg['out_channel'],
                         kernel_size=1,
                         strides=1,
                         padding='same',
                         kernel_initializer=RandomNormal(stddev=0.02),
                         name='C5_reduced',
                         leaky=leaky)(C5)
    # P5到P4上采样
    P5_upsampled = UpsampleLike(name='P5_upsampled')([P5, P4])
    # 上采样后的P5与P4进行特征融合
    P4 = Add(name='P4_merged')([P5_upsampled, P4])
    # 再进行卷积
    P4 = Conv2D_BN_Leaky(cfg['out_channel'],
                         kernel_size=3,
                         strides=1,
                         padding='same',
                         kernel_initializer=RandomNormal(stddev=0.02),
                         name='Conv_P4_merged',
                         leaky=leaky)(P4)
    # P4上采样和P3特征融合
    P4_upsampled = UpsampleLike(name='P4_upsampled')([P4, P3])
    P3 = Add(name='P3_merged')([P4_upsampled, P3])
    P3 = Conv2D_BN_Leaky(cfg['out_channel'], kernel_size=3, strides=1, padding='same',
                         kernel_initializer=RandomNormal(stddev=0.02), name='Conv_P3_merged', leaky=leaky)(P3)

    # 再对三个特征层通过SSH进一步加强特征提取
    SSH1 = SSH(P3, cfg['out_channel'], leaky=leaky)
    SSH2 = SSH(P4, cfg['out_channel'], leaky=leaky)
    SSH3 = SSH(P5, cfg['out_channel'], leaky=leaky)

    SSH_all = [SSH1, SSH2, SSH3]

    # 将所有结果进行堆叠
    # 人脸预测边框
    bbox_regressions = Concatenate(axis=1, name="bbox_reg")([BboxHead(feature) for feature in SSH_all])
    # 是否存在人脸
    classifications = Concatenate(axis=1, name="cls")([ClassHead(feature) for feature in SSH_all])
    # 人脸5个特征点
    ldm_regressions = Concatenate(axis=1, name="ldm_reg")([LandmarkHead(feature) for feature in SSH_all])
    # print(bbox_regressions.shape,classifications.shape,ldm_regressions.shape)
    output = [bbox_regressions, classifications, ldm_regressions]

    model = Model(inputs=inputs, outputs=output)
    return model
