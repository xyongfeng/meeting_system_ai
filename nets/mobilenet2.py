import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import backend


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    Args:
      inputs: Input tensor.
      kernel_size: An integer or tuple/list of 2 integers.
    Returns:
      A tuple.
    """
    img_dim = 2 if backend.image_data_format() == "channels_first" else 1
    input_size = backend.int_shape(inputs)[img_dim: (img_dim + 2)]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return (
        (correct[0] - adjust[0], correct[0]),
        (correct[1] - adjust[1], correct[1]),
    )


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, channel_axis=-1):
    in_channels = backend.int_shape(inputs)[channel_axis]
    prefix = f'block_{block_id}_'
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs

    # pointwise 1x1 升维
    x = layers.Conv2D(
        expansion * in_channels,
        kernel_size=1,
        padding='same',
        use_bias=False,
        activation=None,
        name=prefix + 'expand'
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'expand_BN'
    )(x)
    x = layers.ReLU(6., name=prefix + 'expand_relu')(x)

    # Depthwise 3x3
    if stride == 2:
        x = layers.ZeroPadding2D(
            padding=correct_pad(x, 3),
            name=prefix + "pad"
        )(x)

    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        activation=None,
        use_bias=False,
        padding='same' if stride == 1 else 'valid',
        name=prefix + 'depthwise'
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'depthwise_BN'
    )(x)

    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # pointwise 1x1 降维 线性（无激活函数）
    x = layers.Conv2D(
        pointwise_filters,
        kernel_size=1,
        padding='same',
        use_bias=False,
        activation=None,
        name=prefix + 'project'
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'project_BN'
    )(x)

    if in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x


def MobileNet2_Retinaface(inputs, alpha=1.0, expansion=6):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    first_block_filters = _make_divisible(8 * alpha, 8)
    # 640,640,3 -> 320,320,8
    x = layers.Conv2D(
        first_block_filters,
        kernel_size=3,
        strides=(2, 2),
        padding='same',
        use_bias=False,
        name='Conv1'
    )(inputs)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name='bn_Conv1'
    )(x)
    x = layers.ReLU(6., name='Conv1_relu')(x)
    # 320,320,8 -> 320,320,16

    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)

    # 320,320,16 -> 160,160,32
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2, expansion=expansion, block_id=1)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=expansion, block_id=2)

    # 160,160,32 -> 80,80,64
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2, expansion=expansion, block_id=3)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=expansion, block_id=4)
    feat1 = x

    # 80,80,64 -> 40,40,128
    x = _inverted_res_block(x, filters=128, alpha=alpha, stride=2, expansion=expansion, block_id=5)
    x = _inverted_res_block(x, filters=128, alpha=alpha, stride=1, expansion=expansion, block_id=6)
    x = _inverted_res_block(x, filters=128, alpha=alpha, stride=1, expansion=expansion, block_id=7)
    x = _inverted_res_block(x, filters=128, alpha=alpha, stride=1, expansion=expansion, block_id=8)
    x = _inverted_res_block(x, filters=128, alpha=alpha, stride=1, expansion=expansion, block_id=9)
    x = _inverted_res_block(x, filters=128, alpha=alpha, stride=1, expansion=expansion, block_id=10)
    feat2 = x

    # 40,40,128 -> 20,20,256
    x = _inverted_res_block(x, filters=256, alpha=alpha, stride=2, expansion=expansion, block_id=11)
    x = _inverted_res_block(x, filters=256, alpha=alpha, stride=1, expansion=expansion, block_id=12)
    feat3 = x
    return feat1, feat2, feat3


def MobileNet_FaceNet(inputs, dropout_keep_prob=0.4, alpha=1.0, expansion=6):
    # 160,160,3 -> 80,80,32
    x = layers.Conv2D(32,
                      kernel_size=(3, 3),
                      strides=(2, 2),
                      kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                      padding='same',
                      use_bias=False,
                      name='conv1')(inputs)
    x = layers.BatchNormalization(name='conv1_bn')(x)
    x = layers.ReLU(6., name='Conv1_relu')(x)

    # 80,80,32 -> 80,80,64
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=1, block_id=0)

    # 80,80,64 -> 40,40,128
    x = _inverted_res_block(x, filters=128, alpha=alpha, stride=2, expansion=expansion, block_id=1)
    x = _inverted_res_block(x, filters=128, alpha=alpha, stride=1, expansion=expansion, block_id=2)

    # 40,40,128 -> 20,20,256
    x = _inverted_res_block(x, filters=256, alpha=alpha, stride=2, expansion=expansion, block_id=3)
    x = _inverted_res_block(x, filters=256, alpha=alpha, stride=1, expansion=expansion, block_id=4)

    # 20,20,256 -> 10,10,512
    x = _inverted_res_block(x, filters=512, alpha=alpha, stride=2, expansion=expansion, block_id=5)
    x = _inverted_res_block(x, filters=512, alpha=alpha, stride=1, expansion=expansion, block_id=7)
    x = _inverted_res_block(x, filters=512, alpha=alpha, stride=1, expansion=expansion, block_id=8)
    x = _inverted_res_block(x, filters=512, alpha=alpha, stride=1, expansion=expansion, block_id=9)
    x = _inverted_res_block(x, filters=512, alpha=alpha, stride=1, expansion=expansion, block_id=10)
    x = _inverted_res_block(x, filters=512, alpha=alpha, stride=1, expansion=expansion, block_id=11)

    # 10,10,512 -> 5,5,1024
    x = _inverted_res_block(x, filters=1024, alpha=alpha, stride=2, expansion=expansion, block_id=12)
    x = _inverted_res_block(x, filters=1024, alpha=alpha, stride=1, expansion=expansion, block_id=13)

    # 1024
    # 全局平均池化，只会得到1024
    x = layers.GlobalAveragePooling2D()(x)

    # 随机丢掉一部分神经元
    x = layers.Dropout(rate=1 - dropout_keep_prob, name='Dropout')(x)

    # 全连接层到128
    # 128
    x = layers.Dense(128, use_bias=False, name='Bottleneck')(x)
    x = layers.BatchNormalization(momentum=0.995, epsilon=0.001, scale=False, name='BatchNorm_Bottleneck')(x)

    model = keras.Model(inputs, x, name='mobilenet2')
    return model


if __name__ == '__main__':
    # first_block_filters = _make_divisible(16, 8)
    # print(first_block_filters)
    pass
