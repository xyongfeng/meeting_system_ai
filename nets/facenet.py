import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation, Dense, Input, Lambda
from tensorflow.keras.models import Model
from .mobilenet2 import MobileNet_FaceNet


def facenet(input_shape, num_classes=None, mode='train'):
    inputs = Input(shape=input_shape)
    model = MobileNet_FaceNet(inputs)
    if mode == "train":
        # 训练的话利用交叉熵和triplet_loss,结合一起训练
        logits = Dense(num_classes)(model.output)
        softmax = Activation("softmax", name="Softmax")(logits)
        normalize = Lambda(lambda x: K.l2_normalize(x, axis=1), name="Embedding")(model.output)
        combine_model = Model(inputs, [softmax, normalize])
        return combine_model
    elif mode == 'predict':
        # 预测的时候只需要考虑人脸的特征向量就行了
        x = Lambda(lambda x: K.l2_normalize(x, axis=1), name="Embedding")(model.output)
        model = Model(inputs, x)
        return model
    else:
        raise ValueError('Unsupported mode - `{}`, Use train, predict.'.format(mode))

