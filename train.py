from utils.dataloader import Generator, FacenetDataset
from utils.anchors import Anchors
from utils.config import cfg
from utils.utils_bbox import BBoxUtility
from nets.retinaface import RetinaFace
from nets.facenet import facenet
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from nets.model_training import (box_smooth_l1, conf_loss,
                                 get_lr_scheduler, ldm_smooth_l1, triplet_loss)
from tensorflow.keras.callbacks import (EarlyStopping, LearningRateScheduler,
                                        ModelCheckpoint, TensorBoard, )
from utils.utils import get_num_classes
import numpy as np
import os
import datetime
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
if __name__ == '__main__':

    # retinaface facenet 选择要训练的模型以及对应数据集
    train_model = 'facenet'
    # 指定数据集的路径
    train_path = r'D:\face_data\widerface\train\label.txt'

    # 初始化轮数
    init_epoch = 0
    # 训练总轮数
    epoch = 150
    # 训练批次大小 如果是训练facenet 必须为3的倍数
    batch_size = 96  # 8
    # 权值与日志文件保存的文件夹
    save_dir = f'logs_{train_model}'

    # 权值衰减，可防止过拟合
    # adam会导致weight_decay错误，使用adam时建议设置为0。 5e-4
    weight_decay = 0
    # 优化器内部使用到的momentum参数
    momentum = 0.937
    #   Init_lr         模型的最大学习率
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    Init_lr = 1e-3
    Min_lr = Init_lr * 0.01

    #   lr_decay_type   使用到的学习率下降方式，可选的有'step'、'cos'
    lr_decay_type = 'cos'

    #  facenet 是否开启LFW评估
    # lfw_eval_flag = True

    if train_model == 'retinaface':
        # 生成先验框
        anchors = Anchors(cfg, image_size=(cfg['train_image_size'], cfg['train_image_size'])).get_anchors()
        bbox_util = BBoxUtility(anchors)
        # 收集并处理训练数据
        train_dataloader = Generator(train_path, cfg['train_image_size'], batch_size, bbox_util)
        # 载入模型
        model = RetinaFace(cfg)
    elif train_model == 'facenet':

        input_shape = [160, 160, 3]
        # 读取人脸路径与标签
        annotation_path = "cls_train.txt"
        num_classes = get_num_classes(annotation_path)
        model = facenet(input_shape=input_shape,
                        num_classes=num_classes,
                        mode='train')

        # batch_size = 6
        # 0.01用于验证，0.99用于训练
        val_split = 0.01
        with open(annotation_path, "r") as f:
            lines = f.readlines()
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)
        num_val = int(len(lines) * val_split)
        num_train = len(lines) - num_val

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        # 读取数据集
        train_dataset = FacenetDataset(input_shape, lines[:num_train], batch_size, num_classes, random=True)
        val_dataset = FacenetDataset(input_shape, lines[num_train:], batch_size, num_classes, random=False)

    else:
        raise ValueError("train_model 只能是 retinaface or facenet")

    # for layer in model.layers:
    #     if isinstance(layer, DepthwiseConv2D):
    #         layer.add_loss(l2(weight_decay)(layer.depthwise_kernel))
    #     elif isinstance(layer, Conv2D) or isinstance(layer, Dense):
    #         layer.add_loss(l2(weight_decay)(layer.kernel))

    # -------------------------------------------------------------------#
    #   判断当前batch_size，自适应调整学习率
    # -------------------------------------------------------------------#
    nbs = 64
    lr_limit_max = 1e-3
    lr_limit_min = 3e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    optimizer = Adam(lr=Init_lr_fit, beta_1=momentum)
    if train_model == 'retinaface':
        model.compile(optimizer, loss={
            'bbox_reg': box_smooth_l1(weights=cfg['loc_weight']),
            'cls': conf_loss(),
            'ldm_reg': ldm_smooth_l1()
        })
    else:
        model.compile(
            loss={'Embedding': triplet_loss(batch_size=batch_size // 3), 'Softmax': 'categorical_crossentropy'},
            optimizer=optimizer, metrics={'Softmax': 'categorical_accuracy'}
        )

    #   获得学习率下降的公式
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, epoch)

    # -------------------------------------------------------------------------------#
    #   训练参数的设置
    #   logging         用于设置tensorboard的保存地址
    #   checkpoint      用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   lr_scheduler       用于设置学习率下降的方式
    #   early_stopping  用于设定早停，loss多次不下降自动结束训练，表示模型基本收敛
    # -------------------------------------------------------------------------------#
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    logging = TensorBoard(log_dir)
    # loss_history = LossHistory(log_dir)
    checkpoint = ModelCheckpoint(os.path.join(save_dir, "ep{epoch:03d}-loss{loss:.3f}.h5"),
                                 monitor='loss', save_weights_only=True, save_best_only=False, period=1)
    early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1)
    lr_scheduler = LearningRateScheduler(lr_scheduler_func, verbose=1)
    callbacks = [logging, checkpoint, lr_scheduler]

    if train_model == 'retinaface':
        model.fit(
            train_dataloader,
            steps_per_epoch=train_dataloader.get_len() // batch_size,
            epochs=epoch,
            callbacks=callbacks,
        )
    else:
        model.fit(
            train_dataset,
            steps_per_epoch=epoch_step,
            validation_data=val_dataset,
            validation_steps=epoch_step_val,
            epochs=epoch,
            initial_epoch=init_epoch,
            callbacks=callbacks
        )
