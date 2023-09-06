config = {
    # 模型权重文件路径
    "retinface_weight": 'weight/retinface/ep002-loss5.388.h5',
    "facenet_weight": 'weight/facenet/ep040-loss0.358.h5',
    # 识别阈值，当两个特征差小于tolerance时，则被判断为同一人
    "tolerance": 1,
}
