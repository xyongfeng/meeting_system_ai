cfg = {
    'name': 'mobilenet2',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'clip': False,
    'loc_weight': 2.0,
    'train_image_size': 640,  # 为32倍数，不然keras可能会出现偏差BUG
    'out_channel': 64
}
