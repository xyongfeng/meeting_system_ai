from model_combination import ModelCombination
import cv2 as cv
import tensorflow as tf
import time

# 指定GPU
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

model_combination = ModelCombination()

def get_face_feature(path):
    """
    获取图片面部特征值
    :param path:
    :return:
    """
    img = cv.imread(path)
    feature = model_combination.get_face_feature(img)
    print(feature)


def run_img(path):
    """
    人脸检测一张图片，将识别出来的人脸框出来
    :param path:
    :return:
    """
    img = cv.imread(path)
    t = time.time()
    img = model_combination.detect_draw(img)
    # print(time.time() - t)
    cv.imshow('img', img)
    cv.waitKey(0)
    # print(res)


def run_viedo():
    """
    打开摄像头，进行实时人脸检测
    :return:
    """
    # 0 打开内置摄像头，1 打开外置摄像头
    capture = cv.VideoCapture(0)
    if not capture.isOpened():
        raise Exception("摄像头打开失败")
    fps = 0.0
    while True:
        t1 = time.time()
        ref, img = capture.read()
        if not ref:
            raise Exception("摄像头打开失败")

        img = model_combination.detect_draw(img)
        print(time.time() - t1)
        fps = (fps + (1. / (time.time() + 1e-5 - t1))) / 2
        print("fps= %.2f" % (fps))
        img = cv.putText(img, "fps= %.2f" % (fps), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.imshow('video', img)
        # 按q关闭窗口
        if cv.waitKey(1) & 0xff == ord('q'):
            break
    capture.release()
    # cv.destroyAllWindows()


if __name__ == '__main__':
    # 预测一张图片
    path = r'img_test/retinaface_0.jpg'
    # run_img(path)
    # 打开摄像头实时预测
    run_viedo()


