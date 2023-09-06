from flask import Flask, request
import base64
from my_flask.Service import FlaskService
from my_flask.JsonData import JsonData
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)
flaskService = FlaskService()


def Base64ToImg(imgBase64):
    img = base64.b64decode(imgBase64)
    img = np.asarray(bytearray(img), dtype="uint8")
    img = cv.imdecode(img, cv.IMREAD_COLOR)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


@app.route("/predict", methods=['post'])
def predict():
    """
    预测上传的照片和本地照片是否为同一人
    :return:
    """
    sendImgBase64 = request.json.get('sendImgBase64')
    localFeature = request.json.get('localFeature')
    if sendImgBase64 is None or localFeature is None:
        return JsonData.error("imgBase64和localFeature不能为空")

    sendImg = Base64ToImg(sendImgBase64)

    return flaskService.predict_between_two(sendImg, localImg=None, localFeature=localFeature)


def run_face_feature(funtion):
    """
    公共函数
    :param funtion:
    :return:
    """
    imgBase64 = request.json.get('imgBase64')
    if imgBase64 is None:
        return JsonData.error("imgBase64不能为空")
    img = Base64ToImg(imgBase64)
    return funtion(img, request.json)


@app.route("/feature", methods=['post'])
def up_face_feature():
    """
    获取该图片中人脸的特征
    :return:
    """
    def run(img, json_data):
        return flaskService.get_face_feature(img)

    return run_face_feature(run)


@app.route("/login", methods=['post'])
def login_with_face_feature():
    """
    人脸识别登录
    :return:
    """
    def run(img, json_data):
        features = json_data.get('features')
        if features is None: features = []
        return flaskService.login_with_face_feature(img, features)

    return run_face_feature(run)


if __name__ == '__main__':
    app.run()

