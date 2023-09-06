from model_combination import ModelCombination
from .JsonData import JsonData
from utils.utils import compare_faces
from config import config
model = ModelCombination()


class FlaskService:
    tolerance = config['tolerance']

    def predict_between_two(self, sendImg, *, localImg, localFeature: str = None, tolerance=None):
        """
        预测两张人脸的距离是否小于阈值
        :param sendImg:
        :param localImg: 本地的图片
        :param localFeature: 本地已经提取到的特征
        :param tolerance:
        :return:
        """
        if not tolerance: tolerance = self.tolerance

        sendEncoding = model.get_face_feature(sendImg)
        if localImg != None:
            localEncoding = model.get_face_feature(localImg)
        else:
            localEncoding = list(map(float, localFeature.split(',')))

        if sendEncoding is None:
            return JsonData.error("检测失败，上传的图片中没有人脸")
        if localEncoding is None:
            return JsonData.error("检测失败，请更新个人中心里的面部照片")

        matche, face_distances = compare_faces(sendEncoding, localEncoding,
                                               tolerance=tolerance)
        print(face_distances, f"tolerance:{tolerance}")
        if not matche:
            return JsonData.error("检测失败，不是本人")
        return JsonData.success("检测成功", {'distance': str(face_distances)})

    def __run_face_feature(self, img, funtion):
        feature = model.get_face_feature(img)
        if feature is None:
            return JsonData.error("请上传面部照片")
        return funtion(feature)

    def get_face_feature(self, img):
        def run(fea):
            fea = map(str, list(fea))
            return JsonData.success(data=','.join(fea))

        return self.__run_face_feature(img, run)

    def login_with_face_feature(self, img, users_features):
        def run(fea):
            userid = []
            features = []
            for item in users_features:
                userid.append(item['userId'])
                features.append(list(map(float, item['faceFeature'].split(','))))
            # 计算fea与features中的谁最接近
            isValids, dis = compare_faces(features, fea, tolerance=self.tolerance)
            print(isValids, dis, f"tolerance:{self.tolerance}")
            idmin = userid[dis.argmin()] if isValids[dis.argmin()] else -1
            return JsonData.success(data=idmin)

        return self.__run_face_feature(img, run)
