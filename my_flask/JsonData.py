import json


class JsonData:
    @staticmethod
    def success(message=None, data=None):
        return json.dumps({'code': 200, 'message': message, 'data': data}, ensure_ascii=False)

    @staticmethod
    def error(message=None):
        return json.dumps({'code': 500, 'message': message}, ensure_ascii=False)
