## meeting_system_backend

前端：https://github.com/xyongfeng/meeting_system_front

后端：https://github.com/xyongfeng/meeting_system_backend

### 本项目介绍

一个基于tensorflow的人脸识别项目。

在[bubbliiiing](https://github.com/bubbliiiing)大佬的基础上，更新了tensorflow2和mobilenet2，重新开始进行了训练，并应用于自己的毕设上。

### 开发环境

python3.6

### 技术框架

flask	web框架

tensorflow2.2.0  深度学习框架

### 导入依赖

```
scipy==1.4.1
numpy==1.18.4
matplotlib==3.2.1
opencv_python==4.2.0.34
tensorflow_cpu==2.2.0
tqdm==4.46.1
Pillow==8.2.0
h5py==2.10.0
flask==2.0.3
```

### 权重下载

链接：https://pan.baidu.com/s/13VOVOtbs70vlhPuHCrCJjA?pwd=z7cl 提取码：z7cl
下载之后放入weight文件夹

### 文件说明

**flask_main.py	启动服务**

config.py	设置模型权重路径和识别阈值

predit.py	本地进行人脸检测，可选择检测图片和打开摄像头实时检测

### 参考链接

https://github.com/bubbliiiing/facenet-retinaface-keras