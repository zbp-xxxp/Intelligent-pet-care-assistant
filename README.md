# 基于PaddleHub与Jetson Nano的智能宠物看护助手

本项目部署于Jetson Nano，当发现摄像头前宠物时，设备会立即拍照，为您抓拍精彩时刻

# 一、效果展示

**更快更强更高效快速的目标检测算法PP-YOLO，PaddleHub只需1行代码即可实现调用！！！**

【AI创造营 · 第一期】比赛链接：[https://aistudio.baidu.com/aistudio/competition/detail/72](https://aistudio.baidu.com/aistudio/competition/detail/72)

b站视频链接：[https://www.bilibili.com/video/BV1qy4y177vq](https://www.bilibili.com/video/BV1qy4y177vq)




# 二、实现思路

要让Jetson Nano识别出宠物，需要深度学习模型的辅助，有两种思路，一个是自己训练一个模型，另一种思路是使用PaddleHub的预训练模型。

以下是我在做智能宠物看护助手时想到的一些问题以及相应的解决方案。

## 1.训练一个适用于该需求的模型

- **机器能学会认识宠物吗？**<br>
答：就跟人一样，机器通过学习宠物的特征（浓密的绒毛、长长的胡须、高耸的耳朵、圆圆的小脸等特征）来认识宠物。<br>
<img style="zoom:50%;" src="https://ai-studio-static-online.cdn.bcebos.com/0ae9f8d2b0dc44cf9e64965fcf21a74fc7d95b8397be4209b7928df55cd1085f" alt=""/>
<img style="zoom:40%;" src="https://ai-studio-static-online.cdn.bcebos.com/4c95288aea71482c9033a89994eca3be6a50a03c532340b79e8001d1d7170cd3" alt=""/>

- **分类任务只有一个类别，要怎么做？**<br>
答：增加一个无宠物的类别，即采集没有宠物的背景图片，这样就变成了二分类任务。

- **选择什么算法？为什么**<br>
答：本项目中，我选择使用Ghost Module替换传统的卷积层，以此来提高识别的速度。因为本项目主要是抓拍宠物的照片，因此对于识别的准确率不需要有太高的要求，换句话说就是放弃准确率而选择识别速率。

- **抓拍到宠物照片后，如何发送给用户？**<br>
答：Python的smtplib库提供了一种很方便的途径来发送电子邮件，因此可以以电子邮件的方式将抓拍到的照片保存，并以附件的形式发送给用户。

## 2.使用PaddleHub预训练模型

PaddleHub是飞桨深度学习平台下的预训练模型应用管理工具。旨在为开发者提供丰富的、高质量的、直接可用的预训练模型。无需深度学习背景、无需数据与训练过程，可快速使用AI模型。预训练模型涵盖CV、NLP、Audio、Video主流四大品类，支持一键预测、一键服务化部署和快速迁移学习。全部模型均已开源并提供下载，可离线运行。

<img style="zoom:70%;" src="https://ai-studio-static-online.cdn.bcebos.com/8de30b8d6cf4494eabc26b2c2747cfd466f4a28ae31a42769d08f15218b9f948" alt=""/>

截止2020年底，PaddleHub已经有两百多个预训练模型，这个数量还在不断增加，PaddleHub预训练模型的使用方法也很简单，**一行代码即可调用预训练模型**，下面是一些应用案例：

- [【PaddleHub模型贡献】一行代码实现从彩色图提取素描线稿](https://aistudio.baidu.com/aistudio/projectdetail/1311444)
- [当字幕君的文档丢失了——七年期限](https://aistudio.baidu.com/aistudio/projectdetail/751309)
- [还在担心发朋友圈没文案？快来试试看图写诗吧！](https://aistudio.baidu.com/aistudio/projectdetail/738634)
- [基于PaddleHub的教师节祝福语生成](https://aistudio.baidu.com/aistudio/projectdetail/878918)
- [基于PaddleHub的中秋节看图写诗](https://aistudio.baidu.com/aistudio/projectdetail/920580)
- [手把手教你使用预训练模型ernie_gen进行finetune自己想要的场景](https://aistudio.baidu.com/aistudio/projectdetail/1456984)
- [从零开始将PaddleHub情感倾向性分析模型部署至云服务器](https://zhengbopei.blog.csdn.net/article/details/114317932)


更多资料可参考：

- PaddleHub官网：[https://www.paddlepaddle.org.cn/hub](https://www.paddlepaddle.org.cn/hub)
- PaddleHub源码仓库：[https://github.com/PaddlePaddle/PaddleHub](https://github.com/PaddlePaddle/PaddleHub)


# 三、数据采集

## 硬件部分

使用英伟达的Jetson Nano以及树莓派摄像头采集数据。

![](https://img-blog.csdnimg.cn/img_convert/47f8199ab8cc024ce3ea1069dfd9c9f6.png)


<!--
<img src='https://ai-studio-static-online.cdn.bcebos.com/25aaa4213a764b73b664c35fbd95c0db24fea30ec8ae4fedbb97c695577be39f' style="zoom:30%">
<img src='https://ai-studio-static-online.cdn.bcebos.com/515378eb8bf541d8b07d3fc689f5578357c2660a121e47bfa651627a08ebf46d' style="zoom:29%">
-->

- [Jetson Nano初体验之写入官方Ubuntu镜像](https://zhengbopei.blog.csdn.net/article/details/106027817)
- [Jetson Nano初体验之实现人脸检测（内含更换默认镜像源的方法）](https://zhengbopei.blog.csdn.net/article/details/106057574)


## 代码部分

使用openCV保存图片，以下代码请在Jetson Nano上运行：

```python
import os
import cv2
import numpy as np
import time

path = os.path.split(os.path.realpath(__file__))[0]
save_name="img"

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("----- new folder -----")
    else:
        print('----- there is this folder -----')

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def save_image_process():
    mkdir(path+"/PetsData")
    mkdir(path+"/PetsData/"+save_name)

    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)
        imgInd = 0
        # Window
        while cv2.getWindowProperty("Camera", 0) >= 0:
            ret_val, img = cap.read()
            cv2.imshow("Camera", img)
            cv2.imwrite(path+"/PetsData/"+save_name+"/{}.jpg".format(imgInd), img)
            print("imgInd=",imgInd)
            imgInd+=1
            time.sleep(0.5)
            # This also acts as
            keyCode = cv2.waitKey(30) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")

if __name__ == '__main__':
    save_image_process()
```


<img src='https://ai-studio-static-online.cdn.bcebos.com/42f388736e434ac4ac4692718f390e3e565a110c1be849879ad0344dee094fda' style="zoom:40%">

# 四、数据处理

## 1.解压数据集


```python
!unzip -oq /home/aistudio/data/data69950/PetsData.zip
```

## 2.统一存储

创建临时文件夹temporary，将所有图片保存至该文件夹下


```python
# 将图片整理到一个文件夹，并统一命名
import os
from PIL import Image

categorys = ['pets', 'other']
if not os.path.exists("temporary"):
    os.mkdir("temporary")

for category in categorys:
    # 图片文件夹路径
    path = r"PetsData/{}/".format(category)
    count = 0
    for filename in os.listdir(path):
        img = Image.open(path + filename)
        img = img.resize((1280,720),Image.ANTIALIAS) # 转换图片，图像尺寸变为1280*720
        img = img.convert('RGB') # 保存为.jpg格式才需要
        img.save(r"temporary/{}{}.jpg".format(category, str(count)))
        count += 1
```

用0和1分别代表图片上有宠物和无宠物。


```python
# 获取图片路径与图片标签
import os

# Abbreviation of classification --> ['pets', 'other']
categorys = {'p':0, 'o':1}

train_list = open('train_list.txt',mode='w')
paths = r'temporary/'
# 返回指定路径的文件夹名称
dirs = os.listdir(paths)
# 循环遍历该目录下的照片
for path in dirs:
    # 拼接字符串
    imgPath = paths + path
    train_list.write(imgPath + '\t')
    for category in categorys:
        if category == path[0]:
            train_list.write(str(categorys[category]) + '\n')
train_list.close()
```

## 3.划分训练集和验证集

训练集和验证集将保存到work目录下


```python
# 划分训练集和验证集
import shutil

train_dir = '/home/aistudio/work/trainImages'
eval_dir = '/home/aistudio/work/evalImages'
train_list_path = '/home/aistudio/train_list.txt'
target_path = "/home/aistudio/"

if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(eval_dir):
    os.mkdir(eval_dir) 

with open(train_list_path, 'r') as f:
    data = f.readlines()
    for i in range(len(data)):
        img_path = data[i].split('\t')[0]
        class_label = data[i].split('\t')[1][:-1]
        if i % 5 == 0: # 每5张图片取一个做验证数据
            eval_target_dir = os.path.join(eval_dir, str(class_label)) 
            eval_img_path = os.path.join(target_path, img_path)
            if not os.path.exists(eval_target_dir):
                os.mkdir(eval_target_dir)  
            shutil.copy(eval_img_path, eval_target_dir)                         
        else:
            train_target_dir = os.path.join(train_dir, str(class_label)) 
            train_img_path = os.path.join(target_path, img_path)                     
            if not os.path.exists(train_target_dir):
                os.mkdir(train_target_dir)
            shutil.copy(train_img_path, train_target_dir) 

    print ('划分训练集和验证集完成！')
```

    划分训练集和验证集完成！


## 4.定义数据集


```python
import os
import numpy as np
import paddle
from paddle.io import Dataset
from paddle.vision.datasets import DatasetFolder, ImageFolder
from paddle.vision.transforms import Compose, Resize, BrightnessTransform, Normalize, Transpose

class PetsDataset(Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """
    def __init__(self, mode='train'):
        """
        步骤二：实现构造函数，定义数据读取方式，划分训练和测试数据集
        """
        super(PetsDataset, self).__init__()
        train_image_dir = '/home/aistudio/work/trainImages'
        eval_image_dir = '/home/aistudio/work/evalImages'
        test_image_dir = '/home/aistudio/work/evalImages'

        transform_train = Compose([Normalize(mean=[127.5, 127.5, 127.5],std=[127.5, 127.5, 127.5],data_format='HWC'), Transpose()])
        transform_eval = Compose([Normalize(mean=[127.5, 127.5, 127.5],std=[127.5, 127.5, 127.5],data_format='HWC'), Transpose()])
        train_data_folder = DatasetFolder(train_image_dir, transform=transform_train)
        eval_data_folder = DatasetFolder(eval_image_dir, transform=transform_eval)
        test_data_folder = ImageFolder(test_image_dir, transform=transform_eval)
        self.mode = mode
        if self.mode  == 'train':
            self.data = train_data_folder
        elif self.mode  == 'eval':
            self.data = eval_data_folder
        elif self.mode  == 'test':
            self.data = test_data_folder

    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """
        data = np.array(self.data[index][0]).astype('float32')

        if self.mode  == 'test':
            return data
        else:
            label = np.array([self.data[index][1]]).astype('int64')

            return data, label

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return len(self.data)

train_dataset = PetsDataset(mode='train')
val_dataset = PetsDataset(mode='eval')
# test_dataset = PetsDataset(mode='test')
```


```python
print(len(train_dataset))
```

    2231


# 五、模型组网

## 1.使用resnet搭建PetsNet


```python
import paddle
from paddle.vision.models import resnet18, mobilenet_v2

class FlattenLayer(paddle.nn.Layer):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.reshape((x.shape[0], -1))

class PetsNet(paddle.nn.Layer):
    def __init__(self):
        super(PetsNet, self).__init__()
        self.resnet18_1 = resnet18(num_classes=2, pretrained=True)
        self.Layer1 = paddle.nn.Linear(2, 4)        
        self.Layer2 = paddle.nn.Linear(4, 8)
        self.resnet18_2 = resnet18(num_classes=2, pretrained=False)
        self.Layer3 = paddle.nn.Linear(8, 4)
        self.Softmax = paddle.nn.Softmax()
        self.Dropout = paddle.nn.Dropout(0.5)
        self.FlattenLayer = FlattenLayer()
        self.ReLU = paddle.nn.ReLU()
        self.Layer4 = paddle.nn.Linear(4, 2)

    def forward(self, inputs):    
        out1 = self.resnet18_1(inputs)
        out1 = self.Layer1(out1)
        out1 = self.Layer2(out1)
        out2 = self.resnet18_2(inputs)
        out2 = self.Layer1(out2)
        out2 = self.Layer2(out2)

        # 模型融合
        out = paddle.add(out1, out2)
        out = self.FlattenLayer(out)
        out = self.Layer3(out)
        out = self.ReLU(out)
        # out = self.Dropout(out)
        out = self.Layer4(out)
        out = self.Softmax(out)

        return out

petsnet = PetsNet()
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1263: UserWarning: Skip loading for fc.weight. fc.weight receives a shape [512, 1000], but the expected shape is [512, 2].
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1263: UserWarning: Skip loading for fc.bias. fc.bias receives a shape [1000], but the expected shape is [2].
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))


简单测试模型是否可以跑通：


```python
x = paddle.rand([1, 3, 720, 1280])
out = petsnet(x)

print(out)
```

    Tensor(shape=[1, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
           [[0.23784402, 0.76215595]])


使用paddle.Model完成模型的封装，将网络结构组合成一个可快速使用高层API进行训练和预测的类。


```python
model = paddle.Model(petsnet)
```

## 2.查看模型结构

summary 函数能够打印网络的基础结构和参数信息。


```python
model.summary((-1, 3, 720, 1280))
```

    --------------------------------------------------------------------------------
        Layer (type)         Input Shape          Output Shape         Param #    
    ================================================================================
         Conv2D-337      [[1, 3, 720, 1280]]   [1, 64, 360, 640]        9,408     
      BatchNorm2D-337    [[1, 64, 360, 640]]   [1, 64, 360, 640]         256      
          ReLU-88        [[1, 64, 360, 640]]   [1, 64, 360, 640]          0       
        MaxPool2D-10     [[1, 64, 360, 640]]   [1, 64, 180, 320]          0       
         Conv2D-338      [[1, 64, 180, 320]]   [1, 64, 180, 320]       36,864     
      BatchNorm2D-338    [[1, 64, 180, 320]]   [1, 64, 180, 320]         256      
          ReLU-89        [[1, 64, 180, 320]]   [1, 64, 180, 320]          0       
         Conv2D-339      [[1, 64, 180, 320]]   [1, 64, 180, 320]       36,864     
      BatchNorm2D-339    [[1, 64, 180, 320]]   [1, 64, 180, 320]         256      
       BasicBlock-73     [[1, 64, 180, 320]]   [1, 64, 180, 320]          0       
         Conv2D-340      [[1, 64, 180, 320]]   [1, 64, 180, 320]       36,864     
      BatchNorm2D-340    [[1, 64, 180, 320]]   [1, 64, 180, 320]         256      
          ReLU-90        [[1, 64, 180, 320]]   [1, 64, 180, 320]          0       
         Conv2D-341      [[1, 64, 180, 320]]   [1, 64, 180, 320]       36,864     
      BatchNorm2D-341    [[1, 64, 180, 320]]   [1, 64, 180, 320]         256      
       BasicBlock-74     [[1, 64, 180, 320]]   [1, 64, 180, 320]          0       
         Conv2D-343      [[1, 64, 180, 320]]   [1, 128, 90, 160]       73,728     
      BatchNorm2D-343    [[1, 128, 90, 160]]   [1, 128, 90, 160]         512      
          ReLU-91        [[1, 128, 90, 160]]   [1, 128, 90, 160]          0       
         Conv2D-344      [[1, 128, 90, 160]]   [1, 128, 90, 160]       147,456    
      BatchNorm2D-344    [[1, 128, 90, 160]]   [1, 128, 90, 160]         512      
         Conv2D-342      [[1, 64, 180, 320]]   [1, 128, 90, 160]        8,192     
      BatchNorm2D-342    [[1, 128, 90, 160]]   [1, 128, 90, 160]         512      
       BasicBlock-75     [[1, 64, 180, 320]]   [1, 128, 90, 160]          0       
         Conv2D-345      [[1, 128, 90, 160]]   [1, 128, 90, 160]       147,456    
      BatchNorm2D-345    [[1, 128, 90, 160]]   [1, 128, 90, 160]         512      
          ReLU-92        [[1, 128, 90, 160]]   [1, 128, 90, 160]          0       
         Conv2D-346      [[1, 128, 90, 160]]   [1, 128, 90, 160]       147,456    
      BatchNorm2D-346    [[1, 128, 90, 160]]   [1, 128, 90, 160]         512      
       BasicBlock-76     [[1, 128, 90, 160]]   [1, 128, 90, 160]          0       
         Conv2D-348      [[1, 128, 90, 160]]    [1, 256, 45, 80]       294,912    
      BatchNorm2D-348     [[1, 256, 45, 80]]    [1, 256, 45, 80]        1,024     
          ReLU-93         [[1, 256, 45, 80]]    [1, 256, 45, 80]          0       
         Conv2D-349       [[1, 256, 45, 80]]    [1, 256, 45, 80]       589,824    
      BatchNorm2D-349     [[1, 256, 45, 80]]    [1, 256, 45, 80]        1,024     
         Conv2D-347      [[1, 128, 90, 160]]    [1, 256, 45, 80]       32,768     
      BatchNorm2D-347     [[1, 256, 45, 80]]    [1, 256, 45, 80]        1,024     
       BasicBlock-77     [[1, 128, 90, 160]]    [1, 256, 45, 80]          0       
         Conv2D-350       [[1, 256, 45, 80]]    [1, 256, 45, 80]       589,824    
      BatchNorm2D-350     [[1, 256, 45, 80]]    [1, 256, 45, 80]        1,024     
          ReLU-94         [[1, 256, 45, 80]]    [1, 256, 45, 80]          0       
         Conv2D-351       [[1, 256, 45, 80]]    [1, 256, 45, 80]       589,824    
      BatchNorm2D-351     [[1, 256, 45, 80]]    [1, 256, 45, 80]        1,024     
       BasicBlock-78      [[1, 256, 45, 80]]    [1, 256, 45, 80]          0       
         Conv2D-353       [[1, 256, 45, 80]]    [1, 512, 23, 40]      1,179,648   
      BatchNorm2D-353     [[1, 512, 23, 40]]    [1, 512, 23, 40]        2,048     
          ReLU-95         [[1, 512, 23, 40]]    [1, 512, 23, 40]          0       
         Conv2D-354       [[1, 512, 23, 40]]    [1, 512, 23, 40]      2,359,296   
      BatchNorm2D-354     [[1, 512, 23, 40]]    [1, 512, 23, 40]        2,048     
         Conv2D-352       [[1, 256, 45, 80]]    [1, 512, 23, 40]       131,072    
      BatchNorm2D-352     [[1, 512, 23, 40]]    [1, 512, 23, 40]        2,048     
       BasicBlock-79      [[1, 256, 45, 80]]    [1, 512, 23, 40]          0       
         Conv2D-355       [[1, 512, 23, 40]]    [1, 512, 23, 40]      2,359,296   
      BatchNorm2D-355     [[1, 512, 23, 40]]    [1, 512, 23, 40]        2,048     
          ReLU-96         [[1, 512, 23, 40]]    [1, 512, 23, 40]          0       
         Conv2D-356       [[1, 512, 23, 40]]    [1, 512, 23, 40]      2,359,296   
      BatchNorm2D-356     [[1, 512, 23, 40]]    [1, 512, 23, 40]        2,048     
       BasicBlock-80      [[1, 512, 23, 40]]    [1, 512, 23, 40]          0       
    AdaptiveAvgPool2D-13  [[1, 512, 23, 40]]     [1, 512, 1, 1]           0       
         Linear-37            [[1, 512]]             [1, 2]             1,026     
         ResNet-10       [[1, 3, 720, 1280]]         [1, 2]               0       
         Linear-38             [[1, 2]]              [1, 4]              12       
         Linear-39             [[1, 4]]              [1, 8]              40       
         Conv2D-357      [[1, 3, 720, 1280]]   [1, 64, 360, 640]        9,408     
      BatchNorm2D-357    [[1, 64, 360, 640]]   [1, 64, 360, 640]         256      
          ReLU-97        [[1, 64, 360, 640]]   [1, 64, 360, 640]          0       
        MaxPool2D-11     [[1, 64, 360, 640]]   [1, 64, 180, 320]          0       
         Conv2D-358      [[1, 64, 180, 320]]   [1, 64, 180, 320]       36,864     
      BatchNorm2D-358    [[1, 64, 180, 320]]   [1, 64, 180, 320]         256      
          ReLU-98        [[1, 64, 180, 320]]   [1, 64, 180, 320]          0       
         Conv2D-359      [[1, 64, 180, 320]]   [1, 64, 180, 320]       36,864     
      BatchNorm2D-359    [[1, 64, 180, 320]]   [1, 64, 180, 320]         256      
       BasicBlock-81     [[1, 64, 180, 320]]   [1, 64, 180, 320]          0       
         Conv2D-360      [[1, 64, 180, 320]]   [1, 64, 180, 320]       36,864     
      BatchNorm2D-360    [[1, 64, 180, 320]]   [1, 64, 180, 320]         256      
          ReLU-99        [[1, 64, 180, 320]]   [1, 64, 180, 320]          0       
         Conv2D-361      [[1, 64, 180, 320]]   [1, 64, 180, 320]       36,864     
      BatchNorm2D-361    [[1, 64, 180, 320]]   [1, 64, 180, 320]         256      
       BasicBlock-82     [[1, 64, 180, 320]]   [1, 64, 180, 320]          0       
         Conv2D-363      [[1, 64, 180, 320]]   [1, 128, 90, 160]       73,728     
      BatchNorm2D-363    [[1, 128, 90, 160]]   [1, 128, 90, 160]         512      
          ReLU-100       [[1, 128, 90, 160]]   [1, 128, 90, 160]          0       
         Conv2D-364      [[1, 128, 90, 160]]   [1, 128, 90, 160]       147,456    
      BatchNorm2D-364    [[1, 128, 90, 160]]   [1, 128, 90, 160]         512      
         Conv2D-362      [[1, 64, 180, 320]]   [1, 128, 90, 160]        8,192     
      BatchNorm2D-362    [[1, 128, 90, 160]]   [1, 128, 90, 160]         512      
       BasicBlock-83     [[1, 64, 180, 320]]   [1, 128, 90, 160]          0       
         Conv2D-365      [[1, 128, 90, 160]]   [1, 128, 90, 160]       147,456    
      BatchNorm2D-365    [[1, 128, 90, 160]]   [1, 128, 90, 160]         512      
          ReLU-101       [[1, 128, 90, 160]]   [1, 128, 90, 160]          0       
         Conv2D-366      [[1, 128, 90, 160]]   [1, 128, 90, 160]       147,456    
      BatchNorm2D-366    [[1, 128, 90, 160]]   [1, 128, 90, 160]         512      
       BasicBlock-84     [[1, 128, 90, 160]]   [1, 128, 90, 160]          0       
         Conv2D-368      [[1, 128, 90, 160]]    [1, 256, 45, 80]       294,912    
      BatchNorm2D-368     [[1, 256, 45, 80]]    [1, 256, 45, 80]        1,024     
          ReLU-102        [[1, 256, 45, 80]]    [1, 256, 45, 80]          0       
         Conv2D-369       [[1, 256, 45, 80]]    [1, 256, 45, 80]       589,824    
      BatchNorm2D-369     [[1, 256, 45, 80]]    [1, 256, 45, 80]        1,024     
         Conv2D-367      [[1, 128, 90, 160]]    [1, 256, 45, 80]       32,768     
      BatchNorm2D-367     [[1, 256, 45, 80]]    [1, 256, 45, 80]        1,024     
       BasicBlock-85     [[1, 128, 90, 160]]    [1, 256, 45, 80]          0       
         Conv2D-370       [[1, 256, 45, 80]]    [1, 256, 45, 80]       589,824    
      BatchNorm2D-370     [[1, 256, 45, 80]]    [1, 256, 45, 80]        1,024     
          ReLU-103        [[1, 256, 45, 80]]    [1, 256, 45, 80]          0       
         Conv2D-371       [[1, 256, 45, 80]]    [1, 256, 45, 80]       589,824    
      BatchNorm2D-371     [[1, 256, 45, 80]]    [1, 256, 45, 80]        1,024     
       BasicBlock-86      [[1, 256, 45, 80]]    [1, 256, 45, 80]          0       
         Conv2D-373       [[1, 256, 45, 80]]    [1, 512, 23, 40]      1,179,648   
      BatchNorm2D-373     [[1, 512, 23, 40]]    [1, 512, 23, 40]        2,048     
          ReLU-104        [[1, 512, 23, 40]]    [1, 512, 23, 40]          0       
         Conv2D-374       [[1, 512, 23, 40]]    [1, 512, 23, 40]      2,359,296   
      BatchNorm2D-374     [[1, 512, 23, 40]]    [1, 512, 23, 40]        2,048     
         Conv2D-372       [[1, 256, 45, 80]]    [1, 512, 23, 40]       131,072    
      BatchNorm2D-372     [[1, 512, 23, 40]]    [1, 512, 23, 40]        2,048     
       BasicBlock-87      [[1, 256, 45, 80]]    [1, 512, 23, 40]          0       
         Conv2D-375       [[1, 512, 23, 40]]    [1, 512, 23, 40]      2,359,296   
      BatchNorm2D-375     [[1, 512, 23, 40]]    [1, 512, 23, 40]        2,048     
          ReLU-105        [[1, 512, 23, 40]]    [1, 512, 23, 40]          0       
         Conv2D-376       [[1, 512, 23, 40]]    [1, 512, 23, 40]      2,359,296   
      BatchNorm2D-376     [[1, 512, 23, 40]]    [1, 512, 23, 40]        2,048     
       BasicBlock-88      [[1, 512, 23, 40]]    [1, 512, 23, 40]          0       
    AdaptiveAvgPool2D-14  [[1, 512, 23, 40]]     [1, 512, 1, 1]           0       
         Linear-40            [[1, 512]]             [1, 2]             1,026     
         ResNet-11       [[1, 3, 720, 1280]]         [1, 2]               0       
       FlattenLayer-7          [[1, 8]]              [1, 8]               0       
         Linear-41             [[1, 8]]              [1, 4]              36       
          ReLU-106             [[1, 4]]              [1, 4]               0       
         Dropout-10            [[1, 4]]              [1, 4]               0       
         Linear-42             [[1, 4]]              [1, 2]              10       
         Softmax-7             [[1, 2]]              [1, 2]               0       
    ================================================================================
    Total params: 22,374,374
    Trainable params: 22,335,974
    Non-trainable params: 38,400
    --------------------------------------------------------------------------------
    Input size (MB): 10.55
    Forward/backward pass size (MB): 2097.51
    Params size (MB): 85.35
    Estimated Total Size (MB): 2193.41
    --------------------------------------------------------------------------------
    





    {'total_params': 22374374, 'trainable_params': 22335974}



## 3.模型配置


```python
# 调用飞桨框架的VisualDL模块，保存信息到目录中。
callback = paddle.callbacks.VisualDL(log_dir='visualdl_log_dir')
```


```python
def create_optim(parameters):
    step_each_epoch = 2231 // 32
    lr = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.1,
                                                  T_max=step_each_epoch * 10)

    return paddle.optimizer.Momentum(learning_rate=lr,
                                     parameters=parameters,
                                     weight_decay=paddle.regularizer.L2Decay(0.01))


# 模型训练配置
model.prepare(create_optim(model.parameters()),  # 优化器
              paddle.nn.CrossEntropyLoss(),        # 损失函数
              paddle.metric.Accuracy(topk=(1, 5))) # 评估指标

# model.prepare(optimizer=paddle.optimizer.RMSProp(learning_rate=0.1,
#                                                 momentum=0.1,
#                                                 centered=True,
#                                                 parameters=model.parameters(),
#                                                 weight_decay=0.01),
#               loss=paddle.nn.CrossEntropyLoss(),
#               metrics=paddle.metric.Accuracy(topk=(1, 5)))
```

# 六、训练及评估

## 1.模型训练


```python
model.fit(train_dataset,
          val_dataset,
          epochs=10,
          batch_size=32,
          callbacks=callback,
          verbose=1)
```

    The loss value printed in the log is the current step, and the metric is the average value of previous step.
    Epoch 1/10


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:77: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      return (isinstance(seq, collections.Sequence) and
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/nn/layer/norm.py:636: UserWarning: When training, we now always track global mean and variance.
      "When training, we now always track global mean and variance.")


    step 70/70 [==============================] - loss: 0.4119 - acc_top1: 0.6997 - acc_top5: 1.0000 - 5s/step         
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 18/18 [==============================] - loss: 1.0722 - acc_top1: 0.5735 - acc_top5: 1.0000 - 10s/step           
    Eval samples: 558
    Epoch 2/10
    step 70/70 [==============================] - loss: 0.5428 - acc_top1: 0.8014 - acc_top5: 1.0000 - 4s/step         
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 18/18 [==============================] - loss: 0.3155 - acc_top1: 0.4875 - acc_top5: 1.0000 - 3s/step          
    Eval samples: 558
    Epoch 3/10
    step 70/70 [==============================] - loss: 0.5819 - acc_top1: 0.7916 - acc_top5: 1.0000 - 4s/step         
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 18/18 [==============================] - loss: 0.4183 - acc_top1: 0.7007 - acc_top5: 1.0000 - 3s/step          
    Eval samples: 558
    Epoch 4/10
    step 70/70 [==============================] - loss: 0.4163 - acc_top1: 0.8023 - acc_top5: 1.0000 - 4s/step         
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 18/18 [==============================] - loss: 0.3213 - acc_top1: 0.4946 - acc_top5: 1.0000 - 3s/step          
    Eval samples: 558
    Epoch 5/10
    step 70/70 [==============================] - loss: 0.4296 - acc_top1: 0.8194 - acc_top5: 1.0000 - 4s/step         
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 18/18 [==============================] - loss: 0.3398 - acc_top1: 0.4875 - acc_top5: 1.0000 - 3s/step          
    Eval samples: 558
    Epoch 6/10
    step 70/70 [==============================] - loss: 0.5383 - acc_top1: 0.7996 - acc_top5: 1.0000 - 4s/step         
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 18/18 [==============================] - loss: 0.3653 - acc_top1: 0.6183 - acc_top5: 1.0000 - 3s/step          
    Eval samples: 558
    Epoch 7/10
    step 70/70 [==============================] - loss: 0.4483 - acc_top1: 0.8252 - acc_top5: 1.0000 - 4s/step         
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 18/18 [==============================] - loss: 0.3307 - acc_top1: 0.5108 - acc_top5: 1.0000 - 3s/step          
    Eval samples: 558
    Epoch 8/10
    step 70/70 [==============================] - loss: 0.4600 - acc_top1: 0.8584 - acc_top5: 1.0000 - 4s/step         
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 18/18 [==============================] - loss: 0.3268 - acc_top1: 0.6756 - acc_top5: 1.0000 - 3s/step          
    Eval samples: 558
    Epoch 9/10
    step 70/70 [==============================] - loss: 0.5193 - acc_top1: 0.8853 - acc_top5: 1.0000 - 5s/step         
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 18/18 [==============================] - loss: 0.3779 - acc_top1: 0.8853 - acc_top5: 1.0000 - 3s/step          
    Eval samples: 558
    Epoch 10/10
    step 70/70 [==============================] - loss: 0.3914 - acc_top1: 0.9117 - acc_top5: 1.0000 - 4s/step         
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 18/18 [==============================] - loss: 0.3816 - acc_top1: 0.8907 - acc_top5: 1.0000 - 4s/step          
    Eval samples: 558


## 2.全流程评估


```python
# 用 model.evaluate 在测试集上对模型进行验证
eval_result = model.evaluate(val_dataset, verbose=1)
```

    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 558/558 [==============================] - loss: 0.3612 - acc_top1: 0.8907 - acc_top5: 1.0000 - 87ms/step         
    Eval samples: 558



```python
test_result = model.predict(val_dataset)
# print(test_result)
```

    Predict begin...
    step 558/558 [==============================] - 86ms/step         
    Predict samples: 558



```python
results = test_result
labels = []
for result in results[0]:
    lab = np.argmax(result)
    labels.append(lab)
test_paths = os.listdir('work/evalImages')
final_result=[]
print(test_paths)
print(labels)
```

    ['1', '0']
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]



```python
print(test_result[0])
```

## 3.模型保存


```python
model.save('inferT/Tpets', training=True)
```


```python
model.save('infer/pets', training=False)
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/hapi/model.py:1738: UserWarning: 'inputs' was not specified when Model initialization, so the input shape to be saved will be the shape derived from the user's actual inputs. The input shape to be saved is [[32, 3, 720, 1280]]. For saving correct input shapes, please provide 'inputs' for Model initialization.
      % self._input_info[0])


![](https://img-blog.csdnimg.cn/img_convert/412f6e7acf6baea4a8ea31472f85472c.png)


# 七、将模型部署到Jetson Nano

使用PaddleHub Serving ，只需几个简单的步骤，就可以把模型部署到多种设备中，实现在线服务，使用流程如下所示：（以下代码请在Jetson Nano中运行）

## 1.安装PaddleHub

在安装PaddleHub之前，首先需要安装PaddlePaddle，安装方法可参考该教程：
- [教你如何在三步内Jetson系列上安装PaddlePaddle](https://aistudio.baidu.com/aistudio/projectdetail/1591846)

运行安装指令前，务必运行该指令打开Jetson Nano的小风扇：
```
sudo sh -c "echo 200 > /sys/devices/pwm-fan/target_pwm"
```

成功安装PaddlePaddle以后，只需运行：
```
pip3 install paddlehub
```
安装失败可尝试换源：
```
pip3 install --upgrade paddlehub -i https://pypi.tuna.tsinghua.edu.cn/simple
```
安装的时间比较久，出现如下安装成功的提示时，则安装成功：

<img style="zoom:80%;" src="https://ai-studio-static-online.cdn.bcebos.com/18831593711b454ca40e8e74d71a70e3517d7459b5114bac9b883968b8de30f3" alt=""/>



> 如果不想在Jetson Nano上运行，也可以把模型部署至云服务器端，具体方法请看：
> - [从零开始将PaddleHub情感倾向性分析模型部署至云服务器](https://zhengbopei.blog.csdn.net/article/details/114317932)

## 2.下载模型

本项目使用到的模型是
- 目标检测：yolov3_resnet50_vd_coco2017

使用如下命令即可进行安装：
```
hub install yolov3_resnet50_vd_coco2017==1.0.1
```

## 3.获取实时视频流

将摄像头获取的图像保存下来：

```
import os
import cv2
import numpy as np
import time

path = os.path.split(os.path.realpath(__file__))[0]
save_name="img"

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("----- new folder -----")
    else:
        print('----- there is this folder -----')

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def save_image_process():
    mkdir(path+"/PetsData")
    mkdir(path+"/PetsData/"+save_name)

    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)
        imgInd = 0
        # Window
        while cv2.getWindowProperty("Camera", 0) >= 0:
            ret_val, img = cap.read()
            cv2.imshow("Camera", img)
            cv2.imwrite(path+"/PetsData/"+save_name+"/{}.jpg".format(imgInd), img)
            print("imgInd=",imgInd)
            # imgInd+=1
            time.sleep(0.5)
            # This also acts as
            keyCode = cv2.waitKey(30) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")

if __name__ == '__main__':
    save_image_process()
```

## 4.模型预测

将上一步保存的图片送入目标检测模型


```python
import paddlehub as hub
import cv2

object_detector = hub.Module(name="yolov3_resnet50_vd_coco2017")
result = object_detector.object_detection(images=[cv2.imread('PetsData/pets/891.jpg')])

print("检测标签为：{}".format(result[0]['data'][0]['label']))
ct_detector = hub.Module(name="yolov3_resnet50_vd_coco2017")
result = object_detector.object_detection(images=[cv2.imread('PetsData/pets/891.jpg')])

print("检测标签为：{}".format(result[0]['data'][0]['label']))
print("置信度为：{}".format(result[0]['data'][0]['confidence']))
```

    [2021-03-05 17:00:56,010] [    INFO] - Installing yolov3_resnet50_vd_coco2017 module
    [2021-03-05 17:00:56,013] [    INFO] - Module yolov3_resnet50_vd_coco2017 already installed in /home/aistudio/.paddlehub/modules/yolov3_resnet50_vd_coco2017


    检测标签为：cat
    置信度为：0.7003269195556641


可视化结果：（不得不说，树莓派摄像头的画质感觉有点差，但是PaddleHub的检测结果还是很准的，居然连后面的瓶子都识别出来了）

![](https://img-blog.csdnimg.cn/img_convert/31917547494046fa7abf0fd2172a5a74.png)


## 5.回传照片

检测到照片中有宠物后，比如标签是cat或是dog等，就把照片单独保存起来，并通过邮件的形式发送给用户。

以下代码请在Jetson Nano上运行
```
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header

# 输入Email地址和口令:
from_addr = "2733821739@qq.com"
password = "wkupyhqzvupydfbf"
# 输入收件人地址:
receivers = ["2733821739@qq.com"]
# 输入SMTP服务器地址:
smtp_server = "SMTP.qq.com"
#输入要发送的内容:
mail_msg = """
<p>图片演示：</p>
<p><img src="cid:image1"></p>
"""
msgRoot = MIMEMultipart('related')
msgRoot['From'] = Header("Jetson Nano", 'utf-8')
msgRoot['To'] =  Header("测试", 'utf-8')
subject = '智能宠物看护助手'
msgRoot['Subject'] = Header(subject, 'utf-8')
msgAlternative = MIMEMultipart('alternative')
msgAlternative.attach(MIMEText(mail_msg, 'html', 'utf-8'))
 
# 指定图片为当前目录
fp = open('detection_result/image_numpy_0.jpg', 'rb')
msgImage = MIMEImage(fp.read())
fp.close()
 
# 定义图片 ID，在 HTML 文本中引用
msgImage.add_header('Content-ID', '<image1>')
msgRoot.attach(msgImage)

server = smtplib.SMTP_SSL(smtp_server, 465) # SMTP协议默认端口是25
server.set_debuglevel(1)
server.login(from_addr, password)
try:
    server.sendmail(from_addr, receivers, msgRoot.as_string())
    print("Success: 已成功发送邮件!")
    server.quit()
except smtplib.SMTPException:
    print ("Error: 无法发送邮件")
```

<img style="zoom:50%;" src="https://ai-studio-static-online.cdn.bcebos.com/e27d672f8dcf43a883d0ef7a15b45d606644010a73ad422c9315233051f225bc" alt=""/>

## 6.客户端收到邮件

客户端收到的文件格式是.bin，需要手动将后缀改成.jpg后才能打开

<img style="zoom:80%;" src="https://ai-studio-static-online.cdn.bcebos.com/f8dc23c6373e41e8998c466ca6c14cc0d7da1068167f4107947e4a23217193e0" alt=""/>

