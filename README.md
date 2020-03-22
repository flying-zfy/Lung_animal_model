# 1、项目背景介绍
&emsp;&emsp;对于纤维化肺疾病，传统的Ashcroft评分需要高度训练的病理学家，且容易受到个人主观因素的影响。因此，将深度学习技术应用于组织病理学的图像分析任务中，辅助医生对病情做出诊断。
   
&emsp;&emsp;项目整体上分为两个部分：首先做一个二分类，区分出肺泡组织（记为no-ignore区域）和非肺泡组织(记为ignore区域)，对非肺泡组织，如血管、支气管等进行自动识别，排除在进一步分析之外；
对于肺泡组织（即no-ignore区域），做一个五分类（0，1，3，5，7），对应不同的病变等级，与Ashcroft评分相对应。最后，在ndpi或svs格式的大图上进行预测，验证分类结果的准确性，并统计相应的指标。

&emsp;&emsp;二分类的流程：根据xml文件从ndpi大图切分出ignore和非ignore的区域，大小为512*512，对小图进行染色标准化处理，划分出训练集和验证集，训练模型，然后对大图进行预测和可视化处理，验证准确性；二分类通过之后，
进行五分类。在二分类中，数据集存在一个问题是ignore和非ignore数据不平衡，需要对ignore的数据集进行扩充，因此，首先做的一个操作是从大图上切分出ignore区域，扩充数据集，然后进行染色标准化及其他流程；五分类的数据集用的是现有的存放于（/cptjack/totem/Lung_animal_model/Train/Lung fibrosis (Ashcroft)/）目录下的training和val中的0，1，3，5，7五类数据。先对数据集做染色标准化处理，方法和二分类的方法一样，
然后训练模型，进行大图预测，验证准确性，统计相关的指标。

&emsp;&emsp;本项目中，二分类和五分类在做染色标准化的时候，目标图片选用的是：/cptjack/totem/Lung_animal_model/Train/Lung fiberosis (Ashcroft)/training/7/17_305_18_19_12.png

&emsp;&emsp;项目整体流程图如下所示：

![image](http://192.168.3.126/Flying_zfy/Lung_Animal_model/raw/master/image/Project_flow_chart.PNG)
# 2、运行环境说明
 &emsp;&emsp;<br>虚拟机环境：
 &emsp;&emsp;<br>Cuda == 9.0.176
 &emsp;&emsp;<br>Cudnn == 7.0.5
 &emsp;&emsp;<br>python == 3.6.6
 &emsp;&emsp;<br>numpy == 1.14.3
 &emsp;&emsp;<br>keras == 2.2.0o
 &emsp;&emsp;<br>tensorflow == 1.7.0
 &emsp;&emsp;<br>scikit-learn == 0.19.1
 &emsp;&emsp;<br>scipy == 1.1.0
 &emsp;&emsp;<br>matplotlib == 2.2.2
# 3、项目文件说明

#### 根目录 
#####  /cptjack/totem/zhaofeiyan/Lung model/
该目录主要存放二分类和五分类的所有使用数据，包括数据处理的代码，数据集，训练好的模型，预测的结果，染色标准化的代码和ndpi文件的缩略图；

#### 二分类目录 &emsp; 
#####  ./zhaofeiyan/Lung model/2-class/
该目录存放的是二分类的所有使用数据，里面有四个文件夹：code,model_data,predict_result,select_region；
#####  ./zhaofeiyan/Lung model/2-class/code/
该目录存放的是二分类的所有代码文件，包括切图、训练模型、画模型训练的精度曲线、混淆矩阵和大图预测可视化结果；

##### 2_class_resnet50_train.py ： 二分类的模型训练代码

##### pre_svs_2_class.py ：二分类大图预测代码

##### draw_matrix.py ：画模型的混淆矩阵

##### draw_plot.py ：画模型训练时的精度曲线

##### label_get_patch.py ：从 ndpi 大图中切出 ignore 的区域

##### xml_util.py ：解析出xml文件的坐标信息，生成 ignore 区域的 mask 图

#####  ./zhaofeiyan/Lung model/2-class/medel_data/
该目录存放的是二分类训练好的模型(2-class-model)和训练模型所用的数据集(train-data)，2-class-model中存放模型，精度曲线，混淆矩阵和log文件等，train-data中存放训练集和验证集数据；
#####  ./zhaofeiyan/Lung model/2-class/select_region/
该目录存放的是从大图中读取出来的ignore区域图，映射到大图四级缩略图上的结果，作为辅助文件使用；

#### 五分类目录 
#####  ./zhaofeiyan/Lung model/5-class/
该目录存放的是五分类的所有使用数据，里面有三个文件夹：code,model_data,predict_result；
#####  ./zhaofeiyan/Lung model/5-class/code/
该目录存放的是五分类的所有代码文件，包括训练模型、画模型训练的精度曲线、混淆矩阵和大图预测可视化结果；

##### 5_class_train.py ： 五分类的模型训练代码

##### pre_svs_5_class.py ：五分类大图预测代码

##### draw_matrix.py ：画模型的混淆矩阵

##### draw_plot.py ：画模型训练时的精度曲线

#####  ./zhaofeiyan/Lung model/5-class/model_data/
该目录存放的是五分类训练好的模型(model)和训练模型所用的数据集(train-data)，model中有两个模型，一个是训练过程引用了imagenet_processing()，一个是训练过程未引用imagenet_processing(),train-data中存放训练集和验证集数据；
#####  ./zhaofeiyan/Lung model/5-class/predict_result/
该目录存放的是从大图中读取出来的ignore区域图，映射到大图四级缩略图上的结果，作为辅助文件使用；

#### 染色标准化方法目录
#####  ./zhaofeiyan/Lung model/colorNormalization/
该目录存放的是染色标准化的代码和标准化使用的目标图片；

# 4、项目数据清单

### 数据存放说明

####  原始ndpi大图和xml文件存放位置
##### /cptjack/totem/Lung_animal_model/Test KCI/Left/ 和 /cptjack/totem/Lung_animal_model/Test KCI/Right/

####  ndpi大图的缩略图存放位置
##### /cptjack/totem/zhaofeiyan/Lung model/svs_thumbnail/

####  从大图切出来的ignore小图存放位置
##### /cptjack/totem/zhaofeiyan/Lung model/cut_picture/

####  二分类模型及预测结果存放位置
##### /cptjack/totem/zhaofeiyan/Lung model/2-class/model_data/ 和 /cptjack/totem/zhaofeiyan/Lung model/2-class/predict_result/

####  五分类模型及预测结果存放位置
##### /cptjack/totem/zhaofeiyan/Lung model/5-class/model_data/ 和 /cptjack/totem/zhaofeiyan/Lung model/5-class/predict_result/
##### 现在预测结果存放的是训练和预测都不加imagenet_processing处理的结果

# 5、历史结果展示

#### 二分类和五分类做染色标准化的参考图和染色标准化的结果

![image](http://192.168.3.126/Flying_zfy/Lung_Animal_model/raw/master/image/color_normalization.PNG)

#### 二分类模型的参数设置

![image](http://192.168.3.126/Flying_zfy/Lung_Animal_model/raw/master/image/2_class_dataset.PNG)

#### 二分类模型的训练精度曲线和混淆矩阵

![image](http://192.168.3.126/Flying_zfy/Lung_Animal_model/raw/master/image/2_class_train_result.PNG)

#### 二分类模型的预测结果

![image](http://192.168.3.126/Flying_zfy/Lung_Animal_model/raw/master/image/2_class_predict_result.PNG)

#### 五分类模型的参数设置(未使用imagenet_processing()预处理)

![image](http://192.168.3.126/Flying_zfy/Lung_Animal_model/raw/master/image/5_class_train_data1.PNG)

#### 五分类模型的训练精度曲线和混淆矩阵(未使用imagenet_processing()预处理)

![image](http://192.168.3.126/Flying_zfy/Lung_Animal_model/raw/master/image/5_class_predict_result_noimgnet.PNG)

#### 五分类模型的参数设置(使用imagenet_processing()预处理)

![image](http://192.168.3.126/Flying_zfy/Lung_Animal_model/raw/master/image/5_class_train_data2.PNG)

#### 五分类模型的训练精度曲线和混淆矩阵(使用imagenet_processing()预处理)

![image](http://192.168.3.126/Flying_zfy/Lung_Animal_model/raw/master/image/5_class_predict_result_imgnet.PNG)
