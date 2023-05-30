# Mid_term

# MISSION 1
使用CNN网络模型(自己设计或使用现有的CNN架构，如AlexNet，ResNet-18)作为baseline在CIFAR-100上训练并测试；

对比cutmix, cutout, mixup三种方法以及baseline方法在CIFAR-100图像分类任务中的性能表现；

对三张训练样本分别经过cutmix, cutout, mixup后进行可视化，一共show 9张图像。

### 数据集
数据集为CIFAR100数据集，本数据集共有100个类。每个类有600张大小为32 × 32 的彩色图像，其中500张作为训练集，100张作为测试集。对于每一张图像，它有fine_labels和coarse_labels两个标签，分别代表图像的细粒度和粗粒度标签。

### Requirements
* Python
* pytorch
* torchvision
* matplotlib
* numpy

### Parameters
* lr = 0.001
* epochs = 20
* training proportion = 0.8
* batch_size = 64
* 优化器：adam

训练出的模型已上传至百度网盘（https://pan.baidu.com/s/15RIblwxQqnfF-9ZGizRi3w?pwd=5h5q 提取码: 5h5q）

训练和测试以及可视化结果均可查看.ipynb文件。

# MISSION 2

在VOC数据集上训练并测试目标检测模型 Faster R-CNN 和 FCOS；在四张测试图像上可视化Faster R-CNN第一阶段的proposal box；

两个训练好后的模型分别可视化三张不在VOC数据集内，但是包含有VOC中类别物体的图像的检测结果（类别标签，得分，boundingbox），并进行对比，一共show六张图像。

## VOC data
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html
该数据集的结构如下：
```
 - data
   - VOCdevkit
      - VOC2007
        - Annotations
          - 000001.xml
          - 000002.xml
            ...
        - ImageSets
          - Main
            ...
          - test.txt
            ...
          - trainval.txt
            ...
        - JPEGImages
          - 000001.jpg
          - 000002.jpg
          ...
- ...
```


## Faster R-CNN

<table>
        <tr>
            <th>Backbone</th>
            <th>GPU</th>
            <th>#GPUs</th>
            <th>#Batches/GPU</th>
            <th>Training Speed (FPS)</th>
            <th>Inference Speed (FPS)</th>
            <th>mAP</th>
            <th>image_min_side</th>
            <th>image_max_side</th>
            <th>anchor_ratios</th>
            <th>anchor_sizes</th>
            <th>pooler_mode</th>
            <th>rpn_pre_nms_top_n (train)</th>
            <th>rpn_post_nms_top_n (train)</th>
            <th>rpn_pre_nms_top_n (eval)</th>
            <th>rpn_post_nms_top_n (eval)</th>
            <th>anchor_smooth_l1_loss_beta</th>
            <th>proposal_smooth_l1_loss_beta</th>
            <th>batch_size</th>
            <th>learning_rate</th>
            <th>momentum</th>
            <th>weight_decay</th>
            <th>step_lr_sizes</th>
            <th>step_lr_gamma</th>
            <th>warm_up_factor</th>
            <th>warm_up_num_iters</th>
            <th>num_steps_to_finish</th>
        </tr>
        <tr>
            <td>ResNet-101</td>
            <td>GTX 1080 Ti</td>
            <td>1</td>
            <td>4</td>
            <td>7.12</td>
            <td>15.05</td>
            <td>0.7562</td>
            <td>600</td>
            <td>1000</td>
            <td>[(1, 2), (1, 1), (2, 1)]</td>
            <td>[128, 256, 512]</td>
            <td>align</td>
            <td>12000</td>
            <td>2000</td>
            <td>6000</td>
            <td>300</td>
            <td>1.0</td>
            <td>1.0</td>
            <td><b>4</b></td>
            <td><b>0.004</b></td>
            <td>0.9</td>
            <td>0.0005</td>
            <td><b>[12500, 17500]</b></td>
            <td>0.1</td>
            <td>0.3333</td>
            <td>500</td>
            <td><b>22500</b></td>
        </tr>
    </table>

### Requirements
* Python 3.6
* torch 1.0
* torchvision 0.2.1
* tqdm
* tensorboardX
* OpenCV 3.4
* websockets


### Training

* To apply default configuration (see also `config/`)

```
python train.py -s=voc2007 -b=resnet101
```

* To apply custom configuration (see also `train.py`)

```
python train.py -s=voc2007 -b=resnet101 --weight_decay=0.0001
```
       

* To apply recommended configuration (see also `scripts/`)

```
bash ./scripts/voc2007/train-bs2.sh resnet101 /path/to/outputs/dir
```

训练好的模型（百度网盘链接：链接: https://pan.baidu.com/s/1oJlytX8dvGvaW-PdPUqpzQ?pwd=3yqw 提取码: 3yqw）

### Evaluate
* To apply default configuration (see also `config/`)

```
python eval.py -s=voc2007 -b=resnet101 /path/to/checkpoint.pth
```

* To apply custom configuration (see also `eval.py`)

```
python eval.py -s=voc2007 -b=resnet101 --rpn_post_nms_top_n=1000 /path/to/checkpoint.pth
```


* To apply recommended configuration (see also `scripts/`)

```
bash ./scripts/voc2007/eval.sh resnet101 /path/to/checkpoint.pth
```

### Results
```
aeroplane AP = 0.67292
bicycle AP = 0.7375
bird AP = 0.5668
boat AP = 0.4769
bottle AP = 0.4423
bus AP= 0.7344= 0.7912
car AP = 0.7912
cat AP= 0.7431
chair AP = 0.4208
cow AP = 0.70341
diningtable AP = 0.59432
dog AP = 0.6972
horse AP = 0.7662
motorbike AP = 0.73254
person AP = 0765456
pottedplant AP = 0.3846
sheep AP = 0.6258
sofa AP = 0.6777
train AP = 0.73869
tvmonitor AP = 0.614520
mean AP = 0.6443
```

## FCOS

### Requirements
* opencv-python
* pytorch >= 1.0
* torchvision >= 0.4.
* matplotlib
* numpy == 1.17
* Pillow
* tqdm
* pycocotools

### Training
运行train_voc.py, 训练30 epoch.
初始学习率 lr=0.01

### Evaluate
下载训练好的模型（百度网盘链接：链接: https://pan.baidu.com/s/1PeVgTkxVwk7WJNUIKE44cw?pwd=nqit 提取码: nqit）

放在checkpoint文件夹，然后运行eval_voc.py，可以得到以下结果
```
ap for aeroplane is 0.8404275844161726
ap for bicycle is 0.8538414069634157
ap for bird is 0.8371043868766486
ap for boat is 0.6867630943895144
ap for bottle is 0.7039276923755678
ap for bus is 0.8585650817738049
ap for car is 0.8993155911437366
ap for cat is 0.919100484692941
ap for chair is 0.5575814527810952
ap for cow is 0.8429926423801004
ap for diningtable is 0.6596296818110386
ap for dog is 0.8896160095323242
ap for horse is 0.8436443710873067
ap for motorbike is 0.8114359299817884
ap for person is 0.8525903122141745
ap for pottedplant is 0.47628914937925404
ap for sheep is 0.8257834833986701
ap for sofa is 0.7000391892293902
ap for train is 0.8664281745198105
ap for tvmonitor is 0.8186715890179656
mAP=====>0.787
```
