# MathWorks Radar Drone Classification Project
This project is for MathWorks-St Andrews Radar Drone Classification Project, implemented by [Zhongliang Guo](mailto:zg34@st-andrews.ac.uk).

This project supports a series of advanced neural networks, including:
```python
IMPLEMENTED_NETS = (
    'resnet18',
    'resnet50',
    'convnext_tiny',
    'convnext_base',
    'efficientnet_v2_s',
    'efficientnet_v2_m',
    'resnext50_32x4d',
    'alexnet',
)
```

It can achieve different tasks:
1. binary classification for `drone` and `non-drone`.
2. different drone type classification, including `Autel_Evo_II`, `DJI_Matrice_210`, `DJI_Mavic_3`, `DJI_Mini_2`, and `Yuneec_H520E`.
3. quadruple classification, including `drone`, `bird`, `cluster`, and `noise`.

## Usage
You can run the python script in MATLAB by the [instruction](https://ch.mathworks.com/products/matlab/matlab-and-python.html).

### Train

a simple usage for training a neural networks.
```bash
python train.py \
  --task binary \
  --arch resnet18 \
  --epochs 100 \
  --ckpt_interval 20 \
  --batch_size 64 \
  --lr 1e-05 \
  --random_seed 3407 \
  --data_root /.../data \
  --train_label_path /.../train_binary.csv \
  --eval_label_path /.../eval_binary.csv \
  --output_dir ./logs
```
For more details, please run
```bash
python train.py --help
```

#### How to see the real-time training curve

1. change the directory to the `output_dir`

2. run the following bash
   ```bash
   tensorboard --logdir=./
   ```

   please make sure you've already installed the library `tensorboard`, if not , just run `pip install tensorboard`.

### Inference

```bash
python inference.py \
  --task binary \
  --arch resnet18 \
  --image_path /.../XXX.png \
  --pth_path /.../XXX.pth
```
## Dataset

Download the dataset by clicking [here](https://universityofstandrews907-my.sharepoint.com/:u:/g/personal/zg34_st-andrews_ac_uk/EdsXcV6S6PlNrXaADrUzatYBQnehXgK_CFSz3zlBSQnRuw?e=AG3cOG). The password is `mathworks2024`.

## Results

trained on the machine with

- OS: `Ubuntu 20.04.6 LTS`
- GPU: `NVIDIA A100-SXM4-80GB`
- CPU: `AMD EPYC 7713 64-Core Processor`

### Accuracy

| task                 | AlexNet | ConvNeXt-base | ConvNeXt-tiny | EfficientNetV2-m | EfficientNetV2-s | ResNet18 | ResNet50 | ResNeXt50 |
| -------------------- | ------- | ------------- | ------------- | ---------------- | ---------------- | -------- | -------- | --------- |
| binary               | 100%    | 100%          | 100%          | 100%             | 100%             | 100%     | 100%     | 100%      |
| drone-classification | 97.60%  | 99.02%        | 98.91%        | 96.73%           | 98.04%           | 96.29%   | 94.66%   | 96.84%    |
| four-class           | 94.29%  | 92.09%        | 91.43%        | 99.34%           | 97.36%           | 94.73%   | 93.63%   | 91.43%    |

It's worth noting that all above models used the ImageNet-1K pre-trained weights.

### Runtime (in second)

OS: `Ubuntu 22.04.4 LTS`

#### GPU

tested on `NVIDIA RTX3060 12G`

| task                 | AlexNet | ConvNeXt-base | ConvNeXt-tiny | EfficientNetV2-m | EfficientNetV2-s | ResNet18 | ResNet50 | ResNeXt50 |
| -------------------- | ------- | ------------- | ------------- | ---------------- | ---------------- | -------- | -------- | --------- |
| binary               | 0.0059  | 0.0128        | 0.0094        | 0.0279           | 0.0217           | 0.0078   | 0.0102   | 0.0114    |
| drone-classification | 0.0062  | 0.0144        | 0.0103        | 0.0289           | 0.0218           | 0.0077   | 0.0109   | 0.0109    |
| four-class           | 0.0056  | 0.0136        | 0.0094        | 0.0288           | 0.0223           | 0.0081   | 0.0110   | 0.0107    |

#### CPU

tested on `Intel i5-11400`

| task                 | AlexNet | ConvNeXt-base | ConvNeXt-tiny | EfficientNetV2-m | EfficientNetV2-s | ResNet18 | ResNet50 | ResNeXt50 |
| -------------------- | ------- | ------------- | ------------- | ---------------- | ---------------- | -------- | -------- | --------- |
| binary               | 0.0154  | 0.1149        | 0.0451        | 0.0874           | 0.0537           | 0.0207   | 0.0440   | 0.0519    |
| drone-classification | 0.0164  | 0.1227        | 0.0443        | 0.0889           | 0.0525           | 0.0220   | 0.0459   | 0.0490    |
| four-class           | 0.0155  | 0.1139        | 0.0441        | 0.0864           | 0.0530           | 0.0204   | 0.0441   | 0.0486    |

## Environment

This project requires the Python3 environment with the following libraries:
```text
pandas==1.5.2
Pillow==9.2.0
tensorboardX==2.6.2.2
torch==2.0.0
torchvision==0.15.1
tqdm==4.65.0
```
The docker image that contains the environment I used for this project is available [here](https://hub.docker.com/r/zhongliangguo/custom_torch_image).

## Reference

1. Krizhevsky A, Sutskever I, Hinton G E. Imagenet classification with deep convolutional neural networks[J]. Advances in neural information processing systems, 2012, 25.
2. He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.
3. Xie S, Girshick R, Dollár P, et al. Aggregated residual transformations for deep neural networks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 1492-1500.
4. Tan M, Le Q. Efficientnet: Rethinking model scaling for convolutional neural networks[C]//International conference on machine learning. PMLR, 2019: 6105-6114.
5. Liu Z, Mao H, Wu C Y, et al. A convnet for the 2020s[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022: 11976-11986.

