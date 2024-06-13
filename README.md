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

## Usage
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
  --train_label_path /.../train_label.csv \
  --eval_label_path /.../eval_label.csv \
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

Download the dataset by clicking [here](https://universityofstandrews907-my.sharepoint.com/:u:/g/personal/zg34_st-andrews_ac_uk/ESGPaToyidtHgWNjdL_l-JgB9TKvEYKxzW4JEOiuFAcFZQ?e=duo8iS). The password is `mathworks2024`.

## Results

| task                 | AlexNet | ConvNeXt-base | ConvNeXt-tiny | EfficientNetV2-m | EfficientNetV2-s | ResNet18 | ResNet50 | ResNeXt50 |
| -------------------- | ------- | ------------- | ------------- | ---------------- | ---------------- | -------- | -------- | --------- |
| binary               | 99.76%  | 99.92%        | 99.96%        | 99.88%           | 99.84%           | 99.80%   | 99.68%   | 99.68%    |
| drone-classification | 98.44%  | 99.36%        | 99.45%        | 98.16%           | 98.71%           | 98.44%   | 97.33%   | 97.89%    |

It's worth noting that all above models used the ImageNet-1K pre-trained weights.

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
## Reference

1. Krizhevsky A, Sutskever I, Hinton G E. Imagenet classification with deep convolutional neural networks[J]. Advances in neural information processing systems, 2012, 25.
2. He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.
3. Xie S, Girshick R, Dollár P, et al. Aggregated residual transformations for deep neural networks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 1492-1500.
4. Tan M, Le Q. Efficientnet: Rethinking model scaling for convolutional neural networks[C]//International conference on machine learning. PMLR, 2019: 6105-6114.
5. Liu Z, Mao H, Wu C Y, et al. A convnet for the 2020s[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022: 11976-11986.

