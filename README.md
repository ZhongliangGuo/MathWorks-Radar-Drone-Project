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

### Inference
```bash
python inference.py \
  --task binary \
  --arch resnet18 \
  --image_path /.../XXX.png \
  --pth_path /.../XXX.pth
```
## Environment
This project requires a Python3 environment with the following libraries:
```text
tqdm==4.66.1
pandas==2.0.3
Pillow==9.4.0
torch==2.1.1
torchvision==0.16.1
tensorboardX==2.6.2.2
```
# Reference
1. Krizhevsky A, Sutskever I, Hinton G E. Imagenet classification with deep convolutional neural networks[J]. Advances in neural information processing systems, 2012, 25.
2. He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.
3. Xie S, Girshick R, Dollár P, et al. Aggregated residual transformations for deep neural networks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 1492-1500.
4. Tan M, Le Q. Efficientnet: Rethinking model scaling for convolutional neural networks[C]//International conference on machine learning. PMLR, 2019: 6105-6114.
5. Liu Z, Mao H, Wu C Y, et al. A convnet for the 2020s[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022: 11976-11986.

