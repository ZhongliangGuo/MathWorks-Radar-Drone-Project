import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from models.net import get_model
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam.utils.image import show_cam_on_image
from matplotlib import pyplot as plt
from constant import *

binary_candidate_24GHz = [
    '/home/zg34/datasets/drone_project/data/24GHz/non-drone/interval_0.25_wd_256/2024-02-06_10-24-39/41.png',
    '/home/zg34/datasets/drone_project/data/24GHz/drone/interval_0.25_wd_256/2024-03-01_11-07-57/0.png',
]
binary_candidate_94GHz = [
    '/home/zg34/datasets/drone_project/data/94GHz/non-drone/interval_0.25_wd_256/2024-02-06_10-24-28/27.png',
    '/home/zg34/datasets/drone_project/data/94GHz/drone/interval_0.25_wd_256/2024-03-01_10-45-45/21.png',
]
binary_candidate_207GHz = [
    '/home/zg34/datasets/drone_project/data/207GHz/non-drone/interval_0.25_wd_256/2024-02-06_10-24-23/75.png',
    '/home/zg34/datasets/drone_project/data/207GHz/drone/interval_0.25_wd_256/2024-03-01_09-54-10/0.png',
]

drone_classification_candidate_24GHz = [
    '/home/zg34/datasets/drone_project/data/24GHz/drone/interval_0.25_wd_256/2024-03-01_10-14-20/7.png',
    '/home/zg34/datasets/drone_project/data/24GHz/drone/interval_0.25_wd_256/2024-03-01_11-07-57/21.png',
    '/home/zg34/datasets/drone_project/data/24GHz/drone/interval_0.25_wd_256/2024-03-01_09-33-03/27.png',
    '/home/zg34/datasets/drone_project/data/24GHz/drone/interval_0.25_wd_256/2024-03-01_09-49-52/48.png',
    '/home/zg34/datasets/drone_project/data/24GHz/drone/interval_0.25_wd_256/2024-03-01_10-46-33/6.png',
]
drone_classification_candidate_94GHz = [
    '/home/zg34/datasets/drone_project/data/94GHz/drone/interval_0.25_wd_256/2024-03-01_10-14-24/0.png',
    '/home/zg34/datasets/drone_project/data/94GHz/drone/interval_0.25_wd_256/2024-03-01_11-07-57/49.png',
    '/home/zg34/datasets/drone_project/data/94GHz/drone/interval_0.25_wd_256/2024-03-01_09-33-05/34.png',
    '/home/zg34/datasets/drone_project/data/94GHz/drone/interval_0.25_wd_256/2024-03-01_09-51-34/6.png',
    '/home/zg34/datasets/drone_project/data/94GHz/drone/interval_0.25_wd_256/2024-03-01_10-46-33/48.png',
]
drone_classification_candidate_207GHz = [
    '/home/zg34/datasets/drone_project/data/207GHz/drone/interval_0.25_wd_256/2024-03-01_10-14-20/41.png',
    '/home/zg34/datasets/drone_project/data/207GHz/drone/interval_0.25_wd_256/2024-03-01_12-45-57/38.png',
    '/home/zg34/datasets/drone_project/data/207GHz/drone/interval_0.25_wd_256/2024-03-01_09-33-03/12.png',
    '/home/zg34/datasets/drone_project/data/207GHz/drone/interval_0.25_wd_256/2024-03-01_09-49-45/99.png',
    '/home/zg34/datasets/drone_project/data/207GHz/drone/interval_0.25_wd_256/2024-03-01_10-46-36/5.png',
]

four_class_candidate_24GHz = [
    '/home/zg34/datasets/drone_project/data/24GHz/drone/interval_0.25_wd_256/2024-03-01_11-07-57/0.png',
    '/home/zg34/datasets/drone_project/data/24GHz/bird/interval_0.25_wd_256/2024-02-06_10-26-29/59.png',
    '/home/zg34/datasets/drone_project/data/24GHz/cluster/interval_0.25_wd_256/2024-02-06_10-26-29/1.png',
    '/home/zg34/datasets/drone_project/data/24GHz/noise/interval_0.25_wd_256/2024-03-01_09-30-36/10.png'
]
four_class_candidate_94GHz = [
    '/home/zg34/datasets/drone_project/data/94GHz/drone/interval_0.25_wd_256/2024-03-01_10-45-45/21.png',
    '/home/zg34/datasets/drone_project/data/94GHz/bird/interval_0.25_wd_256/2024-02-06_10-24-28/70.png',
    '/home/zg34/datasets/drone_project/data/94GHz/cluster/interval_0.25_wd_256/2024-02-06_10-26-20/25.png',
    '/home/zg34/datasets/drone_project/data/94GHz/noise/interval_0.25_wd_256/2024-02-06_10-25-12/8.png'
]
four_class_candidate_207GHz = [
    '/home/zg34/datasets/drone_project/data/207GHz/drone/interval_0.25_wd_256/2024-03-01_09-54-10/0.png',
    '/home/zg34/datasets/drone_project/data/207GHz/bird/interval_0.25_wd_256/2024-02-06_10-24-23/67.png',
    '/home/zg34/datasets/drone_project/data/207GHz/cluster/interval_0.25_wd_256/2024-02-06_12-47-38/20.png',
    '/home/zg34/datasets/drone_project/data/207GHz/noise/interval_0.25_wd_256/2024-02-06_13-27-11/0.png'
]
device = torch.device("cuda")


def arch2layer(arch_name, model):
    if arch_name in ['resnet18', 'resnet50', 'resnext50_32x4d']:
        return [model.layer4[-1]]
    elif arch_name in ['convnext_base', 'convnext_tiny']:
        return [model.features[-1][-1].block]
    elif arch_name in ['efficientnet_v2_m', 'efficientnet_v2_s']:
        return [model.features[-1]]
    elif arch_name in ['alexnet']:
        return [model.features[10]]


def draw(task, arch, freq):
    net = get_model(arch, SUPPORTED_TASKS[task], False)
    net.load_state_dict(torch.load(
        f'/home/zg34/Desktop/MathWorks-Radar-Drone-Project/logs/pth_models/{task}/{task}-{arch}.pth'))
    net = net.eval().to(device)

    cam = GradCAM(net, arch2layer(arch, net))

    tf = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])

    multi_fig, axis_list = plt.subplots(SUPPORTED_TASKS[task], SUPPORTED_TASKS[task] + 1,
                                        figsize=((SUPPORTED_TASKS[task] + 1) * 2, SUPPORTED_TASKS[task] * 2))
    if task == 'binary':
        label_mapping = NUM_TO_CLASS_BINARY
    elif task == 'drone-classification':
        label_mapping = NUM_TO_CLASS_DRONES
    elif task == 'four-class':
        label_mapping = NUM_TO_FOUR_CLASS
    else:
        raise Exception
    for label in range(SUPPORTED_TASKS[task]):
        ori_img = Image.open(eval(f'{task.replace("-", "_")}_candidate_{freq}GHz')[label]).convert('RGB')
        img = tf(ori_img).unsqueeze(0)
        axis_list[label][0].imshow(ori_img)
        if label == 0:
            axis_list[label][0].set_title('Input Image')
        axis_list[label][0].set_xticks([])
        axis_list[label][0].set_yticks([])
        axis_list[label][0].set_ylabel(label_mapping[label], rotation=90, va='center')
        for idx, ax in enumerate(axis_list[label][1:]):
            target = [ClassifierOutputTarget(idx)]
            grayscale_cam = cam(input_tensor=img, targets=target)
            grayscale_cam = grayscale_cam[0, :]
            ori_img = img[0].permute(1, 2, 0).cpu().numpy()
            visualization = show_cam_on_image(ori_img, grayscale_cam, use_rgb=True)
            axis_list[label][idx + 1].imshow(visualization)
            if label == 0:
                axis_list[label][idx + 1].set_title(f'label = {label_mapping[idx]}')
            axis_list[label][idx + 1].axis('off')
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.tight_layout()
    plt.savefig(f'figs/{task}_{arch}_GradCAM_{freq}.png')


for task in SUPPORTED_TASKS.keys():
    for arch in IMPLEMENTED_NETS:
        for freq in [24, 94, 207]:
            draw(task, arch, freq)
