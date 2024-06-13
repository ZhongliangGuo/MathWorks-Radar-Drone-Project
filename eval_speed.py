import os
import time
import torch
from os.path import join
from models.net import get_model
from dataset import get_eval_time_loader
from constant import SUPPORTED_TASKS, IMPLEMENTED_NETS

for device in [torch.device("cuda"), torch.device('cpu')]:
    for task in SUPPORTED_TASKS.keys():
        loader = get_eval_time_loader('/home/zg34/datasets/drone_project/eval_label.csv',
                                      '/home/zg34/datasets/drone_project/data',
                                      task, num=200)
        for arch in IMPLEMENTED_NETS:
            net = get_model(arch, SUPPORTED_TASKS[task], False)
            for subfolder in os.listdir(f'./logs/{task}'):
                if arch in subfolder:
                    for pth_name in os.listdir(join(f'./logs/{task}', subfolder)):
                        if 'best' in pth_name:
                            net.load_state_dict(torch.load(join(f'./logs/{task}', subfolder, pth_name)))
                            print('load pth')
                            break
            net.eval()
            net.to(device)
            cost = time.time()
            for idx, (img, _) in enumerate(loader):
                img = img.to(device)
                y_pred = net(img)
                _, cls_pred = torch.max(y_pred, dim=-1)
            cost = time.time() - cost
            print(f'{device.type}-{task}-{arch}-{cost / len(loader):.4f}')
