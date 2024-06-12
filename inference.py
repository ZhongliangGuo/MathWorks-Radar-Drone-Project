import torch
from PIL import Image
from models.net import get_model
from argparse import ArgumentParser
from dataset import get_default_img_tf
from constant import SUPPORTED_TASKS, NUM_TO_CLASS_BINARY, NUM_TO_CLASS_DRONES, IMPLEMENTED_NETS


def predict_a_image(net: torch.nn.Module, task: str, image: torch.Tensor):
    assert task in SUPPORTED_TASKS.keys()
    _, pred_cls = torch.max(net(image), dim=-1)
    if task == 'binary':
        return NUM_TO_CLASS_BINARY[pred_cls.item()]
    elif task == 'drone-classification':
        return NUM_TO_CLASS_DRONES[pred_cls.item()]
    else:
        raise NotImplemented


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task', type=str, choices=SUPPORTED_TASKS.keys(), default='binary')
    parser.add_argument('--arch', type=str, choices=IMPLEMENTED_NETS, default=IMPLEMENTED_NETS[0])
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--pth_path', type=str, required=True)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = get_model(arch=args.arch, num_classes=SUPPORTED_TASKS[args.task], use_pretrained=False)
    net.load_state_dict(torch.load(args.pth_path))
    net = net.to(device).eval()
    img_tf = get_default_img_tf()
    img_tensor = img_tf(Image.open(args.image_path).convert('RGB')).unsqueeze(0).to(device)
    print(predict_a_image(net, args.task, img_tensor))
