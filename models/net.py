import torch.nn as nn
from torchvision import models as tv_models
from constant import IMPLEMENTED_NETS


def get_model(arch, num_classes, use_pretrained=True) -> nn.Module:
    assert arch in IMPLEMENTED_NETS
    if use_pretrained:
        model = eval(f"tv_models.{arch}(weights='DEFAULT')")
        if 'res' in arch:
            model.fc = nn.Linear(in_features=model.fc.in_features,
                                 out_features=num_classes,
                                 bias=model.fc.bias is not None)
        elif 'convnext' in arch or 'efficientnet' in arch or 'alex' in arch:
            model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features,
                                             out_features=num_classes,
                                             bias=model.classifier[-1].bias is not None)
        else:
            raise NotImplemented
        print(f"loaded the pretrained weights for {arch}")
    else:
        model = eval(f"tv_models.{arch}(num_classes={num_classes})")
    return model
