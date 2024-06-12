import torch
import pandas as pd
from PIL import Image
from os.path import join
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from constant import SUPPORTED_TASKS, DRONE_CLASS_INDEX


def get_default_img_tf():
    return transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor()
    ])


class DroneBinaryDataset(Dataset):
    def __init__(self, label_path, data_root, task, img_tf=None):
        super().__init__()
        assert task in SUPPORTED_TASKS.keys()
        self.label = pd.read_csv(label_path)
        self.data_root = data_root
        self.task = task
        if task == "binary":
            pass
        elif task == "drone-classification":
            self.label = self.label[self.label['drone_model'].isin(DRONE_CLASS_INDEX.keys())].reset_index(drop=True)
            self.__getitem__ = self.__drone_classification_get_item__
        else:
            raise NotImplemented
        if img_tf is None:
            self.img_tf = get_default_img_tf()

    def __len__(self):
        return len(self.label)

    def __binary_get_item__(self, index):
        return (self.img_tf(Image.open(join(self.data_root, self.label.iloc[index]['filename'])).convert('RGB')),
                torch.tensor(self.label.iloc[index]['drone_1_else_0']))

    def __drone_classification_get_item__(self, index):
        return (self.img_tf(Image.open(join(self.data_root, self.label.iloc[index]['filename'])).convert('RGB')),
                torch.tensor(DRONE_CLASS_INDEX[self.label.iloc[index]['drone_model']]))

    def __getitem__(self, index):
        if self.task == "binary":
            return self.__binary_get_item__(index)
        elif self.task == "drone-classification":
            return self.__drone_classification_get_item__(index)
        else:
            raise NotImplemented


def get_loader(label_path, data_root, task, img_tf=None, batch_size=64, shuffle=True, drop_last=True,
               num_workers=8) -> DataLoader:
    dataset = DroneBinaryDataset(label_path, data_root, task, img_tf)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return loader
