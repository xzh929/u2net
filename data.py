from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torch


class EyeDataTrain(Dataset):
    def __init__(self, root):
        self.root = root
        self.data_name = os.listdir(os.path.join(self.root, r"images"))
        self.tag_name = os.listdir(os.path.join(self.root, r"1st_manual"))
        self.transforms = T.Compose(
            [T.ToTensor(), T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomCrop((584, 565))])

    def __len__(self):
        return len(self.data_name)

    def __getitem__(self, item):
        data_name = self.data_name[item]
        tag_name = self.tag_name[item]
        data_path, tag_path = os.path.join(self.root, r"images"), os.path.join(self.root, r"1st_manual")
        data = Image.open(os.path.join(data_path, data_name)).convert("RGB")
        tag = Image.open(os.path.join(tag_path, tag_name)).convert("L")
        seed = np.random.randint(214748367)
        torch.manual_seed(seed=seed)
        data = self.transforms(data)
        torch.manual_seed(seed=seed)
        tag = self.transforms(tag)
        return data, tag


class EyeDataTest(Dataset):
    def __init__(self, root):
        self.root = root
        self.data_name = os.listdir(os.path.join(self.root, r"images"))
        self.transforms = T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.data_name)

    def __getitem__(self, item):
        data_name = self.data_name[item]
        data_path = os.path.join(self.root, r"images")
        data = Image.open(os.path.join(data_path, data_name)).convert("RGB")
        data = self.transforms(data)
        return data


if __name__ == '__main__':
    dataset_train = EyeDataTrain(r"D:\data\eye\training")
    dataset_test = EyeDataTest(r"D:\data\eye\test")
    a = dataset_train[0]
    print(len(dataset_train))
    print(a[1].shape)
