import os
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from glob import glob
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from PIL import Image


class BaseDataset(Dataset):
    """
    基本数据集
    """

    def __init__(self, img_paths, mask_paths, x_transform=None, y_transform=None, channels=3):
        assert channels in [1, 3], 'channels should be 1 or 3'
        self.x_transform = transforms.Compose([transforms.ToTensor()]) if x_transform is None else x_transform
        self.y_transform = transforms.Compose([transforms.ToTensor()]) if y_transform is None else y_transform
        self.channels = channels
        self.img_paths = img_paths
        self.mask_paths = mask_paths

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        if self.channels == 1:
            img = Image.open(img_path).convert('L')  # 1 channel
        else:
            img = Image.open(img_path).convert('RGB')  # 3 channels
        mask = Image.open(mask_path)
        if self.x_transform is not None:
            img = self.x_transform(img)
        if self.y_transform is not None:
            mask = self.y_transform(mask)
        return img, mask

    def __len__(self):
        return len(self.img_paths)


def get_path_ratio(data_path, test_ratio=0.3):
    """
    将数据集划分为训练集和测试集
    :param data_path: 数据集跟路径,其目录下包含images，masks两个文件夹，
    :param test_ratio: 测试数据的比例
    :return: 训练集图像及其对应的mask标注路径；测试集图像及其对应的mask标注路径
    """
    img_paths = glob(data_path + r'/images/*')
    mask_paths = glob(data_path + r'/masks/*')
    train_img_paths, test_img_paths, train_mask_paths, test_mask_paths = \
        train_test_split(img_paths, mask_paths, test_size=test_ratio, random_state=41)

    return train_img_paths, train_mask_paths, test_img_paths, test_mask_paths


def get_loaders(dataset_path, batch_size, batch_size_test, x_transforms, y_transforms,
                num_workers=1, val_ratio=0.3, channels=1, loader_shuffle=False):
    """
    获取训练集测试集的dataloader
    :param dataset_path:数据集所在路径
    :param batch_size:训练集batch_size
    :param batch_size_test:测试集batch_size
    :param x_transforms:输入图像变换操作
    :param y_transforms:mask标注变换操作
    :param loader_shuffle: 是否每个epoch打乱loader数据
    :param num_workers: workers数量
    :param val_ratio:验证集占比
    :param channels:训练图像通道数
    :return:
    """
    train_img_paths, train_mask_paths, test_img_paths, test_mask_paths = get_path_ratio(dataset_path, val_ratio)

    train_dataset = BaseDataset(train_img_paths, train_mask_paths, x_transforms, y_transforms, channels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=loader_shuffle,
                              drop_last=False)

    test_dataset = BaseDataset(test_img_paths, test_mask_paths, x_transforms, y_transforms, channels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, num_workers=num_workers, shuffle=loader_shuffle,
                             drop_last=False)

    return train_loader, test_loader
