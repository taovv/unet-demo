import torch
import argparse
import os
import time

from torch import optim
from models.unet import UNet
from utils.trainer import Trainer
from torchvision.transforms import transforms
from utils.loader import get_loaders
from utils.logger import get_logger
from metrics import DSC


def get_args():
    """
    调参，可直接修改default
    :return: args
    """
    parse = argparse.ArgumentParser()
    # task info
    parse.add_argument('--model_name', type=str, default='UNet')
    parse.add_argument('--task_name', type=str, default='Cell_UNet')
    parse.add_argument('--pretrained', type=str, default=None)

    # dataset
    parse.add_argument('--dataset_path', type=str, default=r'datasets/ISBI_cell/train', help='dataset root path')
    parse.add_argument('--dataset_name', type=str, default='ISBI_cell')
    parse.add_argument('--img_size', type=int, default=256)
    parse.add_argument('--in_channels', type=int, default=1, help='image channels')
    parse.add_argument('--num_workers', type=int, default=3)

    # train parameters
    parse.add_argument('--epochs', type=int, default=60, help='epoch number')
    parse.add_argument('--batch_size', type=int, default=8, help='batch size')
    parse.add_argument('--val_epoch', type=int, default=4, help='val every n epoch')
    parse.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parse.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parse.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    # test
    parse.add_argument('--batch_size_test', type=int, default=16, help='batch size test')

    # GPU
    parse.add_argument('--device', type=str, default='cuda')
    parse.add_argument('--DataParallel', type=bool, default=False, help='multi gpus')
    parse.add_argument('--cuda_ids', type=str, default='0', help='0/1/0,1')

    # save
    parse.add_argument('--save_model_epoch', type=int, default=9999, help='save model every n epoch')
    parse.add_argument('--save_path', type=str, default=None, help='the path of model weight file')
    parse.add_argument('--weights', type=str, default=None)
    parse.add_argument('--save_pred_img', type=bool, default=True)

    # log
    parse.add_argument('--tensorboard_dir', type=str, default=None, help='tensorboard dir')
    parse.add_argument('--log_file', type=str, default=None, help='log dir')
    args = parse.parse_args()
    update_args(args)
    return args


def update_args(args):
    """
    更新日志、结果保存路径参数
    :param args:
    :return:
    """
    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.save_path is None:
        args.save_path = os.path.join('./results', args.task_name, 'weights')
    if args.tensorboard_dir is None:
        args.tensorboard_dir = os.path.join('./results', args.task_name, 'runs')
    if args.log_file is None:
        args.log_file = os.path.join('./results', args.task_name, 'logs',
                                     f'{time.strftime("%Y%m%d%H%M", time.localtime(time.time()))}.log')


def train():
    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_ids

    x_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 3通道
        # transforms.Normalize([0.5], [0.5])  # 单通道
    ])
    y_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])

    model = UNet(in_channels=args.in_channels, classes=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    train_loader, val_loader = get_loaders(args.dataset_path, args.batch_size, args.batch_size_test, x_transforms,
                                           y_transforms, args.num_workers, channels=args.in_channels)
    metric = DSC()

    logger = get_logger(args.log_file)
    logger.info(f'''Starting training:
                 Model:           {args.model_name}
                 Dataset:         {args.dataset_name}
                 Input Shape:     {(args.in_channels, args.img_size, args.img_size)}
                 Epochs:          {args.epochs}
                 Batch Size:      {args.batch_size}
                 Learning Rate:   {args.lr}
                 Test Batch Size: {args.batch_size_test}
                 DataParallel:    {args.DataParallel}
                 Device:          {args.device}
                 Cuda Ids:        {args.cuda_ids}
                 Save Model:      {'per ' + str(args.save_model_epoch) + ' epochs'}
             ''')

    trainer = Trainer(model, criterion, optimizer, scheduler, metric, train_loader, val_loader, args, logger)

    start = time.time()
    trainer.train()
    end = time.time()

    logger.info(f'Train Time:{int((end - start) / 60 // 60)}h:{int((end - start) / 60 % 60)}m')


if __name__ == '__main__':
    train()
