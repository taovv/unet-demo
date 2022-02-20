import torch
import os
import cv2
import numpy as np
from tqdm import tqdm
from models.unet import UNet


def predict(model, imgs_path=r'datasets/ISBI_cell/test', img_shape=(1, 256, 256), save_path='pred_results'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img_names = os.listdir(imgs_path)
    for img_name in tqdm(img_names):
        if img_shape[0] < 3:
            img = cv2.imread(os.path.join(imgs_path, img_name), cv2.IMREAD_GRAYSCALE)
            h, w = img.shape
            img = cv2.resize(img, (img_shape[1], img_shape[2]))
            img = torch.tensor(img).unsqueeze(0).unsqueeze(0).to('cuda', torch.float32)
        else:
            img = cv2.imread(os.path.join(imgs_path, img_name), cv2.IMREAD_COLOR)
            h, w, _ = img.shape
            img = cv2.resize(img, (img_shape[1], img_shape[2]))
            img = torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0).to('cuda', torch.float32)
        with torch.no_grad():
            pred = model(img)[0].argmax(0)
        pred = pred.view(img_shape[1], img_shape[1]).cpu().detach().numpy().astype(np.float32)
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)*255
        cv2.imwrite(os.path.join(save_path, f'{img_name.split(".")[0]}.png'), pred)


if __name__ == '__main__':
    net = UNet(in_channels=1, classes=2)
    net.to('cuda')
    net.load_state_dict(torch.load('results/Cell_UNet/weights/best_model.pth'))
    predict(model=net,
            imgs_path=r'datasets/ISBI_cell/test',
            img_shape=(1, 256, 256),
            save_path='pred_results')
