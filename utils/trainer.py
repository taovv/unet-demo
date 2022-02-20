import os
import torch

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def color_string(s, front=50, word=32):
    """
    # 改变字符串颜色的函数
    :param str:
    :param front:
    :param word:
    :return:
    """
    new_str = "\033[0;" + str(int(word)) + ";" + str(int(front)) + "m" + s + "\033[0m"
    return new_str


class Trainer(object):

    def __init__(self, model, criterion, optimizer, scheduler, metric, train_loader, val_loader, args, logger):
        self.model = model.to(args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric = metric
        self.args = args
        self.logger = logger

        if not os.path.exists(args.tensorboard_dir):
            os.makedirs(args.tensorboard_dir)
        self.writer = SummaryWriter(log_dir=args.tensorboard_dir)
        self.inputs = next(iter(self.train_loader))[0]
        self.writer.add_graph(self.model, self.inputs.to(self.args.device, dtype=torch.float32))

        if self.args.DataParallel:
            self.model = torch.nn.DataParallel(self.model)

    def _preload(self):
        """
        若存在预训练权重则加载
        :return:
        """
        if self.args.pretrained is not None and os.path.isfile(self.args.pretrained):
            self.model.load_state_dict(torch.load(self.args.pretrained))
            self.logger.info(color_string(f'load weights:{self.args.pretrained} finish!', word=36))

    def train(self):
        self._preload()
        epochs = self.args.epochs
        n_train = len(self.train_loader)
        step = 0
        best_score = 0.
        for epoch in range(epochs):
            self.model.train()
            # training
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
                for batch in self.train_loader:
                    images, masks = batch[0], batch[1]
                    images = images.to(device=self.args.device, dtype=torch.float32)
                    masks = masks.to(device=self.args.device, dtype=torch.long)

                    self.optimizer.zero_grad()
                    preds = self.model(images)
                    loss = self.criterion(preds, masks.squeeze(1))
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                    self.optimizer.step()

                    self.writer.add_scalar('Loss/train', loss.item(), step)
                    pbar.set_postfix(**{'loss(batch)': loss.item()})
                    pbar.update(1)
                    step = step + 1
            # val
            if (epoch + 1) % self.args.val_epoch == 0:
                score = self.val()
                if score > best_score:
                    best_score = score
                    if self.args.save_path:
                        if not os.path.exists(self.args.save_path):
                            os.makedirs(self.args.save_path)
                        torch.save(self.model.state_dict(), f'{self.args.save_path}/best_model.pth')
                        self.logger.info(color_string(f'best model saved !', word=33))

                self.logger.info(f'Epoch-{epoch + 1}:{self.metric.name}: {score}')
                self.writer.add_scalar(f'Val_{self.metric.name}', best_score, step)
                self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], step)
                if self.args.save_pred_img:
                    self.writer.add_images('images', images, step)
                    self.writer.add_images('masks/true', masks, step)
                    self.writer.add_images('masks/pred', preds.argmax(dim=1, keepdim=True), step)
                self.scheduler.step(score)

            if (epoch + 1) % self.args.save_model_epoch == 0:
                if self.args.save_path:
                    if not os.path.exists(self.args.save_path):
                        os.makedirs(self.args.save_path)
                    model_name = f'{self.args.model}_{self.args.dataset}'
                    torch.save(self.model.state_dict(),
                               f'{self.args.save_path}/{model_name}_{epoch + 1}.pth')
                    self.logger.info(color_string(f'Checkpoint {epoch + 1} saved !'))
        self.writer.close()

    def val(self):
        self.model.train(False)
        self.model.eval()

        val_len = len(self.val_loader)
        batch_score = 0.
        number = 0
        score_list = []
        with torch.no_grad():
            with tqdm(total=val_len, desc=f'val', unit='batch') as pbar:
                for batch in self.val_loader:
                    images, masks = batch[0], batch[1]
                    images = images.to(self.args.device)
                    masks = masks.to(self.args.device)
                    preds = self.model(images).argmax(dim=1, keepdim=True)
                    preds = preds.data.cpu().numpy()
                    masks = masks.data.cpu().numpy()
                    for _ in range(preds.shape[0]):
                        pred_tmp = preds[_, :].reshape(-1)
                        mask_tmp = masks[_, :].reshape(-1)
                        score = self.metric(pred_tmp, mask_tmp)
                        score_list.append(score)
                        batch_score += score
                        number += 1
                    pbar.set_postfix(**{f'{self.metric.name}': batch_score / number})
                    pbar.update(1)

        return batch_score / number
