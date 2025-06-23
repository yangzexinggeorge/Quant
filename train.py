import torch
import torch.optim
import torch.utils.data
from ffquant.strategy.cnn.data.data_entry import select_train_loader, select_eval_loader
from ffquant.strategy.cnn.model.model_entry import select_model
from ffquant.strategy.cnn.options import prepare_train_args
from ffquant.strategy.cnn.utils.logger import Logger
from ffquant.strategy.cnn.utils.torch_utils import load_match_dict
import os
from tqdm import tqdm


# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = 0.7

class Trainer:
    def __init__(self):
        args = prepare_train_args()
        torch.mps.set_per_process_memory_fraction(0.0)
        self.args = args
        torch.manual_seed(args.seed)
        self.logger = Logger(self.args)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print("user device {device}".format(device=self.device))

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        # 获取数据加载器
        self.train_loader = select_train_loader(args)
        self.val_loader = select_eval_loader(args)
        # 获取模型
        self.model = select_model(args=args)
        self.model.to(self.device)
        self.train_loss = []
        self.train_acc = []
        self.validate_loss = []
        self.validate_acc = []
        self.model = torch.nn.DataParallel(self.model)  # ，它可以将一个模型分发到多个 GPU 上并行计算
        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr,
                                          betas=(self.args.momentum, self.args.beta),
                                          weight_decay=self.args.weight_decay)  # 优化器用于更新模型的参数以最小化损失函数

    def train_per_epoch(self, epoch):

        total_loss = 0
        total_correct = 0
        total_data = 0
        train_bar = tqdm(self.train_loader)
        # switch to train mode，切换到测试模式
        self.model.train()
        iteration = 0
        for i, data in enumerate(train_bar):

            # img, pred, label = self.step(data)
            test_x, test_label = data
            test_x = test_x.to(self.device)
            test_label = test_label.to(self.device)

            self.optimizer.zero_grad()
            # compute output
            predict_y = self.model(test_x)

            _, predicted = torch.max(predict_y.data, dim=1)
            total_correct += torch.eq(predicted, test_label).sum().item()
            # compute loss
            loss, metrics = self.compute_metrics(predict_y, test_label, is_train=True)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            total_data += test_label.size(0)
            iteration += 1
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} iteration:{} current_accuracy:{:.2f}%---------".format(
                epoch + 1,
                self.args.epochs,
                loss,
                iteration,
                100.0 * total_correct / total_data)
            # logger record
            for key in metrics.keys():
                self.logger.record_scalar(key, metrics[key])
            # only save img at first step

            # if i == len(self.train_loader) - 1:
            # self.logger.save_imgs(self.gen_imgs_to_write(test_x, predict_y, test_label, True), epoch)

            # monitor training progress
            # if i % self.args.print_freq == 0:
            # print('Train: Epoch {} batch {} Loss {}'.format(epoch, i, loss))

        loss = total_loss / len(self.train_loader)
        acc = 100 * total_correct / total_data
        print('accuracy on train set:%d %%' % acc)

    def train(self):
        num_epochs = self.args.epochs

        for epoch in range(num_epochs):  ## 训练整批数据多少次, 为了节约时间, 我们只训练一次
            self.train_per_epoch(epoch)
            torch.mps.empty_cache()

            self.val_per_epoch(epoch)
            torch.mps.empty_cache()

            self.logger.save_curves(epoch)

            self.logger.save_check_point(self.model, epoch)  # 保存模型参数

    def val_per_epoch(self, epoch):
        total_loss = 0
        total_correct = 0
        total_data = 0
        with torch.no_grad():
            self.model.eval()
            test_bar = tqdm(self.val_loader)
            for i, data in enumerate(test_bar):
                img, pred, label = self.eval_step(data)
                loss, metrics = self.compute_eval_metrics(pred, label, is_train=False)
                _, predicted = torch.max(pred.data, dim=1)
                total_correct += torch.eq(predicted, label).sum().item()
                total_loss += loss.item()
                total_data += label.size(0)
                test_bar.desc = "validate epoch[{}/{}] current_accuracy:{:.2f}%---------".format(epoch + 1,
                                                                                                 self.args.epochs,
                                                                                                 100.0 * total_correct / total_data)
                # for key in metrics.keys():
                for key in metrics.keys():
                    self.logger.record_scalar(key, metrics[key])
            # if i == len(self.val_loader) - 1:
            # self.logger.save_imgs(self.gen_imgs_to_write(img, pred, label, False), epoch)
        loss = total_loss / len(self.val_loader)
        acc = 100 * total_correct / total_data
        self.validate_loss.append(loss)
        self.validate_acc.append(acc)

        print('accuracy on validate set:%d %%\n' % acc)

    def eval_step(self, data):
        img, label = data
        img = img.to(self.device)
        label = label.to(self.device)
        pred = self.model(img)
        return img, pred, label

    def compute_metrics(self, pred, label, is_train):
        # 使用交叉墒函数
        prefix = 'train/' if is_train else 'val/'
        if self.args.loss == 'l1':
            loss = (pred - label).abs().mean()
        elif self.args.loss == 'ce':
            loss = torch.nn.functional.cross_entropy(pred, label)
        else:
            loss = torch.nn.functional.mse_loss(pred, label)
        metrics = {
            prefix + self.args.loss: loss
        }
        return loss, metrics

    def compute_eval_metrics(self, pred, label, is_train):
        # 使用交叉墒函数
        prefix = 'train/' if is_train else 'val/'
        if self.args.eval_loss == 'l1':
            loss = (pred - label).abs().mean()
        elif self.args.eval_loss == 'ce':
            loss = torch.nn.functional.cross_entropy(pred, label)
        else:
            loss = torch.nn.functional.mse_loss(pred, label)
        metrics = {
            prefix + self.args.loss: loss
        }
        return loss, metrics

    def gen_imgs_to_write(self, img, pred, label, is_train):
        # override this method according to your visualization
        prefix = 'train/' if is_train else 'val/'
        return {
            prefix + 'img': img[0],
            prefix + 'pred': pred[0],
            prefix + 'label': label[0]
        }


def main():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()
