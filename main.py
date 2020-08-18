# -*- coding: utf-8 -*-
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import os
import argparse
import torch.nn as nn
from visdom_visualization import Visdom_Visualization

from misc import progress_bar
from resnet import ResNet18
from AlexNet import AlexNet
from VGG import VGG11
from LSTM import LSTM
from MyDataset import MyDataset

def main():
    #设置参数
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs tp train for')
    parser.add_argument('--trainBatchSize', default=50, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    args = parser.parse_args()

    solver = Solver(args)
    solver.run()


class Solver(object):
    def __init__(self, config):
        #初始化变量
        self.model = None
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None

    def load_data(self):
        #读取数据 如何设计自己的数据集有待研究
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        
        #make dataset by myself
        # train_set = MyDataset('cifar-10-batches-py/path.txt',transform=train_transform)#自制数据集读取函数
        train_set = torchvision.datasets.CIFAR10(root=os.getcwd(), train=True, download=False, transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True)
        test_set = torchvision.datasets.CIFAR10(root=os.getcwd(), train=False, download=False, transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)

    def load_model(self):
        #读取模型，设置损失函数，设置优化方法，设置学习率的优化方法
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        # self.model = AlexNet().to(self.device)
        # self.model = VGG11().to(self.device)
        self.model = ResNet18().to(self.device)
        self.useLSTM = False#不用 LSTM的时候 把这个标志置为False
        # self.model = LSTM(32*3,128).to(self.device)
        # self.useLSTM = True
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train(self):
        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0
        #逐batch训练，并显示数据
        window_titles = ["train_batch_loss"]
        # vis = Visdom_Visualization(env_forename='model1',window_titles=window_titles)
        
        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            target = target.long()#为了自己读取数据集时候用的一句话，否则类型不匹配。
            self.optimizer.zero_grad()
            
            if(self.useLSTM):
                data = data.view(-1,32,32*3)
            output = self.model(data)
            
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            #累积loss
            train_loss += loss.item()
            
            # vis.update_line("train_batch_loss", batch_num, loss.item())
            prediction = torch.max(output, 1)  
            #用来计算准确率
            total += target.size(0)
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))

        return train_loss, train_correct / total

    def test(self):
        print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0
        #类似train  不需要计算梯度更新权重
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))

        return test_loss, test_correct / total

    def save(self):
        model_out_path = "model.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        self.load_data()
        self.load_model()
        accuracy = 0
        for epoch in range(1, self.epochs + 1):
            self.scheduler.step(epoch)
            print("\n===> epoch: %d/200" % epoch)
            train_result = self.train()
            print(train_result)
            test_result = self.test()
            accuracy = max(accuracy, test_result[1])
            if epoch == self.epochs:
                print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
                self.save()


if __name__ == '__main__':
    main()
