## 라이브러리 추가하기
import os  # 환경 변수나 디렉토리, 파일 등의 OS 자원 제어 모듈. (파일 저장시 경로 설정, 폴더 생성.)
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn  # 신경망을 위한 기본 블럭 제공.(ex. Conv2d, MaxPool2d)
import torchvision
from torch.utils.data import DataLoader  # MNIST 데이터 셋 다운로드.
from torch.utils.tensorboard import SummaryWriter  # 데이터 저장을 위한 함수.(요약데이터)

from torch.optim import lr_scheduler
from torchvision.models.resnet import resnet18
from torchvision import transforms, datasets

## 트레이닝 필요한 파라메타를 설정
lr = 1e-3  # 0.001(10^-3)
batch_size = 10
num_epoch = 50

LAMBDA_1 = 0.2
LAMBDA_2 = 0.05
START_AGE = 0
END_AGE = 9
VALIDATION_RATE = 0.1

ckpt_dir = './checkpoint'
train_log_dir = './log/train'
val_log_dir = './log/val'
test_log_dir = './log/test'

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')  # cuda(GPU)사용 or CPU 사용.


## 네트워크 구축하기
class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return (x)


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels * 4, kernel_size=1,
                                                          stride=stride),
                                                nn.BatchNorm2d(out_channels * 4))
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels * 4
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)


def ResNet50(img_channels=1, num_classes=10):
    return ResNet(block, [3, 4, 6, 3], img_channels, num_classes)


def test():
    net = ResNet50()
    x = torch.randn(2, 1, 224, 224)
    y = net(x).to('cuda')
    print(y.shape)


test()

## 네트워크를 저장, 불러오는 함수 작성
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

        torch.save({'net': net.state_dict(), 'optim': optim.state_dict()}, './%s/model_epoch%d.pth' % (ckpt_dir, epoch))


def load(ckpt_dir, net, optim):
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort()

    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])

    return net, optim



## face_age 데이터
import glob
import cv2
transf = transforms.Compose([transforms.Grayscale(), transforms.Resize((224, 224)), transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5,), std=(0.5,))])
dataset = torchvision.datasets.ImageFolder(root='./dataset/custom/face_age10', transform = transf)

## face_age 데이터
import glob
import cv2
transf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5,), std=(0.5,))])
dataset = torchvision.datasets.ImageFolder(root='./dataset/face_age/face_age/face_age', transform = transf)

## face_age_test 데이터
import glob
import cv2
transf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5,), std=(0.5,))])
testdataset = torchvision.datasets.ImageFolder(root='./dataset/face_age_testdata', transform = transf)

loader_test = DataLoader(testdataset, batch_size=batch_size, shuffle=True, num_workers=0)

##
from torch.utils.data.sampler import SubsetRandomSampler

def get_subset(indices, start, end):
    return indices[start : start + end]

TRAIN_PCT, VALIDATION_PCT = 0.8, 0.1 # rest will go for test
train_count = int(len(dataset) * TRAIN_PCT)
validation_count = int(len(dataset) * VALIDATION_PCT)
indices = torch.randperm(len(dataset))
train_indices = get_subset(indices, 0, train_count)
validation_indices = get_subset(indices, train_count, validation_count)
test_indices = get_subset(indices, train_count + validation_count, len(dataset))
dataloaders = {
    "train": DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices)
    ),
    "validation": DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(validation_indices)
    ),
    "test": DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices)
    ),
}

num_batch = len(dataloaders['train'])

num_batch_val = len(dataloaders['validation'])

num_batch_test = len(dataloaders['test'])

# 그래프 표시
train_images = [cv2.resize(cv2.imread(file), dsize=(224, 224), interpolation=cv2.INTER_AREA) for file in
                glob.glob("./dataset/custom/face_age5/*.png")]
train_images = np.array(train_images)

train_labels = os.listdir('./dataset/custom/face_age5')

train_labels = np.array(train_labels)

import os
num = []
for i in train_labels:
    list = os.listdir(f'./dataset/custom/face_age5/{i}')  # dir is your directory path
    number_files = len(list)
    print(number_files)
    num.append(number_files)

plt.bar(train_labels, num, 0.5, color='g')
plt.show()


## 이미지를 보여주기 위한 함수
def imshow(img):
    npimg = img.cpu().numpy()
    fig = plt.figure(figsize=(10, 5))
    npimg = np.transpose(npimg, (1, 2, 0))
    # npimg = cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB)
    plt.imshow(npimg, vmin= 0, vmax = 1)
    plt.show()


dataiter = iter(dataloaders['test'])
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images[:4]))
print(' '.join('%5s' % [labels[j]] for j in range(4)))


## meanvarianceloss
class MeanVarianceLoss(nn.Module):

    def __init__(self, lambda_1, lambda_2, start_age, end_age):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.start_age = start_age
        self.end_age = end_age

    def forward(self, input, target):
        N = input.size()[0]
        target = target.type(torch.FloatTensor).cuda()
        m = nn.Softmax(dim=1)
        p = m(input)
        # mean loss
        a = torch.arange(self.start_age, self.end_age + 1, dtype=torch.float32).cuda()
        mean = torch.squeeze((p * a).sum(1, keepdim=True), dim=1)
        mse = (mean - target) ** 2
        mean_loss = mse.mean() / 2.0

        # variance loss
        b = (a[None, :] - mean[:, None]) ** 2
        variance_loss = (p * b).sum(1, keepdim=True).mean()

        return self.lambda_1 * mean_loss, self.lambda_2 * variance_loss


## 네트워크 설정 및 필요한 손실함수 구현하기
net = ResNet50().to(device)
params = net.parameters()

fn_pred = lambda output: torch.softmax(output, dim=1)
fn_acc = lambda pred, label: ((pred.max(dim=1)[1] == label).type(torch.float)).mean()


optim = torch.optim.Adam(params, lr=lr)
criterion1 = MeanVarianceLoss(LAMBDA_1, LAMBDA_2, START_AGE, END_AGE).cuda()
criterion2 = torch.nn.CrossEntropyLoss().cuda()
scheduler = lr_scheduler.MultiStepLR(optim, milestones=[30], gamma=0.1)

writer_train = SummaryWriter(log_dir=train_log_dir)
writer_val = SummaryWriter(log_dir=val_log_dir)
writer_test = SummaryWriter(log_dir=test_log_dir)

## 트레이닝 시작
import matplotlib.pyplot as plt

for epoch in range(1, num_epoch + 1):
    net.train()

    loss_arr = []
    acc_arr = []

    for batch, samples in enumerate(dataloaders['train']):
        input, label = samples

        input = input.float()
        label = label.long()

        input = input.to(device)
        label = label.to(device)

        output = net(input)
        pred = fn_pred(output)

        mean_loss, variance_loss = criterion1(output, label)
        ce_loss = criterion2(output, label)
        loss = mean_loss + variance_loss + ce_loss

        optim.zero_grad()

        # loss = fn_loss(output, label)
        acc = fn_acc(pred, label)

        loss.backward()

        optim.step()

        loss_arr += [loss.item()]
        acc_arr += [acc.item()]
        print('TRAIN : EPOCH %04d/%04d | BATCH %4d/%04d | LOSS : %.4f | ACC %.4f'
              % (epoch, num_epoch, batch, num_batch, np.mean(loss_arr), np.mean(acc_arr)))
    writer_train.add_scalar('train_loss', np.mean(loss_arr), epoch)
    writer_train.add_scalar('train_acc', np.mean(acc_arr), epoch)

    with torch.no_grad():
        net.eval()

        loss_arr = []
        acc_arr = []

        for batch, samples in enumerate(dataloaders.get('validation'), 1):
            input, label = samples

            input = input.to(device)
            label = label.to(device)

            output = net(input)
            pred = fn_pred(output)

            # optim.zero_grad()

            mean_loss, variance_loss = criterion1(output, label)
            ce_loss = criterion2(output, label)
            loss = mean_loss + variance_loss + ce_loss

            # loss = fn_loss(output, label)
            acc = fn_acc(pred, label)
            if epoch == num_epoch:
                plt.scatter((pred.max(dim=1)[1]).cpu().numpy(), label.cpu().numpy(), c='blue')
                plt.plot(label.cpu().numpy(), label.cpu().numpy(), c='red')

            # loss.backward()

            # optim.step()

            loss_arr += [loss.item()]
            acc_arr += [acc.item()]

            print('TEST : BATCH %4d/%04d | LOSS : %.4f | ACC %.4f'
                  % (batch, num_batch_val, np.mean(loss_arr), np.mean(acc_arr)))
        writer_val.add_scalar('val_loss', np.mean(loss_arr), epoch)
        writer_val.add_scalar('val_acc', np.mean(acc_arr), epoch)

        # img_grid = torchvision.utils.make_grid(input[:3])
        # imshow(img_grid)
        print(' '.join('%5s' % label[:5]))
        preds_tensor = torch.max(pred, 1)
        print(' '.join('%5s' % preds_tensor.indices[:5]))

    save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

plt.show()
writer_train.close()
writer_val.close()


## 테스트 시작
import matplotlib.pyplot as plt

for epoch in range(1, num_epoch + 1):

    with torch.no_grad():
        net.eval()

        loss_arr = []
        acc_arr = []

        for batch, samples in enumerate(dataloaders.get('test'), 1):
            input, label = samples

            input = input.to(device)
            label = label.to(device)

            output = net(input)
            pred = fn_pred(output)

            # optim.zero_grad()

            mean_loss, variance_loss = criterion1(output, label)
            ce_loss = criterion2(output, label)
            loss = mean_loss + variance_loss + ce_loss

            # loss = fn_loss(output, label)
            acc = fn_acc(pred, label)
            if epoch == num_epoch:
                plt.scatter((pred.max(dim=1)[1]).cpu().numpy(), label.cpu().numpy(), c='blue')
                plt.plot(label.cpu().numpy(), label.cpu().numpy(), c='red')

            # loss.backward()

            # optim.step()

            loss_arr += [loss.item()]
            acc_arr += [acc.item()]

            print('TEST : BATCH %4d/%04d | LOSS : %.4f | ACC %.4f'
                  % (batch, num_batch_val, np.mean(loss_arr), np.mean(acc_arr)))
        writer_test.add_scalar('test_loss', np.mean(loss_arr), epoch)
        writer_test.add_scalar('test_acc', np.mean(acc_arr), epoch)

        # img_grid = torchvision.utils.make_grid(input[:3])
        # imshow(img_grid)
        print(' '.join('%5s' % label[:5]))
        preds_tensor = torch.max(pred, 1)
        print(' '.join('%5s' % preds_tensor.indices[:5]))

    save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

plt.show()
writer_test.close()