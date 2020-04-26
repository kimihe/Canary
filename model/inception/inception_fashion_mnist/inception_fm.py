import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import model
import time
import numpy as np



EPOCH = 30
BATCH_SIZE = 64
LR = 0.1
USE_GPU = True

if USE_GPU:
    gpu_statue = torch.cuda.is_available()
else:
    gpu_statue = False


def visualization():
    global time_p, tr_acc, ts_acc_top1, loss_p, sum_step, sum_loss, sum_acc, start_time, epoch
    test_acc_top1 = 0.
    net.eval()
    with torch.no_grad():
        for idex,(test_data, test_label) in enumerate(testloader):
            if gpu_statue:
                test_data= test_data.cuda()
            test_out = net(test_data)
            # acc = accuracy(test_out, test_label, topk=(1, 5))
            pred_ts = torch.max(test_out, 1)[1].cpu().data.squeeze()
            acc = (pred_ts==test_label).sum().item()/test_label.size(0)
            test_acc_top1 += acc
            # test_acc_top5 += acc[1]
        end_time = time.time() - start_time
        print('epoch: [{}/{}] | Tr_Loss: {:.4f} | TR_acc: {:.4f} | TS_acc_top1: {:.4f} | Time: {:.1f}'.format(epoch + 1, EPOCH,
                        sum_loss / (sum_step),sum_acc / (sum_step),test_acc_top1/(idex+1),end_time))
        # 可视化部分
        time_p.append(round(end_time, 4))
        tr_acc.append(round(sum_acc / sum_step, 4))
        ts_acc_top1.append(round(test_acc_top1/(idex+1), 4))
        # ts_acc_top5.append(test_acc_top5/(idex+1))
        loss_p.append(round(sum_loss / sum_step, 4))


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = LR * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Loading and normalizing FashionMNIST
train_transform = transforms.Compose(
    [transforms.Resize(32),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

test_transform = transforms.Compose(
    [transforms.Resize(32),
     transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=False, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                       download=False, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)

# Define a Convolution Neural Network
net = model.GoogLeNet()

if gpu_statue:
    net = net.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    print('*'*26, '使用gpu', '*'*26)
else:
    print('*'*26, '使用cpu', '*'*26)
    criterion = nn.CrossEntropyLoss()

# Define a optimizer
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.5, weight_decay=1e-4)

tr_acc, ts_acc_top1, loss_p, time_p = [], [], [], []

# Train and test the network
start_time = time.time()
net.train()
for epoch in range(EPOCH):
    sum_loss, sum_acc, sum_step = 0., 0., 0.
    adjust_learning_rate(optimizer, epoch)

    for i, (data, label) in enumerate(trainloader):
        if gpu_statue:
            data, label = data.cuda(), label.cuda()
        out = net(data)
        loss = criterion(out, label)
        sum_loss += loss.item()*len(label)
        pred_tr = torch.max(out, 1)[1]
        sum_acc += sum(pred_tr==label).item()
        sum_step += label.size(0)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('batch [{}/{}]'.format(i, len(trainloader)))
        # 可视化
        if (i + 1) % 180 == 0:
            visualization()
            sum_loss, sum_acc, sum_step = 0., 0., 0.,
            net.train()
    f = open('./result.txt', 'a')
    f.write('epoch: ' + str(epoch) + '\n' +
            'time: ' +str(time_p) + '\n' +
            'train_accuracy: ' + str(tr_acc) + '\n' +
            'train_loss: ' + str(loss_p) + '\n' +
            'test_accuracy_top1: ' + str(ts_acc_top1) + '\n'
            )
    f.close()
