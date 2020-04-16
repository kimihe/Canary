# MNIST Model
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self, mnist=True):

        super(Net, self).__init__()
        if mnist:
            num_channels = 1
        else:
            num_channels = 3

        self.conv1 = nn.Conv2d(num_channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Training
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # batch_size = 64
    batch_size = 60
    # test_batch_size = 64
    test_batch_size = 60
    epochs = 5
    lr = 0.01
    momentum = 0.5
    seed = 1
    log_interval = 500
    save_model = False
    no_cuda = False

    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                        #    transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    args = {}
    args["log_interval"] = log_interval
    for epoch in range(1, epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")

    return model

# model = main()

########################################################################################################################

# Quantization Trainings

from collections import namedtuple
import torch
import torch.nn as nn

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])


def calcScaleZeroPoint(min_val, max_val,num_bits=8):
  # Calc Scale and zero point of next 
  qmin = 0.
  qmax = 2.**num_bits - 1.

  scale = (max_val - min_val) / (qmax - qmin)

  initial_zero_point = qmin - min_val / scale
  
  zero_point = 0
  if initial_zero_point < qmin:
      zero_point = qmin
  elif initial_zero_point > qmax:
      zero_point = qmax
  else:
      zero_point = initial_zero_point

  zero_point = int(zero_point)

  return scale, zero_point

def quantize_tensor(x, num_bits=8, min_val=None, max_val=None):
    
    if not min_val and not max_val: 
      min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2.**num_bits - 1.

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()
    
    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)

def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)


# Rework Forward pass of Linear and Conv Layers to support Quantisation
def quantizeLayer(x, layer, stat, scale_x, zp_x):
  # for both conv and linear layers

  # cache old values
  W = layer.weight.data
  B = layer.bias.data

  # quantise weights, activations are already quantised
  w = quantize_tensor(layer.weight.data) 
  b = quantize_tensor(layer.bias.data)

  layer.weight.data = w.tensor.float()
  layer.bias.data = b.tensor.float()

  # This is Quantisation Artihmetic
  scale_w = w.scale
  zp_w = w.zero_point
  scale_b = b.scale
  zp_b = b.zero_point
  
  scale_next, zero_point_next = calcScaleZeroPoint(min_val=stat['min'], max_val=stat['max'])

  # Preparing input by shifting
  X = x.float() - zp_x
  layer.weight.data = scale_x * scale_w*(layer.weight.data - zp_w)
  layer.bias.data = scale_b*(layer.bias.data + zp_b)

  # All int computation
  x = (layer(X)/ scale_next) + zero_point_next 
  
  # Perform relu too
  x = F.relu(x)

  # Reset weights for next forward pass
  layer.weight.data = W
  layer.bias.data = B
  
  return x, scale_next, zero_point_next


  # Get Max and Min Stats for Quantising Activations of Network.

  # Get Min and max of x tensor, and stores it

def updateStats(x, stats, key):
  max_val, _ = torch.max(x, dim=1)
  min_val, _ = torch.min(x, dim=1)
  
  
  if key not in stats:
    stats[key] = {"max": max_val.sum(), "min": min_val.sum(), "total": 1}
  else:
    stats[key]['max'] += max_val.sum().item()
    stats[key]['min'] += min_val.sum().item()
    stats[key]['total'] += 1
  
  return stats

# Reworked Forward Pass to access activation Stats through updateStats function
def gatherActivationStats(model, x, stats):

  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1')
  
  x = F.relu(model.conv1(x))

  x = F.max_pool2d(x, 2, 2)
  
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv2')
  
  x = F.relu(model.conv2(x))

  x = F.max_pool2d(x, 2, 2)

  x = x.view(-1, 4*4*50)
  
  stats = updateStats(x, stats, 'fc1')

  x = F.relu(model.fc1(x))
  
  stats = updateStats(x, stats, 'fc2')

  x = model.fc2(x)

  return stats

# Entry function to get stats of all functions.
def gatherStats(model, test_loader):
    # device = 'cuda'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    test_loss = 0
    correct = 0
    stats = {}
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            stats = gatherActivationStats(model, data, stats)
    
    final_stats = {}
    for key, value in stats.items():
      final_stats[key] = { "max" : value["max"] / value["total"], "min" : value["min"] / value["total"] }
    return final_stats


# Forward Pass for Quantised 
def quantForward(model, x, stats):
  
  # Quantise before inputting into incoming layers
  x = quantize_tensor(x, min_val=stats['conv1']['min'], max_val=stats['conv1']['max'])

  x, scale_next, zero_point_next = quantizeLayer(x.tensor, model.conv1, stats['conv2'], x.scale, x.zero_point)

  x = F.max_pool2d(x, 2, 2)
  
  x, scale_next, zero_point_next = quantizeLayer(x, model.conv2, stats['fc1'], scale_next, zero_point_next)

  x = F.max_pool2d(x, 2, 2)

  x = x.view(-1, 4*4*50)

  x, scale_next, zero_point_next = quantizeLayer(x, model.fc1, stats['fc2'], scale_next, zero_point_next)
  
  # Back to dequant for final layer
  x = dequantize_tensor(QTensor(tensor=x, scale=scale_next, zero_point=zero_point_next))
   
  x = model.fc2(x)

  return F.log_softmax(x, dim=1)






# Quantization Training
def train_quanti(args, model, device, train_loader, test_loader, optimizer, epoch, quanti):
    model.train()

    stats = gatherStats(model, test_loader)
    print(stats)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if quanti:
            output = quantForward(model, data, stats)
            # print("output = quantForward(model, data, stats)")
            # print("Quanti")
        else:
            output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
           

def main_quanti():
    # batch_size = 64
    batch_size = 60
    # test_batch_size = 64
    test_batch_size = 60
    epochs = 5
    lr = 0.01
    momentum = 0.5
    seed = 1
    log_interval = 500
    save_model = False
    no_cuda = False
    root = './data'
    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                        #    transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())


    args = {}
    args["log_interval"] = log_interval
    for epoch in range(1, epochs + 1):
        train_quanti(args, model, device, train_loader, test_loader, optimizer, epoch, quanti=True);
        test(args, model, device, test_loader)

    if (save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")

    return model

quanti_model = main_quanti()


'''
main_quanti
{'conv1': {'max': tensor(179.6297, device='cuda:0'), 'min': tensor(-27.0200, device='cuda:0')}, 'conv2': {'max': tensor(211.8767, device='cuda:0'), 'min': tensor(0., device='cuda:0')}, 'fc1': {'max': tensor(109.9994, device='cuda:0'), 'min': tensor(0., device='cuda:0')}, 'fc2': {'max': tensor(46.1353, device='cuda:0'), 'min': tensor(0., device='cuda:0')}}
Train Epoch: 1 [0/60000 (0%)]	Loss: 2.295738
Train Epoch: 1 [30000/60000 (50%)]	Loss: 0.074880
Test set: Average loss: 0.0785, Accuracy: 9767/10000 (98%)
{'conv1': {'max': tensor(179.6297, device='cuda:0'), 'min': tensor(-27.0200, device='cuda:0')}, 'conv2': {'max': tensor(399.7424, device='cuda:0'), 'min': tensor(0., device='cuda:0')}, 'fc1': {'max': tensor(576.9906, device='cuda:0'), 'min': tensor(0., device='cuda:0')}, 'fc2': {'max': tensor(329.1301, device='cuda:0'), 'min': tensor(0., device='cuda:0')}}
Train Epoch: 2 [0/60000 (0%)]	Loss: 0.055436
Train Epoch: 2 [30000/60000 (50%)]	Loss: 0.031404
Test set: Average loss: 0.0580, Accuracy: 9833/10000 (98%)
{'conv1': {'max': tensor(179.6297, device='cuda:0'), 'min': tensor(-27.0200, device='cuda:0')}, 'conv2': {'max': tensor(424.2269, device='cuda:0'), 'min': tensor(0., device='cuda:0')}, 'fc1': {'max': tensor(661.7817, device='cuda:0'), 'min': tensor(0., device='cuda:0')}, 'fc2': {'max': tensor(370.2840, device='cuda:0'), 'min': tensor(0., device='cuda:0')}}
Train Epoch: 3 [0/60000 (0%)]	Loss: 0.005177
Train Epoch: 3 [30000/60000 (50%)]	Loss: 0.044938
Test set: Average loss: 0.0482, Accuracy: 9856/10000 (99%)
{'conv1': {'max': tensor(179.6297, device='cuda:0'), 'min': tensor(-27.0200, device='cuda:0')}, 'conv2': {'max': tensor(441.1009, device='cuda:0'), 'min': tensor(0., device='cuda:0')}, 'fc1': {'max': tensor(696.3938, device='cuda:0'), 'min': tensor(0., device='cuda:0')}, 'fc2': {'max': tensor(382.8173, device='cuda:0'), 'min': tensor(0., device='cuda:0')}}
Train Epoch: 4 [0/60000 (0%)]	Loss: 0.066365
Train Epoch: 4 [30000/60000 (50%)]	Loss: 0.024540
Test set: Average loss: 0.0535, Accuracy: 9832/10000 (98%)
{'conv1': {'max': tensor(179.6297, device='cuda:0'), 'min': tensor(-27.0200, device='cuda:0')}, 'conv2': {'max': tensor(457.6660, device='cuda:0'), 'min': tensor(0., device='cuda:0')}, 'fc1': {'max': tensor(756.7455, device='cuda:0'), 'min': tensor(0., device='cuda:0')}, 'fc2': {'max': tensor(410.8990, device='cuda:0'), 'min': tensor(0., device='cuda:0')}}
Train Epoch: 5 [0/60000 (0%)]	Loss: 0.020183
Train Epoch: 5 [30000/60000 (50%)]	Loss: 0.035602
Test set: Average loss: 0.0445, Accuracy: 9859/10000 (99%)
'''